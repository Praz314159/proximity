//! The profile: one level's envelope as certified brackets, with
//! the threshold grid and the monotone block enclosure for
//! off-grid queries.

use std::collections::BTreeMap;

use crate::error::{Error, Result};
use crate::math::enclosure::Lg;
use rug::float::Round;

/// Fold a term into a possibly-empty running sum.
pub(super) fn add_to(acc: Option<Lg>, term: Lg) -> Lg {
    match acc {
        Some(a) => a.add(&term),
        None => term,
    }
}

/// Default grid resolution: levels whose threshold range fits are
/// computed exactly (stride 1 — every gate-scale level); larger
/// levels are sampled at this many grid points, off-grid thresholds
/// enclosed by monotonicity. At the box this turns a quarter-hour
/// assembly into seconds for a fraction of a bit of slack.
pub const DEFAULT_RESOLUTION: u64 = 1 << 13;

/// A stored bracket together with the smallest threshold it remains
/// an enclosure at (by the profile's monotonicity in `t`).
#[derive(Clone, Copy)]
pub(super) struct BlockBracket {
    pub(super) lo: f64,
    pub(super) hi: f64,
    pub(super) valid_from: u64,
}

impl BlockBracket {
    /// The elementwise max of two blocks as a rebuilt bracket, with
    /// the meet of their validity floors — the deep charge's
    /// channel-max in one place.
    pub(super) fn max(&self, o: &BlockBracket) -> (Lg, u64) {
        (
            Lg::from_f64_bracket(self.lo.max(o.lo), self.hi.max(o.hi)),
            self.valid_from.max(o.valid_from),
        )
    }
}

/// One dimension's slice of a profile: brackets at an explicit grid
/// of thresholds (sorted, first at the curve, last at the level
/// length), stored as directed-rounded `f64` endpoint pairs (a
/// widening, so still an enclosure). Off-grid thresholds are
/// enclosed by monotonicity: the exact profile value is
/// non-increasing in `t`, so
/// `[lo(next grid point), hi(previous grid point)]` contains it.
/// The grid is dense (stride 1) near full agreement, where the
/// profile varies fastest per lattice step and where a ceiling
/// region a few points wide must not be smeared into a coarse
/// block, and uniform across the body; a level whose whole range
/// fits the resolution is entirely exact.
#[derive(Debug)]
pub(super) struct Row {
    grid: Vec<u64>,
    vals: Vec<(f64, f64)>,
}

impl Row {
    pub(super) fn t_min(&self) -> u64 {
        self.grid[0]
    }

    /// The grid index at or below `t`.
    fn idx_below(&self, t: u64) -> usize {
        self.grid.partition_point(|&g| g <= t) - 1
    }

    /// The bracket for `t` together with the smallest threshold the
    /// same bracket is valid at (the run floor). On-grid hits return
    /// the tight bracket, valid only at `t`; off-grid thresholds
    /// return the monotone block enclosure, valid from the grid
    /// point below.
    pub(super) fn bracket(&self, t: u64) -> BlockBracket {
        let i = self.idx_below(t);
        if self.grid[i] == t {
            let (lo, hi) = self.vals[i];
            BlockBracket {
                lo,
                hi,
                valid_from: t,
            }
        } else {
            let j = (i + 1).min(self.vals.len() - 1);
            BlockBracket {
                lo: self.vals[j].0,
                hi: self.vals[i].1,
                valid_from: self.grid[i],
            }
        }
    }
}

/// A profile envelope at one level, as certified brackets: for each
/// dimension `k` in the window, `eval(k, t)` encloses a value that
/// dominates `|Lam_t(w)|` for every word `w` on `mu_n` — provided the
/// hypotheses that built it hold (the interface data of its tower,
/// and nothing else).
#[derive(Debug)]
pub struct Profile {
    /// The level: words live on `mu_n`.
    pub n: u64,
    pub(super) rows: BTreeMap<u64, Row>,
}

impl Profile {
    /// The dimensions of the window.
    pub fn dims(&self) -> impl Iterator<Item = u64> + '_ {
        self.rows.keys().copied()
    }

    /// The threshold curve at dimension `k` (smallest asserted `t`).
    pub fn t_min(&self, k: u64) -> Option<u64> {
        self.rows.get(&k).map(Row::t_min)
    }

    /// The envelope value at `(k, t)`; an error outside the domain
    /// (the hypothesis is silent below the curve, and a threshold
    /// above `n` has no cell). On-grid thresholds return the direct
    /// bracket; off-grid ones the monotone block enclosure.
    pub fn eval(&self, k: u64, t: u64) -> Result<Lg> {
        let row = self
            .rows
            .get(&k)
            .ok_or_else(|| Error::OutOfRange(format!("dimension {k} outside the window")))?;
        if t < row.t_min() || t > self.n {
            return Err(Error::OutOfRange(format!(
                "threshold {t} outside [{}, {}]",
                row.t_min(),
                self.n
            )));
        }
        let b = row.bracket(t);
        Ok(Lg::from_f64_bracket(b.lo, b.hi))
    }

    /// The envelope read in the rows' coordinate: disagreement
    /// `z = n - t`.
    pub fn lg_at_disagreement(&self, k: u64, z: u64) -> Result<Lg> {
        if z >= self.n {
            return Err(Error::OutOfRange("need z < n".into()));
        }
        self.eval(k, self.n - z)
    }

    pub(super) fn insert(&mut self, k: u64, grid: Vec<u64>, vals: Vec<(f64, f64)>) {
        debug_assert_eq!(grid.len(), vals.len());
        self.rows.insert(k, Row { grid, vals });
    }
}

/// The threshold grid for `[t_min, t_max]` at the given resolution:
/// the whole range when it fits, else a uniform coarse body plus a
/// dense stride-1 tail of `res/2` points ending at `t_max` — full
/// agreement is where the profile varies fastest per lattice step.
pub(super) fn build_grid(t_min: u64, t_max: u64, res: u64) -> Vec<u64> {
    let len = t_max - t_min + 1;
    let res = res.max(4);
    if len <= res {
        return (t_min..=t_max).collect();
    }
    let dense = res / 2;
    let dense_from = t_max - dense + 1;
    let body_len = dense_from - t_min;
    let stride = body_len.div_ceil(res - dense).max(1);
    let mut g: Vec<u64> = (t_min..dense_from).step_by(stride as usize).collect();
    g.extend(dense_from..=t_max);
    g
}

/// A bracket as its stored form: `f64` endpoints, rounded outward.
pub(super) fn store(lg: &Lg) -> (f64, f64) {
    (
        lg.lo.to_f64_round(Round::Down),
        lg.hi.to_f64_round(Round::Up),
    )
}
