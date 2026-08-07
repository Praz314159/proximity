//! The envelope: the ceiling face's data layer.
//!
//! A *profile envelope* at level `n` is a function `E(n, k, t)` that
//! dominates the list profile of every word on `mu_n` — every
//! dimension `k` in its window, every threshold `t` above its curve
//! (notes v2, ch. 4, the profile envelope / descent hypothesis). The
//! objects here carry that hypothesis up the tower: [`Profile`] is one
//! level's envelope as certified brackets, [`step`] is the master
//! inequality applied as an operator (level `n` in, level `2n` out),
//! and [`assemble`] is the conditional form of the worst-case bound —
//! the induction from a floor level to the top, checking the
//! compatibility clause at every level and consuming interface data
//! `(D_b, D_c)` along the way. The rows next door
//! ([`super::ceiling`]) judge the assembled envelope against the
//! challenge budget; nothing here knows about thresholds or `eps*`.
//!
//! The master's three charges, per threshold `t` at level `s = 2n`
//! with rung `r = k + 1` and coverage threshold
//! `l* = ceil((n + k - 1)/3)`:
//! deep strata (`l >= l*`) consume the level-`n` profile through the
//! descent injection and anti-squaring; the middle band
//! (`kod <= l < l*`) is charged to `D_b` through the cores; the small
//! strata (`l < kod`) are charged to `D_c` through the cut. Every sum
//! is enclosed term by term; the small-strata tail is truncated only
//! under a certified bound (the divisors grow monotonically as `l`
//! falls, so the remaining terms are dominated by count times the
//! worst numerator over the smallest divisor).
//!
//! Interface data is pluggable ([`Interface`]). [`TrivialInterface`]
//! is the citable floor of the hierarchy: `D_c` by counting the whole
//! stratum, `D_b` by the pencil-agreement lemma's disjointness bound
//! (at most `(s - 2 kod)/a` members per core reach surplus `a`).
//! Real data — the engine's collision bounds and the per-prime
//! envelope at a certified prime — drops in by implementing the same
//! trait. Bases likewise: [`classical_base`] (the default — Johnson
//! above its threshold, interpolation below, exactly 1 at full
//! agreement) seeds the tower with everything classical coding
//! theory grants at the floor; [`assemble_levels_from`] is the seam
//! for sharper bases (the small-level form at a certified prime).
//! The point of running the tower with placeholder inputs is the
//! loss map: where the assembled number is weak tells the
//! compilation chapter where sharpness must come from — with the
//! classical base the two named walls are the sub-Johnson band at
//! the floor (nonempty above rate 1/4; the useful radius halves per
//! level until the band is closed) and the trivial `D_b` flood in
//! the middle charge.

use std::collections::{BTreeMap, BTreeSet};

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::math::enclosure::{lg_binom, Lg};
use rug::float::Round;
use rug::ops::{AddAssignRound, SubAssignRound};
use rug::{Float, Integer};

/// Working precision for rebuilt endpoints (matches the enclosure
/// module's own).
const PREC: u32 = 192;

/// Fold a term into an optional accumulator (empty sums stay empty).
fn acc_add(acc: Option<Lg>, term: &Lg) -> Option<Lg> {
    Some(match acc {
        Some(a) => a.add(term),
        None => term.clone(),
    })
}

/// Default grid resolution: levels whose threshold range fits are
/// computed exactly (stride 1 — every gate-scale level); larger
/// levels are sampled at this many grid points, off-grid thresholds
/// enclosed by monotonicity. At the box this turns a quarter-hour
/// assembly into seconds for a fraction of a bit of slack.
pub const DEFAULT_RESOLUTION: u64 = 1 << 13;

/// Certified interface data at a level (notes v2, ch. 4, the derived
/// interface): `D_b(a)` bounds, over all cores, the number of family
/// members with agreement surplus at least `a`; `D_c(l)` bounds the
/// cut stratum at pair count `l`. Both must hold for every word at
/// the level. Implementations receive the full cell context
/// `(s, k, ·)` so one object can serve every level of a tower.
///
/// Contract: `d_b` must be non-increasing in `a` (demanding more
/// surplus admits fewer members — true of any honest bound). The
/// step's coarse-grid enclosure rests on the master's right-hand side
/// being non-increasing in the threshold, and `d_b` is the only
/// data-supplied factor in that monotonicity; the step guards the
/// computed profile and refuses on a violation.
pub trait Interface: Sync {
    /// Bound on the core-summed family-member count at surplus `a >= 1`.
    fn d_b(&self, s: u64, k: u64, a: u64) -> Lg;
    /// Bound on the cut stratum at pair count `l < k/2`.
    fn d_c(&self, s: u64, k: u64, l: u64) -> Lg;
    /// Bound on `max(d_c(l'))` over `l' <= l` — the small-strata tail
    /// bound's numerator. The default scans, which is correct but
    /// linear in `l`; providers with structure should override.
    fn d_c_sup(&self, s: u64, k: u64, l: u64) -> Lg {
        let mut best = self.d_c(s, k, 0);
        for lp in 1..=l {
            let v = self.d_c(s, k, lp);
            if v.hi > best.hi {
                best = v;
            }
        }
        best
    }
}

/// The citable floor of the data hierarchy — no engine, no prime.
///
/// `D_c(l) = C(s/2, l)`: the whole stratum (every `l`-subset of the
/// slots counted as consistent). `D_b(a) = C(s/2, kod) * m` with
/// `m = floor((s - 2 kod)/a)` at odd `k` — the pencil-agreement
/// lemma's disjointness bound: the level sets of members reaching
/// surplus `a` are disjoint subsets, each of size at least `a`, of
/// the `s - 2 kod` available points — and `m = 1` at even `k`, where
/// the family through a core is a single interpolant.
pub struct TrivialInterface;

impl Interface for TrivialInterface {
    fn d_b(&self, s: u64, k: u64, a: u64) -> Lg {
        let kod = k / 2;
        let per_core = if k % 2 == 1 {
            (s - 2 * kod) / a.max(1)
        } else {
            1
        };
        lg_binom(s / 2, kod).mul(&Lg::from_u64(per_core.max(1)))
    }

    fn d_c(&self, s: u64, _k: u64, l: u64) -> Lg {
        lg_binom(s / 2, l)
    }

    // C(s/2, ·) increases up to its peak at s/4, so the prefix
    // maximum is the value at min(l, s/4)
    fn d_c_sup(&self, s: u64, _k: u64, l: u64) -> Lg {
        lg_binom(s / 2, l.min(s / 4))
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
struct Row {
    grid: Vec<u64>,
    vals: Vec<(f64, f64)>,
}

impl Row {
    fn t_min(&self) -> u64 {
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
    fn bracket(&self, t: u64) -> ((f64, f64), u64) {
        let i = self.idx_below(t);
        if self.grid[i] == t {
            (self.vals[i], t)
        } else {
            let j = (i + 1).min(self.vals.len() - 1);
            ((self.vals[j].0, self.vals[i].1), self.grid[i])
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
    rows: BTreeMap<u64, Row>,
}

impl Profile {
    /// The dimensions of the window.
    pub fn dims(&self) -> impl Iterator<Item = u64> + '_ {
        self.rows.keys().copied()
    }

    /// The threshold curve at dimension `k` (smallest asserted `t`).
    pub fn t_min(&self, k: u64) -> Option<u64> {
        self.rows.get(&k).map(|r| r.t_min())
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
        let ((lo, hi), _) = row.bracket(t);
        Ok(Lg {
            lo: Float::with_val(PREC, lo),
            hi: Float::with_val(PREC, hi),
        })
    }

    /// The envelope read in the rows' coordinate: disagreement
    /// `z = n - t`.
    pub fn lg_at_disagreement(&self, k: u64, z: u64) -> Result<Lg> {
        if z >= self.n {
            return Err(Error::OutOfRange("need z < n".into()));
        }
        self.eval(k, self.n - z)
    }

    fn insert(&mut self, k: u64, grid: Vec<u64>, vals: Vec<(f64, f64)>) {
        debug_assert_eq!(grid.len(), vals.len());
        self.rows.insert(k, Row { grid, vals });
    }
}

/// The threshold grid for `[t_min, t_max]` at the given resolution:
/// the whole range when it fits, else a uniform coarse body plus a
/// dense stride-1 tail of `res/2` points ending at `t_max` — full
/// agreement is where the profile varies fastest per lattice step.
fn build_grid(t_min: u64, t_max: u64, res: u64) -> Vec<u64> {
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

fn store(lg: &Lg) -> (f64, f64) {
    (
        lg.lo.to_f64_round(Round::Down),
        lg.hi.to_f64_round(Round::Up),
    )
}

/// The interpolation base: `E(n0, k, t) = C(n0, k)` for `t >= k + 1`
/// — a member agreeing on at least the rung is the interpolant of `w`
/// on any `k`-subset of its agreement set, so the map to that subset
/// is injective and the list is at most the number of subsets. The
/// crudest citable base; the base chapter's classical seeding will
/// replace it through the same type.
pub fn interpolation_base(n0: u64, dims: &BTreeSet<u64>) -> Result<Profile> {
    let mut prof = Profile {
        n: n0,
        rows: BTreeMap::new(),
    };
    for &k in dims {
        if k == 0 || k + 1 > n0 {
            return Err(Error::OutOfRange(format!(
                "base dimension {k} needs 1 <= k < n0 = {n0}"
            )));
        }
        let val = store(&lg_binom(n0, k));
        prof.insert(k, (k + 1..=n0).collect(), vec![val; (n0 - k) as usize]);
    }
    Ok(prof)
}

/// The classical base: at each threshold the smaller of the
/// interpolation bound `C(n0, k)` and the agreement-form Johnson
/// bound
/// `floor( n (t - k + 1) / (t^2 - n (k - 1)) )`, valid once
/// `t^2 > n (k - 1)` — the quadratic argument: `m` members agreeing
/// on `>= t` points each, pairwise on `<= k - 1`, force
/// `m t (m t - n) / n <= m (m - 1)(k - 1)` by convexity, and solving
/// for `m` gives the display. At `t = n` it reads exactly 1, so the
/// tower's loss-free transport carries a one-word list to the top.
/// This is the ch. 4 base section's classical seeding as code (the
/// notes' opening chapter recovers the same count as the Corradi
/// degeneration); what it cannot cover is the band between the
/// coverage threshold `(1 + 2 rho) / 3` and the Johnson fraction
/// `sqrt(rho)` — nonempty exactly above rate 1/4 — where the
/// interpolation fallback still floods. Closing that band is the
/// base statements' remaining content, not a defect of this
/// constructor.
pub fn classical_base(n0: u64, dims: &BTreeSet<u64>) -> Result<Profile> {
    let mut prof = Profile {
        n: n0,
        rows: BTreeMap::new(),
    };
    for &k in dims {
        if k == 0 || k + 1 > n0 {
            return Err(Error::OutOfRange(format!(
                "base dimension {k} needs 1 <= k < n0 = {n0}"
            )));
        }
        let interp = Integer::from(Integer::binomial_u(n0 as u32, k as u32));
        let vals: Vec<(f64, f64)> = (k + 1..=n0)
            .map(|t| {
                let mut best = interp.clone();
                if t * t > n0 * (k - 1) {
                    let johnson = Integer::from(n0 * (t - k + 1) / (t * t - n0 * (k - 1)));
                    if johnson < best {
                        best = johnson;
                    }
                }
                store(&Lg::from_integer(&best.max(Integer::from(1))))
            })
            .collect();
        prof.insert(k, (k + 1..=n0).collect(), vals);
    }
    Ok(prof)
}

/// The channel dimensions of `k`: the fold splits degree-below-`k`
/// polynomials into even and odd parts of dimensions `ceil(k/2)` and
/// `floor(k/2)`.
pub fn channel_dims(k: u64) -> (u64, u64) {
    (k.div_ceil(2), k / 2)
}

/// The master inequality as an operator: an envelope at level
/// `prev.n`, plus interface data at level `s = 2 prev.n`, yields an
/// envelope at level `s` for the given dimensions. The compatibility
/// clause is enforced, not assumed: the previous window must contain
/// each dimension's channel pair, and the previous curve must reach
/// the coverage threshold — a violation is an error naming the cell.
pub fn step(
    prev: &Profile,
    dims: &BTreeSet<u64>,
    data: &dyn Interface,
    res: u64,
) -> Result<Profile> {
    let n = prev.n;
    let s = 2 * n;
    let mut out = Profile {
        n: s,
        rows: BTreeMap::new(),
    };
    for &k in dims {
        if k < 2 || k > s - 2 {
            return Err(Error::OutOfRange(format!(
                "step dimension {k} needs 2 <= k <= s - 2 at s = {s}"
            )));
        }
        let (kev, kod) = channel_dims(k);
        let r = k + 1;
        let lstar = (n + k - 1).div_ceil(3);
        // the compatibility clause of the master theorem
        for kc in [kev, kod] {
            match prev.t_min(kc) {
                None => {
                    return Err(Error::Unsupported(format!(
                        "compatibility: dimension {kc} missing from the level-{n} window"
                    )))
                }
                Some(curve) if curve > lstar => {
                    return Err(Error::Unsupported(format!(
                        "compatibility: curve {curve} at dimension {kc} does not reach \
                         the coverage threshold {lstar} of level {s}, dimension {k}"
                    )))
                }
                Some(_) => {}
            }
        }

        // deep strata: the level-below profile is block-constant on
        // its grid, so the suffix over l in [lstar, n] compresses to
        // runs — stretches where both channel brackets are constant —
        // each contributing width times its max. A run's record keeps
        // the sum strictly above it, so any lower limit is answered
        // with one partial-width multiply.
        struct DeepRun {
            lo: u64,
            hi: u64,
            m: Lg,
            above: Option<Lg>,
        }
        let mut deep_runs: Vec<DeepRun> = Vec::new();
        if lstar <= n {
            let row_e = prev.rows.get(&kev).expect("checked above");
            let row_o = prev.rows.get(&kod).expect("checked above");
            let mut hi_l = n;
            let mut above: Option<Lg> = None;
            loop {
                // each channel's bracket comes with the lowest
                // threshold it stays valid at; the run is their meet
                let (be, floor_e) = row_e.bracket(hi_l);
                let (bo, floor_o) = row_o.bracket(hi_l);
                let m = Lg {
                    lo: Float::with_val(PREC, be.0.max(bo.0)),
                    hi: Float::with_val(PREC, be.1.max(bo.1)),
                };
                let lo_l = floor_e.max(floor_o).max(lstar);
                let width = hi_l - lo_l + 1;
                let contribution = Lg::from_u64(width).mul(&m);
                let next_above = acc_add(above.clone(), &contribution);
                deep_runs.push(DeepRun {
                    lo: lo_l,
                    hi: hi_l,
                    m,
                    above,
                });
                above = next_above;
                if lo_l == lstar {
                    break;
                }
                hi_l = lo_l - 1;
            }
        }
        // runs are built from l = n downward, so they sit in the vec
        // in descending order; binary-search the one containing l0
        let deep_at = |l0: u64| -> Lg {
            let j = deep_runs.partition_point(|run| run.lo > l0);
            let run = &deep_runs[j];
            debug_assert!(run.lo <= l0 && l0 <= run.hi);
            let partial = Lg::from_u64(run.hi - l0 + 1).mul(&run.m);
            match &run.above {
                Some(a) => a.add(&partial),
                None => partial,
            }
        };

        // middle band: the suffix of 1/C(l, kod) over l in [lo, lstar).
        // Small ranges get the exact term-by-term array; large ones
        // the telescoping identity sum 1/C(l, m) over l >= l0 =
        // m/((m-1) C(l0-1, m-1)), enclosed as [first term, closed
        // form] — the width is lg(l0/(m-1)), a fraction of a bit at
        // rate 1/2, for two binomials per query instead of a
        // level-length precomputation.
        let mid_len = lstar.saturating_sub(kod) as usize;
        let mid_exact: Option<Vec<Lg>> = if mid_len <= 1 << 12 {
            let mut suffix: Vec<Lg> = Vec::with_capacity(mid_len);
            for i in 0..mid_len {
                let l = lstar - 1 - i as u64;
                let inv = Lg::zero().div(&lg_binom(l, kod));
                suffix.push(match suffix.last() {
                    Some(acc) => acc.add(&inv),
                    None => inv,
                });
            }
            Some(suffix)
        } else {
            None
        };
        let mid_at = |l0: u64| -> Lg {
            match &mid_exact {
                Some(suffix) => suffix[(lstar - 1 - l0) as usize].clone(),
                None => {
                    let hi = Lg::from_u64(kod)
                        .div(&Lg::from_u64(kod - 1))
                        .div(&lg_binom(l0 - 1, kod - 1));
                    let lo = Lg::zero().div(&lg_binom(l0, kod));
                    Lg {
                        lo: lo.lo,
                        hi: hi.hi,
                    }
                }
            }
        };

        // small strata: the cut prices of the term-capped window are
        // threshold-independent for most cells — cache them
        const CAP: u64 = 16;
        let dc_lo = kod.saturating_sub(CAP + 2);
        let dc_cache: Vec<(Lg, Lg)> = (dc_lo..kod)
            .map(|l| (data.d_c(s, k, l), data.d_c_sup(s, k, l)))
            .collect();
        let dc_at = |l: u64| -> Lg {
            if l >= dc_lo {
                dc_cache[(l - dc_lo) as usize].0.clone()
            } else {
                data.d_c(s, k, l)
            }
        };
        let dc_sup_at = |l: u64| -> Lg {
            if l >= dc_lo {
                dc_cache[(l - dc_lo) as usize].1.clone()
            } else {
                data.d_c_sup(s, k, l)
            }
        };

        let g = build_grid(r, s, res);
        let cell = |&t: &u64| -> Result<(f64, f64)> {
            let lmin = t.saturating_sub(n);
            let mut total: Option<Lg> = None;

            // charge 1: descent + anti-squaring into the level-n profile
            let lo1 = lmin.max(lstar);
            if lo1 <= n {
                total = acc_add(total, &deep_at(lo1));
            }

            // charge 2: cores to family members, priced by D_b
            let lo2 = lmin.max(kod);
            if lo2 < lstar {
                let a = t - 2 * kod;
                let term = data.d_b(s, k, a).mul(&mid_at(lo2));
                total = acc_add(total, &term);
            }

            // charge 3: canonical subsets to the cut, priced by D_c,
            // summed downward with a certified tail bound. The tail —
            // count times the worst remaining numerator over the
            // smallest remaining divisor (divisors grow as l falls) —
            // closes the sum either when it is provably negligible or
            // at the term cap. Near t = r the divisors grow slowly and
            // the cap bites, but there the remaining terms are
            // near-equal and the bound is tight; everywhere else the
            // terms decay fast and the tail is dust. Cost per
            // threshold stays O(cap), keeping a level linear in its
            // length.
            if lmin < kod {
                let mut acc: Option<Lg> = None;
                let mut l = kod - 1;
                loop {
                    let term = dc_at(l).div(&lg_binom(t - 2 * l, r - 2 * l));
                    acc = acc_add(acc, &term);
                    if l == lmin {
                        break;
                    }
                    let count = l - lmin;
                    let cur = acc.as_ref().expect("nonempty");
                    let mut tail_hi = Lg::from_u64(count).hi;
                    tail_hi.add_assign_round(&dc_sup_at(l - 1).hi, Round::Up);
                    tail_hi.sub_assign_round(
                        &lg_binom(t - 2 * (l - 1), r - 2 * (l - 1)).lo,
                        Round::Up,
                    );
                    let mut margin = cur.hi.clone();
                    margin -= 80u32;
                    if tail_hi <= margin || kod - 1 - l >= CAP {
                        let tail = Lg {
                            lo: Float::with_val(PREC, f64::NEG_INFINITY),
                            hi: tail_hi,
                        };
                        let bounded = cur.add(&tail);
                        acc = Some(Lg {
                            lo: cur.lo.clone(),
                            hi: bounded.hi,
                        });
                        break;
                    }
                    l -= 1;
                }
                total = acc_add(total, &acc.expect("nonempty range"));
            }

            let total = total.ok_or_else(|| {
                Error::Unsupported(format!("no charge covers cell ({s}, {k}, {t})"))
            })?;
            Ok(store(&total))
        };
        let vals: Vec<(f64, f64)> = g.par_iter().map(cell).collect::<Result<Vec<_>>>()?;
        // the grid's enclosure rests on the exact right-hand side
        // being non-increasing in t; a bracket that certifiably
        // rises means the data broke its d_b contract
        for w in vals.windows(2) {
            if w[1].0 > w[0].1 {
                return Err(Error::Unsupported(format!(
                    "interface data violates monotonicity at level {s}, dimension {k}"
                )));
            }
        }
        out.insert(k, g, vals);
    }
    Ok(out)
}

/// The conditional form of the worst-case bound, as computation: from
/// the interpolation base at floor level `n0`, apply the step once
/// per level up to `s`, evaluating only the dimensions the top
/// dimension `k` folds down to. The result is the envelope at level
/// `s`, valid under the interface data supplied — the assumptions of
/// the conditional corollary, checked where checkable (the
/// compatibility clause at every level) and priced by `data`
/// everywhere else.
pub fn assemble(s: u64, k: u64, n0: u64, data: &dyn Interface, res: u64) -> Result<Profile> {
    Ok(assemble_levels(s, k, n0, data, res)?
        .pop()
        .expect("at least the base level"))
}

/// The dimension windows of a tower, base level first: each level
/// needs its dimensions' channel pairs one level below.
fn windows(s: u64, k: u64, n0: u64) -> Result<Vec<BTreeSet<u64>>> {
    if !s.is_power_of_two() || !n0.is_power_of_two() || n0 < 4 || s < n0 {
        return Err(Error::OutOfRange(
            "need power-of-two levels with 4 <= n0 <= s".into(),
        ));
    }
    let d = (s / n0).ilog2() as usize;
    let mut down: Vec<BTreeSet<u64>> = vec![BTreeSet::from([k])];
    for _ in 0..d {
        let mut below = BTreeSet::new();
        for &kc in down.last().expect("nonempty") {
            let (e, o) = channel_dims(kc);
            below.insert(e);
            below.insert(o);
        }
        down.push(below);
    }
    down.reverse();
    Ok(down)
}

/// [`assemble`], keeping every intermediate level (base first, top
/// last) — the loss map's instrument: where the tower's numbers turn
/// weak locates which input the compilation chapter must sharpen.
/// Seeds from the classical base; [`assemble_levels_from`] accepts
/// any base profile.
pub fn assemble_levels(
    s: u64,
    k: u64,
    n0: u64,
    data: &dyn Interface,
    res: u64,
) -> Result<Vec<Profile>> {
    let w = windows(s, k, n0)?;
    assemble_levels_from(classical_base(n0, &w[0])?, s, k, data, res)
}

/// The tower from a caller-supplied base profile — the seam for base
/// providers beyond the built-in constructors (the small-level form
/// at a certified prime, when it lands, plugs in here). The base
/// must sit at a level dividing `s` by a power of two and contain
/// the bottom window's dimensions; the step's compatibility clause
/// checks the curves.
pub fn assemble_levels_from(
    base: Profile,
    s: u64,
    k: u64,
    data: &dyn Interface,
    res: u64,
) -> Result<Vec<Profile>> {
    let w = windows(s, k, base.n)?;
    for &kc in &w[0] {
        if base.t_min(kc).is_none() {
            return Err(Error::OutOfRange(format!(
                "base profile is missing dimension {kc}"
            )));
        }
    }
    let mut levels = vec![base];
    for win in &w[1..] {
        let next = step(levels.last().expect("nonempty"), win, data, res)?;
        levels.push(next);
    }
    Ok(levels)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::{Integer, Rational};

    fn binom(a: u64, b: u64) -> Integer {
        if b > a {
            return Integer::from(0);
        }
        let mut v = Integer::from(1);
        for i in 0..b {
            v *= a - i;
            v /= i + 1;
        }
        v
    }

    /// One full step against exact rational arithmetic: level 8 to 16
    /// at dimension 7 with trivial data, every threshold. Exercises
    /// all three charges and the suffix plumbing.
    #[test]
    fn step_encloses_exact_rational_at_16_7() {
        let (n, k) = (8u64, 7u64);
        let s = 2 * n;
        let (kev, kod) = channel_dims(k); // (4, 3)
        let r = k + 1;
        let lstar = (n + k - 1).div_ceil(3); // 5
        let base = interpolation_base(n, &BTreeSet::from([kev, kod])).expect("base");
        let prof = step(&base, &BTreeSet::from([k]), &TrivialInterface, u64::MAX).expect("step");
        for t in r..=s {
            let lmin = t.saturating_sub(n);
            let mut exact = Rational::new();
            for _l in lmin.max(lstar)..=n {
                // base values: C(8, 4) vs C(8, 3), max is C(8, 4) = 70
                exact += Rational::from(binom(n, kev).max(binom(n, kod)));
            }
            if lmin.max(kod) < lstar {
                let a = t - 2 * kod;
                let db = Rational::from(binom(n, kod) * ((s - 2 * kod) / a));
                for l in lmin.max(kod)..lstar {
                    exact += db.clone() / Rational::from(binom(l, kod));
                }
            }
            for l in lmin..kod {
                exact += Rational::from((binom(n, l), binom(t - 2 * l, r - 2 * l)));
            }
            let got = prof.eval(k, t).expect("in domain");
            let want = exact.to_f64().log2();
            let lo = got.lo.to_f64_round(Round::Down);
            let hi = got.hi.to_f64_round(Round::Up);
            assert!(
                lo <= want && want <= hi,
                "t = {t}: [{lo}, {hi}] vs exact {want}"
            );
            assert!(hi - lo < 0.01, "t = {t}: bracket too wide");
        }
    }

    /// The assembled tower dominates the measured record at the base
    /// cell: the census maximum 2674 at (32, 15, 17) is a true list,
    /// so any admissible envelope must certifiably exceed it.
    #[test]
    fn tower_dominates_the_record_cell() {
        let prof = assemble(32, 15, 8, &TrivialInterface, DEFAULT_RESOLUTION).expect("tower");
        let at = prof.eval(15, 17).expect("in domain");
        let record = (2674f64).log2();
        assert!(
            at.lo.to_f64_round(Round::Down) >= record,
            "envelope must dominate the measured record"
        );
    }

    /// The compatibility clause is a hard error: at dimension 13 over
    /// level 16, the channel curve (8) overshoots the coverage
    /// threshold (7), and the step must refuse rather than assert an
    /// unbacked bound.
    #[test]
    fn compatibility_violation_is_an_error() {
        let err = assemble(16, 13, 8, &TrivialInterface, DEFAULT_RESOLUTION).unwrap_err();
        assert!(
            err.to_string().contains("compatibility"),
            "wrong error: {err}"
        );
    }

    /// The domain is exactly the stated window [r, s] at the top
    /// dimension, silent outside it, and every asserted value is a
    /// finite bracket.
    #[test]
    fn profile_domain_is_the_stated_window() {
        let prof = assemble(32, 15, 8, &TrivialInterface, DEFAULT_RESOLUTION).expect("tower");
        assert_eq!(prof.t_min(15), Some(16));
        assert!(prof.eval(15, 15).is_err());
        assert!(prof.eval(15, 32).is_ok());
        assert!(prof.eval(15, 33).is_err());
        assert!(prof.eval(14, 20).is_err(), "dimension outside window");
        for t in 16..=32 {
            let v = prof.eval(15, t).expect("in domain");
            assert!(v.hi.is_finite());
        }
    }

    /// The classical base pins: at (8, 4) the sharp agreement-form
    /// Johnson bound n(t - k + 1)/(t^2 - n(k - 1)) reads 16, 2, 1, 1
    /// across t = 5..8 (all under the interpolation 70), and full
    /// agreement is exactly one word — zero bits.
    #[test]
    fn classical_base_pins() {
        let base = classical_base(8, &BTreeSet::from([4, 3])).expect("base");
        let want = [(5u64, 4.0), (6, 1.0), (7, 0.0), (8, 0.0)];
        for &(t, bits) in &want {
            let v = base.eval(4, t).expect("in domain");
            assert!(
                (v.hi.to_f64() - bits).abs() < 1e-9,
                "t = {t}: {} vs {bits}",
                v.hi.to_f64()
            );
        }
        // dimension 3: interpolation at t = 4, Johnson from t = 5
        assert!((base.eval(3, 4).unwrap().hi.to_f64() - (56f64).log2()).abs() < 1e-9);
        assert!((base.eval(3, 5).unwrap().hi.to_f64() - 1.0).abs() < 1e-9);
    }

    /// Full agreement transports one word: the classical base reads
    /// exactly 1 at t = n0, and the tower's loss-free deep charge
    /// carries zero bits to the top unchanged.
    #[test]
    fn full_agreement_transports_one_word() {
        let prof = assemble(
            1 << 12,
            (1 << 11) - 1,
            64,
            &TrivialInterface,
            DEFAULT_RESOLUTION,
        )
        .expect("tower");
        let v = prof.eval((1 << 11) - 1, 1 << 12).expect("in domain");
        assert!(v.hi.to_f64() < 1e-9, "got {} bits", v.hi.to_f64());
    }

    /// With the classical base the ceiling exists at the reduced box:
    /// a positive certified radius under the 2^-128 budget — the
    /// first nonvacuous ceiling row. The value is tiny (the
    /// sub-Johnson band halves the useful radius per level); the
    /// assertion is existence, not strength.
    #[test]
    fn classical_ceiling_exists_at_reduced_box() {
        use rug::ops::Pow;
        let total = 1u64 << 12;
        let k = total / 2 - 1;
        let ext = Integer::from(crate::field::named::KOALABEAR).pow(6);
        let prof = assemble(total, k, 64, &TrivialInterface, DEFAULT_RESOLUTION).expect("tower");
        let row = crate::soundness::ceiling::list_ceiling_row(
            1,
            total,
            total - k - 1,
            &ext,
            -128.0,
            |z| prof.lg_at_disagreement(k, z),
        )
        .expect("a positive ceiling");
        assert!(row.z_star >= 5, "z* = {}", row.z_star);
    }

    /// The coarse grid encloses the exact computation: a stride-1
    /// tower against an aggressively coarsened one at (512, 255),
    /// spot-checked across the domain. The coarse bracket must
    /// contain the exact bracket (up to f64 store jitter) — the
    /// monotone block enclosure is a widening, never a shift.
    #[test]
    fn coarse_grid_encloses_exact() {
        let exact = assemble(512, 255, 8, &TrivialInterface, u64::MAX).expect("exact");
        let coarse = assemble(512, 255, 8, &TrivialInterface, 32).expect("coarse");
        for t in (256..=512).step_by(7) {
            let e = exact.eval(255, t).expect("in domain");
            let c = coarse.eval(255, t).expect("in domain");
            let (elo, ehi) = (e.lo.to_f64_round(Round::Down), e.hi.to_f64_round(Round::Up));
            let (clo, chi) = (c.lo.to_f64_round(Round::Down), c.hi.to_f64_round(Round::Up));
            assert!(
                clo <= elo + 1e-6 && chi >= ehi - 1e-6,
                "t = {t}: coarse [{clo}, {chi}] vs exact [{elo}, {ehi}]"
            );
        }
    }

    /// A deeper tower with a longer small-strata range (s = 64 at
    /// rate 1/2, kod = 15) assembles finite ordered brackets — the
    /// smoke over the truncation path; the (16, 7) oracle above is
    /// the exactness gate.
    #[test]
    fn small_strata_truncation_smoke() {
        let prof = assemble(64, 31, 8, &TrivialInterface, DEFAULT_RESOLUTION).expect("tower");
        // spot thresholds across the domain
        for t in [32u64, 40, 48, 56, 64] {
            let v = prof.eval(31, t).expect("in domain");
            assert!(v.hi.is_finite() && v.lo.is_finite());
            assert!(v.hi >= v.lo);
        }
    }
}
