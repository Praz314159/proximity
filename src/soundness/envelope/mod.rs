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
//! trait. Bases likewise: [`analytic_base`] (the default — the ch. 4
//! base section's unconditional statement: interpolation, sharp
//! Johnson, and the ownership shower bound, pointwise min) seeds the
//! tower; [`assemble_levels_from`] is the seam for sharper bases
//! (the certified floor values of the base section's companion
//! statement, when the register lands). With the analytic base the
//! floor holds no flood at small `n0`.
//!
//! THE MEASURED WALL (re-affirmed, SPLICE round 17b). Every
//! provider's gauge pins at the Johnson clamp because the mid band
//! is priced through `D_b` at `l0 = kod`, where no data can be
//! small. A "band-collapse" charge that briefly certified the box
//! at `delta = 1/3` was RETRACTED the same day: its fiber-fold
//! instantiation of thm:tc-ft2 misread the theorem's joint list
//! (see the retraction note at `Charges::rhs`), and the standalone
//! verifier gate_band_collapse.py refuted it with planted words.
//! Beyond-Johnson radii remain gated on genuine mid-band data —
//! the deep-capacity program (STAR-MAXIMUM-GENERAL.md secs. 12,
//! 14) is the standing route.

mod base;
mod charges;
mod interface;
mod profile;
#[cfg(test)]
mod tests;

pub use base::{analytic_base, interpolation_base};
pub use charges::channel_dims;
pub use interface::{
    Interface, RigidityInterface, ShowerInterface, SpeciesInterface, StarInterface,
    TrivialInterface,
};
pub use profile::{Profile, DEFAULT_RESOLUTION};

use base::analytic_brackets;
use charges::{Cell, CutCharge, DeepCharge, MidSuffix};
use profile::{build_grid, store};

use std::collections::{BTreeMap, BTreeSet};

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::math::enclosure::Lg;

/// The master inequality's machinery for one `(level, dimension)`:
/// the three charges with their precomputed structures, evaluated
/// per threshold by [`Charges::rhs`]. Construction performs the
/// dimension-range check and the compatibility clause — a violation
/// is an error naming the cell, never a silent assumption.
struct Charges<'a> {
    cell: Cell,
    data: &'a dyn Interface,
    deep: DeepCharge,
    mid: MidSuffix,
    cut: CutCharge,
}

impl<'a> Charges<'a> {
    /// How many of the lowest middle-band strata are consumed
    /// through the per-stratum datum before the aggregate's closed
    /// form takes over. Small: the ownership divisor `C(l, kod)`
    /// makes higher strata negligible, and the box's `l*` is far too
    /// large to sum term by term.
    const SPECIES_WINDOW: u64 = 4;

    fn build(prev: &Profile, k: u64, data: &'a dyn Interface) -> Result<Self> {
        let cell = Cell::new(prev.n, k);
        if k < 2 || k > cell.s - 2 {
            return Err(Error::OutOfRange(format!(
                "step dimension {k} needs 2 <= k <= s - 2 at s = {}",
                cell.s
            )));
        }
        // the compatibility clause of the master theorem
        for kc in [cell.kev, cell.kod] {
            match prev.t_min(kc) {
                None => {
                    return Err(Error::Unsupported(format!(
                        "compatibility: dimension {kc} missing from the level-{} window",
                        cell.n
                    )))
                }
                Some(curve) if curve > cell.lstar => {
                    return Err(Error::Unsupported(format!(
                        "compatibility: curve {curve} at dimension {kc} does not reach \
                         the coverage threshold {} of level {}, dimension {k}",
                        cell.lstar, cell.s
                    )))
                }
                Some(_) => {}
            }
        }
        Ok(Charges {
            deep: DeepCharge::build(prev, &cell),
            mid: MidSuffix::build(&cell),
            cut: CutCharge::build(&cell, data),
            cell,
            data,
        })
    }

    /// Charge 2 at threshold `t`, `None` when the band is empty.
    ///
    /// The first [`Self::SPECIES_WINDOW`] strata are consumed through
    /// the PER-STRATUM datum [`Interface::d_b_at`] and the rest
    /// through the aggregate's closed form. The split is exact: with
    /// the default `d_b_at` (the ownership division) the two forms
    /// sum to the old single-term product, so shipped providers are
    /// unaffected. It exists because the strata are not alike — the
    /// dominant term is `l = kod`, where the ownership divisor is
    /// one, and that is exactly the PAIR-POOR species; the strata
    /// above it are pair-rich and already discounted by
    /// `C(l, kod)`. A provider that prices the species separately
    /// (SpeciesInterface) is read here and nowhere else.
    fn mid_at(&self, t: u64) -> Option<Lg> {
        let cell = &self.cell;
        let l0 = t.saturating_sub(cell.n).max(cell.kod);
        if l0 >= cell.lstar {
            return None;
        }
        let a = t - 2 * cell.kod;
        // strata above the far-branch pair limit are empty (a
        // theorem, Cell::far_pair_limit); the near branch is
        // floored at one in `rhs`
        let top = cell.lstar.min(cell.far_pair_limit(t) + 1);
        if l0 >= top {
            return None;
        }
        let hi = (l0 + Self::SPECIES_WINDOW).min(top);
        let mut acc: Option<Lg> = None;
        for l in l0..hi {
            let term = self.data.d_b_at(cell.s, cell.k, l, a);
            acc = Some(match acc {
                Some(x) => x.add(&term),
                None => term,
            });
        }
        if hi < top {
            let tail = self
                .data
                .d_b(cell.s, cell.k, a)
                .mul(&self.mid.suffix_from(cell, hi));
            acc = Some(match acc {
                Some(x) => x.add(&tail),
                None => tail,
            });
        }
        acc
    }

    /// [RETRACTED 2026-08-12, SPLICE round 17b] A "band collapse"
    /// candidate briefly lived here: it priced the whole sub-`l*`
    /// range by the child rows at fiber agreement `F0 = t - l* + 1`
    /// via thm:tc-ft2, and certified the box at `delta = 1/3`. The
    /// standalone verifier (gate_band_collapse.py) REFUTED it with
    /// planted words at every audited cell: tc-ft2's joint list
    /// `J_F` takes channel pairs with a COMMON F-fiber witness of
    /// PLAIN agreement in both channels (07_topcut.tex), and an
    /// unfolded member's plain both-channel agreement is exactly
    /// its PAIR count `l` — the deep charge's `F = l`
    /// instantiation, valid only at `l >= l*`. Single-fiber
    /// (marked) agreement never enters `J_F`; there is no
    /// fiber-fold instantiation, and the sub-`l*` strata remain
    /// priced by `D_b`/`D_c` data alone.
    ///
    /// The master's right-hand side at threshold `t`: the minimum
    /// over split candidates of the three charges. The split between
    /// the middle band and the deep range is a FREE parameter
    /// `lambda in [l*, n]` — the core charge holds at every
    /// `l >= kod` and the round-9 nesting + FT-2 chain holds at
    /// every `F >= l*` — and the candidates bracket its extremes:
    /// `lambda = l*` (the ch. 4 form) and `lambda = n` (the whole
    /// middle range through cores, deep reduced to the fully-paired
    /// class). Each candidate is non-increasing in `t`, so the min
    /// preserves the grid's enclosure contract.
    fn rhs(&self, t: u64) -> Result<Lg> {
        let cell = &self.cell;
        let small = self.cut.at(cell, self.data, t);
        let classic = [self.deep.at(cell, t), self.mid_at(t), small.clone()]
            .into_iter()
            .flatten()
            .reduce(|a, b| a.add(&b));
        let full_mid = if cell.kod >= 2 && cell.lstar < cell.n {
            let l0m = t.saturating_sub(cell.n).max(cell.kod);
            // per-stratum over the window, aggregate for the tail —
            // the same treatment as `mid_at`, so a provider that
            // prices strata separately is consulted by BOTH
            // candidates. (Before round 19n this candidate read only
            // the aggregate, and since it wins the min it made the
            // per-stratum seam invisible.)
            let topm = cell.n.min(cell.far_pair_limit(t) + 1);
            let mid = if l0m < topm {
                let a = t - 2 * cell.kod;
                let hi = (l0m + Self::SPECIES_WINDOW).min(topm);
                let mut acc: Option<Lg> = None;
                for l in l0m..hi {
                    let term = self.data.d_b_at(cell.s, cell.k, l, a);
                    acc = Some(match acc {
                        Some(x) => x.add(&term),
                        None => term,
                    });
                }
                if hi < topm {
                    let tail = self
                        .data
                        .d_b(cell.s, cell.k, a)
                        .mul(&self.mid.range_bracket(cell, hi));
                    acc = Some(match acc {
                        Some(x) => x.add(&tail),
                        None => tail,
                    });
                }
                acc
            } else {
                None
            };
            [self.deep.single_at(cell.n), mid, small]
                .into_iter()
                .flatten()
                .reduce(|a, b| a.add(&b))
        } else {
            None
        };
        // THE NEAR BRANCH (desc:interface-scope's companion, SPLICE
        // round 20b). At threshold `t` a word with
        // `t_max >= s + k - t` has list EXACTLY ONE: its best
        // codeword `g` and any member `f` agree with each other on
        // `>= t + t_max - s >= k` points, and two distinct
        // polynomials of degree `< k` agree at most `k - 1` times.
        // That branch consumes no charge, so the envelope is the
        // MAX of one and the far branch's bound — which is why an
        // all-empty right-hand side is the value one, not an error.
        Ok([classic, full_mid]
            .into_iter()
            .flatten()
            .reduce(|a, b| a.min(&b))
            .map_or_else(Lg::zero, |v| v.max(&Lg::zero())))
    }
}

/// The master inequality as an operator: an envelope at level
/// `prev.n`, plus interface data at level `s = 2 prev.n`, yields an
/// envelope at level `s` for the given dimensions. Per dimension:
/// build the three charges, evaluate the right-hand
/// side on the threshold grid in parallel, and refuse if the
/// computed profile certifiably rises — the grid's enclosure rests
/// on the exact right-hand side being non-increasing in `t`, and a
/// rise means the data broke its `d_b` contract.
pub fn step(
    prev: &Profile,
    dims: &BTreeSet<u64>,
    data: &dyn Interface,
    res: u64,
) -> Result<Profile> {
    let mut out = Profile {
        n: 2 * prev.n,
        rows: BTreeMap::new(),
    };
    for &k in dims {
        let charges = Charges::build(prev, k, data)?;
        let grid = build_grid(
            charges.cell.r,
            charges.cell.s,
            2 * charges.cell.lstar,
            res,
        );
        // the envelope is the min of every theorem in hand: the
        // master's right-hand side, clamped by the analytic counts at
        // this level (Johnson and the shower bound hold at every
        // level, not only the floor). Without the clamp the deep
        // charge compounds near full agreement — at t = s - z it
        // sums z + 1 classes at unit-or-more each even where the true
        // classes are empty, and iterated over the tower that phantom
        // union-bound mass grows like C(z + d, d). A pointwise min of
        // valid upper bounds is a valid upper bound, and min of
        // non-increasing functions is non-increasing, so the grid's
        // enclosure contract survives.
        let vals: Vec<(f64, f64)> = grid
            .par_iter()
            .map(|&t| {
                let mut best = charges.rhs(t)?;
                for a in analytic_brackets(charges.cell.s, k, t) {
                    best = best.min(&a);
                }
                Ok(store(&best))
            })
            .collect::<Result<Vec<_>>>()?;
        if vals.windows(2).any(|w| w[1].0 > w[0].1) {
            return Err(Error::Unsupported(format!(
                "interface data violates monotonicity at level {}, dimension {k}",
                2 * prev.n
            )));
        }
        out.insert(k, grid, vals);
    }
    Ok(out)
}

/// The conditional form of the worst-case bound, as computation: from
/// the analytic base at floor level `n0`, apply the step once
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
/// Seeds from the analytic base; [`assemble_levels_from`] accepts
/// any base profile.
pub fn assemble_levels(
    s: u64,
    k: u64,
    n0: u64,
    data: &dyn Interface,
    res: u64,
) -> Result<Vec<Profile>> {
    let w = windows(s, k, n0)?;
    assemble_levels_from(analytic_base(n0, &w[0])?, s, k, data, res)
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
