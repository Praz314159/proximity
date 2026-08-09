//! Interface data: the master step's external hypothesis as a
//! plug-in seam, and the shipped providers — the citable ladder
//! from unconditional counting to the graded-rigidity hypothesis.

use std::collections::BTreeMap;

use rayon::prelude::*;

use crate::math::enclosure::{lg_binom, Lg, LgFactorials};

use super::profile::store;

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
    ///
    /// Contract (load-bearing): the bound must hold for EVERY word at
    /// the level, and must be non-increasing in `a`. The step's
    /// off-grid block enclosure rests on the exact right-hand side
    /// being non-increasing in the threshold, and `d_b` is the only
    /// data-supplied factor in that monotonicity. The step's guard
    /// catches only violations larger than a bracket width — a
    /// provider that violates monotonicity by less silently
    /// invalidates every off-grid bracket at coarse resolution, so
    /// the contract cannot be discharged by the guard alone.
    fn d_b(&self, s: u64, k: u64, a: u64) -> Lg;
    /// Bound on the cut stratum `|Z^(l)(b)|` at pair count
    /// `l < k/2`, valid for every word at the level. `None` asserts
    /// the stratum is PROVABLY EMPTY (no geometry admits a member) —
    /// which a log-domain bracket cannot express.
    fn d_c(&self, s: u64, k: u64, l: u64) -> Option<Lg>;
    /// Bound on the number of REALIZED partial cores at stratum `l`:
    /// `l`-subsets `Y` of the slots whose rank-`(k - 2l)` derived
    /// list at agreement `>= m` is nonempty (gate_graded_pencils,
    /// Lemma D) — the graded interface datum. Must hold for every
    /// word and be non-increasing in `m`. The default counts every
    /// `l`-subset — word-free and weak; the graded-rigidity
    /// hypothesis sharpens it. The charge multiplies this by the
    /// derived-Johnson multiplicity (Lemma J, a theorem), so `d_r`
    /// carries the entire hypothesis content of the graded route.
    fn d_r(&self, s: u64, _k: u64, l: u64, _m: u64) -> Option<Lg> {
        Some(lg_binom(s / 2, l))
    }
    /// Bound on `max(d_r(l', t - 2 l'))` over `l' <= l` — the graded
    /// tail's numerator. The default covers the default `d_r` (every
    /// `l'`-subset; the prefix max of `C(s/2, ·)` peaks at `s/4`).
    /// Must be non-increasing in `t` at fixed `l`.
    fn d_r_sup(&self, s: u64, _k: u64, l: u64, _t: u64) -> Option<Lg> {
        Some(lg_binom(s / 2, l.min(s / 4)))
    }
    /// Bound on `max(d_c(l'))` over `l' <= l`, `None` when every
    /// stratum up to `l` is empty — the small-strata tail bound's
    /// numerator. Domain: `l < k/2`, matching `d_c`; providers may
    /// enforce it (the shower provider asserts, the default scan
    /// tolerates). The default is correct but linear in `l`;
    /// providers with structure should override.
    fn d_c_sup(&self, s: u64, k: u64, l: u64) -> Option<Lg> {
        prefix_max_by_hi((0..=l).map(|lp| self.d_c(s, k, lp)))
            .pop()
            .flatten()
    }
}

/// The running prefix maximum of a sequence of optional brackets,
/// compared by `hi` ONLY: the returned `lo` is the `lo` of the
/// argmax-by-`hi`, deliberately not tightened to `max(lo_i)` — the
/// consumers read `.hi` (the tail bound's numerator), and keeping
/// the pair from one element preserves a genuine bracket of that
/// element's quantity.
fn prefix_max_by_hi(vals: impl Iterator<Item = Option<Lg>>) -> Vec<Option<Lg>> {
    let mut best: Option<Lg> = None;
    vals.map(|v| {
        match (v, &best) {
            (Some(v), Some(b)) if v.hi > b.hi => best = Some(v),
            (Some(v), None) => best = Some(v),
            _ => {}
        }
        best.clone()
    })
    .collect()
}

/// The configuration geometry of cut stratum `l` at cell `(s, k)`
/// with rung `r = k + 1`, in the `l'`-corrected bookkeeping of the
/// general cut bound (ch. 3 scope note; gate_cut_shower): a
/// stratum-`l` member's complement has `l' = s/2 + l - r` full
/// slots, and the folded functionals pair against `l'`-sets. Returns
/// `(n' = s/2, h = r - 2l, l')`, or `None` when the stratum is empty
/// (no complement geometry exists).
fn stratum_geometry(s: u64, k: u64, l: u64) -> Option<(u64, u64, u64)> {
    let np = s / 2;
    let r = k + 1;
    if 2 * l > r {
        return None;
    }
    let h = r - 2 * l;
    let lp = (np + l).checked_sub(r)?;
    if lp > np || h > np - lp {
        return None;
    }
    Some((np, h, lp))
}

/// The citable floor of the data hierarchy — no engine, no prime.
///
/// `D_c(l) = 2^h C(s/2, l') C(s/2 - l', h)`: the full stratum of the
/// cut by configurations — every configuration charged its whole
/// section cube. (The earlier `C(s/2, l)` here was unsound against
/// the master's contract, which needs `D_c(l) >= |Z^(l)(b)|`: at
/// (16,7) the top word's stratum 1 is 256 against `C(8,1) = 8`.
/// Corrected 2026-08-09 with the shower/window work, issue #61.)
/// `D_b(a) = C(s/2, kod) * m` with `m = floor((s - 2 kod)/a)` at odd
/// `k` — the pencil-agreement lemma's disjointness bound: the level
/// sets of members reaching surplus `a` are disjoint subsets, each
/// of size at least `a`, of the `s - 2 kod` available points — and
/// `m = 1` at even `k`, where the family through a core is a single
/// interpolant.
pub struct TrivialInterface;

impl Interface for TrivialInterface {
    fn d_b(&self, s: u64, k: u64, a: u64) -> Lg {
        let kod = k / 2;
        assert!(
            2 * kod < s,
            "d_b needs k < s (got s = {s}, k = {k}); the step enforces \
             k <= s - 2 — direct callers must too"
        );
        let per_core = if k % 2 == 1 {
            (s - 2 * kod) / a.max(1)
        } else {
            1
        };
        // a surplus larger than the available points admits no
        // members, but the log bracket cannot say zero; one is still
        // a valid (and immaterial) upper bound there
        lg_binom(s / 2, kod).mul(&Lg::from_u64(per_core.max(1)))
    }

    fn d_c(&self, s: u64, k: u64, l: u64) -> Option<Lg> {
        let (np, h, lp) = stratum_geometry(s, k, l)?;
        Some(
            Lg::from_u64(2)
                .pow(h)
                .mul(&lg_binom(np, lp))
                .mul(&lg_binom(np - lp, h)),
        )
    }
}

/// The per-`(s, k)` prefix-maximum store of [`ShowerInterface`]'s
/// suprema, as outward-rounded `f64` endpoint pairs (`None` = every
/// stratum up to that index provably empty).
type SupCache = std::sync::Mutex<BTreeMap<(u64, u64), Vec<Option<(f64, f64)>>>>;

/// The word-free cut strata of the shower assembly (issue #61,
/// gate_cut_shower): the general cut bound in `l'`-corrected form,
/// with the joint counts word-freed by the low-strata pencil theorem
/// where it applies and by counting where it does not:
///
/// `D_c(l) = 2^(h-1) ( C(n', l') C(n' - l', h) + Jbar C(n' - l', h) )`
///
/// with `Jbar = C(n' - 1, l' - 1)` when `2 l' <= h + 1` (Theorem P:
/// the joint sets are a pencil over a nonempty base locus) and
/// `Jbar = C(n', l')` otherwise. Unconditional at every prime.
/// `D_b` is the same pencil-agreement disjointness bound as
/// [`TrivialInterface`] — this provider sharpens only the cut face.
///
/// The prefix suprema the small-strata tail bound consumes are
/// computed once per `(s, k)` from a parallel factorial table and
/// cached; the assembler's window queries become lookups.
pub struct ShowerInterface {
    sup: SupCache,
}

impl ShowerInterface {
    /// A provider with an empty supremum cache; the first tower
    /// assembly at each `(s, k)` fills it.
    #[must_use]
    pub fn new() -> Self {
        ShowerInterface {
            sup: std::sync::Mutex::new(BTreeMap::new()),
        }
    }
}

impl Default for ShowerInterface {
    fn default() -> Self {
        Self::new()
    }
}

/// The shower stratum bound `2^(h-1) (C(np,lp) C(np-lp,h) + Jbar
/// C(np-lp,h))`, with the joint count from the low-strata pencil
/// theorem where it applies (`2 lp <= h + 1`) and by counting
/// elsewhere; `binom` abstracts the binomial source so the cached
/// scan can substitute its factorial table.
fn shower_d_c(np: u64, h: u64, lp: u64, binom: &dyn Fn(u64, u64) -> Lg) -> Lg {
    let sections = Lg::from_u64(2).pow(h.saturating_sub(1));
    let halves = binom(np - lp, h);
    let config = binom(np, lp).mul(&halves);
    if lp == 0 {
        // the empty missed-set is joint only at B = 0, excluded:
        // no joint term (Lg cannot say zero, so omit it)
        return sections.mul(&config);
    }
    let jbar = if 2 * lp <= h + 1 {
        binom(np - 1, lp - 1)
    } else {
        binom(np, lp)
    };
    sections.mul(&config.add(&jbar.mul(&halves)))
}

impl Interface for ShowerInterface {
    fn d_b(&self, s: u64, k: u64, a: u64) -> Lg {
        TrivialInterface.d_b(s, k, a)
    }

    fn d_c(&self, s: u64, k: u64, l: u64) -> Option<Lg> {
        let (np, h, lp) = stratum_geometry(s, k, l)?;
        Some(shower_d_c(np, h, lp, &lg_binom))
    }

    /// Precondition: `l < k/2` (the master's small-strata range) and
    /// `k >= 2` — matching the trait's stated `d_c` domain.
    fn d_c_sup(&self, s: u64, k: u64, l: u64) -> Option<Lg> {
        let kod = k / 2;
        assert!(
            l < kod,
            "d_c_sup asked at l = {l} outside the small-strata range \
             [0, {kod}) of k = {k}"
        );
        let mut cache = self
            .sup
            .lock()
            // the cache holds only outward-rounded endpoint pairs,
            // valid regardless of where another thread panicked
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prefix = cache.entry((s, k)).or_insert_with(|| {
            // one factorial table per (s, k), built in parallel,
            // turns the O(kod) scan of three-lgamma binomials into
            // table lookups — the profiled 94%-of-wall hot spot
            let np = s / 2;
            let facts = LgFactorials::new(np);
            let binom = |n: u64, k: u64| -> Lg { facts.binom(n, k) };
            let vals: Vec<Option<Lg>> = (0..kod)
                .into_par_iter()
                .map(|lp| {
                    stratum_geometry(s, k, lp).map(|(np, h, lpp)| shower_d_c(np, h, lpp, &binom))
                })
                .collect();
            prefix_max_by_hi(vals.into_iter())
                .into_iter()
                .map(|v| v.as_ref().map(store))
                .collect()
        });
        let (lo, hi) = prefix[l as usize]?;
        Some(Lg::from_f64_bracket(lo, hi))
    }
}

/// The bucket-rigidity interface — CONDITIONAL. The cut face is the
/// unconditional [`ShowerInterface`]; the middle face is the
/// program's single external hypothesis in interface form, the
/// tail-rigidity statement (the SBC face): derived buckets have
/// bounded surplus. Concretely,
///
/// `D_b(a) <= C(s/2, kod) * 2^(4 - a)` for `a <= a_max`, and
/// `D_b(a) <= 1` for `a > a_max`
///
/// — the measured-transport shape (both measured cells: (16,7)
/// D_b = 128/28/20, (32,15) D_b = 28348/824/400, cap at surplus 4,
/// per-core densities ~2.4 constant across the tower), with the cap
/// taken at `a_max` rather than the measured 4. Every tower
/// assembled with this data is valid EXACTLY UNDER that hypothesis;
/// its rows are the conditional form of the worst-case bound, and
/// the hypothesis is the one theorem the program still owes.
pub struct RigidityInterface {
    shower: ShowerInterface,
    /// The hypothesized surplus cap (measured: 4).
    pub a_max: u64,
}

impl RigidityInterface {
    /// The conditional provider at hypothesized surplus cap `a_max`
    /// (measured cap at the gate cells: 4).
    #[must_use]
    pub fn new(a_max: u64) -> Self {
        RigidityInterface {
            shower: ShowerInterface::new(),
            a_max,
        }
    }
}

impl Interface for RigidityInterface {
    /// The graded face of the same hypothesis: a realized partial
    /// core at stratum `l` demands derived agreement `m` on a
    /// rank-`k' = k - 2l` family, i.e. graded surplus `m - k'` —
    /// which equals `t - k`, INDEPENDENT of `l` (measured at the
    /// record cell: every populated stratum sits at graded surplus
    /// 2). The hypothesis caps it: beyond `a_max`, at most one
    /// realized core per stratum; within the cap, all cores allowed.
    fn d_r(&self, s: u64, k: u64, l: u64, m: u64) -> Option<Lg> {
        let kp = k - 2 * l;
        if m.saturating_sub(kp) > self.a_max {
            return Some(Lg::zero()); // at most one realized core
        }
        Some(lg_binom(s / 2, l))
    }

    /// Beyond the cap the graded surplus `t - k` (l-independent)
    /// caps every stratum's realized cores at one, so the sup is one.
    fn d_r_sup(&self, s: u64, k: u64, l: u64, t: u64) -> Option<Lg> {
        if t.saturating_sub(k) > self.a_max {
            return Some(Lg::zero());
        }
        Some(lg_binom(s / 2, l.min(s / 4)))
    }

    fn d_b(&self, s: u64, k: u64, a: u64) -> Lg {
        if a > self.a_max {
            return Lg::zero();
        }
        let kod = k / 2;
        let shape = Lg::from_u64(16).div(&Lg::from_u64(2).pow(a.min(4)));
        lg_binom(s / 2, kod).mul(&shape)
    }

    fn d_c(&self, s: u64, k: u64, l: u64) -> Option<Lg> {
        self.shower.d_c(s, k, l)
    }

    fn d_c_sup(&self, s: u64, k: u64, l: u64) -> Option<Lg> {
        self.shower.d_c_sup(s, k, l)
    }
}
