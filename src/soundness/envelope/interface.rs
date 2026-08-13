//! Interface data: the master step's external hypothesis as a
//! plug-in seam, and the shipped providers — the citable ladder
//! from unconditional counting to the graded-rigidity hypothesis.

use std::collections::BTreeMap;

use rayon::prelude::*;

use crate::math::enclosure::{lg_binom, lg_binom_memo, Lg, LgFactorials};

use super::profile::store;

/// Certified interface data at a level (notes v2, ch. 4, the derived
/// interface): `D_b(a)` bounds, over all cores, the number of family
/// members with agreement surplus at least `a`; `D_c(l)` bounds the
/// cut stratum at pair count `l`. Implementations receive the full
/// cell context `(s, k, ·)` so one object can serve every level of a
/// tower.
///
/// SCOPE (desc:interface with desc:interface-scope): both displays
/// must hold for every word at the level whose syndrome is NONZERO
/// — that is, every non-codeword. The restriction is forced: at
/// `b = 0` every `r`-subset is incident, so the cut stratum is the
/// FULL configuration count and any sharper `d_c` fails outright
/// (measured at (16,7): a codeword has `|Z^(0)(0)| = 2^8 = 256`
/// against the 128 supplied by both the shower and star faces).
/// It costs nothing: a zero-syndrome word is a codeword `g`, two
/// distinct codewords cannot agree at `k` points, so `|Lam_t(g)| = 1`
/// for `t >= r` and the master's conclusion holds with no interface
/// data consumed. gate_provider_battery.py enforces this tiering —
/// TrivialInterface is clean even at `b = 0`, ShowerInterface and
/// StarInterface are clean exactly off it.
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
    /// PER-STRATUM form of the middle-band datum: a bound on the
    /// number of family members with EXACTLY `l` pairs at surplus
    /// `a`. The default is the ownership division of [`d_b`] — the
    /// identity `sum_Y N_Y(t) = sum_f C(pairs(f), kod)` says a
    /// member with `l` pairs is counted `C(l, kod)` times, so
    /// `d_b / C(l, kod)` bounds how many such members there are —
    /// which reproduces the aggregate charge exactly. Providers
    /// override it to price the SPECIES separately: `l = kod` is the
    /// pair-poor face (the capacity theorem's object, weight one
    /// each) and `l > kod` the pair-rich face (weight `C(l, kod)`,
    /// where the mass sits, and the class Transport recurses).
    /// Must be non-increasing in `a` for the same reason `d_b` is.
    fn d_b_at(&self, s: u64, k: u64, l: u64, a: u64) -> Lg {
        self.d_b(s, k, a).div(&lg_binom(l, k / 2))
    }
    /// Bound on the cut stratum `|Z^(l)(b)|` at pair count
    /// `l < k/2`, valid for every word at the level. `None` asserts
    /// the stratum is PROVABLY EMPTY (no geometry admits a member) —
    /// which a log-domain bracket cannot express.
    fn d_c(&self, s: u64, k: u64, l: u64) -> Option<Lg>;
    /// Bound on the number of REALIZED partial cores at stratum `l`:
    /// `l`-subsets `Y` of the slots whose rank-`(k - 2l)` derived
    /// list at agreement `>= m` is nonempty (gate_graded_pencils,
    /// Lemma D) — the graded interface datum. `None` asserts the
    /// count is PROVABLY ZERO — which a log-domain bracket cannot
    /// express — and hence, since every class member realizes its
    /// own fiber set (the graded charge's decomposition,
    /// unconditional), that the class at `(l, m)` is EMPTY: the
    /// charge consumes this with no multiplicity factor and no
    /// derived-Johnson validity condition (issue #65; the emptiness
    /// form crosses the coverage curve). Must hold for every word,
    /// and be non-increasing in `m` (`None` = bottom, so
    /// empty-beyond-a-surplus is monotone-safe). The default counts
    /// every `l`-subset — word-free and weak; the graded-rigidity
    /// hypothesis sharpens it. Where the count is positive, the
    /// charge multiplies it by the derived-Johnson multiplicity
    /// (Lemma J, a theorem), so `d_r` carries the entire hypothesis
    /// content of the graded route.
    fn d_r(&self, s: u64, _k: u64, l: u64, _m: u64) -> Option<Lg> {
        Some(lg_binom_memo(s / 2, l))
    }
    /// Bound on `max(d_r(l', t - 2 l'))` over `l' <= l` — the graded
    /// tail's numerator. `None` asserts every stratum up to `l` has
    /// provably zero realized cores at threshold `t` (all their
    /// classes empty — the tail vanishes outright). The default
    /// covers the default `d_r` (every `l'`-subset; the prefix max
    /// of `C(s/2, ·)` peaks at `s/4`). Must be non-increasing in
    /// `t` at fixed `l`.
    fn d_r_sup(&self, s: u64, _k: u64, l: u64, _t: u64) -> Option<Lg> {
        Some(lg_binom_memo(s / 2, l.min(s / 4)))
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
        lg_binom_memo(s / 2, kod).mul(&Lg::from_u64(per_core.max(1)))
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

    /// The prefix scan of the trait default is `O(l)` binomials per
    /// query — at box scale that linear scan was the profiled wall
    /// (37 of 40 assembly minutes). Same values, table-driven and
    /// cached per `(s, k)`, the shower/star pattern.
    fn d_c_sup(&self, s: u64, k: u64, l: u64) -> Option<Lg> {
        let kod = k / 2;
        assert!(
            l < kod,
            "d_c_sup asked at l = {l} outside the small-strata range \
             [0, {kod}) of k = {k}"
        );
        static SUP: std::sync::OnceLock<SupCache> = std::sync::OnceLock::new();
        let mut cache = SUP
            .get_or_init(|| std::sync::Mutex::new(BTreeMap::new()))
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prefix = cache.entry((s, k)).or_insert_with(|| {
            let np = s / 2;
            let facts = LgFactorials::new(np);
            let binom = |n: u64, kk: u64| -> Lg { facts.binom(n, kk) };
            let vals: Vec<Option<Lg>> = (0..kod)
                .into_par_iter()
                .map(|lp| {
                    stratum_geometry(s, k, lp).map(|(np, h, lpp)| {
                        Lg::from_u64(2)
                            .pow(h)
                            .mul(&binom(np, lpp))
                            .mul(&binom(np - lpp, h))
                    })
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

/// The star-maximum cut face — UNCONDITIONAL and EXACT. The
/// per-stratum star-maximum theorem (STAR-MAXIMUM-GENERAL.md,
/// SPLICE rounds 5-9; gates star_maximum / stratum_rate /
/// secant_general) evaluates the word-free stratum supremum in
/// closed form: with `(np, h, lp)` the stratum geometry,
///
/// `D_c(l) = sup_b |Z^(l)(b)|
///         = 2^h C(np-1, lp-1) C(np-lp, h)
///           + 2^(h-1) C(np-1, lp) C(np-lp-1, h-1)`
///
/// (Pascal's identity absorbs the subtraction in the raw form),
/// equal to `(k+1)/s` times the full stratum — the stratum-uniform
/// rate law. The supremum is ATTAINED (by the point-divisor
/// syndromes, the same word at every stratum simultaneously), so
/// this is the exact worst case, not merely a bound; and the
/// theorem is field-uniform — valid at every prime `p = 1 mod s`
/// with no accident hypotheses. `D_b` remains the unconditional
/// pencil-disjointness bound: this provider sharpens (to exactness)
/// only the cut face.
pub struct StarInterface {
    sup: SupCache,
}

impl StarInterface {
    /// A provider with an empty supremum cache; the first tower
    /// assembly at each `(s, k)` fills it.
    #[must_use]
    pub fn new() -> Self {
        StarInterface {
            sup: std::sync::Mutex::new(BTreeMap::new()),
        }
    }
}

impl Default for StarInterface {
    fn default() -> Self {
        Self::new()
    }
}

/// The star-maximum stratum supremum `2^h C(np-1,lp-1) C(np-lp,h)
/// + 2^(h-1) C(np-1,lp) C(np-lp-1,h-1)`; the `lp = 0` stratum has
/// no core face and keeps only the point term
/// `2^(h-1) C(np-1, h-1)`; the `h = 0` stratum (outside the d_c
/// domain at odd `k`, kept for completeness) is Lemma B alone:
/// `C(np-1, lp-1)`.
fn star_d_c(np: u64, h: u64, lp: u64, binom: &dyn Fn(u64, u64) -> Lg) -> Lg {
    if h == 0 {
        return if lp == 0 {
            Lg::from_u64(1)
        } else {
            binom(np - 1, lp - 1)
        };
    }
    let point = Lg::from_u64(2)
        .pow(h - 1)
        .mul(&binom(np - 1, lp))
        .mul(&binom(np - lp - 1, h - 1));
    if lp == 0 {
        // C(np-1, lp) = 1 and C(np-lp-1, h-1) = C(np-1, h-1)
        return point;
    }
    let star = Lg::from_u64(2)
        .pow(h)
        .mul(&binom(np - 1, lp - 1))
        .mul(&binom(np - lp, h));
    star.add(&point)
}

impl Interface for StarInterface {
    fn d_b(&self, s: u64, k: u64, a: u64) -> Lg {
        TrivialInterface.d_b(s, k, a)
    }

    fn d_c(&self, s: u64, k: u64, l: u64) -> Option<Lg> {
        let (np, h, lp) = stratum_geometry(s, k, l)?;
        Some(star_d_c(np, h, lp, &lg_binom))
    }

    /// Precondition: `l < k/2` — matching the trait's stated `d_c`
    /// domain (same as the shower provider).
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
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let prefix = cache.entry((s, k)).or_insert_with(|| {
            let np = s / 2;
            let facts = LgFactorials::new(np);
            let binom = |n: u64, kk: u64| -> Lg { facts.binom(n, kk) };
            let vals: Vec<Option<Lg>> = (0..kod)
                .into_par_iter()
                .map(|lp| {
                    stratum_geometry(s, k, lp).map(|(np, h, lpp)| star_d_c(np, h, lpp, &binom))
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
    /// which equals `t - k`, INDEPENDENT of `l` (the witness
    /// identity; measured at the record cell: every populated
    /// stratum sits at graded surplus 2). The hypothesis empties it:
    /// beyond `a_max`, ZERO realized cores (the census measures
    /// zero, not few, at every cell and stratum), expressed as
    /// `None` — the emptiness form of the charge, which needs no
    /// derived-Johnson factor (issue #65). Within the cap, all
    /// cores allowed.
    fn d_r(&self, s: u64, k: u64, l: u64, m: u64) -> Option<Lg> {
        let kp = k - 2 * l;
        if m.saturating_sub(kp) > self.a_max {
            return None; // zero realized cores: the class is empty
        }
        Some(lg_binom(s / 2, l))
    }

    /// Beyond the cap the graded surplus `t - k` (l-independent)
    /// empties every stratum's realized cores at once, so the sup
    /// is `None` and the small-strata tail vanishes outright.
    fn d_r_sup(&self, s: u64, k: u64, l: u64, t: u64) -> Option<Lg> {
        if t.saturating_sub(k) > self.a_max {
            return None;
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

/// The SPECIES-SPLIT middle face — a PROBE, not a theorem, and
/// NOT CURRENTLY EXERCISED BY THE ASSEMBLY (round 19n): the
/// master's right-hand side takes a MIN over two candidates, and
/// the winning one (`full_mid`, the `lambda = n` extended split)
/// prices the whole middle range with the AGGREGATE [`Interface::d_b`],
/// bypassing [`Interface::d_b_at`] entirely. Disabling that
/// candidate drops every configuration to the Johnson clamp, which
/// is the proof that it, not the per-stratum data, carries the
/// gauge. Making the species split meaningful requires teaching
/// `full_mid` the per-stratum datum over its whole range — real
/// surgery, not a knob. Until then this provider's rows measure
/// the aggregate cap and the graded cap, nothing finer. The (32,15) measurement showed the middle-band datum
/// is a multiplicity-weighted count whose mass sits on PAIR-RICH
/// members: at threshold 18 the charge is 400, of which ten
/// nine-pair members contribute 360 and the forty pair-poor ones
/// contribute 40. The two species have different mechanisms —
/// pair-poor is the capacity theorem's object, pair-rich transports
/// — so this provider prices them separately and the scan asks
/// which one the gauge actually responds to.
///
/// Cut face: the exact [`StarInterface`]. Pair-rich strata
/// (`l > kod`): the unconditional pigeonhole with an optional
/// emptiness cap. Pair-poor stratum (`l = kod`): a free parameter
/// (`poor_lg` bits, empty past `poor_cap`) — the knob the deep
/// capacity theorem would have to fill.
pub struct SpeciesInterface {
    star: StarInterface,
    /// log2 of the pair-poor bound (the `l = kod` stratum).
    pub poor_lg: f64,
    /// Surplus past which the PAIR-POOR face is empty.
    pub poor_cap: u64,
    /// Surplus past which the PAIR-RICH face is empty (`None` =
    /// never: the unconditional pigeonhole all the way up).
    pub rich_cap: Option<u64>,
    /// Surplus past which the GRADED face (`d_r`, which prices the
    /// SMALL-STRATA cut charge, a different charge entirely) is
    /// empty. Independent of the species caps: round 19m wired it
    /// to the conjunction of both and so measured this face while
    /// believing it measured the species split.
    pub graded_cap: Option<u64>,
}

impl SpeciesInterface {
    #[must_use]
    pub fn new(
        poor_lg: f64,
        poor_cap: u64,
        rich_cap: Option<u64>,
        graded_cap: Option<u64>,
    ) -> Self {
        SpeciesInterface {
            star: StarInterface::new(),
            poor_lg,
            poor_cap,
            rich_cap,
            graded_cap,
        }
    }
}

impl Interface for SpeciesInterface {
    fn d_b(&self, s: u64, k: u64, a: u64) -> Lg {
        // the aggregate (used only by the charge's far tail): the
        // unconditional pigeonhole, emptied past the rich cap
        if self.rich_cap.is_some_and(|c| a > c) {
            return Lg::zero();
        }
        TrivialInterface.d_b(s, k, a)
    }

    fn d_b_at(&self, s: u64, k: u64, l: u64, a: u64) -> Lg {
        let kod = k / 2;
        if l == kod {
            // the pair-poor face: weight one per member
            if a > self.poor_cap {
                return Lg::zero();
            }
            return Lg::from_f64_bracket(self.poor_lg, self.poor_lg);
        }
        // the pair-rich face: ownership division of the aggregate
        self.d_b(s, k, a).div(&lg_binom(l, kod))
    }

    fn d_r(&self, s: u64, k: u64, l: u64, m: u64) -> Option<Lg> {
        let a = m.saturating_sub(k - 2 * l);
        if self.graded_cap.is_some_and(|c| a > c) {
            return None;
        }
        Some(lg_binom_memo(s / 2, l))
    }

    fn d_r_sup(&self, s: u64, k: u64, l: u64, t: u64) -> Option<Lg> {
        let a = t.saturating_sub(k);
        if self.graded_cap.is_some_and(|c| a > c) {
            return None;
        }
        Some(lg_binom_memo(s / 2, l.min(s / 4)))
    }

    fn d_c(&self, s: u64, k: u64, l: u64) -> Option<Lg> {
        self.star.d_c(s, k, l)
    }

    fn d_c_sup(&self, s: u64, k: u64, l: u64) -> Option<Lg> {
        self.star.d_c_sup(s, k, l)
    }
}
