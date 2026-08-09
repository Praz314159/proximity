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
//! floor holds no flood at small `n0`. The loss map's measured wall
//! (box run, 2026-08-09) is the SMALL-STRATA cut charge: past the
//! Johnson radius the classes at `l` just below `kod` activate, and
//! the configuration-count `D_c` floods at scale in both data modes
//! — beyond-Johnson radii are gated on a scale-correct small-strata
//! charge (the graded-pencil route) before the `D_b` supply even
//! binds.

use std::collections::{BTreeMap, BTreeSet};

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::math::enclosure::{lg_binom, lg_factorial, Lg};
use rug::float::Round;
use rug::ops::{AddAssignRound, SubAssignRound};
use rug::{Float, Integer};

/// Working precision for rebuilt endpoints (matches the enclosure
/// module's own).
const PREC: u32 = 192;

/// Fold a term into a possibly-empty running sum.
fn add_to(acc: Option<Lg>, term: Lg) -> Lg {
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
    /// Bound on `max(d_c(l'))` over `l' <= l`, `None` when every
    /// stratum up to `l` is empty — the small-strata tail bound's
    /// numerator. The default scans, which is correct but linear in
    /// `l`; providers with structure should override.
    fn d_c_sup(&self, s: u64, k: u64, l: u64) -> Option<Lg> {
        let mut best: Option<Lg> = None;
        for lp in 0..=l {
            match (self.d_c(s, k, lp), &best) {
                (Some(v), Some(b)) if v.hi > b.hi => best = Some(v),
                (Some(v), None) => best = Some(v),
                _ => {}
            }
        }
        best
    }
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
            let table: Vec<Lg> = (0..=np).into_par_iter().map(lg_factorial).collect();
            let binom = |n: u64, k: u64| -> Lg {
                table[n as usize]
                    .div(&table[k as usize])
                    .div(&table[(n - k) as usize])
            };
            let vals: Vec<Option<Lg>> = (0..kod)
                .into_par_iter()
                .map(|lp| {
                    stratum_geometry(s, k, lp).map(|(np, h, lpp)| shower_d_c(np, h, lpp, &binom))
                })
                .collect();
            let mut out = Vec::with_capacity(kod as usize);
            let mut best: Option<Lg> = None;
            for v in vals {
                match (v, &best) {
                    (Some(v), Some(b)) if v.hi > b.hi => best = Some(v),
                    (Some(v), None) => best = Some(v),
                    _ => {}
                }
                out.push(best.as_ref().map(store));
            }
            out
        });
        let (lo, hi) = prefix[l as usize]?;
        Some(Lg {
            lo: Float::with_val(PREC, lo),
            hi: Float::with_val(PREC, hi),
        })
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

/// A stored bracket together with the smallest threshold it remains
/// an enclosure at (by the profile's monotonicity in `t`).
#[derive(Clone, Copy)]
struct BlockBracket {
    lo: f64,
    hi: f64,
    valid_from: u64,
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
    fn bracket(&self, t: u64) -> BlockBracket {
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
    rows: BTreeMap<u64, Row>,
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
        Ok(Lg {
            lo: Float::with_val(PREC, b.lo),
            hi: Float::with_val(PREC, b.hi),
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

/// A bracket as its stored form: `f64` endpoints, rounded outward.
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
/// crudest citable base, kept as the reference floor;
/// [`analytic_base`] dominates it pointwise and is [`assemble`]'s
/// default.
pub fn interpolation_base(n0: u64, dims: &BTreeSet<u64>) -> Result<Profile> {
    base_from_counts(n0, dims, |_, _, interp| interp.clone())
}

/// Shared base construction: validate each dimension, then fill its
/// stride-1 row from an exact per-threshold count
/// `(k, t, interpolation count) -> count`, stored as a tight bracket.
fn base_from_counts(
    n0: u64,
    dims: &BTreeSet<u64>,
    count: impl Fn(u64, u64, &Integer) -> Integer,
) -> Result<Profile> {
    let mut prof = Profile {
        n: n0,
        rows: BTreeMap::new(),
    };
    let n0_32 =
        u32::try_from(n0).map_err(|_| Error::OutOfRange(format!("base level {n0} exceeds u32")))?;
    for &k in dims {
        if k == 0 || k + 1 > n0 {
            return Err(Error::OutOfRange(format!(
                "base dimension {k} needs 1 <= k < n0 = {n0}"
            )));
        }
        let interp = Integer::from(Integer::binomial_u(n0_32, k as u32));
        let vals = (k + 1..=n0)
            .map(|t| {
                store(&Lg::from_integer(
                    &count(k, t, &interp).max(Integer::from(1)),
                ))
            })
            .collect();
        prof.insert(k, (k + 1..=n0).collect(), vals);
    }
    Ok(prof)
}

/// The analytic base (ch. 4, the base of the tower): at each
/// threshold the smallest of three unconditional, prime-free counts.
/// **Interpolation**: `C(n0, k)`. **Johnson**, in the sharp
/// agreement form `floor( n (t - k + 1) / (t^2 - n (k - 1)) )`,
/// valid once `t^2 > n (k - 1)` — the quadratic argument: `m`
/// members agreeing on `>= t` points each, pairwise on `<= k - 1`,
/// force `m t (m t - n) / n <= m (m - 1)(k - 1)` by convexity. At
/// `t = n` it reads exactly 1, so the tower's loss-free transport
/// carries a one-word list to the top. **The shower bound**
/// (dictionary, ownership): the `t`-cliques of the cut decompose
/// disjointly by list member, so
/// `|Lam_t(w)| <= |Z(b)| / C(t, k + 1) <= C(n, k + 1) / C(t, k + 1)`
/// — weak, but it closes the band between the coverage curve and
/// the Johnson threshold at any floor, and slack at the base
/// inflates only the final constant, never the induction. On the
/// integer grid at floors 8 and 16 (rate 1/2) the sharp Johnson
/// form already covers the coverage curve and the band is empty;
/// the shower term guards every other configuration. The certified
/// sharpening — exact floor values as register-backed certificates —
/// is the base section's companion statement, consumed only where
/// the compilation chapter wants sharp seeds; the mainline rests on
/// this analytic statement.
pub fn analytic_base(n0: u64, dims: &BTreeSet<u64>) -> Result<Profile> {
    base_from_counts(n0, dims, |k, t, interp| {
        let mut best = interp.clone();
        analytic_refine(n0, k, t, &mut best);
        best
    })
}

/// Lower `best` to the sharper of the Johnson and shower counts at
/// `(n, k, t)` where they apply — the analytic statement's two
/// nontrivial clauses in exact integers, for the base constructor.
fn analytic_refine(n: u64, k: u64, t: u64, best: &mut Integer) {
    // u128 throughout: at levels past 2^32 the u64 products wrap in
    // release and a wrapped Johnson clause could clamp BELOW the
    // truth — the one failure mode this module must never have
    let (tw, nw, kw) = (t as u128, n as u128, k as u128);
    if tw * tw > nw * (kw - 1) {
        let johnson = Integer::from(nw * (tw - kw + 1) / (tw * tw - nw * (kw - 1)));
        if johnson < *best {
            *best = johnson;
        }
    }
    let shower = Integer::from(Integer::binomial_u(n as u32, (k + 1) as u32))
        / Integer::from(Integer::binomial_u(t as u32, (k + 1) as u32));
    if shower < *best {
        *best = shower;
    }
}

/// The analytic counts at `(n, k, t)` as log brackets — the same
/// three clauses as [`analytic_refine`] without the floors (a valid
/// loosening), in log-gamma arithmetic so the per-level clamp costs
/// microseconds where the exact binomials would cost million-bit
/// integers.
fn analytic_brackets(n: u64, k: u64, t: u64) -> Vec<Lg> {
    let mut out = vec![lg_binom(n, k), lg_binom(n, k + 1).div(&lg_binom(t, k + 1))];
    let (tw, nw, kw) = (t as u128, n as u128, k as u128);
    if tw * tw > nw * (kw - 1) {
        out.push(
            Lg::from_integer(&Integer::from(nw * (tw - kw + 1)))
                .div(&Lg::from_integer(&Integer::from(tw * tw - nw * (kw - 1)))),
        );
    }
    out
}

/// The channel dimensions of `k`: the fold splits degree-below-`k`
/// polynomials into even and odd parts of dimensions `ceil(k/2)` and
/// `floor(k/2)`.
#[must_use]
pub fn channel_dims(k: u64) -> (u64, u64) {
    (k.div_ceil(2), k / 2)
}

/// The cell geometry one step application works in: level `s = 2n`,
/// dimension `k`, and the derived quantities every charge reads —
/// channel dimensions, rung, coverage threshold.
#[derive(Clone, Copy)]
struct Cell {
    n: u64,
    s: u64,
    k: u64,
    kev: u64,
    kod: u64,
    r: u64,
    lstar: u64,
}

impl Cell {
    fn new(n: u64, k: u64) -> Self {
        let (kev, kod) = channel_dims(k);
        Cell {
            n,
            s: 2 * n,
            k,
            kev,
            kod,
            r: k + 1,
            lstar: (n + k - 1).div_ceil(3),
        }
    }
}

/// Charge 1 — deep strata (`l >= l*`): the descent injection places
/// the class in the joint list, anti-squaring collapses it to the
/// larger channel list, and the level-below profile prices it. The
/// profile is block-constant on its grid, so the suffix over
/// `[l*, n]` compresses to runs — stretches where both channel
/// brackets are constant — each contributing width times its max,
/// with the sum strictly above each run precomputed so any lower
/// limit is answered with one partial-width multiply.
struct DeepCharge {
    /// Runs in descending `l` order (built from `l = n` down).
    runs: Vec<DeepRun>,
}

struct DeepRun {
    lo: u64,
    hi: u64,
    max: Lg,
    sum_above: Option<Lg>,
}

impl DeepCharge {
    fn build(prev: &Profile, cell: &Cell) -> Self {
        let mut runs = Vec::new();
        if cell.lstar > cell.n {
            return DeepCharge { runs };
        }
        let row_e = &prev.rows[&cell.kev];
        let row_o = &prev.rows[&cell.kod];
        let mut hi = cell.n;
        let mut sum_above: Option<Lg> = None;
        loop {
            // each channel's bracket comes with the lowest threshold
            // it stays valid at; the run is their meet
            let e = row_e.bracket(hi);
            let o = row_o.bracket(hi);
            let max = Lg {
                lo: Float::with_val(PREC, e.lo.max(o.lo)),
                hi: Float::with_val(PREC, e.hi.max(o.hi)),
            };
            let lo = e.valid_from.max(o.valid_from).max(cell.lstar);
            let contribution = Lg::from_u64(hi - lo + 1).mul(&max);
            let next_above = Some(add_to(sum_above.clone(), contribution));
            runs.push(DeepRun {
                lo,
                hi,
                max,
                sum_above,
            });
            sum_above = next_above;
            if lo == cell.lstar {
                break;
            }
            hi = lo - 1;
        }
        DeepCharge { runs }
    }

    /// The charge at threshold `t`: the profile suffix from
    /// `max(l*, t - n)`, or `None` when the stratum range is empty.
    fn at(&self, cell: &Cell, t: u64) -> Option<Lg> {
        let l0 = t.saturating_sub(cell.n).max(cell.lstar);
        if l0 > cell.n || self.runs.is_empty() {
            return None;
        }
        let j = self.runs.partition_point(|run| run.lo > l0);
        let run = &self.runs[j];
        debug_assert!(run.lo <= l0 && l0 <= run.hi);
        let partial = Lg::from_u64(run.hi - l0 + 1).mul(&run.max);
        Some(match &run.sum_above {
            Some(above) => above.add(&partial),
            None => partial,
        })
    }
}

/// Charge 2 — the middle band (`kod <= l < l*`): the core charge
/// prices the class by `D_b` against the suffix of `1/C(l, kod)`.
/// Small ranges get the exact term-by-term array; large ones the
/// telescoping identity `sum over l >= l0 of 1/C(l, m) =
/// m/((m - 1) C(l0 - 1, m - 1))`, enclosed as [first term, closed
/// form] — width `lg(l0/(m - 1))`, a fraction of a bit at rate 1/2,
/// for two binomials per query instead of a level-length
/// precomputation. The identity needs `m >= 2`, so `kod = 1` always
/// takes the exact route (the closed form would divide by zero).
enum MidSuffix {
    Exact(Vec<Lg>),
    Telescoped,
}

impl MidSuffix {
    const EXACT_LIMIT: usize = 1 << 12;

    fn build(cell: &Cell) -> Self {
        let len = cell.lstar.saturating_sub(cell.kod) as usize;
        if len > Self::EXACT_LIMIT && cell.kod >= 2 {
            return MidSuffix::Telescoped;
        }
        let mut suffix: Vec<Lg> = Vec::with_capacity(len);
        for i in 0..len {
            let l = cell.lstar - 1 - i as u64;
            let inv = Lg::zero().div(&lg_binom(l, cell.kod));
            suffix.push(match suffix.last() {
                Some(acc) => acc.add(&inv),
                None => inv,
            });
        }
        MidSuffix::Exact(suffix)
    }

    fn suffix_from(&self, cell: &Cell, l0: u64) -> Lg {
        match self {
            MidSuffix::Exact(suffix) => suffix[(cell.lstar - 1 - l0) as usize].clone(),
            MidSuffix::Telescoped => {
                let hi = Lg::from_u64(cell.kod)
                    .div(&Lg::from_u64(cell.kod - 1))
                    .div(&lg_binom(l0 - 1, cell.kod - 1));
                let lo = Lg::zero().div(&lg_binom(l0, cell.kod));
                Lg {
                    lo: lo.lo,
                    hi: hi.hi,
                }
            }
        }
    }
}

/// Charge 3 — small strata (`l < kod`): canonical subsets to the cut,
/// priced by `D_c` over divisors `C(t - 2l, r - 2l)`, summed downward
/// with a certified tail bound: count times the worst remaining
/// numerator over the smallest remaining divisor (divisors grow as
/// `l` falls). The sum closes either when the tail is provably
/// negligible or at the term cap — near `t = r` the divisors grow
/// slowly and the cap bites, but there the remaining terms are
/// near-equal and the bound is tight; everywhere else the terms decay
/// fast and the tail is dust. Cost per threshold stays O(cap). The
/// `D_c` values of the capped window are threshold-independent, so
/// they are fetched once.
struct CutCharge {
    /// `(d_c, d_c_sup)` for `l` in `[window_lo, kod)`; `None` = the
    /// provider certifies the stratum (resp. every stratum up to
    /// here) empty, and the charge skips it.
    window: Vec<(Option<Lg>, Option<Lg>)>,
    window_lo: u64,
}

impl CutCharge {
    const CAP: u64 = 16;
    const NEGLIGIBLE_BITS: u32 = 80;

    fn build(cell: &Cell, data: &dyn Interface) -> Self {
        let window_lo = cell.kod.saturating_sub(Self::CAP + 2);
        let window = (window_lo..cell.kod)
            .map(|l| (data.d_c(cell.s, cell.k, l), data.d_c_sup(cell.s, cell.k, l)))
            .collect();
        CutCharge { window, window_lo }
    }

    fn dc(&self, cell: &Cell, data: &dyn Interface, l: u64) -> Option<Lg> {
        if l >= self.window_lo {
            self.window[(l - self.window_lo) as usize].0.clone()
        } else {
            data.d_c(cell.s, cell.k, l)
        }
    }

    fn dc_sup(&self, cell: &Cell, data: &dyn Interface, l: u64) -> Option<Lg> {
        if l >= self.window_lo {
            self.window[(l - self.window_lo) as usize].1.clone()
        } else {
            data.d_c_sup(cell.s, cell.k, l)
        }
    }

    fn at(&self, cell: &Cell, data: &dyn Interface, t: u64) -> Option<Lg> {
        let lmin = t.saturating_sub(cell.n);
        if lmin >= cell.kod {
            return None;
        }
        let (r, kod) = (cell.r, cell.kod);
        let mut acc: Option<Lg> = None;
        let mut l = kod - 1;
        loop {
            if let Some(dc) = self.dc(cell, data, l) {
                let term = dc.div(&lg_binom(t - 2 * l, r - 2 * l));
                acc = Some(add_to(acc, term));
            }
            if l == lmin {
                break;
            }
            // a provider that certifies everything below `l` empty
            // closes the sum exactly — no tail term at all
            let Some(sup) = self.dc_sup(cell, data, l - 1) else {
                break;
            };
            let count = l - lmin;
            let mut tail_hi = Lg::from_u64(count).hi;
            tail_hi.add_assign_round(&sup.hi, Round::Up);
            tail_hi.sub_assign_round(&lg_binom(t - 2 * (l - 1), r - 2 * (l - 1)).lo, Round::Up);
            let closable = match &acc {
                Some(cur) => {
                    let mut margin = cur.hi.clone();
                    margin -= Self::NEGLIGIBLE_BITS;
                    tail_hi <= margin
                }
                None => false,
            };
            if closable || kod - 1 - l >= Self::CAP {
                let tail = Lg {
                    lo: Float::with_val(PREC, f64::NEG_INFINITY),
                    hi: tail_hi,
                };
                acc = Some(match &acc {
                    Some(cur) => Lg {
                        lo: cur.lo.clone(),
                        hi: cur.add(&tail).hi,
                    },
                    // every visited stratum was empty: the tail
                    // alone bounds the rest, with no mass below
                    None => tail,
                });
                break;
            }
            l -= 1;
        }
        acc
    }
}

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
    fn mid_at(&self, t: u64) -> Option<Lg> {
        let cell = &self.cell;
        let l0 = t.saturating_sub(cell.n).max(cell.kod);
        if l0 >= cell.lstar {
            return None;
        }
        let a = t - 2 * cell.kod;
        Some(
            self.data
                .d_b(cell.s, cell.k, a)
                .mul(&self.mid.suffix_from(cell, l0)),
        )
    }

    /// The master's right-hand side at threshold `t`: the sum of the
    /// three charges over their (possibly empty) stratum ranges.
    fn rhs(&self, t: u64) -> Result<Lg> {
        let cell = &self.cell;
        [
            self.deep.at(cell, t),
            self.mid_at(t),
            self.cut.at(cell, self.data, t),
        ]
        .into_iter()
        .flatten()
        .reduce(|a, b| a.add(&b))
        .ok_or_else(|| {
            Error::Unsupported(format!(
                "no charge covers cell ({}, {}, {t})",
                cell.s, cell.k
            ))
        })
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
        let grid = build_grid(charges.cell.r, charges.cell.s, res);
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
                    best = Lg {
                        lo: if a.lo < best.lo { a.lo } else { best.lo },
                        hi: if a.hi < best.hi { a.hi } else { best.hi },
                    };
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
                // the corrected trivial D_c: full stratum by
                // configurations, 2^h C(n, l') C(n - l', h) with
                // l' = l on the rung (r = n here)
                let h = (r - 2 * l) as u32;
                let dc = (Integer::from(1) << h) * binom(n, l) * binom(n - l, r - 2 * l);
                exact += Rational::from((dc, binom(t - 2 * l, r - 2 * l)));
            }
            // the profile is the master's RHS clamped by the analytic
            // brackets at this level (unfloored ratios) — mirror them
            let mut clamps = vec![
                Rational::from(binom(s, k)),
                Rational::from((binom(s, k + 1), binom(t, k + 1))),
            ];
            if t * t > s * (k - 1) {
                clamps.push(Rational::from((
                    Integer::from(s * (t - k + 1)),
                    Integer::from(t * t - s * (k - 1)),
                )));
            }
            for c in clamps {
                if c < exact {
                    exact = c;
                }
            }
            let got = prof.eval(k, t).expect("in domain");
            let want = exact.to_f64().log2();
            let lo = got.lo.to_f64_round(Round::Down);
            let hi = got.hi.to_f64_round(Round::Up);
            // `want` is itself a two-rounding f64 approximation of the
            // exact rational, so compare with an ulp-scale tolerance
            assert!(
                lo <= want + 1e-12 && want <= hi + 1e-12,
                "t = {t}: [{lo}, {hi}] vs exact {want}"
            );
            assert!(hi - lo < 0.01, "t = {t}: bracket too wide");
        }
    }

    /// The step against exact rational arithmetic with the SHOWER
    /// provider — the same oracle as the trivial-data gate below, with
    /// the shower `D_c` mirrored independently (configuration count
    /// plus the pencil-or-counting joint term), so a transcription
    /// error in either the provider or the charge plumbing shows as a
    /// bracket miss.
    #[test]
    fn step_encloses_exact_rational_at_16_7_shower() {
        let (n, k) = (8u64, 7u64);
        let s = 2 * n;
        let (kev, kod) = channel_dims(k); // (4, 3)
        let r = k + 1;
        let lstar = (n + k - 1).div_ceil(3); // 5
        let base = interpolation_base(n, &BTreeSet::from([kev, kod])).expect("base");
        let data = ShowerInterface::new();
        let prof = step(&base, &BTreeSet::from([k]), &data, u64::MAX).expect("step");
        for t in r..=s {
            let lmin = t.saturating_sub(n);
            let mut exact = Rational::new();
            for _l in lmin.max(lstar)..=n {
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
                // the shower D_c, mirrored: r = n here (on the rung),
                // so l' = l and h = r - 2l
                let h = r - 2 * l;
                let halves = binom(n - l, h);
                let jbar = if l == 0 {
                    Integer::from(0)
                } else if 2 * l <= h + 1 {
                    binom(n - 1, l - 1)
                } else {
                    binom(n, l)
                };
                let dc =
                    (Integer::from(1) << (h - 1) as u32) * (binom(n, l) * &halves + jbar * &halves);
                exact += Rational::from((dc, binom(t - 2 * l, r - 2 * l)));
            }
            let mut clamps = vec![
                Rational::from(binom(s, k)),
                Rational::from((binom(s, k + 1), binom(t, k + 1))),
            ];
            if t * t > s * (k - 1) {
                clamps.push(Rational::from((
                    Integer::from(s * (t - k + 1)),
                    Integer::from(t * t - s * (k - 1)),
                )));
            }
            for c in clamps {
                if c < exact {
                    exact = c;
                }
            }
            let got = prof.eval(k, t).expect("in domain");
            let want = exact.to_f64().log2();
            let lo = got.lo.to_f64_round(Round::Down);
            let hi = got.hi.to_f64_round(Round::Up);
            assert!(
                lo <= want + 1e-12 && want <= hi + 1e-12,
                "t = {t}: [{lo}, {hi}] vs exact {want}"
            );
            assert!(hi - lo < 0.01, "t = {t}: bracket too wide");
        }
    }

    /// The provider pins: the shower `D_c` reproduces the word-free
    /// stratum bounds of gate_cut_shower at (32,15) exactly, and both
    /// providers dominate the measured strata at (16,7) (top word:
    /// 256 and 416 at l = 1, 2 — the counts that exposed the old
    /// `C(s/2, l)` as unsound).
    #[test]
    fn provider_pins() {
        let sh = ShowerInterface::new();
        let dc = |l: u64| sh.d_c(32, 15, l).expect("nonempty stratum");
        // l = 0: 2^15 (config 1, pencil-J empty)
        assert!((dc(0).hi.to_f64() - 15.0).abs() < 1e-9);
        // l = 4: 2^7 * C(12,8) * (C(16,4) + C(15,3))
        let want = (128f64 * 495.0 * (1820.0 + 455.0)).log2();
        assert!((dc(4).hi.to_f64() - want).abs() < 1e-9);
        // l = 6: trivial-J branch (2l > h + 1): 2^3 C(10,4) * 2 C(16,6)
        let want = (8f64 * 210.0 * 2.0 * 8008.0).log2();
        assert!((dc(6).hi.to_f64() - want).abs() < 1e-9);
        // soundness against measured strata at (16,7), top word
        for (l, meas) in [(1u64, 256f64), (2, 416.0)] {
            assert!(sh.d_c(16, 7, l).expect("nonempty").hi.to_f64() >= meas.log2());
            assert!(
                TrivialInterface
                    .d_c(16, 7, l)
                    .expect("nonempty")
                    .hi
                    .to_f64()
                    >= meas.log2()
            );
        }
        // the sup is a prefix maximum: never below the pointwise value
        for l in 0..7 {
            assert!(sh.d_c_sup(32, 15, l).expect("nonempty").hi >= dc(l).hi);
        }
        // rigidity: cut face delegates, middle face caps
        let rig = RigidityInterface::new(8);
        assert_eq!(
            rig.d_c(32, 15, 3).expect("nonempty").hi.to_f64(),
            dc(3).hi.to_f64()
        );
        assert!(rig.d_b(32, 15, 9).hi.to_f64() < 1e-9); // beyond the cap: one
        assert!(rig.d_b(32, 15, 2).hi.to_f64() >= (28348f64).log2()); // measured
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

    /// The analytic base pins: at (8, 4) the sharp agreement-form
    /// Johnson bound n(t - k + 1)/(t^2 - n(k - 1)) reads 16, 2, 1, 1
    /// across t = 5..8 (all under the interpolation 70 and the
    /// shower 56), and full agreement is exactly one word — zero
    /// bits. At (32, 16, 21) — a genuine band point, Johnson invalid
    /// (441 <= 480) — the shower bound C(32, 17)/C(21, 17) = 94523
    /// takes over from the interpolation 601080390.
    #[test]
    fn analytic_base_pins() {
        let base = analytic_base(8, &BTreeSet::from([4, 3])).expect("base");
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
        // the band point: shower bound active where Johnson is not
        let band = analytic_base(32, &BTreeSet::from([16])).expect("base");
        let v = band.eval(16, 21).unwrap();
        assert!((v.hi.to_f64() - (94523f64).log2()).abs() < 1e-9);
    }

    /// Full agreement transports one word: the analytic base reads
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
