//! The derived interface: the external data the master step consumes,
//! as a trait, with the shipped providers.
//!
//! Throughout, `s` is the level, `k` the dimension, `kod = k/2`,
//! `r = k + 1` the rung, and `n = s/2` the number of fibers. Every
//! bound is a log-domain bracket [`Lg`]; `None`, where a method
//! returns `Option<Lg>`, asserts that the quantity is provably zero,
//! which a bracket cannot express.
//!
//! Scope. A provider's bounds must hold for every word at the level
//! whose syndrome is nonzero. Codewords are excluded on purpose: at
//! `b = 0` the cut stratum is the full configuration count, so any
//! sharpened cut face fails there, and nothing is lost, because a
//! codeword's list at `t >= r` is exactly itself.

use std::collections::BTreeMap;

use rayon::prelude::*;

use crate::math::enclosure::{lg_binom, lg_binom_memo, Lg, LgFactorials};

use super::profile::store;

/// Interface data at a level. One object serves every level of a
/// tower, so each method takes the cell `(s, k)` explicitly.
///
/// Monotonicity is part of the contract: `d_b`, `d_b_at` are
/// non-increasing in `a`; `d_r`, `d_r_sup` in `m` and `t`. The step's
/// off-grid enclosure rests on the master's right-hand side being
/// non-increasing in the threshold, and these are its only
/// data-supplied factors. The step's guard catches violations larger
/// than a bracket width only.
pub trait Interface: Sync {
    /// `D_b(a)`: bounds, summed over all `kod`-cores `Y`, the number
    /// of members of the family through `Y` with agreement surplus at
    /// least `a >= 1`.
    fn d_b(&self, s: u64, k: u64, a: u64) -> Lg;

    /// Per-stratum form of `D_b`: bounds the number of members with
    /// exactly `l` pairs at surplus `a`. The default divides `D_b` by
    /// `C(l, kod)`, the number of cores owning such a member.
    fn d_b_at(&self, s: u64, k: u64, l: u64, a: u64) -> Lg {
        self.d_b(s, k, a).div(&lg_binom(l, k / 2))
    }

    /// `D_c(l)`: bounds the cut stratum `|Z^(l)(b)|` at pair count
    /// `l < kod`. `None`: the stratum is empty for every word.
    fn d_c(&self, s: u64, k: u64, l: u64) -> Option<Lg>;

    /// Bounds the number of realized partial cores at stratum `l`:
    /// `l`-subsets of fibers whose rank-`(k - 2l)` derived list at
    /// agreement `>= m` is nonempty. `None`: zero realized cores, so
    /// the class at `(l, m)` is empty. The default counts every
    /// `l`-subset.
    fn d_r(&self, s: u64, _k: u64, l: u64, _m: u64) -> Option<Lg> {
        Some(lg_binom_memo(s / 2, l))
    }

    /// Bounds `max_{l' <= l} d_r(l', t - 2l')`. `None`: every stratum
    /// up to `l` has zero realized cores at threshold `t`. The default
    /// matches the default `d_r` (`C(n, ·)` peaks at `n/2`).
    fn d_r_sup(&self, s: u64, _k: u64, l: u64, _t: u64) -> Option<Lg> {
        Some(lg_binom_memo(s / 2, l.min(s / 4)))
    }

    /// Bounds `max_{l' <= l} d_c(l')`, for `l < kod`. `None`: every
    /// stratum up to `l` is empty. The default is a linear scan;
    /// providers with a closed form override it with a cached table.
    fn d_c_sup(&self, s: u64, k: u64, l: u64) -> Option<Lg> {
        prefix_max_by_hi((0..=l).map(|lp| self.d_c(s, k, lp)))
            .pop()
            .flatten()
    }
}

/// Running prefix maximum by `hi`. The `lo` returned is that of the
/// argmax element, not `max(lo_i)`, so each entry is a genuine
/// bracket of one element's quantity; consumers read `hi`.
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

/// The geometry of cut stratum `l` at `(s, k)`: `(n, h, l')` with
/// `h = r - 2l` the number of unpaired points of an `r`-subset and
/// `l' = n + l - r` the number of full fibers in its complement.
/// `None` when no such subset exists.
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

/// A stratum bound as a function of `(n, h, l')` and a binomial
/// source; the source is abstracted so the cached table can use a
/// factorial table instead of per-call `lgamma`.
type StratumFormula = fn(u64, u64, u64, &dyn Fn(u64, u64) -> Lg) -> Lg;

/// Per-`(s, k)` table of prefix maxima of a cut face, as outward-
/// rounded endpoints; `None` marks a prefix of empty strata.
type SupCache = std::sync::Mutex<BTreeMap<(u64, u64), Vec<Option<(f64, f64)>>>>;

/// `max_{l' <= l} D_c(l')` for a stratum formula, tabulated once per
/// `(s, k)` over `[0, kod)` in parallel and looked up thereafter.
/// Requires `l < kod`.
fn cached_prefix_sup(
    cache: &SupCache,
    s: u64,
    k: u64,
    l: u64,
    formula: StratumFormula,
) -> Option<Lg> {
    let kod = k / 2;
    assert!(l < kod, "d_c_sup at l = {l} outside [0, {kod}) for k = {k}");
    // the table holds only rounded endpoints, so a poisoned lock is
    // still valid
    let mut cache = cache
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    let prefix = cache.entry((s, k)).or_insert_with(|| {
        let np = s / 2;
        let facts = LgFactorials::new(np);
        let binom = |n: u64, kk: u64| -> Lg { facts.binom(n, kk) };
        let vals: Vec<Option<Lg>> = (0..kod)
            .into_par_iter()
            .map(|lp| stratum_geometry(s, k, lp).map(|(np, h, lpp)| formula(np, h, lpp, &binom)))
            .collect();
        prefix_max_by_hi(vals.into_iter())
            .into_iter()
            .map(|v| v.as_ref().map(store))
            .collect()
    });
    let (lo, hi) = prefix[l as usize]?;
    Some(Lg::from_f64_bracket(lo, hi))
}

/// Word-free counting, valid at every prime.
///
/// * `D_b(a) = C(n, kod) * floor((s - 2 kod) / a)` at odd `k`: the
///   level sets of distinct members through a core are disjoint
///   subsets of the `s - 2 kod` free points, each of size `>= a`.
///   At even `k` the family through a core is one interpolant.
/// * `D_c(l) = 2^h C(n, l') C(n - l', h)`: the full stratum, every
///   configuration with its whole section cube.
pub struct TrivialInterface;

impl Interface for TrivialInterface {
    fn d_b(&self, s: u64, k: u64, a: u64) -> Lg {
        let kod = k / 2;
        assert!(2 * kod < s, "d_b needs k < s (got s = {s}, k = {k})");
        let per_core = if k % 2 == 1 {
            (s - 2 * kod) / a.max(1)
        } else {
            1
        };
        // past the available points the true count is zero; one is
        // still a valid bound and the bracket cannot say zero
        lg_binom_memo(s / 2, kod).mul(&Lg::from_u64(per_core.max(1)))
    }

    fn d_c(&self, s: u64, k: u64, l: u64) -> Option<Lg> {
        let (np, h, lp) = stratum_geometry(s, k, l)?;
        Some(trivial_d_c(np, h, lp, &lg_binom))
    }

    /// A unit struct has no field for a cache; the table is
    /// process-global.
    fn d_c_sup(&self, s: u64, k: u64, l: u64) -> Option<Lg> {
        static SUP: std::sync::OnceLock<SupCache> = std::sync::OnceLock::new();
        let cache = SUP.get_or_init(|| std::sync::Mutex::new(BTreeMap::new()));
        cached_prefix_sup(cache, s, k, l, trivial_d_c)
    }
}

/// `2^h C(n, l') C(n - l', h)`.
fn trivial_d_c(np: u64, h: u64, lp: u64, binom: &dyn Fn(u64, u64) -> Lg) -> Lg {
    Lg::from_u64(2)
        .pow(h)
        .mul(&binom(np, lp))
        .mul(&binom(np - lp, h))
}

/// The general cut bound with the joint count made word-free: the
/// joint sets of a stratum form a pencil over a nonempty base locus
/// whenever `2l' <= h + 1`, and are counted outright otherwise.
/// Sharpens only the cut face; `D_b` is [`TrivialInterface`]'s.
///
/// `D_c(l) = 2^(h-1) C(n - l', h) ( C(n, l') + J )`, with
/// `J = C(n - 1, l' - 1)` when `2l' <= h + 1` (the joint sets form a
/// pencil) and `J = C(n, l')` otherwise.
pub struct ShowerInterface {
    sup: SupCache,
}

impl ShowerInterface {
    /// A provider with an empty table; the first query at each
    /// `(s, k)` fills it.
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

/// `2^(h-1) C(n - l', h) (C(n, l') + J)`; at `l' = 0` the joint term
/// occurs only at `b = 0`, which is out of scope, so it is omitted.
fn shower_d_c(np: u64, h: u64, lp: u64, binom: &dyn Fn(u64, u64) -> Lg) -> Lg {
    let sections = Lg::from_u64(2).pow(h.saturating_sub(1));
    let halves = binom(np - lp, h);
    let config = binom(np, lp).mul(&halves);
    if lp == 0 {
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

    fn d_c_sup(&self, s: u64, k: u64, l: u64) -> Option<Lg> {
        cached_prefix_sup(&self.sup, s, k, l, shower_d_c)
    }
}

/// The exact cut face: `sup_b |Z^(l)(b)|` in closed form, valid at
/// every prime and attained by the point-divisor syndromes (the
/// syndromes orthogonal to every core through one fiber, with the
/// secant line through one of its square roots). `D_b` is
/// [`TrivialInterface`]'s.
///
/// `D_c(l) = 2^h C(n-1, l'-1) C(n-l', h) + 2^(h-1) C(n-1, l') C(n-l'-1, h-1)`,
/// which is `(k+1)/s` times the full stratum.
pub struct StarInterface {
    sup: SupCache,
}

impl StarInterface {
    /// A provider with an empty table; the first query at each
    /// `(s, k)` fills it.
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

/// The star term `2^h C(n-1, l'-1) C(n-l', h)` (absent at `l' = 0`)
/// plus the point term `2^(h-1) C(n-1, l') C(n-l'-1, h-1)`. At
/// `h = 0`, outside the `d_c` domain at odd `k`, only the window
/// lemma's `C(n-1, l'-1)` remains.
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

    fn d_c_sup(&self, s: u64, k: u64, l: u64) -> Option<Lg> {
        cached_prefix_sup(&self.sup, s, k, l, star_d_c)
    }
}
