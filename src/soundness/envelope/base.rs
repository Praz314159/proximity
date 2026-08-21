//! The base of the tower: unconditional, prime-free seeds
//! (interpolation, sharp Johnson, the ownership shower bound) and
//! the per-level analytic clamps the step applies pointwise.

use std::collections::{BTreeMap, BTreeSet};

use crate::error::{Error, Result};
use crate::math::enclosure::{lg_binom, lg_binom_memo, Lg};
use rug::Integer;

use super::profile::{store, Profile};

/// The interpolation base: `E(n0, k, t) = C(n0, k)` for `t >= k + 1`
/// — a member agreeing on at least the rung is the interpolant of `w`
/// on any `k`-subset of its agreement set, so the map to that subset
/// is injective and the list is at most the number of subsets. The
/// crudest citable base, kept as the reference floor;
/// [`analytic_base`] dominates it pointwise and is [`super::assemble`]'s
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

/// The Johnson agreement bound's kernel: the unreduced
/// `(numerator, denominator) = (n (t - k + 1), t^2 - n (k - 1))`
/// pair in `u128` (u64 products wrap in release at levels past
/// 2^32, and a wrapped clause could sit BELOW the truth — the one
/// failure mode this module must never have), gated on the
/// quadratic validity condition. The single home of the formula:
/// the base's floored refinement, the per-level clamp, the derived
/// (graded) multiplicity, and the test mirrors all wrap this pair.
/// Callers add their own regime gates (e.g. the derived charge's
/// monotone-safety) — those are NOT validity conditions and stay at
/// the call sites.
pub(super) fn johnson_agreement(n: u64, k: u64, t: u64) -> Option<(u128, u128)> {
    let (tw, nw, kw) = (t as u128, n as u128, k as u128);
    if kw == 0 || tw * tw <= nw * (kw - 1) {
        return None;
    }
    Some((nw * (tw - kw + 1), tw * tw - nw * (kw - 1)))
}

/// Lower `best` to the sharper of the Johnson and shower counts at
/// `(n, k, t)` where they apply — the analytic statement's two
/// nontrivial clauses in exact integers, for the base constructor.
fn analytic_refine(n: u64, k: u64, t: u64, best: &mut Integer) {
    if let Some((num, den)) = johnson_agreement(n, k, t) {
        let johnson = Integer::from(num / den);
        if johnson < *best {
            *best = johnson;
        }
    }
    // skipping the shower clause on a u32 overflow is conservative
    // (the clamp is a min); do not rely on the caller's validation
    let (Ok(n32), Ok(r32), Ok(t32)) = (u32::try_from(n), u32::try_from(k + 1), u32::try_from(t))
    else {
        return;
    };
    let shower =
        Integer::from(Integer::binomial_u(n32, r32)) / Integer::from(Integer::binomial_u(t32, r32));
    if shower < *best {
        *best = shower;
    }
}

/// The analytic counts at `(n, k, t)` as log brackets — the same
/// three clauses as [`analytic_refine`] without the floors (a valid
/// loosening), in log-gamma arithmetic so the per-level clamp costs
/// microseconds where the exact binomials would cost million-bit
/// integers.
pub(super) fn analytic_brackets(n: u64, k: u64, t: u64) -> Vec<Lg> {
    // the two `(n, k)` binomials are per-level constants queried at
    // every grid point — memoized; only `C(t, k+1)` varies with `t`
    let mut out = vec![
        lg_binom_memo(n, k),
        lg_binom_memo(n, k + 1).div(&lg_binom(t, k + 1)),
    ];
    if let Some((num, den)) = johnson_agreement(n, k, t) {
        out.push(Lg::from_integer(&Integer::from(num)).div(&Lg::from_integer(&Integer::from(den))));
    }
    out
}
