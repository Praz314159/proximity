//! Exact soundness analysis of the Proximity-Prize survey's Section-6 toy
//! protocol (Constructions 6.2/6.9).
//!
//! For the canonical attack pair `f1 = x^r`, `f2 = x^{r-1}` on `mu_s` with
//! `k = r - 1`, a challenge `gamma` lies in the winning set (Definition 6.11,
//! `x = (0, 0, 0)`) iff `f1 + gamma * f2` agrees with a codeword on `>= r`
//! points iff some `r`-subset `S` has `e_1(S) = -gamma`. Hence the winning
//! set is exactly the (negated) support of the `q = 1` bucket distribution,
//! and the protocol's exact soundness under this attack is
//! `occupied_buckets / p` — identically `1` while `p` is below the structural
//! class count (every challenge wins), then decaying like `classes / p`.

use crate::buckets::dp;
use crate::domain::MultiplicativeSubgroup;
use crate::error::Result;
use crate::field::binom;

/// Exact toy-protocol soundness data at `(p, s, r)`.
#[derive(Debug, Clone, Copy)]
pub struct ToySoundness {
    /// Number of winning challenges `|Omega|` (occupied buckets).
    pub winning: u64,
    /// Exact soundness error `winning / p` for the canonical attack.
    pub soundness: f64,
    /// The characteristic-zero ceiling on `winning`: the number of nonempty
    /// structural classes (the phase-transition scale).
    pub classes: u64,
}

/// Number of nonempty structural `e_1`-classes for `r`-subsets of `mu_s`
/// (`s` a power of two): `sum_w C(s/2, w) 2^w` over feasible weights.
#[must_use]
pub fn classes_count(s: usize, r: usize) -> u64 {
    let half = (s / 2) as u64;
    let mut tot = 0u64;
    let mut w = (r % 2) as u64;
    while w <= r.min(s / 2) as u64 {
        if (r as u64 - w) / 2 <= half - w {
            tot += binom(half, w) * (1u64 << w);
        }
        w += 2;
    }
    tot
}

/// Exact winning-set size and soundness for the canonical attack pair at
/// `(p, s, r)` (cost scales with `p` via the full DP distribution).
pub fn exact_soundness(sg: &MultiplicativeSubgroup, r: usize) -> Result<ToySoundness> {
    let d = dp::distribution_q1(sg, r)?;
    let winning = d.occupied();
    Ok(ToySoundness {
        winning,
        soundness: winning as f64 / sg.p() as f64,
        classes: classes_count(sg.order(), r),
    })
}
