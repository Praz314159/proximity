//! The negacyclic ring `Z[zeta_s] = Z[x]/(x^{s/2} + 1)` (`s` a power of
//! two) — the characteristic-zero home of the program's exact values,
//! and the single authority for its conventions.
//!
//! The recurring bug class this module retires is the *negacyclic fold*:
//! reducing an exponent past `s/2` while dropping the sign from
//! `zeta^{s/2} = -1`. The [`fold`] primitive defines that operation once;
//! [`Cyclo`] is the element type for construction, orbit/norm reasoning,
//! and the Python boundary.
//!
//! Division of labor (the design rule that shaped the migration of
//! `census` / `norms` / `certify` onto this module): hot enumeration
//! loops — kernel censuses, norm tables — keep their flat, zero-alloc
//! representations and route only their exponent reduction through
//! [`fold`]; a `Cyclo` per enumerated item would be a severe regression.
//! `Cyclo` is the layer for construction, reasoning, and boundaries, and
//! glue tests pin each flat loop to the ring's own arithmetic
//! (`norm_table` to [`Cyclo::norm_i128`], the censuses to the
//! [`Cyclo::eval_at`] kernel, certification tiers to
//! [`Cyclo::norm_mod`] divisibility). The bucket side (`buckets::mitm`,
//! symmetric-function signs in `F_p[Y]`) is deliberately out of scope —
//! a different ring with its own institutionalized convention.
//!
//! Norms: [`Cyclo::norm_mod`] computes `N(v) mod p` — the workhorse of
//! accident manufacturing and per-prime cleanliness certificates
//! (`p` divides `N(v)` iff `norm_mod(p) == 0`) — with no big-integer
//! arithmetic. [`Cyclo::norm_i128`] is the exact norm when it fits;
//! larger norms are reconstructed by the caller via CRT over
//! `norm_mod` values. Known perf headroom, if norm batches ever become
//! a bottleneck: HEXL-style preconditioned butterflies in [`ntt`].

mod cyclo;
pub mod foldunits;
pub mod ntt;
pub mod primes;

pub use cyclo::Cyclo;
pub use foldunits::{
    alpha_certificate, fold_unit, rank_certificate, AlphaCertificate, RankCertificate,
};

/// THE fold: reduce `zeta^exp` on the half-basis. Returns
/// `(index, sign)` with `zeta^exp = sign * zeta^index`, `index < half`.
#[inline]
pub fn fold(half: usize, exp: usize) -> (usize, i64) {
    // one division + one branch measures fastest on the hot paths
    // (branch-chain variants mispredict on irregular exponents —
    // measured 2026-07-24, see hotpath_baseline).
    let e = exp % (2 * half);
    if e < half {
        (e, 1)
    } else {
        (e - half, -1)
    }
}
