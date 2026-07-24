//! The negacyclic ring `Z[zeta_s] = Z[x]/(x^{s/2} + 1)` (`s` a power of
//! two) — the characteristic-zero home of the program's exact values
//! (see `design/negacyclic_ring.md`).
//!
//! The recurring bug class this module retires is the *negacyclic fold*:
//! reducing an exponent past `s/2` while dropping the sign from
//! `zeta^{s/2} = -1`. The [`fold`] primitive defines that operation once;
//! [`Cyclo`] is the element type for construction, orbit/norm reasoning,
//! and the Python boundary. Hot enumeration loops keep their flat
//! representations and call [`fold`] (see `vs::exact_value_census`).
//!
//! Norms: [`Cyclo::norm_mod`] computes `N(v) mod p` — the workhorse of
//! accident manufacturing and per-prime cleanliness certificates
//! (`p` divides `N(v)` iff `norm_mod(p) == 0`) — with no big-integer
//! arithmetic. [`Cyclo::norm_i128`] is the exact norm when it fits;
//! larger norms are reconstructed by the caller via CRT over
//! `norm_mod` values.

mod cyclo;
pub mod ntt;

pub use cyclo::Cyclo;

/// THE fold: reduce `zeta^exp` on the half-basis. Returns
/// `(index, sign)` with `zeta^exp = sign * zeta^index`, `index < half`.
#[inline]
pub fn fold(half: usize, exp: usize) -> (usize, i64) {
    let e = exp % (2 * half);
    if e < half {
        (e, 1)
    } else {
        (e - half, -1)
    }
}
