//! The pure-arithmetic substrate, independent of any code or protocol:
//!
//! - [`field`]: prime-field arithmetic — `mulmod`/`powmod`, primality,
//!   factoring, prime enumeration.
//! - [`poly`]: dense univariate polynomials over `F_p` — evaluation,
//!   interpolation, modular arithmetic, gcd, root finding.
//! - [`ring`]: the cyclotomic (negacyclic) ring `Z[zeta_s]` — the
//!   [`ring::fold`] primitive, [`ring::Cyclo`], and the NTTs
//!   ([`ring::ntt`]).
//! - `enclosure` (behind the `certified` feature): certified bracket
//!   arithmetic — interval enclosures of base-2 logarithms under MPFR
//!   directed rounding, for quantities too large for the other two.
//!
//! The coding-theory layers (`crate::rs`, `crate::smooth`)
//! consume these; both are re-exported at the crate root, so
//! `crate::field` and `crate::ring` remain valid paths.

#[cfg(feature = "certified")]
pub mod enclosure;
pub mod field;
pub mod poly;
pub mod ring;
