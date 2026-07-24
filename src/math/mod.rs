//! The pure-arithmetic substrate, independent of any code or protocol:
//!
//! - [`field`]: prime-field arithmetic — `mulmod`/`powmod`, primality,
//!   factoring, prime enumeration.
//! - [`ring`]: the cyclotomic (negacyclic) ring `Z[zeta_s]` — the
//!   [`ring::fold`] primitive, [`ring::Cyclo`], and the NTTs
//!   ([`ring::ntt`]).
//!
//! The coding-theory layers (`crate::rs`, `crate::vs`, `crate::smooth`)
//! consume these; both are re-exported at the crate root, so
//! `crate::field` and `crate::ring` remain valid paths.

pub mod field;
pub mod ring;
