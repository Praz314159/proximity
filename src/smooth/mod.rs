//! The **smooth multiplicative subgroup** program: the accident analysis
//! specific to `mu_s <= F_p^*` with `s` a power of two — the rung/ladder
//! combinatorics ([`rung`]), the bad-set pipeline over the kernel census
//! ([`norms`]), and the structural certificates ([`certify`]). The
//! counting engines themselves (buckets, kernel vectors) live in
//! [`crate::census`].

pub mod certify;
/// Moved to [`crate::ring::primes::norms`] (pure number theory —
/// its interface never involved a subgroup); re-export keeps paths
/// working.
pub use crate::ring::primes::norms;
pub mod rung;
