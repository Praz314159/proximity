//! The **smooth multiplicative subgroup** program: the accident analysis
//! specific to `mu_s <= F_p^*` with `s` a power of two — the rung/ladder
//! combinatorics ([`rung`]), the bad-set pipeline over the kernel census
//! ([`norms`]), and the structural certificates ([`certify`]). The
//! counting engines themselves (buckets, kernel vectors) live in
//! [`crate::census`].

pub mod certify;
pub mod norms;
pub mod rung;
