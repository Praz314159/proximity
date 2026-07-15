//! The **smooth multiplicative subgroup** program: the bucket / accident
//! analysis specific to `mu_s <= F_p^*` with `s` a power of two — bucket sizes
//! ([`buckets`]), the rung/ladder combinatorics ([`rung`]), the arithmetic
//! accidents that inflate buckets ([`census`], [`norms`]), and the structural
//! certificates ([`certify`]).

pub mod buckets;
pub mod census;
pub mod certify;
pub mod norms;
pub mod rung;
