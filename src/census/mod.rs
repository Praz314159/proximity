//! The census kernels — every exact counting engine of the program in
//! one place, organized by *what is counted*:
//!
//! - [`kernel`]: **kernel vectors** — bounded-coefficient solutions of
//!   `sum v_i w^i = 0 (mod p)`, the arithmetic accidents whose weighted
//!   count is exactly the bucket inflation.
//! - [`value`]: **exact ring values** — the integer-exact `Z[zeta_s]`
//!   census of an elementary-symmetric coordinate over all `r`-subsets
//!   ([`exact_value_census`]): distinct values and the prime-independent
//!   multiplicity floor `M_inf`.
//! - [`valuemap`]: **value-map fibers mod `p`** — MITM subset-product
//!   fiber statistics (totals, maxima, the second moment / collision
//!   count), fiber members, histograms, prime sweeps.
//! - [`buckets`]: **bucket fibers** — `#{ |S| = r : e_i(S) = lambda_i }`,
//!   exact list sizes of the extremal C.5 words (DP and MITM engines).
//! - [`skeleton`]: **criterion solutions** — the G1 skeleton census:
//!   unit-equation solution counts derived through the proven
//!   budget/congruence/box/realization stack rather than enumeration.
//!
//! The MITM engines share one vocabulary: enumerate a *side* (half the
//! configuration space) incrementally, bucket it by a *key*, and join
//! each side entry against the *complementary* key of the target. The
//! sides and complement laws are algebra-specific (subsets with
//! products, coefficient vectors with sums, slot codes with addresses).

pub mod buckets;
pub mod kernel;
pub mod skeleton;
pub mod value;
pub mod valuemap;

pub use value::exact_value_census;
