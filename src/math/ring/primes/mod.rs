//! The prime axis of `Z[zeta_s]`: how rational primes behave under the
//! ring — which `p = 1 (mod s)` admit small kernel vectors, and hence
//! which create bucket coincidences beyond the structural floor.
//!
//! The criterion chain is arithmetic end to end: a bounded-coefficient
//! vector `v` lies in the kernel of `Z[zeta_s] -> F_p` iff `p` divides
//! its norm; norms of bounded-height vectors are confined by the height
//! law `N(v) <= (sum v_i^2)^{s/4}`; enumerating and factoring them
//! inverts to the complete per-prime accident inventory. The modules:
//!
//! - [`kernel`]: the kernel-vector engines (direct and MitM counting) —
//!   moved here from `census` because they count *ring elements*, not
//!   subsets; `census::kernel` remains as a re-export.
//! - [`norms`]: exact norm tables and bad-set enumeration — the
//!   enumerate -> norm -> factor -> invert pipeline producing complete
//!   per-prime accident inventories, with [`norms::ingest`] streaming
//!   GPU-computed norm tables; moved from `smooth::norms` (its
//!   interface never involved a subgroup), which remains as a
//!   re-export.
//!
//! The bucket-level consequences of these facts (class merges, ladder
//! values, certificates with bucket semantics) live above, in
//! `smooth::certify`, which delegates its criterion here.

pub mod kernel;
pub mod norms;
