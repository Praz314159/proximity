//! # vanish
//!
//! An exact computational toolkit for exploring **proximity gaps, correlated
//! agreement, and list decoding near capacity** for the smooth-domain
//! Reed–Solomon codes used in SNARKs (the setting of the Proximity Prize
//! survey, ePrint 2026/680).
//!
//! ## The mathematical objects, bottom-up
//!
//! - [`field`]: arithmetic over `F_p` (primality, generators, factorization).
//! - [`domain`]: the core object — a multiplicative subgroup `mu_s <= F_p^*`
//!   ([`domain::Subgroup`]), with its cosets and dilation structure.
//! - [`code`]: Reed–Solomon codes on a subgroup domain
//!   ([`code::ReedSolomon`]), their radii (capacity, Johnson), the C.5-form
//!   extremal words, and the quantized-ladder combinatorics
//!   ([`code::m_struct`], [`code::rung_lambda`]).
//! - [`buckets`]: the central analysis — *bucket* sizes
//!   `#{ |S| = r : e_i(S) = lambda_i }`, which by the exactness theorem are
//!   exact list sizes of the extremal words beyond the Johnson radius. Full
//!   distributions ([`buckets::dp`], cost ~ `p`) and `p`-independent single
//!   buckets / decompositions ([`buckets::mitm`]).
//! - [`census`]: kernel censuses — the arithmetic accidents that inflate
//!   buckets, in dilation orbits, governed by the anticorrelation law
//!   `N(v) <= (sum v_i^2)^{s/4}`.
//! - [`norms`]: cyclotomic norms and complete *bad sets* — every prime any
//!   bounded-weight accident can ever visit, by enumerating and factoring
//!   norms; [`norms::ingest`] streams externally computed (GPU) norm tables
//!   through the same pipeline at the billions-of-entries scale.
//! - [`certify`]: tiered, `p`-independent structural certificates — prove
//!   that no accident inflates any bucket at given parameters, or report the
//!   exact inflated anatomy.
//! - [`toy`]: exact soundness of the survey's Section-6 toy protocol via the
//!   winning-set identity (winning challenges = occupied buckets).
//! - [`attack`]: the parameter-space calculator — best attack radius over the
//!   quantized ladder, the antipodal baseline, the structural ceiling, and
//!   the Elias threshold. Deliberately float-domain and standalone; its exact
//!   integer counterpart is [`code::m_struct`].
//!
//! ## Validation contract
//!
//! Every kernel is pinned to exhaustively-verified golden values
//! (`tests/golden.rs`) and property tests (mass `= C(s,r)`, dilation
//! invariance, DP <-> MitM agreement). Contributions must keep the suite
//! green and extend it; see `CONTRIBUTING.md`.
//!
//! ## Example
//!
//! ```
//! use vanish::{domain::Subgroup, buckets};
//!
//! let sg = Subgroup::new(3457, 32).unwrap();
//! let dist = buckets::dp::distribution_q1(&sg, 16).unwrap();
//! let (max, lambda) = dist.max();
//! assert_eq!(max, 220134);          // exhaustively verified golden value
//! assert_eq!(lambda, 0);
//! // p-independent cross-check via meet-in-the-middle:
//! let tables = buckets::mitm::HalfTables::build(&sg, 16, 1).unwrap();
//! assert_eq!(tables.bucket(&[lambda]).unwrap(), max);
//! ```

// `n % d == 0` is the standard idiom in number-theoretic code; clippy's
// `is_multiple_of` suggestion hurts readability here.
#![allow(clippy::manual_is_multiple_of)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod attack;
pub mod buckets;
pub mod census;
pub mod certify;
pub mod classify;
pub mod cluster;
pub mod code;
pub mod decode;
pub mod domain;
pub mod error;
pub mod field;
pub mod norms;
pub mod toy;

pub use error::{Error, Result};

#[cfg(feature = "python")]
mod py;
