//! # vanish
//!
//! An exact computational toolkit for exploring **proximity gaps, correlated
//! agreement, and list decoding near capacity** for the smooth-domain
//! Reed–Solomon codes used in SNARKs (the setting of the Proximity Prize
//! survey, ePrint 2026/680).
//!
//! ## Layout
//!
//! **Foundation.** [`field`] (`F_p` arithmetic), [`domain`] — the
//! [`MultiplicativeSubgroup`](domain::MultiplicativeSubgroup) `mu_s <= F_p^*`
//! (cosets, dilation) and the [`EvalDomain`](domain::EvalDomain) an RS code
//! sits on. Shared by both families below.
//!
//! **[`rs`] — generic Reed–Solomon + list-decoding discovery.**
//! [`ReedSolomon`](rs::code::ReedSolomon) over *any* evaluation domain; exact
//! and sampled list decoding ([`rs::decode`]); bottom-up cluster growth and
//! optimization ([`rs::cluster`]); the graded structure diagnostic
//! ([`rs::classify`]). Decoupled from the subgroup structure.
//!
//! **[`smooth`] — the smooth multiplicative-subgroup program.** *Bucket* sizes
//! `#{ |S| = r : e_i(S) = lambda_i }` ([`census::buckets`]), which by the
//! exactness theorem are exact list sizes beyond the Johnson radius; the
//! rung/ladder combinatorics ([`smooth::rung`]); the arithmetic accidents that
//! inflate buckets ([`census::kernel`], [`smooth::norms`], with
//! [`smooth::norms::ingest`] streaming GPU norm tables); and the structural
//! certificates ([`smooth::certify`]).
//!
//! **Applications.** [`toy`] (Section-6 toy-protocol soundness) and
//! [`attack`], in two rigor tiers: the float parameter-space explorer
//! (`attack::ladder`) and — behind the `certified` feature — the
//! certified tier (`attack::certified`), whose brackets come from the
//! third arithmetic, `math::enclosure`. Reproductions of published
//! attack tables are configuration in `examples/`.
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
//! use vanish::{census::buckets, domain::MultiplicativeSubgroup};
//!
//! let sg = MultiplicativeSubgroup::new(3457, 32).unwrap();
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

// Foundation.
pub mod census;
pub mod domain;
pub mod error;
pub mod math;

// Generic Reed–Solomon codes + list-decoding discovery.
/// The vanishing-syndrome geometry (dual view of RS on a subgroup).
pub mod rs;

// Smooth multiplicative subgroup: bucket & accident program.
pub mod smooth;

// Applications.
pub mod attack;
pub mod toy;

pub use error::{Error, Result};
#[cfg(feature = "certified")]
pub use math::enclosure;
pub use math::{field, ring};

#[cfg(feature = "python")]
mod py;
