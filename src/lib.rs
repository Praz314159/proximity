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
//! sits on. Shared by every layer above.
//!
//! **[`ring`] — exact `Z[zeta_s]` arithmetic, the characteristic-zero
//! home.** [`Cyclo`](ring::Cyclo) on the negacyclic half-basis, with norms
//! at every height (`norm_mod` / `norm_i128` / `norm_crt`); the fold; the
//! fold units and their certified alpha tables ([`ring::foldunits`]); exact
//! negacyclic NTT. What is computed here descends to every good prime at
//! once.
//!
//! **[`rs`] — Reed–Solomon, both views.** The primal view
//! [`ReedSolomon`](rs::code::ReedSolomon) over *any* evaluation domain (the
//! frozen-head words, the ladder), with exact and sampled list decoding
//! ([`rs::decode`]); the dual view [`VsSpace`](rs::vs::VsSpace) — syndromes,
//! cuts, cliques — which is the crate's **convention authority**: its
//! certificate (subset ranking, moment rows, domain order, syndrome signs)
//! is what every accelerated or external view must reproduce before it is
//! trusted. Alongside: the descent — the level-halving operation, channel
//! words, and derived words ([`rs::descent`]) — the moment cloud and cut
//! kernels ([`rs::moments`]), cluster growth ([`rs::cluster`]), and the
//! graded structure diagnostic ([`rs::classify`]).
//!
//! **[`census`] — every exact counting kernel**, organized by what is
//! counted: [`census::buckets`] (bucket sizes, which by the exactness
//! theorem are exact list sizes beyond the Johnson radius),
//! [`census::kernel`] (accident vectors), [`census::value`] (exact ring
//! values), [`census::valuemap`] (value-map fibers mod `p`),
//! [`census::skeleton`] (the unit-equation census), over the shared MITM
//! keyed-table layer [`census::join`].
//!
//! **[`smooth`] — the smooth-subgroup layer.** [`smooth::rung`] (the
//! quantized ladder and the closed-form top-word / GS-class layer),
//! [`smooth::norms`] (per-prime accident inventories via cyclotomic norms;
//! [`smooth::norms::ingest`] streams GPU norm tables), [`smooth::certify`]
//! (per-prime structural certificates).
//!
//! **Applications.** [`toy`] (toy-protocol soundness) and [`attack`], in
//! two rigor tiers: the float parameter-space explorer (`attack::ladder`)
//! and — behind the `certified` feature — the certified tier
//! (`attack::certified`), whose brackets come from the third arithmetic,
//! `math::enclosure`; reproductions of published attack tables are
//! configuration in `examples/`. GPU campaign drivers live in `gpu/`
//! (Python; each certifies itself against the `VsSpace` certificate
//! before use).
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
#[cfg(feature = "certified")]
pub mod soundness;
pub mod toy;

pub use error::{Error, Result};
#[cfg(feature = "certified")]
pub use math::enclosure;
pub use math::{field, ring};

#[cfg(feature = "python")]
mod py;
