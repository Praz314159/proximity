//! Attack-threshold calculators, in two rigor tiers.
//!
//! [`ladder`] is the float parameter-space explorer: fast, uncertified,
//! for scanning where the best known attacks sit. Its public API is
//! re-exported here, so existing `crate::attack::*` paths are unchanged.
//! The certified tier (`certified`, behind the `certified` feature)
//! computes the same kind of statement — parameters in, attack radius
//! out — as machine-checked interval brackets fit to cite; explore with
//! the ladder, cite from the certified tier.

#[cfg(feature = "certified")]
pub use crate::soundness as certified;
pub mod ladder;
pub use ladder::*;
