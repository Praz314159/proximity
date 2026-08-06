//! The soundness chain — one spine, two faces.
//!
//! Everything here converts certified counts into certified soundness
//! statements at the challenge box, as machine-checked interval
//! brackets ([`Lg`](crate::math::enclosure::Lg) enclosures). The
//! layering: [`volumes`] counts (balls, expected lists, the exact
//! Elias count); [`chain`] converts (the Lemma 6.12 soundness map and
//! the z-lattice crossing reports); [`floor`] consumes counts on the
//! attack side — what adversaries certifiably achieve. [`ceiling`]
//! consumes the master theorem's list envelope through the
//! identical chain ([`ceiling`]), and the prize's pinch is one
//! testable assertion: the floor's certified crossing lands one step
//! above the ceiling's.
//!
//! The flat namespace is preserved: every item re-exports here, and
//! `attack::certified` remains an alias of this module.

pub mod ceiling;
pub mod chain;
pub mod floor;
pub mod volumes;

pub use ceiling::*;
pub use chain::*;
pub use floor::*;
pub use volumes::*;
