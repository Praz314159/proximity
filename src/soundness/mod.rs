//! Threshold calculus for the grand challenge — two faces, two tiers.
//!
//! Everything here is a judgment about parameters in the challenge's
//! list currency: where does a list cross the `eps* |F|` budget? The
//! two faces converge on the answer from opposite sides. The **floor**
//! face is the attack side — certified *lower* bounds on lists
//! (counting and construction rows); past its crossing, lists provably
//! exceed the budget. The **ceiling** face is the defense side —
//! certified *upper* bounds from the master theorem's envelope; up to
//! its crossing, lists provably fit. The challenge is resolved at a
//! radius where the two crossings meet on the lattice.
//!
//! Two rigor tiers cut across both faces. [`explore`] is the float
//! tier: fast `f64` scans of the parameter space, always compiled,
//! nothing citable. The certified tier — behind the `certified`
//! feature — computes machine-checked interval brackets
//! (`Lg` enclosures): `volumes` counts
//! (balls, expected lists, the exact Elias count); `chain` converts
//! (the Lemma 6.12 soundness map, the budget bracket, and the
//! z-lattice crossing reports); `floor` and `ceiling` are the two
//! faces as rows. The module rule: every certified row cites the named
//! theorem (or the challenge statement itself) backing its comparison.
//!
//! The namespace is flat: every item re-exports here.

pub mod explore;
pub use explore::*;

#[cfg(feature = "certified")]
pub mod ceiling;
#[cfg(feature = "certified")]
pub mod chain;
#[cfg(feature = "certified")]
pub mod floor;
#[cfg(feature = "certified")]
pub mod volumes;

#[cfg(feature = "certified")]
pub use ceiling::*;
#[cfg(feature = "certified")]
pub use chain::*;
#[cfg(feature = "certified")]
pub use floor::*;
#[cfg(feature = "certified")]
pub use volumes::*;
