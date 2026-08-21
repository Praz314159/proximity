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
//! (`Lg` enclosures): `volumes` counts for the attack side (balls,
//! expected lists, the exact Elias count); `envelope` assembles the
//! defense side's object (the profile tower of the conditional
//! corollary); `chain` converts (the soundness map, the
//! budget bracket, and the z-lattice crossing reports); `floor` and
//! `ceiling` are the two faces as rows. Every certified row names the
//! statement backing its comparison.
//!
//! Throughout, *the box* is the deployment cell: level `s = 2^21` at
//! rate one half (`k = s/2 - 1`) over the degree-6 extension of
//! KoalaBear, with challenge budget `eps* = 2^-128`; *the reduced box*
//! is the same at `s = 2^12`.
//!
//! The namespace is flat: every item re-exports here.
//!
//! Reference. ABF26: G. Arnon, D. Boneh, G. Fenzi, *Open Problems
//! in List Decoding and Correlated Agreement*, 2026. Lemma and
//! theorem numbers cited in this module are to that paper.

pub mod explore;
pub use explore::*;

#[cfg(feature = "certified")]
pub mod ceiling;
#[cfg(feature = "certified")]
pub mod chain;
#[cfg(feature = "certified")]
pub mod envelope;
#[cfg(feature = "certified")]
pub mod floor;
#[cfg(feature = "certified")]
pub mod volumes;

#[cfg(feature = "certified")]
pub use ceiling::*;
#[cfg(feature = "certified")]
pub use chain::*;
#[cfg(feature = "certified")]
pub use envelope::*;
#[cfg(feature = "certified")]
pub use floor::*;
#[cfg(feature = "certified")]
pub use volumes::*;
