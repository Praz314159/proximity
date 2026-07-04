//! bucketlab: fast exact kernels for the bucket-landscape program.
//!
//! Validation discipline (see notes/anatomy.tex): every kernel here carries
//! golden tests pinned to exhaustively-verified Python values (tests/golden.rs)
//! and property checks (mass = C(s,r), dilation invariance, DP <-> MitM
//! agreement). Run `cargo test --release` (default features; no Python needed).
//! Build the Python module with `maturin develop --release` (enables the
//! `python` feature).

pub mod census;
pub mod dp;
pub mod field;
pub mod mitm;

#[cfg(feature = "python")]
mod py;
