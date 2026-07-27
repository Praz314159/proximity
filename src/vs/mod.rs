//! The vanishing-syndrome geometry `VS(s, k)` — module root.
//!
//! - [`space`]: the dual (quotient) view of Reed-Solomon — [`VsSpace`],
//!   the convention authority, and [`VsCertificate`].
//! - [`census`]: the integer-exact `Z[zeta_s]` value census
//!   ([`exact_value_census`]) — the floor preamble of the pointwise
//!   L^2 program.
//!
//! Public paths are re-exported here so downstream code and the Python
//! bindings are unaffected by the file layout.

mod census;
mod space;
pub mod valuemap;

pub use census::exact_value_census;
pub use space::{VsCertificate, VsSpace};
