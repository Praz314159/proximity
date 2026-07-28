//! The vanishing-syndrome geometry `VS(s, k)` — module root.
//!
//! [`space`] holds the dual (quotient) view of Reed-Solomon —
//! [`VsSpace`], the convention authority, and [`VsCertificate`].
//! The census kernels that count over this geometry live in
//! [`crate::census`].

mod space;

pub use space::{VsCertificate, VsSpace};
