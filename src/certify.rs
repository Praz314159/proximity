//! Structural certification: prove, for a given `(p, s, r)`, that bucket sizes
//! are *exactly* their characteristic-zero (ladder) values — i.e. that no
//! arithmetic accident inflates any list at these parameters.
//!
//! The logic rests on the (exactly validated) decomposition law: two
//! structural classes merge mod `p` iff their `eps`-difference — a vector with
//! coefficients in `[-2, 2]` — lies in the kernel of `Z[zeta_s] -> F_p`; the
//! zero-class bucket in particular can only merge via `{-1, 0, 1}`-valued
//! kernel vectors. Hence:
//!
//! - kernel census empty at `cmax = 2` ⟹ **every** `q = 1` bucket is a single
//!   structural class; the maximum bucket equals the ladder value.
//! - kernel census empty at `cmax = 1` ⟹ the zero-coset bucket (the rung /
//!   C.6 word's exact list, by the exactness theorem) is exactly structural.
//!
//! When a census is nonzero, the certificate reports the accident orbits and
//! the *exact* inflated zero-bucket via the decomposition engine instead —
//! the output is exact either way; only its classification differs.

use crate::buckets::mitm::decompose_bucket_q1;
use crate::census;
use crate::code::{class_size, m_struct};
use crate::domain::MultiplicativeSubgroup;
use crate::error::{Error, Result};

/// Certification verdict for `(p, s, r)`, `q = 1`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// Kernel census empty at coefficient range `[-2, 2]`: every bucket is a
    /// single structural class; max bucket = ladder value, provably.
    AllBucketsStructural,
    /// `{-1,0,1}` census empty (zero-bucket exactly structural = the rung
    /// word's list), but some `[-2,2]` kernel vectors exist, so *other*
    /// buckets may merge classes.
    ZeroBucketStructural {
        /// Nonzero `[-2, 2]` census counts by weight.
        census2_by_weight: Vec<u64>,
    },
    /// `{-1,0,1}` kernel vectors exist: the zero bucket is inflated. Contains
    /// the exact accounting.
    Inflated {
        /// `{-1,0,1}` census counts by weight (vectors, in dilation orbits of
        /// size `s`).
        census1_by_weight: Vec<u64>,
        /// Exact zero-coset bucket (sum of merged structural classes).
        zero_bucket: u64,
        /// Per-weight class counts inside the zero bucket.
        zero_profile: Vec<u64>,
    },
}

/// A certificate for the `q = 1` bucket landscape at `(p, s, r)`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Certificate {
    /// The verdict (see [`Verdict`]).
    pub verdict: Verdict,
    /// The structural ladder value `M_struct(s, r, 1)` (the certified max
    /// bucket when the verdict is [`Verdict::AllBucketsStructural`]).
    pub m_struct: u64,
    /// The zero-class size `C(s/2, ...)` = the structural zero bucket.
    pub zero_class: u64,
}

/// Certify the `q = 1` landscape at `(p, s, r)`. Requires power-of-two
/// `s <= 32` (the full-census meet-in-the-middle range); the census cost is
/// `p`-independent, so `p` may be arbitrarily large.
pub fn certify_q1(sg: &MultiplicativeSubgroup, r: usize) -> Result<Certificate> {
    if !sg.is_two_smooth() || sg.order() > 32 {
        return Err(Error::Unsupported(
            "certification requires power-of-two s <= 32".into(),
        ));
    }
    if r == 0 || r >= sg.order() {
        return Err(Error::OutOfRange("need 1 <= r < s".into()));
    }
    let s = sg.order();
    let m1 = census::mitm(sg, 1)?;
    let cert = Certificate {
        verdict: Verdict::AllBucketsStructural, // provisional
        m_struct: m_struct(s, r, 1),
        zero_class: class_size(s, r, 0),
    };
    if m1.iter().any(|&c| c > 0) {
        let (zero_bucket, zero_profile) = decompose_bucket_q1(sg, r, 0)?;
        return Ok(Certificate {
            verdict: Verdict::Inflated {
                census1_by_weight: m1,
                zero_bucket,
                zero_profile,
            },
            ..cert
        });
    }
    let m2 = census::mitm(sg, 2)?;
    if m2.iter().any(|&c| c > 0) {
        return Ok(Certificate {
            verdict: Verdict::ZeroBucketStructural {
                census2_by_weight: m2,
            },
            ..cert
        });
    }
    Ok(cert)
}
