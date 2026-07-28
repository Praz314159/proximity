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
//!
//! The ring `Z[zeta_s]` here is [`crate::ring`]: an `eps`-difference is a
//! bounded-height [`crate::ring::Cyclo`], "in the kernel" means its
//! [`eval_at`](crate::ring::Cyclo::eval_at) vanishes at the subgroup
//! generator, and `p` admits such a vector iff `p` divides a
//! [`norm`](crate::ring::Cyclo::norm_mod) — the glue tests below pin the
//! census verdicts to the ring's own arithmetic.

use crate::census::buckets::mitm::decompose_bucket_q1;
use crate::census::kernel as census;
use crate::domain::MultiplicativeSubgroup;
use crate::error::{Error, Result};
use crate::smooth::rung::{class_size, m_struct};

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::Cyclo;

    /// The s=16 die-out witness (`N = 9986 = 2 * 4993`, pinned in the ring
    /// tests): at its accident prime the certificate must degrade exactly
    /// one notch — the `{-1,0,1}` census stays empty (Parseval:
    /// `N <= 8^4 < 4993`), the `[-2,2]` census fires — and the ring
    /// explains the accident: `p | N(v)`.
    #[test]
    fn accident_prime_degrades_certificate_and_ring_explains_it() {
        let v = Cyclo::from_coeffs(vec![2, 2, 0, 2, -1, 0, 0, -1]).unwrap();
        assert_eq!(v.norm_mod(4993).unwrap(), 0);
        let sg = MultiplicativeSubgroup::new(4993, 16).unwrap();
        let cert = certify_q1(&sg, 8).unwrap();
        match cert.verdict {
            Verdict::ZeroBucketStructural { census2_by_weight } => {
                assert!(census2_by_weight.iter().sum::<u64>() > 0);
            }
            other => panic!("expected ZeroBucketStructural, got {other:?}"),
        }
    }

    /// Above the realized accident range (`P_max(16) = 4993`) the full
    /// certificate holds: every bucket structural, max = ladder value.
    #[test]
    fn clean_prime_certifies_all_buckets_structural() {
        let sg = MultiplicativeSubgroup::new(65537, 16).unwrap();
        let cert = certify_q1(&sg, 8).unwrap();
        assert_eq!(cert.verdict, Verdict::AllBucketsStructural);
        assert_eq!(cert.m_struct, m_struct(16, 8, 1));
        assert_eq!(cert.zero_class, class_size(16, 8, 0));
    }
}
