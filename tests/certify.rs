//! Golden tests for structural certification, pinned to the exhaustively
//! verified landscape at s = 32 (and cross-checked against the DP engine,
//! which is independent of the census machinery).

use vanish::buckets::dp;
use vanish::certify::{certify_q1, Verdict};
use vanish::domain::Subgroup;

#[test]
fn golden_inflated_prime() {
    // p = 89633: one weight-6 orbit inflates the zero bucket to 29382
    let sg = Subgroup::new(89633, 32).unwrap();
    let c = certify_q1(&sg, 16).unwrap();
    match c.verdict {
        Verdict::Inflated {
            zero_bucket,
            ref zero_profile,
            ..
        } => {
            assert_eq!(zero_bucket, 29382);
            assert_eq!(zero_profile[6], 32, "one dilation orbit at weight 6");
            // independent cross-check: DP max at this prime is the zero bucket
            let d = dp::distribution_q1(&sg, 16).unwrap();
            assert_eq!(d.max().0, zero_bucket);
        }
        ref v => panic!("expected Inflated, got {v:?}"),
    }
}

#[test]
fn golden_fermat_prime_tiers() {
    // p = 65537: {-1,0,1} census empty (zero bucket exactly structural 12870),
    // but [-2,2] vectors exist (w=2: 32) -> tier 2, matching the measured
    // landscape (DP max = 12870 = M_struct).
    let sg = Subgroup::new(65537, 32).unwrap();
    let c = certify_q1(&sg, 16).unwrap();
    match c.verdict {
        Verdict::ZeroBucketStructural {
            ref census2_by_weight,
        } => {
            assert_eq!(census2_by_weight[2], 32);
            assert_eq!(c.zero_class, 12870);
        }
        ref v => panic!("expected ZeroBucketStructural, got {v:?}"),
    }
}

#[test]
fn certified_structural_at_large_prime() {
    // The [-2,2] norm law caps accident coefficients at (sum v_i^2)^{s/4}
    // <= (4*16)^8 = 2^48 for s = 32, so above ~2^48 the full census empties
    // and every bucket is provably structural. The p-independent engines make
    // this a fast check even at 50-bit primes.
    let p = 562949953421729u64; // prime, = 1 mod 32, > 2^49
    assert!(vanish::field::is_prime(p) && (p - 1) % 32 == 0);
    let sg = Subgroup::new(p, 32).unwrap();
    let c = certify_q1(&sg, 16).unwrap();
    assert_eq!(c.verdict, Verdict::AllBucketsStructural);
    assert_eq!(c.m_struct, 12870);

    // just above 2^32 the {-1,0,1} census is empty but [-2,2] vectors persist:
    // tier 2 (zero bucket exactly structural, other buckets unprotected).
    let sg = Subgroup::new(4294967681, 32).unwrap();
    let c = certify_q1(&sg, 16).unwrap();
    assert!(matches!(c.verdict, Verdict::ZeroBucketStructural { .. }));
}

#[test]
fn certify_error_paths() {
    let s6 = Subgroup::new(31, 6).unwrap();
    assert!(certify_q1(&s6, 3).is_err(), "non-two-smooth rejected");
    let sg = Subgroup::new(97, 32).unwrap();
    assert!(certify_q1(&sg, 0).is_err());
    assert!(certify_q1(&sg, 32).is_err());
}
