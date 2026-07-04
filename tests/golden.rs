//! Golden tests pinned to exhaustively-verified values from the Python
//! experiment suite (proximity_explorations/useful_families/exp17..exp20).
//! These are the regression contract for every optimization pass.

use bucketlab::census::{census_direct, census_mitm};
use bucketlab::dp::{bucket_dist_q1, bucket_dist_q2, max_and_argmax};
use bucketlab::field::{binom, subgroup};
use bucketlab::mitm::{decompose_bucket_q1, rung_lambda_e, HalfTables};

fn mass_check(dist: &[u64], s: u64, r: u64) {
    let sum: u64 = dist.iter().sum();
    assert_eq!(sum, binom(s, r), "total mass must be C(s, r)");
}

#[test]
fn golden_q1_s16() {
    // exp17 s=16, r=8: (p, maxN)
    for (p, want) in [(17u64, 758u64), (193, 102), (241, 70), (257, 70), (577, 102)] {
        let d = bucket_dist_q1(p, 16, 8);
        mass_check(&d, 16, 8);
        assert_eq!(max_and_argmax(&d).0, want, "maxN at p={p}");
    }
}

#[test]
fn golden_q1_s32() {
    // exp17/exp19/exp20a s=32, r=16
    for (p, want) in [
        (97u64, 6196723u64),
        (3457, 220134),
        (47041, 33862),
        (65537, 12870),
        (77569, 30598),
        (89633, 29382),
        (180001, 16006),
    ] {
        let d = bucket_dist_q1(p, 32, 16);
        mass_check(&d, 32, 16);
        assert_eq!(max_and_argmax(&d).0, want, "maxN at p={p}");
    }
}

#[test]
fn golden_q2_s32_p97() {
    // exp18: joint (e1, e2) distribution at p=97: max 65089, min 63339
    let d = bucket_dist_q2(97, 32, 16);
    mass_check(&d, 32, 16);
    let mx = *d.iter().max().unwrap();
    let mn = *d.iter().min().unwrap();
    assert_eq!(mx, 65089);
    assert_eq!(mn, 63339);
}

#[test]
fn golden_census() {
    // exp20a full [-2,2] census by weight
    let c = census_mitm(89633, 32, 2);
    assert_eq!(c[6], 480, "p=89633 weight-6 census");
    assert_eq!(c[7], 1728, "p=89633 weight-7 census");
    assert_eq!(c[8], 9760, "p=89633 weight-8 census");
    let c = census_mitm(65537, 32, 2);
    assert_eq!((c[2], c[3], c[4]), (32, 32, 448), "p=65537 low-weight census");
    // direct engine must agree with MitM on the capped range
    let cd = census_direct(89633, 32, 2, 7);
    assert_eq!(cd[6], 480);
    assert_eq!(cd[7], 1728);
}

#[test]
fn golden_mitm_rung_buckets() {
    // exp20c (corrected): exact rung-lambda buckets
    for (p, q, want) in [
        (3457u64, 2usize, 422u64),
        (97, 4, 38),
        (47041, 2, 70),
        (89633, 3, 70),
        (97, 8, 2),
        (3457, 8, 2),
    ] {
        let lam = rung_lambda_e(p, 32, 16, q);
        let t = HalfTables::build(p, 32, 16, q);
        assert_eq!(t.bucket_e(&lam), want, "rung bucket at p={p}, q={q}");
    }
}

#[test]
fn property_dp_vs_mitm_q1() {
    // MitM bucket at the DP argmax must equal the DP max (any prime).
    for p in [97u64, 3457, 89633] {
        let d = bucket_dist_q1(p, 32, 16);
        let (mx, arg) = max_and_argmax(&d);
        let t = HalfTables::build(p, 32, 16, 1);
        assert_eq!(t.bucket_e(&[arg as u64]), mx, "DP/MitM disagree at p={p}");
    }
}

#[test]
fn property_decomposition_law() {
    // exp20a anatomy: decomposition must reproduce the DP bucket to the unit,
    // with the pinned weight profile at p=77569.
    let d = bucket_dist_q1(77569, 32, 16);
    let (mx, arg) = max_and_argmax(&d);
    let (total, per_w) = decompose_bucket_q1(77569, 32, 16, arg as u64);
    assert_eq!(total, mx);
    assert_eq!(mx, 30598);
    assert_eq!(per_w[0], 1);
    assert_eq!(per_w[6], 32);
    assert_eq!(per_w[8], 96);
    assert_eq!(per_w[10], 128);
    assert_eq!(per_w[12], 64);
}

#[test]
fn property_dilation_invariance() {
    // N(g * lambda) = N(lambda) for g in the subgroup.
    let p = 3457u64;
    let d = bucket_dist_q1(p, 32, 16);
    let els = subgroup(p, 32);
    for lam in [1u64, 5, 100, 2000] {
        let base = d[lam as usize];
        for &g in &els[1..4] {
            let gl = (g as u128 * lam as u128 % p as u128) as usize;
            assert_eq!(d[gl], base, "dilation symmetry broken at lam={lam}, g={g}");
        }
    }
}

#[test]
fn golden_s64_mass() {
    // s=64 row from exp20b range: exact mass check (maxN values were reported
    // in float; the mass identity is the exact anchor).
    let d = bucket_dist_q1(193, 64, 32);
    mass_check(&d, 64, 32);
}
