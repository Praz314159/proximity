//! Golden tests: the validation contract. Every value here is pinned to an
//! exhaustively-verified independent computation (a Python/numpy reference
//! implementation run over full subset enumerations / DP distributions).
//! Optimization passes and refactors must keep this suite green; new kernels
//! must add their own pins. See CONTRIBUTING.md.

use vanish::domain::MultiplicativeSubgroup;
use vanish::field::{binom, factor, is_prime};
use vanish::smooth::buckets::{dp, mitm};
use vanish::smooth::census;
use vanish::smooth::rung::{class_size, m_struct, rung_lambda};

fn sg(p: u64, s: usize) -> MultiplicativeSubgroup {
    MultiplicativeSubgroup::new(p, s).unwrap()
}

fn mass_check(dist: &vanish::smooth::buckets::BucketDistribution, s: u64, r: u64) {
    assert_eq!(dist.total(), binom(s, r), "total mass must be C(s, r)");
}

#[test]
fn golden_q1_s16() {
    for (p, want) in [
        (17u64, 758u64),
        (193, 102),
        (241, 70),
        (257, 70),
        (577, 102),
    ] {
        let d = dp::distribution_q1(&sg(p, 16), 8).unwrap();
        mass_check(&d, 16, 8);
        assert_eq!(d.max().0, want, "maxN at p={p}");
    }
}

#[test]
fn golden_q1_s32() {
    for (p, want) in [
        (97u64, 6196723u64),
        (3457, 220134),
        (47041, 33862),
        (65537, 12870),
        (77569, 30598),
        (89633, 29382),
        (180001, 16006),
    ] {
        let d = dp::distribution_q1(&sg(p, 32), 16).unwrap();
        mass_check(&d, 32, 16);
        assert_eq!(d.max().0, want, "maxN at p={p}");
    }
}

#[test]
fn golden_q2_s32_p97() {
    let d = dp::distribution_q2(&sg(97, 32), 16).unwrap();
    assert_eq!(d.iter().sum::<u64>(), binom(32, 16));
    assert_eq!(*d.iter().max().unwrap(), 65089);
    assert_eq!(*d.iter().min().unwrap(), 63339);
}

#[test]
fn golden_census() {
    let c = census::mitm(&sg(89633, 32), 2).unwrap();
    assert_eq!((c[6], c[7], c[8]), (480, 1728, 9760));
    let c = census::mitm(&sg(65537, 32), 2).unwrap();
    assert_eq!((c[2], c[3], c[4]), (32, 32, 448));
    let cd = census::direct(&sg(89633, 32), 2, 7).unwrap();
    assert_eq!(
        (cd[6], cd[7]),
        (480, 1728),
        "direct engine must agree with MitM"
    );
}

#[test]
fn golden_mitm_rung_buckets() {
    for (p, q, want) in [
        (3457u64, 2usize, 422u64),
        (97, 4, 38),
        (47041, 2, 70),
        (89633, 3, 70),
        (97, 8, 2),
        (3457, 8, 2),
    ] {
        let s = sg(p, 32);
        let lam = rung_lambda(&s, 16, q).unwrap();
        let t = mitm::HalfTables::build(&s, 16, q).unwrap();
        assert_eq!(t.bucket(&lam).unwrap(), want, "rung bucket at p={p}, q={q}");
    }
}

#[test]
fn golden_ladder_values() {
    // exhaustively verified structural maxima (exp16): refined ladder formula
    for (s, r, q, want) in [
        (16usize, 8usize, 1usize, 70u64),
        (16, 8, 2, 6),
        (16, 8, 4, 2),
        (16, 5, 1, 21),
        (32, 16, 1, 12870),
        (32, 16, 3, 70),
        (32, 6, 1, 560),
    ] {
        assert_eq!(m_struct(s, r, q), want, "M_struct({s},{r},{q})");
    }
    assert_eq!(class_size(32, 16, 0), 12870);
    assert_eq!(class_size(32, 16, 6), 252);
    assert_eq!(class_size(32, 16, 7), 0, "odd weight infeasible at even r");
}

#[test]
fn property_dp_vs_mitm_q1() {
    for p in [97u64, 3457, 89633] {
        let s = sg(p, 32);
        let d = dp::distribution_q1(&s, 16).unwrap();
        let (mx, arg) = d.max();
        let t = mitm::HalfTables::build(&s, 16, 1).unwrap();
        assert_eq!(t.bucket(&[arg]).unwrap(), mx, "DP/MitM disagree at p={p}");
    }
}

#[test]
fn property_decomposition_law() {
    let s = sg(77569, 32);
    let d = dp::distribution_q1(&s, 16).unwrap();
    let (mx, arg) = d.max();
    let (total, per_w) = mitm::decompose_bucket_q1(&s, 16, arg).unwrap();
    assert_eq!(total, mx);
    assert_eq!(mx, 30598);
    assert_eq!(
        (per_w[0], per_w[6], per_w[8], per_w[10], per_w[12]),
        (1, 32, 96, 128, 64)
    );
}

#[test]
fn property_dilation_invariance() {
    let p = 3457u64;
    let s = sg(p, 32);
    let d = dp::distribution_q1(&s, 16).unwrap();
    let v = d.values();
    for lam in [1u64, 5, 100, 2000] {
        let base = v[lam as usize];
        for &g in &s.elements()[1..4] {
            let gl = (g as u128 * lam as u128 % p as u128) as usize;
            assert_eq!(v[gl], base, "dilation symmetry broken at lam={lam}, g={g}");
        }
    }
}

#[test]
fn property_domain_and_field() {
    let s = sg(3457, 32);
    assert_eq!(s.elements().len(), 32);
    assert_eq!(s.elements()[16], 3457 - 1, "w^(s/2) must be -1 for even s");
    let cosets = s.cosets(2).unwrap();
    assert_eq!(cosets.len(), 8);
    let mut all: Vec<u64> = cosets.into_iter().flatten().collect();
    all.sort_unstable();
    let mut els = s.elements().to_vec();
    els.sort_unstable();
    assert_eq!(all, els, "cosets must partition the subgroup");
    // construction errors
    assert!(
        MultiplicativeSubgroup::new(3458, 32).is_err(),
        "composite p rejected"
    );
    assert!(
        MultiplicativeSubgroup::new(97, 31).is_err(),
        "s must divide p - 1"
    );
    // factorization: known case + product reconstruction on a norm-scale value
    assert_eq!(factor(6 * 35), vec![2, 3, 5, 7]);
    let n = 1_331_716u64; // max weight-6 cyclotomic norm at s = 32
    let fs = factor(n);
    assert_eq!(fs.iter().product::<u64>(), n);
    assert!(fs.iter().all(|&f| is_prime(f)));
    assert!(is_prime(2130706433));
}

#[test]
fn golden_s64_mass() {
    let d = dp::distribution_q1(&sg(193, 64), 32).unwrap();
    assert_eq!(d.total(), binom(64, 32));
}

/// Theorem B_mult pins (2026-07-21): the top word's exact list is the maximal
/// Graham-Sloane class — field-independently, including at the worst s=16
/// accident prime. And the class-count DP reproduces the headline integers.
#[test]
fn golden_top_word_and_gs_classes() {
    use vanish::rs::decode::{DecodeOracle, ListOracle, Radius};
    let n16 = vanish::smooth::rung::gs_class_counts(16, 8).unwrap();
    assert_eq!(n16.iter().sum::<u64>(), 12870);
    assert_eq!(*n16.iter().max().unwrap(), 810);
    assert_eq!((n16[4], n16[12]), (810, 810));
    assert_eq!(
        vanish::smooth::rung::gs_class_counts(16, 9).unwrap()[3],
        715
    );
    assert_eq!(
        *vanish::smooth::rung::gs_class_counts(32, 17)
            .unwrap()
            .iter()
            .max()
            .unwrap(),
        17_678_835
    );
    for p in [65537u64, 97] {
        let sg = vanish::domain::MultiplicativeSubgroup::new(p, 16).unwrap();
        let w = vanish::smooth::rung::top_word(&sg, 8, 4).unwrap();
        let rs = vanish::rs::code::ReedSolomon::on_subgroup(&sg, 7).unwrap();
        let l = DecodeOracle::new(&rs)
            .list_size(&w, Radius::agreement(8))
            .unwrap();
        assert_eq!(l, 810, "top word exact list at p = {p}");
    }
    // syndrome-word convention: b supported on e_1 gives the additive cut (70)
    let sg = vanish::domain::MultiplicativeSubgroup::new(65537, 16).unwrap();
    let w = vanish::smooth::rung::word_from_syndrome(65537, sg.elements(), 8, &[0, 1]);
    let rs = vanish::rs::code::ReedSolomon::on_subgroup(&sg, 7).unwrap();
    let l = DecodeOracle::new(&rs)
        .list_size(&w, Radius::agreement(8))
        .unwrap();
    assert_eq!(l, 70, "e_1 coordinate cut = the additive bucket");
}
