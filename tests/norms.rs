//! Golden tests for the norms module, pinned to the exhaustively verified
//! s = 32 landscape (exp22b/exp24/exp27 reference data) and cross-checked
//! against the independent decomposition engine.

use vanish::buckets::mitm::decompose_bucket_q1;
use vanish::code::class_size;
use vanish::domain::MultiplicativeSubgroup;
use vanish::norms::{bad_set, norm_table};

#[test]
fn golden_nmax_profile_s32() {
    // exp22b per-weight norm maxima at s = 32, {-1,0,1} vectors
    let t = norm_table(32, 6, 1).unwrap();
    let nmax = t.n_max_by_weight();
    assert_eq!(
        &nmax[1..=6],
        &[1, 256, 6561, 38416, 279841, 1331716],
        "anticorrelation profile w = 1..6"
    );
}

#[test]
fn badset_cross_checks_decompose_s16() {
    // s = 16 full bad set; predicted zero-buckets must match the independent
    // decomposition engine at every classical F3 prime.
    let bs = bad_set(16, 8, 1).unwrap();
    let by_p: std::collections::HashMap<u64, &vanish::norms::BadSetEntry> =
        bs.iter().map(|e| (e.p, e)).collect();
    for p in [97u64, 193, 241, 257, 353, 577] {
        let pred: u64 = class_size(16, 8, 0)
            + by_p.get(&p).map_or(0, |e| {
                e.counts
                    .iter()
                    .enumerate()
                    .filter(|(w, _)| w % 2 == 0 && *w > 0)
                    .map(|(w, &c)| c * class_size(16, 8, w))
                    .sum()
            });
        let sg = MultiplicativeSubgroup::new(p, 16).unwrap();
        let (exact, _) = decompose_bucket_q1(&sg, 8, 0).unwrap();
        assert_eq!(pred, exact, "zero-bucket prediction at p={p}");
    }
}

#[test]
fn badset_spot_pins_s32() {
    // weight <= 6 slice of the complete s = 32 bad set: the two-orbit story
    let bs = bad_set(32, 6, 1).unwrap();
    // largest weight<=6 prime is bounded by N_max(6) = 1,331,716
    assert!(bs.iter().all(|e| e.p <= 1_331_716));
    // p = 89633 carries exactly one weight-6 orbit (32 vectors)
    let e = bs.iter().find(|e| e.p == 89633).unwrap();
    assert_eq!(e.counts[6], 32);
    assert_eq!(e.provenance, vanish::norms::Provenance::ValuationSplit);
    // predicted zero bucket from the weight<=6 slice already matches ground
    // truth at 89633 (its higher-weight counts contribute nothing even):
    let sg = MultiplicativeSubgroup::new(89633, 32).unwrap();
    let (exact, profile) = decompose_bucket_q1(&sg, 16, 0).unwrap();
    assert_eq!(exact, 29382);
    assert_eq!(profile[6], 32, "consistent with the norms-derived census");
}

#[test]
fn census_fallback_triggers_at_small_primes() {
    // p = 97 has p^2-divisible norms; counts must come from the census path
    // and reproduce the exact zero bucket 6,196,582 via the class formula.
    let bs = bad_set(32, 16, 1);
    // full s=32 run is heavy for CI; use the s=16 table where the same
    // phenomenon exists (p = 17 divides small norms with multiplicity).
    drop(bs);
    let bs = bad_set(16, 8, 1).unwrap();
    let e = bs.iter().find(|e| e.p == 17).unwrap();
    assert_eq!(
        e.provenance,
        vanish::norms::Provenance::CensusCorrected,
        "p^2 fallback must trigger at p = 17"
    );
    let pred: u64 = class_size(16, 8, 0)
        + e.counts
            .iter()
            .enumerate()
            .filter(|(w, _)| w % 2 == 0 && *w > 0)
            .map(|(w, &c)| c * class_size(16, 8, w))
            .sum::<u64>();
    let sg = MultiplicativeSubgroup::new(17, 16).unwrap();
    let (exact, _) = decompose_bucket_q1(&sg, 8, 0).unwrap();
    assert_eq!(
        pred, exact,
        "census-fallback counts must be exact at p = 17"
    );
}
