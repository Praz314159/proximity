//! Hot-path baselines — the no-regression gate for ring/kernel refactors.
//!
//! Run `cargo bench` before touching any hot loop (e.g. the planned
//! `Cyclo`/`fold` migration of `design/negacyclic_ring.md`) and compare
//! after; criterion stores the previous run and reports the delta.
//!
//! What is measured and why:
//!   - `exact_census_16_8`: the negacyclic DFS (rotation-only ring
//!     arithmetic) end to end — the loop the `fold` primitive must not
//!     slow down. (16, 8) is the largest cell that runs in milliseconds.
//!   - `syndrome_s32` / `moment_row_s32`: the per-word and per-subset
//!     convention kernels every campaign calls millions of times.
//!   - `cut_counts_s16`: the streamed CPU counter (rayon) at census
//!     scale-model size.

use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;
use vanish::census::exact_value_census;
use vanish::domain::MultiplicativeSubgroup;
use vanish::vs::VsSpace;

fn space(p: u64, s: usize, k: usize) -> VsSpace {
    let sg = MultiplicativeSubgroup::new(p, s).unwrap();
    VsSpace::new(&sg, k).unwrap()
}

fn benches(c: &mut Criterion) {
    c.bench_function("exact_census_16_8_coord4", |b| {
        b.iter(|| exact_value_census(black_box(16), 8, 4).unwrap())
    });

    let vs = space(2130706433, 32, 15);
    let word: Vec<u64> = (0..32).map(|i| (i * i * 977 + 5) % 2130706433).collect();
    c.bench_function("syndrome_s32", |b| {
        b.iter(|| vs.syndrome(black_box(&word)).unwrap())
    });

    let subset: Vec<usize> = (0..16).map(|i| 2 * i).collect();
    c.bench_function("moment_row_s32", |b| {
        b.iter(|| vs.moment_row(black_box(&subset)).unwrap())
    });

    let big_a = vanish::ring::Cyclo::from_coeffs(
        (0..1024)
            .map(|i| ((i * 37 + 11) % 20011) as i64 - 10000)
            .collect(),
    )
    .unwrap();
    let big_b = vanish::ring::Cyclo::from_coeffs(
        (0..1024)
            .map(|i| ((i * 61 + 7) % 20011) as i64 - 10000)
            .collect(),
    )
    .unwrap();
    c.bench_function("cyclo_mul_schoolbook_1024", |b| {
        b.iter(|| black_box(&big_a).mul(black_box(&big_b)).unwrap())
    });
    c.bench_function("cyclo_mul_ntt_1024", |b| {
        b.iter(|| black_box(&big_a).mul_ntt(black_box(&big_b)).unwrap())
    });

    let small_a =
        vanish::ring::Cyclo::from_coeffs((0..16).map(|i| (i * 37 + 11) % 2003 - 1000).collect())
            .unwrap();
    let small_b =
        vanish::ring::Cyclo::from_coeffs((0..16).map(|i| (i * 61 + 7) % 2003 - 1000).collect())
            .unwrap();
    c.bench_function("cyclo_mul_ntt_16", |b| {
        b.iter(|| black_box(&small_a).mul_ntt(black_box(&small_b)).unwrap())
    });

    let vs16 = space(65537, 16, 7);
    let b0 = vs16
        .syndrome(&(0..16).map(|i| (7 * i + 3) as u64).collect::<Vec<_>>())
        .unwrap();
    c.bench_function("cut_counts_s16", |b| {
        b.iter(|| {
            vs16.cut_counts(black_box(std::slice::from_ref(&b0)))
                .unwrap()
        })
    });
}

criterion_group! {
    name = hotpaths;
    config = Criterion::default().sample_size(20);
    targets = benches
}
criterion_main!(hotpaths);
