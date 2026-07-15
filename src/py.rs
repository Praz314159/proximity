//! Python bindings (`pip install maturin && maturin develop --release
//! --features python`). Function names and signatures are kept stable across
//! internal refactors so downstream experiment scripts do not change.

// pyo3 macro expansion trips clippy::useless_conversion on every #[pyfunction];
// false positive at this expansion site.
#![allow(clippy::useless_conversion)]

use crate::buckets::{dp, mitm};
use crate::code;
use crate::domain::MultiplicativeSubgroup;
use crate::{census, field};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn sub(p: u64, s: usize) -> PyResult<MultiplicativeSubgroup> {
    MultiplicativeSubgroup::new(p, s).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn err<T>(r: crate::Result<T>) -> PyResult<T> {
    r.map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Full exact q=1 bucket distribution: `out[lam]` counts r-subsets S of mu_s with `e_1(S) = lam`. Cost/memory scale with p.
#[pyfunction]
fn bucket_dist_q1(py: Python<'_>, p: u64, s: usize, r: usize) -> PyResult<Py<PyArray1<u64>>> {
    let d = err(dp::distribution_q1(&sub(p, s)?, r))?;
    Ok(d.into_values().into_pyarray_bound(py).into())
}

/// Full exact q=2 joint distribution over `(e_1, e_2)`, shape (p, p). Intended for p <= ~700.
#[pyfunction]
fn bucket_dist_q2(py: Python<'_>, p: u64, s: usize, r: usize) -> PyResult<Py<PyArray2<u64>>> {
    let v = err(dp::distribution_q2(&sub(p, s)?, r))?;
    let pp = p as usize;
    let arr = numpy::ndarray::Array2::from_shape_vec((pp, pp), v)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py).into())
}

/// Kernel census, weight-capped: `out[w]` counts nonzero vectors with coefficients in `[-cmax, cmax]` and weight w <= wmax such that `sum v_i w^i = 0 (mod p)`.
#[pyfunction]
fn census_direct(p: u64, s: usize, cmax: u64, wmax: usize) -> PyResult<Vec<u64>> {
    let cmax = i64::try_from(cmax).map_err(|_| PyValueError::new_err("cmax too large"))?;
    err(census::direct(&sub(p, s)?, cmax, wmax))
}

/// Full kernel census by weight (meet-in-the-middle; s <= 32 at cmax = 2).
#[pyfunction]
fn census_mitm(p: u64, s: usize, cmax: i64) -> PyResult<Vec<u64>> {
    err(census::mitm(&sub(p, s)?, cmax))
}

/// Exact single bucket at e-values `lam` (any q = len(lam) <= 8, s <= 32). Cost is p-independent.
#[pyfunction]
fn bucket_e(p: u64, s: usize, r: usize, lam: Vec<u64>) -> PyResult<u64> {
    let sg = sub(p, s)?;
    let t = err(mitm::HalfTables::build(&sg, r, lam.len()))?;
    err(t.bucket(&lam))
}

/// Exact buckets for many lambdas sharing one table build (any q <= 8, s <= 32).
#[pyfunction]
fn buckets_e(p: u64, s: usize, r: usize, q: usize, lams: Vec<Vec<u64>>) -> PyResult<Vec<u64>> {
    let sg = sub(p, s)?;
    let t = err(mitm::HalfTables::build(&sg, r, q))?;
    lams.iter().map(|l| err(t.bucket(l))).collect()
}

/// The common `(e_1..e_q)` of the Theorem-A rung family (the optimal structural construction).
#[pyfunction]
fn rung_lambda_e(p: u64, s: usize, r: usize, q: usize) -> PyResult<Vec<u64>> {
    err(code::rung_lambda(&sub(p, s)?, r, q))
}

/// Anatomy of a q=1 bucket: returns `(total, per_weight_class_counts)`; total equals the DP bucket exactly.
#[pyfunction]
fn decompose_bucket_q1(p: u64, s: usize, r: usize, lam: u64) -> PyResult<(u64, Vec<u64>)> {
    err(mitm::decompose_bucket_q1(&sub(p, s)?, r, lam))
}

/// The quantized-ladder structural maximum `C(s/2^t - [r0!=0], floor(r/2^t))`, `t = ceil(log2(q+1))`.
#[pyfunction]
fn m_struct(s: usize, r: usize, q: usize) -> u64 {
    code::m_struct(s, r, q)
}

/// Elements of the order-s subgroup of F_p^* as consecutive powers `[w^0, ..., w^{s-1}]`.
#[pyfunction]
fn subgroup(p: u64, s: usize) -> PyResult<Vec<u64>> {
    Ok(sub(p, s)?.elements().to_vec())
}

/// Deterministic Miller-Rabin primality test for n < 2^64.
#[pyfunction]
fn is_prime(n: u64) -> bool {
    field::is_prime(n)
}

/// Full prime factorization (trial division + Pollard rho), sorted with multiplicity.
#[pyfunction]
fn factor(n: u64) -> Vec<u64> {
    field::factor(n)
}

/// One-call sweep statistics for q=1: returns (max, argmax, occupied, total,
/// second_moment) — the sweep-workload API (avoids marshaling full
/// distributions). second_moment is exact (u128 internally), returned as u128.
#[pyfunction]
fn dist_stats_q1(p: u64, s: usize, r: usize) -> PyResult<(u64, u64, u64, u64, u128)> {
    let d = err(dp::distribution_q1(&sub(p, s)?, r))?;
    let (mx, arg) = d.max();
    Ok((mx, arg, d.occupied(), d.total(), d.second_moment()))
}

/// A row of sweep statistics: (p, max, argmax, occupied, total, second_moment).
type SweepRow = (u64, u64, u64, u64, u64, u128);

/// Parallel (rayon) prime sweep of `dist_stats_q1` — the campaign driver.
#[pyfunction]
fn sweep_stats_q1(py: Python<'_>, s: usize, r: usize, primes: Vec<u64>) -> PyResult<Vec<SweepRow>> {
    use rayon::prelude::*;
    py.allow_threads(|| {
        primes
            .par_iter()
            .map(|&p| {
                let d = dp::distribution_q1(&MultiplicativeSubgroup::new(p, s)?, r)?;
                let (mx, arg) = d.max();
                Ok((p, mx, arg, d.occupied(), d.total(), d.second_moment()))
            })
            .collect::<crate::Result<Vec<_>>>()
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Tiered structural certificate for (p, s, r), q = 1 (p-independent).
/// Returns (tier, m_struct, zero_bucket): tier 1 = all buckets structural,
/// tier 2 = zero bucket structural, tier 3 = inflated (zero_bucket = exact
/// inflated value; tiers 1-2 return the structural zero-class size).
#[pyfunction]
fn certify_q1(p: u64, s: usize, r: usize) -> PyResult<(u8, u64, u64)> {
    use crate::certify::{certify_q1 as cert, Verdict};
    let c = err(cert(&sub(p, s)?, r))?;
    Ok(match c.verdict {
        Verdict::AllBucketsStructural => (1, c.m_struct, c.zero_class),
        Verdict::ZeroBucketStructural { .. } => (2, c.m_struct, c.zero_class),
        Verdict::Inflated { zero_bucket, .. } => (3, c.m_struct, zero_bucket),
    })
}

/// Primes p = 1 (mod s) in [lo, hi) — the sweep-population helper every
/// experiment script was rewriting locally.
#[pyfunction]
fn primes_1_mod(s: u64, lo: u64, hi: u64) -> Vec<u64> {
    field::primes_one_mod(s, lo)
        .take_while(|&p| p < hi)
        .collect()
}

/// Structural class size C(s/2 - w, (r - w)/2) (0 when parity/range infeasible).
#[pyfunction]
fn class_size(s: usize, r: usize, w: usize) -> u64 {
    code::class_size(s, r, w)
}

/// A decomposition row: (p, total, per-weight class counts).
type DecompRow = (u64, u64, Vec<u64>);

/// Parallel zero-bucket decompositions across primes (the audit workload:
/// 23k primes in seconds instead of minutes).
#[pyfunction]
fn decompose_many(
    py: Python<'_>,
    s: usize,
    r: usize,
    primes: Vec<u64>,
) -> PyResult<Vec<DecompRow>> {
    use rayon::prelude::*;
    py.allow_threads(|| {
        primes
            .par_iter()
            .map(|&p| {
                let sg = MultiplicativeSubgroup::new(p, s)?;
                let (t, pw) = mitm::decompose_bucket_q1(&sg, r, 0)?;
                Ok((p, t, pw))
            })
            .collect::<crate::Result<Vec<_>>>()
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// An attack row: (delta_star, deficit, t, s_g, r, log2_list).
type AttackRow = (f64, f64, u32, u64, u64, f64);

/// Best ladder attack: (delta_star, deficit, t, s_g, r, log2_list), or None.
#[pyfunction]
fn attack_best(n: u64, k: u64, list_bits: f64) -> PyResult<Option<AttackRow>> {
    let p = crate::attack::AttackParams { n, k, list_bits };
    let a = err(crate::attack::best_attack(&p))?;
    Ok(a.map(|a| (a.delta_star, a.deficit, a.t, a.s_g, a.r, a.log2_list)))
}

/// Antipodal (survey Table-5) baseline attack, same tuple shape as attack_best.
#[pyfunction]
fn attack_antipodal(n: u64, k: u64, list_bits: f64) -> PyResult<Option<AttackRow>> {
    let p = crate::attack::AttackParams { n, k, list_bits };
    let a = err(crate::attack::antipodal_attack(&p))?;
    Ok(a.map(|a| (a.delta_star, a.deficit, a.t, a.s_g, a.r, a.log2_list)))
}

/// Structural-framework ceiling delta_min - H2(rate)/list_bits.
#[pyfunction]
fn attack_ceiling(n: u64, k: u64, list_bits: f64) -> PyResult<f64> {
    err(crate::attack::hyperbola_ceiling(
        &crate::attack::AttackParams { n, k, list_bits },
    ))
}

/// Exact toy-protocol soundness: (winning, soundness, classes).
#[pyfunction]
fn toy_soundness(p: u64, s: usize, r: usize) -> PyResult<(u64, f64, u64)> {
    let t = err(crate::toy::exact_soundness(&sub(p, s)?, r))?;
    Ok((t.winning, t.soundness, t.classes))
}

/// A rung-sweep row: (p, one exact rung bucket per requested q).
type RungRow = (u64, Vec<u64>);

/// Rayon-parallel rung-bucket sweep: for each prime, build the MitM tables
/// once per q and evaluate the rung lambda's exact bucket. The q-axis
/// campaign driver (23k primes x several q in minutes, not hours).
#[pyfunction]
fn rung_buckets_many(
    py: Python<'_>,
    s: usize,
    r: usize,
    qs: Vec<usize>,
    primes: Vec<u64>,
) -> PyResult<Vec<RungRow>> {
    use rayon::prelude::*;
    py.allow_threads(|| {
        primes
            .par_iter()
            .map(|&p| {
                let sg = MultiplicativeSubgroup::new(p, s)?;
                let mut row = Vec::with_capacity(qs.len());
                for &q in &qs {
                    let lam = code::rung_lambda(&sg, r, q)?;
                    let t = mitm::HalfTables::build(&sg, r, q)?;
                    row.push(t.bucket(&lam)?);
                }
                Ok((p, row))
            })
            .collect::<crate::Result<Vec<_>>>()
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Rayon-parallel certificates: rows of (p, tier, m_struct, zero_bucket).
#[pyfunction]
fn certify_many(
    py: Python<'_>,
    s: usize,
    r: usize,
    primes: Vec<u64>,
) -> PyResult<Vec<(u64, u8, u64, u64)>> {
    use crate::certify::{certify_q1 as cert, Verdict};
    use rayon::prelude::*;
    py.allow_threads(|| {
        primes
            .par_iter()
            .map(|&p| {
                let c = cert(&MultiplicativeSubgroup::new(p, s)?, r)?;
                Ok(match c.verdict {
                    Verdict::AllBucketsStructural => (p, 1u8, c.m_struct, c.zero_class),
                    Verdict::ZeroBucketStructural { .. } => (p, 2, c.m_struct, c.zero_class),
                    Verdict::Inflated { zero_bucket, .. } => (p, 3, c.m_struct, zero_bucket),
                })
            })
            .collect::<crate::Result<Vec<_>>>()
    })
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// A bad-set row: (p, per-weight Galois-normalized counts, census_fallback);
/// the bool is reconstructed from the row's provenance at this boundary so
/// the Python tuple shape stays frozen.
type BadRow = (u64, Vec<u64>, bool);

/// Complete bad set for weights <= wmax, coefficients in [-cmax, cmax]:
/// every prime p = 1 mod s that any such kernel vector can visit, with
/// exact per-weight counts (p^2-divisibility handled by census fallback).
#[pyfunction]
fn norms_bad_set(py: Python<'_>, s: usize, wmax: usize, cmax: i64) -> PyResult<Vec<BadRow>> {
    py.allow_threads(|| crate::norms::bad_set(s, wmax, cmax))
        .map(|v| {
            v.into_iter()
                .map(|e| {
                    let cf = e.provenance == crate::norms::Provenance::CensusCorrected;
                    (e.p, e.counts, cf)
                })
                .collect()
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Per-weight maximum cyclotomic norms (the anticorrelation profile),
/// as decimal strings (values can exceed u64 at large s).
#[pyfunction]
fn norms_n_max(py: Python<'_>, s: usize, wmax: usize, cmax: i64) -> PyResult<Vec<String>> {
    py.allow_threads(|| crate::norms::norm_table(s, wmax, cmax))
        .map(|t| t.n_max_by_weight().iter().map(|n| n.to_string()).collect())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Ingest GPU-campaign norm-table JSON shards into the s-64-scale bad set.
/// Returns (rows, mass_by_weight, n_max_by_weight, entries_parsed) where
/// rows = list of (p, counts_by_weight, unsafe_split).
/// Writes <out_prefix>.primes.bin (u64 le), .counts.bin (u64 le, row-major
/// n x (wmax+1)), .flags.bin (u8) and returns
/// (n_rows, mass_by_weight, n_max_by_weight, entries_parsed).
/// Counts are u64: at w = 12 the smallest primes carry per-weight counts
/// beyond u32 (a u32 format cost a full ingest run to discover).
#[pyfunction]
fn badset_from_gpu_json(
    py: Python<'_>,
    paths: Vec<String>,
    s: usize,
    wmax: usize,
    out_prefix: String,
) -> PyResult<(u64, Vec<u64>, Vec<u64>, u64)> {
    let (rows, stats) = py
        .allow_threads(|| {
            crate::norms::ingest::badset_from_gpu_json(&paths, s, wmax, Some(&out_prefix))
        })
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let n = rows.len() as u64;
    let werr = |e: std::io::Error| PyValueError::new_err(e.to_string());
    let mut pb = Vec::with_capacity(rows.len() * 8);
    let mut cb = Vec::with_capacity(rows.len() * (wmax + 1) * 8);
    let mut fb = Vec::with_capacity(rows.len());
    for e in &rows {
        pb.extend_from_slice(&e.p.to_le_bytes());
        for &c in &e.counts {
            cb.extend_from_slice(&c.to_le_bytes());
        }
        fb.push(u8::from(!e.provenance.is_exact()));
    }
    std::fs::write(format!("{out_prefix}.primes.bin"), pb).map_err(werr)?;
    std::fs::write(format!("{out_prefix}.counts.bin"), cb).map_err(werr)?;
    std::fs::write(format!("{out_prefix}.flags.bin"), fb).map_err(werr)?;
    // outputs are durable: the crash-recovery checkpoint has served its purpose
    crate::norms::ingest::clear_checkpoint(&out_prefix);
    Ok((
        n,
        stats.mass_by_weight,
        stats.n_max_by_weight,
        stats.entries_parsed,
    ))
}

/// Exact list decode of a generic RS code `RS[F_p, domain, k]`: every codeword
/// (as its evaluation vector) agreeing with `word` on at least `t` of the
/// `n = len(domain)` coordinates. `domain` is any list of distinct field
/// elements (e.g. `subgroup(p, s)`, or an arbitrary set). Requires `t >= k`.
#[pyfunction]
fn list_decode(
    p: u64,
    domain: Vec<u64>,
    k: usize,
    word: Vec<u64>,
    t: usize,
) -> PyResult<Vec<Vec<u64>>> {
    let rs = err(code::ReedSolomon::on_domain(p, domain, k))?;
    let oracle = crate::decode::DecodeOracle::new(&rs);
    err(oracle.list(&word, crate::decode::Radius::agreement(t)))
}

/// One code-first optimization run on `RS[F_p, domain, k]`: build a random
/// pencil seed and anneal it to maximize list size. Returns
/// `(center, members, size_trajectory)` — the raw cluster (member codewords as
/// evaluation vectors) plus the per-move list-size trajectory. Loop over `seed`
/// in Python to collect a discovery dataset. `domain` is any distinct-point set.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn anneal_pencil(
    p: u64,
    domain: Vec<u64>,
    k: usize,
    t: usize,
    petals: usize,
    steps: usize,
    seed: u64,
) -> PyResult<(Vec<u64>, Vec<Vec<u64>>, Vec<usize>)> {
    let rs = err(code::ReedSolomon::on_domain(p, domain, k))?;
    let rad = crate::decode::Radius::agreement(t);
    let seedw = err(crate::cluster::random_pencil_seed(&rs, petals, seed))?;
    let (cl, tr) = err(crate::cluster::anneal(&rs, &seedw, rad, steps, 2.0, 0.92, seed))?;
    Ok((cl.center().to_vec(), cl.members().to_vec(), tr.sizes.clone()))
}

#[pymodule]
fn vanish(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(list_decode, m)?)?;
    m.add_function(wrap_pyfunction!(anneal_pencil, m)?)?;
    m.add_function(wrap_pyfunction!(bucket_dist_q1, m)?)?;
    m.add_function(wrap_pyfunction!(bucket_dist_q2, m)?)?;
    m.add_function(wrap_pyfunction!(census_direct, m)?)?;
    m.add_function(wrap_pyfunction!(census_mitm, m)?)?;
    m.add_function(wrap_pyfunction!(bucket_e, m)?)?;
    m.add_function(wrap_pyfunction!(buckets_e, m)?)?;
    m.add_function(wrap_pyfunction!(rung_lambda_e, m)?)?;
    m.add_function(wrap_pyfunction!(decompose_bucket_q1, m)?)?;
    m.add_function(wrap_pyfunction!(m_struct, m)?)?;
    m.add_function(wrap_pyfunction!(subgroup, m)?)?;
    m.add_function(wrap_pyfunction!(is_prime, m)?)?;
    m.add_function(wrap_pyfunction!(factor, m)?)?;
    m.add_function(wrap_pyfunction!(dist_stats_q1, m)?)?;
    m.add_function(wrap_pyfunction!(sweep_stats_q1, m)?)?;
    m.add_function(wrap_pyfunction!(certify_q1, m)?)?;
    m.add_function(wrap_pyfunction!(primes_1_mod, m)?)?;
    m.add_function(wrap_pyfunction!(class_size, m)?)?;
    m.add_function(wrap_pyfunction!(decompose_many, m)?)?;
    m.add_function(wrap_pyfunction!(attack_best, m)?)?;
    m.add_function(wrap_pyfunction!(attack_antipodal, m)?)?;
    m.add_function(wrap_pyfunction!(attack_ceiling, m)?)?;
    m.add_function(wrap_pyfunction!(toy_soundness, m)?)?;
    m.add_function(wrap_pyfunction!(rung_buckets_many, m)?)?;
    m.add_function(wrap_pyfunction!(certify_many, m)?)?;
    m.add_function(wrap_pyfunction!(norms_bad_set, m)?)?;
    m.add_function(wrap_pyfunction!(badset_from_gpu_json, m)?)?;
    m.add_function(wrap_pyfunction!(norms_n_max, m)?)?;
    Ok(())
}
