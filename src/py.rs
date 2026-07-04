//! Python bindings (`pip install maturin && maturin develop --release
//! --features python`). Function names and signatures are kept stable across
//! internal refactors so downstream experiment scripts do not change.

// pyo3 macro expansion trips clippy::useless_conversion on every #[pyfunction];
// false positive at this expansion site.
#![allow(clippy::useless_conversion)]

use crate::buckets::{dp, mitm};
use crate::code;
use crate::domain::Subgroup;
use crate::{census, field};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn sub(p: u64, s: usize) -> PyResult<Subgroup> {
    Subgroup::new(p, s).map_err(|e| PyValueError::new_err(e.to_string()))
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
                let d = dp::distribution_q1(&Subgroup::new(p, s)?, r)?;
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

#[pymodule]
fn vanish(m: &Bound<'_, PyModule>) -> PyResult<()> {
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
    Ok(())
}
