use crate::{census, dp, field, mitm};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;

#[pyfunction]
fn bucket_dist_q1(py: Python<'_>, p: u64, s: usize, r: usize) -> Py<PyArray1<u64>> {
    dp::bucket_dist_q1(p, s, r).into_pyarray_bound(py).into()
}

#[pyfunction]
fn bucket_dist_q2(py: Python<'_>, p: u64, s: usize, r: usize) -> Py<PyArray2<u64>> {
    let v = dp::bucket_dist_q2(p, s, r);
    let pp = p as usize;
    let arr = numpy::ndarray::Array2::from_shape_vec((pp, pp), v).unwrap();
    arr.into_pyarray_bound(py).into()
}

#[pyfunction]
#[pyo3(signature = (p, s, cmax, wmax))]
fn census_direct(p: u64, s: usize, cmax: u64, wmax: usize) -> Vec<u64> {
    census::census_direct(p, s, cmax, wmax)
}

#[pyfunction]
fn census_mitm(p: u64, s: usize, cmax: i64) -> Vec<u64> {
    census::census_mitm(p, s, cmax)
}

#[pyfunction]
fn bucket_e(p: u64, s: usize, r: usize, lam: Vec<u64>) -> u64 {
    let q = lam.len();
    mitm::HalfTables::build(p, s, r, q).bucket_e(&lam)
}

/// Build tables once, query many lambdas: returns Vec of bucket sizes.
#[pyfunction]
fn buckets_e(p: u64, s: usize, r: usize, q: usize, lams: Vec<Vec<u64>>) -> Vec<u64> {
    let t = mitm::HalfTables::build(p, s, r, q);
    lams.iter().map(|l| t.bucket_e(l)).collect()
}

#[pyfunction]
fn rung_lambda_e(p: u64, s: usize, r: usize, q: usize) -> Vec<u64> {
    mitm::rung_lambda_e(p, s, r, q)
}

#[pyfunction]
fn decompose_bucket_q1(p: u64, s: usize, r: usize, lam: u64) -> (u64, Vec<u64>) {
    mitm::decompose_bucket_q1(p, s, r, lam)
}

#[pyfunction]
fn subgroup(p: u64, s: usize) -> Vec<u64> {
    field::subgroup(p, s)
}

#[pyfunction]
fn is_prime(n: u64) -> bool {
    field::is_prime(n)
}

#[pymodule]
fn bucketlab(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(bucket_dist_q1, m)?)?;
    m.add_function(wrap_pyfunction!(bucket_dist_q2, m)?)?;
    m.add_function(wrap_pyfunction!(census_direct, m)?)?;
    m.add_function(wrap_pyfunction!(census_mitm, m)?)?;
    m.add_function(wrap_pyfunction!(bucket_e, m)?)?;
    m.add_function(wrap_pyfunction!(buckets_e, m)?)?;
    m.add_function(wrap_pyfunction!(rung_lambda_e, m)?)?;
    m.add_function(wrap_pyfunction!(decompose_bucket_q1, m)?)?;
    m.add_function(wrap_pyfunction!(subgroup, m)?)?;
    m.add_function(wrap_pyfunction!(is_prime, m)?)?;
    Ok(())
}
