//! Python bindings (`pip install maturin && maturin develop --release
//! --features python`). Function names and signatures are kept stable across
//! internal refactors so downstream experiment scripts do not change.

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

#[pyfunction]
fn bucket_dist_q1(py: Python<'_>, p: u64, s: usize, r: usize) -> PyResult<Py<PyArray1<u64>>> {
    let d = err(dp::distribution_q1(&sub(p, s)?, r))?;
    Ok(d.into_values().into_pyarray_bound(py).into())
}

#[pyfunction]
fn bucket_dist_q2(py: Python<'_>, p: u64, s: usize, r: usize) -> PyResult<Py<PyArray2<u64>>> {
    let v = err(dp::distribution_q2(&sub(p, s)?, r))?;
    let pp = p as usize;
    let arr = numpy::ndarray::Array2::from_shape_vec((pp, pp), v)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py).into())
}

#[pyfunction]
fn census_direct(p: u64, s: usize, cmax: u64, wmax: usize) -> PyResult<Vec<u64>> {
    err(census::direct(&sub(p, s)?, cmax, wmax))
}

#[pyfunction]
fn census_mitm(p: u64, s: usize, cmax: i64) -> PyResult<Vec<u64>> {
    err(census::mitm(&sub(p, s)?, cmax))
}

#[pyfunction]
fn bucket_e(p: u64, s: usize, r: usize, lam: Vec<u64>) -> PyResult<u64> {
    let sg = sub(p, s)?;
    let t = err(mitm::HalfTables::build(&sg, r, lam.len()))?;
    err(t.bucket(&lam))
}

#[pyfunction]
fn buckets_e(p: u64, s: usize, r: usize, q: usize, lams: Vec<Vec<u64>>) -> PyResult<Vec<u64>> {
    let sg = sub(p, s)?;
    let t = err(mitm::HalfTables::build(&sg, r, q))?;
    lams.iter().map(|l| err(t.bucket(l))).collect()
}

#[pyfunction]
fn rung_lambda_e(p: u64, s: usize, r: usize, q: usize) -> PyResult<Vec<u64>> {
    err(code::rung_lambda(&sub(p, s)?, r, q))
}

#[pyfunction]
fn decompose_bucket_q1(p: u64, s: usize, r: usize, lam: u64) -> PyResult<(u64, Vec<u64>)> {
    err(mitm::decompose_bucket_q1(&sub(p, s)?, r, lam))
}

#[pyfunction]
fn m_struct(s: usize, r: usize, q: usize) -> u64 {
    code::m_struct(s, r, q)
}

#[pyfunction]
fn subgroup(p: u64, s: usize) -> PyResult<Vec<u64>> {
    Ok(sub(p, s)?.elements().to_vec())
}

#[pyfunction]
fn is_prime(n: u64) -> bool {
    field::is_prime(n)
}

#[pyfunction]
fn factor(n: u64) -> Vec<u64> {
    field::factor(n)
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
    Ok(())
}
