//! Python bindings (`pip install maturin && maturin develop --release
//! --features python`). Function names and signatures are kept stable across
//! internal refactors so downstream experiment scripts do not change.

// pyo3 macro expansion trips clippy::useless_conversion on every #[pyfunction];
// false positive at this expansion site.
#![allow(clippy::useless_conversion)]

use crate::census::buckets::{dp, mitm};
use crate::domain::MultiplicativeSubgroup;
use crate::rs::code;
use crate::smooth::rung;
use crate::{census::kernel, field};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::{PyIOError, PyNotImplementedError, PyValueError};
use pyo3::prelude::*;

/// Map a library error onto the closest builtin Python exception:
/// I/O failures raise `IOError`, engine/parameter-regime limits raise
/// `NotImplementedError`, and every validation failure raises
/// `ValueError`. (The enum is `#[non_exhaustive]`; unknown future
/// variants default to `ValueError`.)
fn to_py(e: crate::Error) -> PyErr {
    match e {
        crate::Error::Io { .. } => PyIOError::new_err(e.to_string()),
        crate::Error::Unsupported(_) => PyNotImplementedError::new_err(e.to_string()),
        _ => PyValueError::new_err(e.to_string()),
    }
}

fn sub(p: u64, s: usize) -> PyResult<MultiplicativeSubgroup> {
    MultiplicativeSubgroup::new(p, s).map_err(to_py)
}

fn err<T>(r: crate::Result<T>) -> PyResult<T> {
    r.map_err(to_py)
}

/// Full exact q=1 bucket distribution: `out[lam]` counts r-subsets S of mu_s with `e_1(S) = lam`. Cost/memory scale with p.
#[pyfunction]
fn bucket_dist_q1(py: Python<'_>, p: u64, s: usize, r: usize) -> PyResult<Py<PyArray1<u64>>> {
    let d = err(py.allow_threads(|| dp::distribution_q1(&MultiplicativeSubgroup::new(p, s)?, r)))?;
    Ok(d.into_values().into_pyarray_bound(py).into())
}

/// Full exact q=2 joint distribution over `(e_1, e_2)`, shape (p, p). Intended for p <= ~700.
#[pyfunction]
fn bucket_dist_q2(py: Python<'_>, p: u64, s: usize, r: usize) -> PyResult<Py<PyArray2<u64>>> {
    let v = err(py.allow_threads(|| dp::distribution_q2(&MultiplicativeSubgroup::new(p, s)?, r)))?;
    let pp = p as usize;
    let arr = numpy::ndarray::Array2::from_shape_vec((pp, pp), v)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py).into())
}

/// Kernel census, weight-capped: `out[w]` counts nonzero vectors with coefficients in `[-cmax, cmax]` and weight w <= wmax such that `sum v_i w^i = 0 (mod p)`.
#[pyfunction]
fn census_direct(p: u64, s: usize, cmax: i64, wmax: usize) -> PyResult<Vec<u64>> {
    err(kernel::direct(&sub(p, s)?, cmax, wmax))
}

/// Full kernel census by weight (meet-in-the-middle; s <= 32 at cmax = 2).
#[pyfunction]
fn census_mitm(p: u64, s: usize, cmax: i64) -> PyResult<Vec<u64>> {
    err(kernel::mitm(&sub(p, s)?, cmax))
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
fn buckets_e(
    py: Python<'_>,
    p: u64,
    s: usize,
    r: usize,
    q: usize,
    lams: Vec<Vec<u64>>,
) -> PyResult<Py<PyArray1<u64>>> {
    use rayon::prelude::*;
    let v = err(py.allow_threads(|| {
        let sg = MultiplicativeSubgroup::new(p, s)?;
        let t = mitm::HalfTables::build(&sg, r, q)?;
        lams.par_iter()
            .map(|l| t.bucket(l))
            .collect::<crate::Result<Vec<_>>>()
    }))?;
    Ok(v.into_pyarray_bound(py).into())
}

/// The common `(e_1..e_q)` of the Theorem-A rung family (the optimal structural construction).
#[pyfunction]
fn rung_lambda_e(p: u64, s: usize, r: usize, q: usize) -> PyResult<Vec<u64>> {
    err(rung::rung_lambda(&sub(p, s)?, r, q))
}

/// Anatomy of a q=1 bucket: returns `(total, per_weight_class_counts)`; total equals the DP bucket exactly.
#[pyfunction]
fn decompose_bucket_q1(p: u64, s: usize, r: usize, lam: u64) -> PyResult<(u64, Vec<u64>)> {
    err(mitm::decompose_bucket_q1(&sub(p, s)?, r, lam))
}

/// The quantized-ladder structural maximum `C(s/2^t - [r0!=0], floor(r/2^t))`, `t = ceil(log2(q+1))`.
#[pyfunction]
fn m_struct(s: usize, r: usize, q: usize) -> u64 {
    rung::m_struct(s, r, q)
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
fn dist_stats_q1(
    py: Python<'_>,
    p: u64,
    s: usize,
    r: usize,
) -> PyResult<(u64, u64, u64, u64, u128)> {
    let d = err(py.allow_threads(|| dp::distribution_q1(&MultiplicativeSubgroup::new(p, s)?, r)))?;
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
    .map_err(to_py)
}

/// Tiered structural certificate for (p, s, r), q = 1 (p-independent).
/// Returns (tier, m_struct, zero_bucket): tier 1 = all buckets structural,
/// tier 2 = zero bucket structural, tier 3 = inflated (zero_bucket = exact
/// inflated value; tiers 1-2 return the structural zero-class size).
#[pyfunction]
fn certify_q1(p: u64, s: usize, r: usize) -> PyResult<(u8, u64, u64)> {
    use crate::smooth::certify::{certify_q1 as cert, Verdict};
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
    rung::class_size(s, r, w)
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
    .map_err(to_py)
}

/// An attack row: (delta_star, deficit, t, s_g, r, log2_list).
type AttackRow = (f64, f64, u32, u64, u64, f64);

/// Best ladder attack: (delta_star, deficit, t, s_g, r, log2_list), or None.
#[pyfunction]
fn attack_best(n: u64, k: u64, list_bits: f64) -> PyResult<Option<AttackRow>> {
    let p = crate::soundness::AttackParams { n, k, list_bits };
    let a = err(crate::soundness::best_attack(&p))?;
    Ok(a.map(|a| (a.delta_star, a.deficit, a.t, a.s_g, a.r, a.log2_list)))
}

/// Antipodal (survey Table-5) baseline attack, same tuple shape as attack_best.
#[pyfunction]
fn attack_antipodal(n: u64, k: u64, list_bits: f64) -> PyResult<Option<AttackRow>> {
    let p = crate::soundness::AttackParams { n, k, list_bits };
    let a = err(crate::soundness::antipodal_attack(&p))?;
    Ok(a.map(|a| (a.delta_star, a.deficit, a.t, a.s_g, a.r, a.log2_list)))
}

/// Structural-framework ceiling delta_min - H2(rate)/list_bits.
#[pyfunction]
fn attack_ceiling(n: u64, k: u64, list_bits: f64) -> PyResult<f64> {
    err(crate::soundness::hyperbola_ceiling(
        &crate::soundness::AttackParams { n, k, list_bits },
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
                    let lam = rung::rung_lambda(&sg, r, q)?;
                    let t = mitm::HalfTables::build(&sg, r, q)?;
                    row.push(t.bucket(&lam)?);
                }
                Ok((p, row))
            })
            .collect::<crate::Result<Vec<_>>>()
    })
    .map_err(to_py)
}

/// Rayon-parallel certificates: rows of (p, tier, m_struct, zero_bucket).
#[pyfunction]
fn certify_many(
    py: Python<'_>,
    s: usize,
    r: usize,
    primes: Vec<u64>,
) -> PyResult<Vec<(u64, u8, u64, u64)>> {
    use crate::smooth::certify::{certify_q1 as cert, Verdict};
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
    .map_err(to_py)
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
    py.allow_threads(|| crate::smooth::norms::bad_set(s, wmax, cmax))
        .map(|v| {
            v.into_iter()
                .map(|e| {
                    let cf = e.provenance == crate::smooth::norms::Provenance::CensusCorrected;
                    (e.p, e.counts, cf)
                })
                .collect()
        })
        .map_err(to_py)
}

/// Per-weight maximum cyclotomic norms (the anticorrelation profile),
/// as decimal strings (values can exceed u64 at large s).
#[pyfunction]
fn norms_n_max(py: Python<'_>, s: usize, wmax: usize, cmax: i64) -> PyResult<Vec<String>> {
    py.allow_threads(|| crate::smooth::norms::norm_table(s, wmax, cmax))
        .map(|t| t.n_max_by_weight().iter().map(|n| n.to_string()).collect())
        .map_err(to_py)
}

/// An accident-event row in the ratified field order: (level, p, weight,
/// norm, valuation, cofactor, orbit_rep, orbit_size, height, max_coeff,
/// provenance, source); provenance 0 = exhaustive orbits, 1 = exemplar
/// only; source 0 = CPU norm table, 1 = GPU ingest.
type EventRow = (
    usize,
    u64,
    usize,
    u128,
    u64,
    u128,
    Vec<i64>,
    u64,
    u64,
    i64,
    u8,
    u8,
);

fn event_rows(events: Vec<crate::smooth::norms::events::AccidentEvent>) -> Vec<EventRow> {
    use crate::smooth::norms::events::{EventProvenance, EventSource};
    events
        .into_iter()
        .map(|e| {
            (
                e.level,
                e.p,
                e.weight,
                e.norm,
                e.valuation,
                e.cofactor,
                e.orbit_rep,
                e.orbit_size,
                e.height,
                e.max_coeff,
                u8::from(e.provenance == EventProvenance::ExemplarOnly),
                u8::from(e.source == EventSource::GpuIngest),
            )
        })
        .collect()
}

/// The complete accident inventory of the (s, wmax, cmax) enumeration:
/// one [`EventRow`] per (prime, symmetry orbit) incidence, exhaustive
/// and occupancy-certified (norms::events::accident_events).
#[pyfunction]
fn accident_events(py: Python<'_>, s: usize, wmax: usize, cmax: i64) -> PyResult<Vec<EventRow>> {
    py.allow_threads(|| {
        let table = crate::smooth::norms::norm_table(s, wmax, cmax)?;
        crate::smooth::norms::events::accident_events(&table)
    })
    .map(event_rows)
    .map_err(to_py)
}

/// The ingest summary returned to Python:
/// (n_rows, mass_by_weight, n_max_by_weight, entries_parsed, events).
type IngestSummary = (u64, Vec<u64>, Vec<u64>, u64, Vec<EventRow>);

/// The events-retaining ingest: [`badset_from_gpu_json`]'s outputs plus
/// accident-event rows for retained primes (listed in `event_primes`, or
/// at/above `event_min_p`), built from binary dumps with row-aligned
/// `.exemplars.bin` files. Returns an [`IngestSummary`].
#[pyfunction]
#[allow(clippy::too_many_arguments)] // the Python API's parameter list
fn badset_and_events_from_gpu_bin(
    py: Python<'_>,
    paths: Vec<String>,
    s: usize,
    wmax: usize,
    cmax: i64,
    out_prefix: String,
    event_primes: Vec<u64>,
    event_min_p: u64,
) -> PyResult<IngestSummary> {
    use crate::smooth::norms::events::EventFilter;
    let filter = EventFilter {
        primes: event_primes,
        min_p: event_min_p,
    };
    let (rows, events, stats) = py
        .allow_threads(|| {
            crate::smooth::norms::ingest::badset_and_events_from_gpu_bin(
                &paths,
                s,
                wmax,
                cmax,
                Some(&out_prefix),
                filter,
            )
        })
        .map_err(to_py)?;
    let n = rows.len() as u64;
    let werr = |e: std::io::Error| PyIOError::new_err(e.to_string());
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
    crate::smooth::norms::ingest::clear_checkpoint(&out_prefix);
    Ok((
        n,
        stats.mass_by_weight,
        stats.n_max_by_weight,
        stats.entries_parsed,
        event_rows(events),
    ))
}

/// Ingest GPU-campaign norm-table JSON shards into the s-64-scale bad set.
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
            crate::smooth::norms::ingest::badset_from_gpu_json(&paths, s, wmax, Some(&out_prefix))
        })
        .map_err(to_py)?;
    let n = rows.len() as u64;
    let werr = |e: std::io::Error| PyIOError::new_err(e.to_string());
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
    crate::smooth::norms::ingest::clear_checkpoint(&out_prefix);
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
    py: Python<'_>,
    p: u64,
    domain: Vec<u64>,
    k: usize,
    word: Vec<u64>,
    t: usize,
) -> PyResult<Py<PyArray2<u64>>> {
    let members = err(py.allow_threads(|| {
        let rs = code::ReedSolomon::on_domain(p, domain, k)?;
        let oracle = crate::rs::decode::DecodeOracle::new(&rs);
        oracle.list(&word, crate::rs::decode::Radius::agreement(t))
    }))?;
    rows_to_array(py, &members)
}

/// Exact list decode past the fiber count on a paired domain
/// (`points[i + n] = -points[i]`), by core enumeration and
/// Guruswami-Sudan residual decodes: complete for `t > n` whenever
/// `(t - 2l)^2 > (s - 2l)(k - 2l - 1)` with `l = t - n` (errors
/// otherwise, and on unpaired domains). Reaches cells the
/// information-set engine cannot. GIL released.
#[pyfunction]
fn list_decode_paired(
    py: Python<'_>,
    p: u64,
    points: Vec<u64>,
    k: usize,
    word: Vec<u64>,
    t: usize,
) -> PyResult<Py<PyArray2<u64>>> {
    let members = err(py.allow_threads(|| {
        let dom = crate::rs::core_residual::PairedDomain::from_points(p, points)?;
        crate::rs::core_residual::list_paired(&dom, k, &word, t)
    }))?;
    rows_to_array(py, &members)
}

/// Sampled lower bound for the paired-domain decode: the distinct
/// members found through `samples` uniform cores — a subset of the
/// true list, deterministic in `seed`. The optimizer's objective.
#[pyfunction]
fn list_decode_paired_sampled(
    py: Python<'_>,
    p: u64,
    points: Vec<u64>,
    k: usize,
    word: Vec<u64>,
    t: usize,
    samples: u64,
    seed: u64,
) -> PyResult<Py<PyArray2<u64>>> {
    let members = err(py.allow_threads(|| {
        let dom = crate::rs::core_residual::PairedDomain::from_points(p, points)?;
        crate::rs::core_residual::list_paired_sampled(&dom, k, &word, t, samples, seed)
    }))?;
    rows_to_array(py, &members)
}

/// Exact distinct list sizes for a batch of words in one sweep (issue
/// #45): shared barycentric tables per information set, lex-first
/// dedup, rayon inside — the CPU campaign confirm path without the
/// per-word round-trips. GIL released for the whole computation.
#[pyfunction]
fn list_sizes(
    py: Python<'_>,
    p: u64,
    domain: Vec<u64>,
    k: usize,
    words: Vec<Vec<u64>>,
    t: usize,
) -> PyResult<Vec<u64>> {
    err(py.allow_threads(|| {
        let rs = code::ReedSolomon::on_domain(p, domain, k)?;
        let oracle = crate::rs::decode::DecodeOracle::new(&rs);
        oracle.list_sizes(&words, crate::rs::decode::Radius::agreement(t))
    }))
}

/// One code-first optimization run on `RS[F_p, domain, k]`: build a random
/// pencil seed and anneal it to maximize list size. Returns
/// `(center, members, size_trajectory)` — the raw cluster (member codewords as
/// evaluation vectors) plus the per-move list-size trajectory. Loop over `seed`
/// in Python to collect a discovery dataset. `domain` is any distinct-point set.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn anneal_pencil(
    py: Python<'_>,
    p: u64,
    domain: Vec<u64>,
    k: usize,
    t: usize,
    petals: usize,
    steps: usize,
    seed: u64,
) -> PyResult<(Vec<u64>, Vec<Vec<u64>>, Vec<usize>)> {
    err(py.allow_threads(|| {
        let rs = code::ReedSolomon::on_domain(p, domain, k)?;
        let rad = crate::rs::decode::Radius::agreement(t);
        let seedw = crate::rs::cluster::random_pencil_seed(&rs, petals, seed)?;
        let (cl, tr) = crate::rs::cluster::anneal(&rs, &seedw, rad, steps, 2.0, 0.92, seed)?;
        Ok((
            cl.center().to_vec(),
            cl.members().to_vec(),
            tr.sizes.clone(),
        ))
    }))
}

/// One code-first run to **convergence**: build a random pencil seed and greedily
/// hill-climb (boundary-alignment flips) until no flip increases the list — a
/// true local maximum of the list size. `max_flips` is a safety cap (set it
/// well above the achievable list size; convergence normally halts first).
/// Returns `(center, members, size_trajectory)`; the trajectory is monotone and
/// its last value is the converged local-max list size. Deterministic in `seed`.
#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn optimize_pencil(
    py: Python<'_>,
    p: u64,
    domain: Vec<u64>,
    k: usize,
    t: usize,
    petals: usize,
    max_flips: usize,
    seed: u64,
) -> PyResult<(Vec<u64>, Vec<Vec<u64>>, Vec<usize>)> {
    err(py.allow_threads(|| {
        let rs = code::ReedSolomon::on_domain(p, domain, k)?;
        let rad = crate::rs::decode::Radius::agreement(t);
        let seedw = crate::rs::cluster::random_pencil_seed(&rs, petals, seed)?;
        let (cl, tr) = crate::rs::cluster::optimize(&rs, &seedw, rad, 1, max_flips)?;
        Ok((
            cl.center().to_vec(),
            cl.members().to_vec(),
            tr.sizes.clone(),
        ))
    }))
}

/// Greedy list-size climb to convergence FROM A GIVEN WORD (warm start): the
/// binding for `rs::cluster::optimize` with an explicit seed. Returns
/// `(center, members, size_trajectory)` with `members` as an `(L, n)` uint64
/// array. Deterministic.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn optimize_word(
    py: Python<'_>,
    p: u64,
    domain: Vec<u64>,
    k: usize,
    t: usize,
    word: Vec<u64>,
    max_flips: usize,
) -> PyResult<(Vec<u64>, Py<PyArray2<u64>>, Vec<usize>)> {
    let (center, members, sizes) = err(py.allow_threads(|| {
        let rs = code::ReedSolomon::on_domain(p, domain, k)?;
        let rad = crate::rs::decode::Radius::agreement(t);
        let (cl, tr) = crate::rs::cluster::optimize(&rs, &word, rad, 1, max_flips)?;
        Ok((
            cl.center().to_vec(),
            cl.members().to_vec(),
            tr.sizes.clone(),
        ))
    }))?;
    Ok((center, rows_to_array(py, &members)?, sizes))
}

/// A random pencil seed word (random `(k-1)`-core + `petals` petal codewords),
/// the unbiased code-first start for the search engines. Deterministic in
/// `seed`. The Rust engine `optimize_word` climbs from any word, this one
/// included.
#[pyfunction]
fn pencil_seed(p: u64, domain: Vec<u64>, k: usize, petals: usize, seed: u64) -> PyResult<Vec<u64>> {
    let rs = err(code::ReedSolomon::on_domain(p, domain, k))?;
    err(crate::rs::cluster::random_pencil_seed(&rs, petals, seed))
}

/// One symmetric-function stat row: (index, entropy_bits, distinct,
/// max_class_fraction, mode_value, distribution as (value, count) pairs).
type SymRow = (usize, f64, usize, f64, u64, Vec<(u64, u64)>);

/// Decode with the full structural profile in one call: returns
/// `(members, agreement_sizes, size_entropy, sym_stats, joint_entropy,
/// joint_distinct)` where `members` is an `(L, n)` uint64 array,
/// `agreement_sizes` is `[(size, count)]`, and `sym_stats` has one
/// [`SymRow`] per `e_i` (`e_1` first). The canonical "is anything frozen?"
/// probe — replaces the hand-rolled post-processing after `list_decode`.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn decode_profile(
    py: Python<'_>,
    p: u64,
    domain: Vec<u64>,
    k: usize,
    word: Vec<u64>,
    t: usize,
) -> PyResult<(
    Py<PyArray2<u64>>,
    Vec<(usize, u64)>,
    f64,
    Vec<SymRow>,
    f64,
    usize,
)> {
    let (members, st) = err(py.allow_threads(|| {
        let rs = code::ReedSolomon::on_domain(p, domain, k)?;
        let rad = crate::rs::decode::Radius::agreement(t);
        let members = crate::rs::decode::DecodeOracle::new(&rs).list(&word, rad)?;
        let st = crate::rs::classify::structure(&rs, &word, rad)?;
        Ok((members, st))
    }))?;
    let sym = st
        .symmetric
        .into_iter()
        .map(|s| {
            (
                s.index,
                s.entropy,
                s.distinct,
                s.max_class_fraction,
                s.mode_value,
                s.distribution,
            )
        })
        .collect();
    Ok((
        rows_to_array(py, &members)?,
        st.agreement_sizes,
        st.size_entropy,
        sym,
        st.joint_entropy,
        st.joint_distinct,
    ))
}

/// The additive frozen-head word `sum_i (-1)^i lambda_i x^{r-i}` on `mu_s` (the
/// Theorem-B word class; `lam` holds `(e_1..e_q)`).
#[pyfunction]
fn c5_word(p: u64, s: usize, r: usize, lam: Vec<u64>) -> PyResult<Vec<u64>> {
    let sg = sub(p, s)?;
    let k = r.saturating_sub(lam.len()).max(1);
    let rs = err(code::ReedSolomon::on_subgroup(&sg, k))?;
    err(rs.c5_word(r, &lam))
}

/// The proven multiplicative extremal word `x^{r-1} - (-1)^{r+1} zeta^c
/// x^{s-1}` (Theorem B_mult): its exact list at agreement `r` over code degree
/// `< r-1` is the Graham-Sloane class of `c`, at every prime containing mu_s.
#[pyfunction]
fn top_word(p: u64, s: usize, r: usize, c: usize) -> PyResult<Vec<u64>> {
    err(rung::top_word(&sub(p, s)?, r, c))
}

/// The word of a syndrome vector: `w = sum_j (-1)^j b_j x^{r-1+j}` on
/// `domain`, pinned to the convention `D_S(w) = sum_j b_j e_j(complement)`.
#[pyfunction]
fn word_from_syndrome(p: u64, domain: Vec<u64>, r: usize, b: Vec<u64>) -> Vec<u64> {
    rung::word_from_syndrome(p, &domain, r, &b)
}

/// The negacyclic fold: `zeta^exp = sign * zeta^index` on the
/// half-basis. THE primitive — campaigns must call this instead of
/// re-deriving exponent reduction (conventions: `crate::ring`).
#[pyfunction]
fn fold(half: usize, exp: usize) -> (usize, i64) {
    crate::ring::fold(half, exp)
}

/// An element of `Z[zeta_s]` (s a power of two) on the half-basis.
/// The characteristic-zero home of exact values: norms, Galois action,
/// dilation, per-prime cleanliness tests.
#[pyclass(name = "Cyclo")]
#[derive(Clone)]
struct PyCyclo {
    inner: crate::ring::Cyclo,
}

#[pymethods]
impl PyCyclo {
    #[new]
    fn new(coeffs: Vec<i64>) -> PyResult<Self> {
        Ok(PyCyclo {
            inner: err(crate::ring::Cyclo::from_coeffs(coeffs))?,
        })
    }
    #[staticmethod]
    fn monomial(s: usize, exp: usize) -> PyResult<Self> {
        Ok(PyCyclo {
            inner: err(crate::ring::Cyclo::monomial(s, exp))?,
        })
    }
    /// `1 - zeta^exp` — the content map.
    #[staticmethod]
    fn one_minus(s: usize, exp: usize) -> PyResult<Self> {
        Ok(PyCyclo {
            inner: err(crate::ring::Cyclo::one_minus(s, exp))?,
        })
    }
    /// Exact `prod(1 - zeta^e)` over `exps` — the A-map value.
    #[staticmethod]
    fn prod_one_minus(s: usize, exps: Vec<usize>) -> PyResult<Self> {
        Ok(PyCyclo {
            inner: err(crate::ring::Cyclo::prod_one_minus(s, &exps))?,
        })
    }
    /// Exact elementary symmetric functions `e_0..e_m` of `{zeta^e}`.
    #[staticmethod]
    fn e_vector(s: usize, exps: Vec<usize>, m: usize) -> PyResult<Vec<PyCyclo>> {
        Ok(err(crate::ring::Cyclo::e_vector(s, &exps, m))?
            .into_iter()
            .map(|inner| PyCyclo { inner })
            .collect())
    }
    fn coeffs(&self) -> Vec<i64> {
        self.inner.coeffs().to_vec()
    }
    fn s(&self) -> usize {
        self.inner.s()
    }
    fn add(&self, o: &PyCyclo) -> PyResult<Self> {
        Ok(PyCyclo {
            inner: err(self.inner.add(&o.inner))?,
        })
    }
    fn sub(&self, o: &PyCyclo) -> PyResult<Self> {
        Ok(PyCyclo {
            inner: err(self.inner.sub(&o.inner))?,
        })
    }
    fn mul(&self, o: &PyCyclo) -> PyResult<Self> {
        Ok(PyCyclo {
            inner: err(self.inner.mul(&o.inner))?,
        })
    }
    fn neg(&self) -> Self {
        PyCyclo {
            inner: self.inner.neg(),
        }
    }
    fn mul_ntt(&self, o: &PyCyclo) -> PyResult<Self> {
        Ok(PyCyclo {
            inner: err(self.inner.mul_ntt(&o.inner))?,
        })
    }
    fn dilate(&self, d: usize) -> Self {
        PyCyclo {
            inner: self.inner.dilate(d),
        }
    }
    fn galois(&self, m: usize) -> PyResult<Self> {
        Ok(PyCyclo {
            inner: err(self.inner.galois(m))?,
        })
    }
    fn conj(&self) -> Self {
        PyCyclo {
            inner: self.inner.conj(),
        }
    }
    fn eval_at(&self, x: u64, p: u64) -> u64 {
        self.inner.eval_at(x, p)
    }
    fn norm_mod(&self, p: u64) -> PyResult<u64> {
        err(self.inner.norm_mod(p))
    }
    fn norm_i128(&self) -> PyResult<i128> {
        err(self.inner.norm_i128())
    }
    /// Exact field norm at any height (CRT over `norm_mod` at 62-bit
    /// good primes; arbitrary-precision result). The wrapper that
    /// `norm_i128`'s overflow error points to.
    fn norm_crt(&self, py: Python<'_>) -> PyResult<num_bigint::BigInt> {
        err(py.allow_threads(|| self.inner.norm_crt()))
    }
    fn weight(&self) -> usize {
        self.inner.weight()
    }
    fn sq_sum(&self) -> i128 {
        self.inner.sq_sum()
    }
    fn height(&self) -> i64 {
        self.inner.height()
    }
    fn is_zero(&self) -> bool {
        self.inner.is_zero()
    }
    /// The integer value of this element, or None if it is not rational.
    fn as_int(&self) -> Option<i64> {
        self.inner.as_int()
    }
    /// Equality against a rational integer.
    fn eq_int(&self, v: i64) -> bool {
        self.inner.eq_int(v)
    }
    fn __repr__(&self) -> String {
        format!("Cyclo(s={}, {:?})", self.inner.s(), self.inner.coeffs())
    }
    fn __eq__(&self, o: &PyCyclo) -> bool {
        self.inner == o.inner
    }
    fn __hash__(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        self.inner.hash(&mut h);
        h.finish()
    }
}

/// MITM value-map census: `(total, distinct, max_fiber, argmax,
/// second_moment)` of `prod(point - dom[e])` over `size`-subsets of
/// `h1 + h2` with exponent sum = `class` (mod level). The collision
/// count is `second_moment`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn valuemap_census(
    py: Python<'_>,
    p: u64,
    level: usize,
    h1: Vec<usize>,
    h2: Vec<usize>,
    size: usize,
    class: usize,
    point: u64,
) -> PyResult<(u64, u64, u64, u64, u128)> {
    let c = err(py.allow_threads(|| {
        let dom = MultiplicativeSubgroup::new(p, level)?.elements().to_vec();
        let a = crate::census::valuemap::half_table(&dom, &h1, level, point, p)?;
        let b = crate::census::valuemap::half_table(&dom, &h2, level, point, p)?;
        crate::census::valuemap::join_census(&a, &b, size, class, p)
    }))?;
    Ok((c.total, c.distinct, c.max_fiber, c.argmax, c.second_moment))
}

/// The fiber size of one target value in the MITM value-map census.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn valuemap_fiber(
    py: Python<'_>,
    p: u64,
    level: usize,
    h1: Vec<usize>,
    h2: Vec<usize>,
    size: usize,
    class: usize,
    point: u64,
    value: u64,
) -> PyResult<u64> {
    err(py.allow_threads(|| {
        let dom = MultiplicativeSubgroup::new(p, level)?.elements().to_vec();
        let a = crate::census::valuemap::half_table(&dom, &h1, level, point, p)?;
        let b = crate::census::valuemap::half_table(&dom, &h2, level, point, p)?;
        crate::census::valuemap::fiber_count(&a, &b, size, class, p, value)
    }))
}

/// Members of one fiber as sorted exponent lists (up to `cap`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn valuemap_fiber_members(
    py: Python<'_>,
    p: u64,
    level: usize,
    h1: Vec<usize>,
    h2: Vec<usize>,
    size: usize,
    class: usize,
    point: u64,
    value: u64,
    cap: usize,
) -> PyResult<Vec<Vec<usize>>> {
    err(py.allow_threads(|| {
        let dom = MultiplicativeSubgroup::new(p, level)?.elements().to_vec();
        let a = crate::census::valuemap::half_table(&dom, &h1, level, point, p)?;
        let b = crate::census::valuemap::half_table(&dom, &h2, level, point, p)?;
        crate::census::valuemap::fiber_members(&a, &b, size, class, p, value, cap)
    }))
}

/// Fiber-size histogram of the MITM census: `out[k]` = number of
/// values with fiber size `k`. The lightest full-shape output.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn valuemap_histogram(
    py: Python<'_>,
    p: u64,
    level: usize,
    h1: Vec<usize>,
    h2: Vec<usize>,
    size: usize,
    class: usize,
    point: u64,
) -> PyResult<Py<PyArray1<u64>>> {
    let h = err(py.allow_threads(|| {
        let dom = MultiplicativeSubgroup::new(p, level)?.elements().to_vec();
        let a = crate::census::valuemap::half_table(&dom, &h1, level, point, p)?;
        let b = crate::census::valuemap::half_table(&dom, &h2, level, point, p)?;
        crate::census::valuemap::join_histogram(&a, &b, size, class, p)
    }))?;
    Ok(h.into_pyarray_bound(py).into())
}

/// A pair of u64 arrays (values, multiplicities).
type U64ArrayPair = (Py<PyArray1<u64>>, Py<PyArray1<u64>>);

/// Value-resolved MITM census: `(values, multiplicities)` sorted by
/// value — the input of the spectrum pipeline and per-value figures.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn valuemap_distribution(
    py: Python<'_>,
    p: u64,
    level: usize,
    h1: Vec<usize>,
    h2: Vec<usize>,
    size: usize,
    class: usize,
    point: u64,
) -> PyResult<U64ArrayPair> {
    let (v, c) = err(py.allow_threads(|| {
        let dom = MultiplicativeSubgroup::new(p, level)?.elements().to_vec();
        let a = crate::census::valuemap::half_table(&dom, &h1, level, point, p)?;
        let b = crate::census::valuemap::half_table(&dom, &h2, level, point, p)?;
        crate::census::valuemap::join_distribution(&a, &b, size, class, p)
    }))?;
    Ok((
        v.into_pyarray_bound(py).into(),
        c.into_pyarray_bound(py).into(),
    ))
}

/// A valuemap sweep row: (p, total, distinct, max_fiber, argmax,
/// second_moment).
type ValuemapRow = (u64, u64, u64, u64, u64, u128);

/// Parallel prime sweep of the MITM census summary — the
/// floors-vs-accidents campaign driver.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn valuemap_sweep(
    py: Python<'_>,
    level: usize,
    h1: Vec<usize>,
    h2: Vec<usize>,
    size: usize,
    class: usize,
    point: u64,
    primes: Vec<u64>,
) -> PyResult<Vec<ValuemapRow>> {
    use rayon::prelude::*;
    py.allow_threads(|| {
        primes
            .par_iter()
            .map(|&p| {
                let dom = MultiplicativeSubgroup::new(p, level)?.elements().to_vec();
                let a = crate::census::valuemap::half_table(&dom, &h1, level, point, p)?;
                let b = crate::census::valuemap::half_table(&dom, &h2, level, point, p)?;
                let c = crate::census::valuemap::join_census(&a, &b, size, class, p)?;
                Ok((
                    p,
                    c.total,
                    c.distinct,
                    c.max_fiber,
                    c.argmax,
                    c.second_moment,
                ))
            })
            .collect::<crate::Result<Vec<_>>>()
    })
    .map_err(to_py)
}

/// Exact skeleton-DP totals of the G1 census at `level` (32 or 64):
/// `(window, budget_pairs, budget_skeletons)`.
#[pyfunction]
fn skeleton_totals(py: Python<'_>, level: usize) -> PyResult<(u128, u128, u128)> {
    err(py.allow_threads(|| crate::census::skeleton::skeleton_totals(level)))
}

/// The exact G1-criterion census at `level` (32 or 64) via the MITM
/// join: `(m1_pairs, m2_pairs, solvable_pairs, solutions)`. Level 64
/// is a minutes-scale many-core computation (S4: 262s on 252 threads).
#[pyfunction]
fn skeleton_census(py: Python<'_>, level: usize) -> PyResult<(u64, u64, u64, u64)> {
    let c = err(py.allow_threads(|| crate::census::skeleton::skeleton_census(level)))?;
    Ok((c.m1_pairs, c.m2_pairs, c.solvable_pairs, c.solutions))
}

/// The fold unit `u_e = (1 + zeta^e)/(1 - zeta^e)` as an exact ring
/// element (closed form; the descent calculus's bookkeeping currency).
#[pyfunction]
fn fold_unit(s: usize, e: usize) -> PyResult<PyCyclo> {
    Ok(PyCyclo {
        inner: err(crate::ring::fold_unit(s, e))?,
    })
}

/// Certified multiplicative independence (mod torsion) of the free
/// fold units at level `s`: `(det_lo, det_hi, independent)` from the
/// interval-arithmetic determinant of the log-embedding matrix.
#[pyfunction]
fn foldunit_rank_certificate(s: usize) -> PyResult<(f64, f64, bool)> {
    let c = err(crate::ring::rank_certificate(s))?;
    Ok((c.det_lo, c.det_hi, c.independent))
}

/// A certified alpha table:
/// `(denom, alpha rows, torsion2s, residual_bound, height_gap)`.
type AlphaCertRow = (i64, Vec<Vec<i64>>, Vec<usize>, f64, f64);

/// The certified atom-address table at `level` (a power of two,
/// 16..=8192): `(denom, alpha, torsion2s, residual_bound, height_gap)`
/// with `alpha[j-1]` = the integer vector `denom * alpha_j` of
/// `A_j = (1 - zeta^j)/(1 - zeta)^{gcd(j, level)}` in the fold-unit
/// basis, certified exact (interval residual below the height gap +
/// two-camera torsion pin).
#[pyfunction]
fn foldunit_alpha_certificate(py: Python<'_>, level: usize) -> PyResult<AlphaCertRow> {
    let c = err(py.allow_threads(|| crate::ring::foldunits::alpha_certificate(level)))?;
    Ok((
        c.denom,
        c.alpha,
        c.torsion2s,
        c.residual_bound,
        c.height_gap,
    ))
}

/// Batch `N(v) mod p` over many half-basis coefficient vectors (one
/// shared subgroup construction, rayon-parallel): the extremal-norm
/// campaign workhorse.
#[pyfunction]
fn norms_mod_batch(py: Python<'_>, coeffs: Vec<Vec<i64>>, p: u64, s: usize) -> PyResult<Vec<u64>> {
    err(py.allow_threads(|| {
        let sg = MultiplicativeSubgroup::new(p, s)?;
        crate::ring::Cyclo::norms_mod_batch(&coeffs, &sg)
    }))
}

/// Exact `Z[zeta_s]` value census of coordinate `coord` over all
/// `r`-subsets of the s-th roots of unity: `(distinct, intrinsic_floor,
/// top5_multiplicities)`. The prime-independent floors of the pointwise
/// L^2 census; integer-exact, no field involved.
#[pyfunction]
fn exact_value_census(s: usize, r: usize, coord: usize) -> PyResult<(u64, u64, Vec<u64>)> {
    err(crate::census::exact_value_census(s, r, coord))
}

/// Tower cleanliness certificate at a named prime (ring::primes):
/// one (s, tier, counts) triple per level from s_top down to 8.
/// tier: 0 = Certified (full census empty), 1 = CertifiedAtUnit
/// (zero class exact; counts = [-2,2] census by weight),
/// 2 = CertifiedToWeight (counts = [wmax]), 3 = Dirty (counts =
/// {-1,0,1} census by weight).
#[pyfunction]
fn certify_clean(
    py: Python<'_>,
    p: u64,
    s_top: usize,
    wmax_large: usize,
) -> PyResult<Vec<(usize, u8, Vec<u64>)>> {
    use crate::ring::primes::{certify_clean as cert, LevelCleanliness as L};
    let c = err(py.allow_threads(|| cert(p, s_top, wmax_large)))?;
    Ok(c.levels
        .into_iter()
        .map(|l| match l.verdict {
            L::Certified => (l.s, 0, vec![]),
            L::CertifiedAtUnit { census2_by_weight } => (l.s, 1, census2_by_weight),
            L::CertifiedToWeight { wmax } => (l.s, 2, vec![wmax as u64]),
            L::Dirty { census_by_weight } => (l.s, 3, census_by_weight),
        })
        .collect())
}

/// One certified Table-4-style row (present only when the crate is built
/// with the `certified` feature): the smallest radius z* whose exact-Elias +
/// Lemma 6.12 chain certifies soundness >= 2^target_bits, for interleaving
/// width s, block length total_len / s, rate 1/2, base alphabet size
/// base_q, extension |F| = base_q^ext_degree. Returns
/// (z_star, n, delta_star, lg_sound_lo, lg_sound_hi, crossing_pinned).
#[cfg(feature = "certified")]
#[pyfunction]
fn elias_row(
    py: Python<'_>,
    s: u64,
    total_len: u64,
    base_q: u64,
    ext_degree: u32,
    target_bits: f64,
) -> PyResult<(u64, u64, f64, f64, f64, bool)> {
    use rug::ops::Pow;
    let base = rug::Integer::from(base_q);
    let ext = base.clone().pow(ext_degree);
    let r =
        err(py.allow_threads(|| {
            crate::soundness::elias_row(s, total_len, &base, &ext, target_bits)
        }))?;
    Ok((
        r.z_star,
        r.n,
        r.delta_star,
        r.lg_sound_lo,
        r.lg_sound_hi,
        r.crossing_pinned,
    ))
}

/// Graham-Sloane class counts `out[c] = #{T in C(Z_s, t) : sum T = c mod s}`.
#[pyfunction]
fn gs_class_counts(py: Python<'_>, s: usize, t: usize) -> PyResult<Py<PyArray1<u64>>> {
    Ok(err(rung::gs_class_counts(s, t))?
        .into_pyarray_bound(py)
        .into())
}

/// The moment cloud of `(p, domain, r)`: an `(C(n,r), n-r+1)` uint64 array;
/// row `S` (lex order over index subsets, matching
/// `itertools.combinations(range(n), r)`) holds `(e_0..e_{n-r})` of the
/// complement of `S`. The fixed point set every syndrome-hyperplane
/// experiment slices.
#[pyfunction]
fn moment_cloud(py: Python<'_>, p: u64, domain: Vec<u64>, r: usize) -> PyResult<Py<PyArray2<u64>>> {
    let (flat, rows, cols) =
        err(py.allow_threads(|| crate::rs::moments::moment_cloud(p, &domain, r)))?;
    let arr = numpy::ndarray::Array2::from_shape_vec((rows, cols), flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py).into())
}

/// Cut sizes `|Z(b)|` for many syndrome vectors at once (streaming; no cloud
/// materialization). Each `b` has `n - r + 1` coordinates in the moment
/// convention `D_S(w) = sum_j b_j e_j(complement)`.
#[pyfunction]
fn cut_counts(
    py: Python<'_>,
    p: u64,
    domain: Vec<u64>,
    r: usize,
    bs: Vec<Vec<u64>>,
) -> PyResult<Py<PyArray1<u64>>> {
    let v = err(py.allow_threads(|| crate::rs::moments::cut_counts(p, &domain, r, &bs)))?;
    Ok(v.into_pyarray_bound(py).into())
}

/// Exhaustive sparse-cut maximum over all words on a 3- or 4-coordinate
/// moment support (last coefficient normalized to -1; zero-last-coefficient
/// words live on smaller supports). Returns `(max_cut, coeffs_on_support)`.
/// The audited certification kernel (rayon; p^{|support|-2} slope tuples).
#[pyfunction]
fn cut_max_sparse(
    py: Python<'_>,
    p: u64,
    domain: Vec<u64>,
    r: usize,
    support: Vec<usize>,
) -> PyResult<(u64, Vec<u64>)> {
    err(py.allow_threads(|| crate::rs::moments::cut_max_sparse(p, &domain, r, &support)))
}

/// Reduced row echelon form mod p: returns `(rank, rref_rows, pivot_cols)`.
#[pyfunction]
#[allow(clippy::type_complexity)]
fn rref_mod(mut rows: Vec<Vec<u64>>, p: u64) -> PyResult<(usize, Vec<Vec<u64>>, Vec<usize>)> {
    let (rank, pivots) = err(crate::rs::linalg::rref_mod(&mut rows, p))?;
    Ok((rank, rows, pivots))
}

/// A basis of the right nullspace of the row span mod p.
#[pyfunction]
fn nullspace_mod(rows: Vec<Vec<u64>>, p: u64) -> PyResult<Vec<Vec<u64>>> {
    err(crate::rs::linalg::nullspace_mod(&rows, p))
}

/// Residues of `vecs` modulo the row span of `span` (RREF elimination).
#[pyfunction]
fn reduce_mod_span(vecs: Vec<Vec<u64>>, span: Vec<Vec<u64>>, p: u64) -> PyResult<Vec<Vec<u64>>> {
    err(crate::rs::linalg::reduce_mod_span(&vecs, &span, p))
}

/// Batch modular inverses (Montgomery's trick; one Fermat exponentiation).
#[pyfunction]
fn inv_mod(py: Python<'_>, vals: Vec<u64>, p: u64) -> PyResult<Py<PyArray1<u64>>> {
    let v = err(crate::rs::linalg::inv_mod(&vals, p))?;
    let _ = py;
    Ok(v.into_pyarray_bound(py).into())
}

/// Elementary-symmetric vectors `(e_0..e_m)` for many value rows at once.
#[pyfunction]
fn e_syms(py: Python<'_>, p: u64, rows: Vec<Vec<u64>>) -> PyResult<Vec<Py<PyArray1<u64>>>> {
    rows.iter()
        .map(|r| {
            Ok(crate::rs::moments::e_vec(r, p)
                .into_pyarray_bound(py)
                .into())
        })
        .collect()
}

/// Divided-difference functional rows for index subsets of the domain:
/// `row[x] = 1/prod_{y != x}(x - y)` on `T`, so `D_T(w) = row . w`.
#[pyfunction]
fn dd_rows(p: u64, domain: Vec<u64>, subsets: Vec<Vec<usize>>) -> PyResult<Vec<Vec<u64>>> {
    err(crate::rs::linalg::dd_rows(p, &domain, &subsets))
}

/// Rows -> (L, n) uint64 array (empty -> shape (0, 0)).
fn rows_to_array(py: Python<'_>, rows: &[Vec<u64>]) -> PyResult<Py<PyArray2<u64>>> {
    let n = rows.first().map_or(0, Vec::len);
    let flat: Vec<u64> = rows.iter().flatten().copied().collect();
    let arr = numpy::ndarray::Array2::from_shape_vec((rows.len(), n), flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py).into())
}

/// The vanishing-syndrome geometry `VS(s, k)` — stateful handle carrying
/// `(p, s, k)`; the convention authority for syndromes, moments, cuts,
/// strata, and the theorem-backed word constructors. See `src/vs.rs`.
#[pyclass(name = "VsSpace")]
struct PyVsSpace {
    inner: crate::rs::vs::VsSpace,
}

#[pymethods]
impl PyVsSpace {
    #[new]
    fn new(p: u64, s: usize, k: usize) -> PyResult<Self> {
        let sg = MultiplicativeSubgroup::new(p, s).map_err(to_py)?;
        Ok(PyVsSpace {
            inner: crate::rs::vs::VsSpace::new(&sg, k).map_err(to_py)?,
        })
    }

    #[getter]
    fn p(&self) -> u64 {
        self.inner.p()
    }
    #[getter]
    fn s(&self) -> usize {
        self.inner.s()
    }
    #[getter]
    fn k(&self) -> usize {
        self.inner.k()
    }
    #[getter]
    fn r(&self) -> usize {
        self.inner.r()
    }
    #[getter]
    fn syndrome_dim(&self) -> usize {
        self.inner.syndrome_dim()
    }
    /// The domain, ordered as generator powers.
    fn domain(&self) -> Vec<u64> {
        self.inner.subgroup().elements().to_vec()
    }

    fn syndrome(&self, word: Vec<u64>) -> PyResult<Vec<u64>> {
        err(self.inner.syndrome(&word))
    }
    fn word(&self, b: Vec<u64>) -> PyResult<Vec<u64>> {
        err(self.inner.word(&b))
    }
    fn moment_row(&self, subset: Vec<usize>) -> PyResult<Vec<u64>> {
        err(self.inner.moment_row(&subset))
    }
    fn incident(&self, b: Vec<u64>, subset: Vec<usize>) -> PyResult<bool> {
        err(self.inner.incident(&b, &subset))
    }
    fn divided_difference(&self, word: Vec<u64>, subset: Vec<usize>) -> PyResult<u64> {
        err(self.inner.divided_difference(&word, &subset))
    }
    fn subset_rank(&self, subset: Vec<usize>) -> PyResult<u64> {
        err(self.inner.subset_rank(&subset))
    }
    fn subset_unrank(&self, rank: u64) -> PyResult<Vec<usize>> {
        err(self.inner.subset_unrank(rank))
    }
    fn twist_subset(&self, subset: Vec<usize>) -> Vec<usize> {
        self.inner.twist_subset(&subset)
    }
    fn invert_subset(&self, subset: Vec<usize>) -> Vec<usize> {
        self.inner.invert_subset(&subset)
    }
    fn subset_orbit_canon(&self, subset: Vec<usize>) -> Vec<usize> {
        self.inner.subset_orbit_canon(&subset)
    }
    fn core(&self, subset: Vec<usize>) -> PyResult<(Vec<usize>, usize)> {
        err(self.inner.core(&subset))
    }
    fn strata_counts(&self, py: Python<'_>, b: Vec<u64>) -> PyResult<Vec<u64>> {
        err(py.allow_threads(|| self.inner.strata_counts(&b)))
    }
    fn top_word(&self, c: usize) -> PyResult<Vec<u64>> {
        err(self.inner.top_word(c))
    }
    fn coordinate_word(&self, j: usize) -> PyResult<Vec<u64>> {
        err(self.inner.coordinate_word(j))
    }
    fn fold_ladder_word(&self) -> PyResult<Vec<u64>> {
        err(self.inner.fold_ladder_word())
    }
    fn gs_class_counts(&self) -> PyResult<Vec<u64>> {
        err(self.inner.gs_class_counts())
    }
    fn cut_counts(&self, py: Python<'_>, bs: Vec<Vec<u64>>) -> PyResult<Vec<u64>> {
        err(py.allow_threads(|| self.inner.cut_counts(&bs)))
    }
    /// Cut-driven exact list sizes for a batch of words at agreement
    /// `t >= r` — the ownership identity as an algorithm (issue #48);
    /// GIL released.
    fn list_sizes_cut(&self, py: Python<'_>, words: Vec<Vec<u64>>, t: usize) -> PyResult<Vec<u64>> {
        err(py.allow_threads(|| self.inner.list_sizes_cut(&words, t)))
    }
    fn cut_max_sparse(&self, py: Python<'_>, support: Vec<usize>) -> PyResult<(u64, Vec<u64>)> {
        err(py.allow_threads(|| self.inner.cut_max_sparse(&support)))
    }
    /// The convention certificate as a dict (accelerated views gate on it).
    fn certificate<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let cert = err(self.inner.certificate())?;
        let d = pyo3::types::PyDict::new_bound(py);
        d.set_item("version", cert.version)?;
        d.set_item("p", cert.p)?;
        d.set_item("s", cert.s)?;
        d.set_item("k", cert.k)?;
        d.set_item("ranking", cert.ranking)?;
        d.set_item("moment_rows", cert.moment_rows)?;
        d.set_item("domain_head", cert.domain_head)?;
        d.set_item("coordinate_cuts", cert.coordinate_cuts)?;
        if let Some(dp) = cert.descent_pins {
            let dd = pyo3::types::PyDict::new_bound(py);
            dd.set_item("wev_head", dp.wev_head)?;
            dd.set_item("wod_head", dp.wod_head)?;
            dd.set_item("slice_heads", dp.slice_heads.to_vec())?;
            dd.set_item("psi_sample", dp.psi_sample)?;
            d.set_item("descent", dd)?;
        }
        Ok(d)
    }
}

/// The descent handle: the level-halving operation `s -> s/2` at
/// `(p, s, k)` — channel splits, channel syndromes, derived words.
/// Build once per cell; build per-word views with `word()`.
#[pyclass(name = "Descent")]
struct PyDescent {
    inner: std::sync::Arc<crate::rs::descent::Descent>,
}

#[pymethods]
impl PyDescent {
    #[new]
    fn new(p: u64, s: usize, k: usize) -> PyResult<Self> {
        let sg = err(MultiplicativeSubgroup::new(p, s))?;
        Ok(PyDescent {
            inner: std::sync::Arc::new(err(crate::rs::descent::Descent::new(&sg, k))?),
        })
    }
    fn k_even(&self) -> usize {
        self.inner.k_even()
    }
    fn k_odd(&self) -> usize {
        self.inner.k_odd()
    }
    fn half_points(&self) -> Vec<u64> {
        self.inner.half_points().to_vec()
    }
    fn channels(&self, word: Vec<u64>) -> PyResult<(Vec<u64>, Vec<u64>)> {
        err(self.inner.channels(&word))
    }
    fn unfold(&self, wev: Vec<u64>, wod: Vec<u64>) -> PyResult<Vec<u64>> {
        err(self.inner.unfold(&wev, &wod))
    }
    fn monomial_coeffs(&self, word: Vec<u64>) -> PyResult<Vec<u64>> {
        err(self.inner.monomial_coeffs(&word))
    }
    /// The per-word view: the interpolant transform and channel split,
    /// computed once and cached across queries.
    fn word(&self, word: Vec<u64>) -> PyResult<PyWordView> {
        let coeffs = err(self.inner.monomial_coeffs(&word))?;
        let (wev, wod) = err(self.inner.channels(&word))?;
        Ok(PyWordView {
            d: self.inner.clone(),
            word,
            coeffs,
            wev,
            wod,
        })
    }
}

/// Per-word descent data (built by `Descent.word`): cached interpolant
/// coefficients and channel words; per-pair and per-core queries.
#[pyclass(name = "WordView")]
struct PyWordView {
    d: std::sync::Arc<crate::rs::descent::Descent>,
    word: Vec<u64>,
    coeffs: Vec<u64>,
    wev: Vec<u64>,
    wod: Vec<u64>,
}

impl PyWordView {
    fn view(&self) -> crate::rs::descent::WordView<'_> {
        crate::rs::descent::WordView::from_parts(
            &self.d,
            self.word.clone(),
            self.coeffs.clone(),
            self.wev.clone(),
            self.wod.clone(),
        )
    }
}

#[pymethods]
impl PyWordView {
    fn coeffs(&self) -> Vec<u64> {
        self.coeffs.clone()
    }
    fn channel_words(&self) -> (Vec<u64>, Vec<u64>) {
        (self.wev.clone(), self.wod.clone())
    }
    fn channel_syndromes(&self) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        let [b0, b1, b2] = self.view().channel_syndromes();
        (b0, b1, b2)
    }
    fn effective_syndrome(&self, x: u64, xp: u64) -> PyResult<Vec<u64>> {
        err(self.view().effective_syndrome(x, xp))
    }
    fn psi_y(&self, core: Vec<usize>) -> PyResult<Vec<(usize, u64)>> {
        err(self.view().psi_y(&core))
    }
    /// `(total, distinct, max_fiber, collisions)` of `psi_Y`.
    fn psi_y_stats(&self, core: Vec<usize>) -> PyResult<(usize, usize, usize, u64)> {
        let s = err(self.view().psi_y_stats(&core))?;
        Ok((s.total, s.distinct, s.max_fiber, s.collisions))
    }
    /// Both sides of the stratum identity (GIL released; rayon inside).
    fn stratum_identity_check(&self, py: Python<'_>) -> PyResult<(u64, u64)> {
        err(py.allow_threads(|| self.view().stratum_identity_check()))
    }
    fn member_functional(&self, core: Vec<usize>, i1: usize, i2: usize) -> PyResult<u64> {
        err(self.view().member_functional(&core, i1, i2))
    }
}

#[pymodule]
fn vanish(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyVsSpace>()?;
    m.add_class::<PyDescent>()?;
    m.add_class::<PyWordView>()?;
    m.add_function(wrap_pyfunction!(list_decode, m)?)?;
    m.add_function(wrap_pyfunction!(list_decode_paired, m)?)?;
    m.add_function(wrap_pyfunction!(list_decode_paired_sampled, m)?)?;
    m.add_function(wrap_pyfunction!(list_sizes, m)?)?;
    m.add_function(wrap_pyfunction!(anneal_pencil, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_pencil, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_word, m)?)?;
    m.add_function(wrap_pyfunction!(pencil_seed, m)?)?;
    m.add_function(wrap_pyfunction!(decode_profile, m)?)?;
    m.add_function(wrap_pyfunction!(c5_word, m)?)?;
    m.add_function(wrap_pyfunction!(top_word, m)?)?;
    m.add_function(wrap_pyfunction!(word_from_syndrome, m)?)?;
    m.add_class::<PyCyclo>()?;
    m.add_function(wrap_pyfunction!(valuemap_census, m)?)?;
    m.add_function(wrap_pyfunction!(valuemap_histogram, m)?)?;
    m.add_function(wrap_pyfunction!(valuemap_distribution, m)?)?;
    m.add_function(wrap_pyfunction!(valuemap_sweep, m)?)?;
    m.add_function(wrap_pyfunction!(skeleton_totals, m)?)?;
    m.add_function(wrap_pyfunction!(skeleton_census, m)?)?;
    m.add_function(wrap_pyfunction!(valuemap_fiber, m)?)?;
    m.add_function(wrap_pyfunction!(valuemap_fiber_members, m)?)?;
    m.add_function(wrap_pyfunction!(fold_unit, m)?)?;
    m.add_function(wrap_pyfunction!(foldunit_rank_certificate, m)?)?;
    m.add_function(wrap_pyfunction!(foldunit_alpha_certificate, m)?)?;
    m.add_function(wrap_pyfunction!(fold, m)?)?;
    m.add_function(wrap_pyfunction!(norms_mod_batch, m)?)?;
    m.add_function(wrap_pyfunction!(exact_value_census, m)?)?;
    m.add_function(wrap_pyfunction!(certify_clean, m)?)?;
    #[cfg(feature = "certified")]
    m.add_function(wrap_pyfunction!(elias_row, m)?)?;
    m.add("KOALABEAR", crate::field::named::KOALABEAR)?;
    m.add("GOLDILOCKS", crate::field::named::GOLDILOCKS)?;
    m.add("BABYBEAR", crate::field::named::BABYBEAR)?;
    m.add_function(wrap_pyfunction!(gs_class_counts, m)?)?;
    m.add_function(wrap_pyfunction!(moment_cloud, m)?)?;
    m.add_function(wrap_pyfunction!(cut_counts, m)?)?;
    m.add_function(wrap_pyfunction!(cut_max_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(rref_mod, m)?)?;
    m.add_function(wrap_pyfunction!(nullspace_mod, m)?)?;
    m.add_function(wrap_pyfunction!(reduce_mod_span, m)?)?;
    m.add_function(wrap_pyfunction!(inv_mod, m)?)?;
    m.add_function(wrap_pyfunction!(e_syms, m)?)?;
    m.add_function(wrap_pyfunction!(dd_rows, m)?)?;
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
    m.add_function(wrap_pyfunction!(accident_events, m)?)?;
    m.add_function(wrap_pyfunction!(badset_and_events_from_gpu_bin, m)?)?;
    Ok(())
}
