//! Exact list decoding on paired domains at thresholds past the
//! fiber count, by core enumeration: a codeword agreeing on `t > n`
//! of the `s = 2n` points must fully agree on at least `l = t - n`
//! fibers, so it contains an `l`-subset of fibers ("core") on which
//! it matches the word at both points. Enumerating cores and
//! Guruswami–Sudan-decoding the residual is therefore complete:
//! for a core `Y`, a member `f` factors as `f = q_Y + V_Y g` with
//! `q_Y` the interpolant of the word on the core points,
//! `V_Y = prod_{y in Y} (x^2 - y)`, and `deg g < k - 2l`; on the
//! free points `g` must agree with `(w - q_Y)/V_Y` at least
//! `t - 2l` times, a decoding problem [`crate::rs::gs`] solves
//! whenever `t - 2l` exceeds its Johnson agreement.
//!
//! This reaches cells the information-set engine cannot: its cost is
//! `C(n, l)` residual decodes instead of `C(s, k)` interpolations.
//! At `(64, 31, 43)` that is 129M decodes against 9e17 subsets.

use std::collections::HashSet;

use rayon::prelude::*;

use crate::error::{Error, Result};
use crate::field::{batch_inv, checked_binom, mulmod};
use crate::poly;
use crate::rs::combi::unrank_combination;
use crate::rs::gs::{gs_list, gs_params};

/// A domain of `s = 2n` points closed under negation, stored as
/// `points[i]` and `points[i + n] = -points[i]`; fiber `i` is the
/// pair `{points[i], points[i + n]}` over `points[i]^2`.
pub struct PairedDomain {
    p: u64,
    points: Vec<u64>,
    n: usize,
}

impl PairedDomain {
    /// The paired form of the order-`s` subgroup (even `s`): fiber
    /// `i` is `{g^i, g^{i + s/2}} = {g^i, -g^i}`.
    pub fn from_points(p: u64, points: Vec<u64>) -> Result<Self> {
        if points.len() % 2 != 0 || points.is_empty() {
            return Err(Error::OutOfRange("paired domain needs even size".into()));
        }
        let n = points.len() / 2;
        for i in 0..n {
            if (points[i] + points[i + n]) % p != 0 {
                return Err(Error::OutOfRange(format!(
                    "points {i} and {} are not an antipodal pair",
                    i + n
                )));
            }
        }
        let mut sorted = points.clone();
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            return Err(Error::OutOfRange("repeated points".into()));
        }
        Ok(PairedDomain { p, points, n })
    }

    /// The number of fibers `n = s/2`.
    #[must_use]
    pub fn fibers(&self) -> usize {
        self.n
    }

    /// The points, fiber-major: `[x_0..x_{n-1}, -x_0..-x_{n-1}]`.
    #[must_use]
    pub fn points(&self) -> &[u64] {
        &self.points
    }
}

/// The members through one core: decode the residual and reassemble.
/// Returns evaluation vectors over the full domain, verified to
/// agree with `word` on at least `t` points.
fn members_through_core(
    dom: &PairedDomain,
    k: usize,
    word: &[u64],
    t: usize,
    core: &[usize],
) -> Vec<Vec<u64>> {
    let p = dom.p;
    let n = dom.n;
    let l = core.len();
    let in_core = {
        let mut m = vec![false; n];
        for &i in core {
            m[i] = true;
        }
        m
    };
    // interpolant of the word on the 2l core points
    let mut cxs = Vec::with_capacity(2 * l);
    let mut cys = Vec::with_capacity(2 * l);
    for &i in core {
        for j in [i, i + n] {
            cxs.push(dom.points[j]);
            cys.push(word[j]);
        }
    }
    let q = poly::interpolate(&cxs, &cys, p);
    // V_Y(x) = prod_{y in Y} (x^2 - y)
    let v_at = |x: u64| {
        let x2 = mulmod(x, x, p);
        core.iter().fold(1, |acc, &i| {
            let y = mulmod(dom.points[i], dom.points[i], p);
            mulmod(acc, (x2 + p - y) % p, p)
        })
    };
    // free points, the residual targets (w - q_Y)/V_Y there
    let (fxs, fidx): (Vec<u64>, Vec<usize>) = in_core
        .iter()
        .enumerate()
        .filter(|&(_, &inc)| !inc)
        .flat_map(|(i, _)| [i, i + n])
        .map(|j| (dom.points[j], j))
        .unzip();
    let qf = poly::evaluate(&q, &fxs, p);
    let mut v_inv: Vec<u64> = fxs.iter().map(|&x| v_at(x)).collect();
    batch_inv(&mut v_inv, p);
    let targets: Vec<u64> = fidx
        .iter()
        .zip(&qf)
        .zip(&v_inv)
        .map(|((&j, &qx), &vi)| mulmod((word[j] + p - qx) % p, vi, p))
        .collect();
    let Ok(gs) = gs_list(p, &fxs, &targets, (k - 2 * l) as u64, (t - 2 * l) as u64) else {
        return Vec::new();
    };
    // reassemble f = q_Y + V_Y g and keep it if it truly agrees
    gs.into_iter()
        .filter_map(|g| {
            let f: Vec<u64> = dom
                .points
                .iter()
                .map(|&x| {
                    let member = mulmod(v_at(x), poly::horner(&g, x, p), p);
                    (poly::horner(&q, x, p) + member) % p
                })
                .collect();
            let agree = f.iter().zip(word).filter(|(a, b)| a == b).count();
            (agree >= t).then_some(f)
        })
        .collect()
}

/// The exact list of `RS[F_p, dom, k]` at agreement `t > n`: every
/// codeword agreeing with `word` on at least `t` points. Refuses when
/// the residual cell is not Guruswami–Sudan-decodable, i.e. unless
/// `(t - 2l)^2 > (s - 2l)(k - 2l - 1)` with `l = t - n` — the caller
/// sees the exact reason.
pub fn list_paired(dom: &PairedDomain, k: usize, word: &[u64], t: usize) -> Result<Vec<Vec<u64>>> {
    let total = core_count(dom, k, t)?;
    list_paired_range(dom, k, word, t, 0..total)
}

/// The number of cores `C(n, t - n)` the exact decode enumerates —
/// the index space of [`list_paired_range`]. Validates the cell the
/// same way [`list_paired`] does.
pub fn core_count(dom: &PairedDomain, k: usize, t: usize) -> Result<u64> {
    let n = dom.n;
    if t <= n || t > 2 * n {
        return Err(Error::Unsupported(format!(
            "core enumeration needs n < t <= 2n (got t = {t}, n = {n})"
        )));
    }
    let l = t - n;
    if k <= 2 * l + 1 {
        return Err(Error::OutOfRange(format!(
            "residual dimension k - 2l = {} below 2 at (k, l) = ({k}, {l})",
            k as i64 - 2 * l as i64
        )));
    }
    // certify the residual cell up front, so a sweep cannot start
    // on an undecodable cell
    gs_params(
        (2 * n - 2 * l) as u64,
        (k - 2 * l) as u64,
        (t - 2 * l) as u64,
    )?;
    checked_binom(n as u64, l as u64)
        .ok_or_else(|| Error::OutOfRange("core count overflows u64".into()))
}

/// One shard of the exact decode: the members found through the
/// cores with indices in `cores` (a sub-range of
/// `0..core_count(..)`, colexicographic order). The union of the
/// members over a partition of the full range, deduplicated, is the
/// exact list — a long sweep can run as resumable chunks.
pub fn list_paired_range(
    dom: &PairedDomain,
    k: usize,
    word: &[u64],
    t: usize,
    cores: std::ops::Range<u64>,
) -> Result<Vec<Vec<u64>>> {
    if word.len() != 2 * dom.n {
        return Err(Error::OutOfRange("word length != domain size".into()));
    }
    let total = core_count(dom, k, t)?;
    if cores.end > total {
        return Err(Error::OutOfRange(format!(
            "core range ends at {} of {total}",
            cores.end
        )));
    }
    let l = (t - dom.n) as u64;
    let found: Vec<Vec<u64>> = cores
        .into_par_iter()
        .flat_map_iter(|idx| {
            let core = unrank_combination(idx, dom.n as u64, l);
            members_through_core(dom, k, word, t, &core)
        })
        .collect();
    Ok(dedup(found))
}

/// Keep the first occurrence of each member, preserving order.
fn dedup(found: Vec<Vec<u64>>) -> Vec<Vec<u64>> {
    let mut seen = HashSet::new();
    found
        .into_iter()
        .filter(|f| seen.insert(f.clone()))
        .collect()
}

/// A sampled lower bound: the distinct members found through
/// `samples` uniformly drawn cores. A subset of the true list;
/// deterministic in `seed`. The optimizer's objective.
pub fn list_paired_sampled(
    dom: &PairedDomain,
    k: usize,
    word: &[u64],
    t: usize,
    samples: u64,
    seed: u64,
) -> Result<Vec<Vec<u64>>> {
    if word.len() != 2 * dom.n {
        return Err(Error::OutOfRange("word length != domain size".into()));
    }
    let total = core_count(dom, k, t)?;
    let l = t - dom.n;
    let mut rng = crate::rs::combi::SplitMix64::new(seed);
    let idxs: Vec<u64> = (0..samples).map(|_| rng.next_u64() % total).collect();
    let found: Vec<Vec<u64>> = idxs
        .into_par_iter()
        .flat_map_iter(|idx| {
            let core = unrank_combination(idx, dom.n as u64, l as u64);
            members_through_core(dom, k, word, t, &core)
        })
        .collect();
    Ok(dedup(found))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::MultiplicativeSubgroup;
    use crate::rs::code::ReedSolomon;
    use crate::rs::decode::{DecodeOracle, Radius};

    fn paired_subgroup(p: u64, s: usize) -> PairedDomain {
        let sg = MultiplicativeSubgroup::new(p, s).expect("subgroup");
        let pts = sg.elements().to_vec();
        let n = pts.len() / 2;
        // subgroup as powers of a generator g: g^{i+n} = -g^i
        PairedDomain::from_points(p, pts.clone()).unwrap_or_else(|_| {
            // reorder into fiber-major antipodal layout
            let mut points: Vec<u64> = pts[..n].to_vec();
            points.extend(pts[..n].iter().map(|&x| (p - x) % p));
            PairedDomain::from_points(p, points).expect("paired layout")
        })
    }

    fn rng_word(p: u64, len: usize, seed: u64) -> Vec<u64> {
        crate::rs::combi::SplitMix64::new(seed).word(p, len)
    }

    /// The core-residual list equals the information-set engine's at
    /// every decodable cell small enough to cross-check, on random
    /// words, a planted-pair word, and a codeword, at two primes.
    #[test]
    fn agrees_with_exact_engine() {
        for p in [65537u64, 97] {
            for (s, k, t) in [(16usize, 7usize, 10usize), (16, 5, 9)] {
                let dom = paired_subgroup(p, s);
                let rs = ReedSolomon::on_domain(p, dom.points().to_vec(), k).expect("code");
                let oracle = DecodeOracle::new(&rs);
                let mut words: Vec<Vec<u64>> = (0..4).map(|i| rng_word(p, s, 11 + i)).collect();
                // a codeword, and a word one step from it
                let msg = rng_word(p, k, 5);
                let cw = rs.encode(&msg).expect("encode");
                let mut near = cw.clone();
                near[3] = (near[3] + 1) % p;
                words.push(cw);
                words.push(near);
                for w in words {
                    let mut truth = oracle.list(&w, Radius::agreement(t)).expect("exact");
                    truth.sort();
                    let mut got = list_paired(&dom, k, &w, t).expect("core-residual");
                    got.sort();
                    assert_eq!(got, truth, "p = {p}, (s, k, t) = ({s}, {k}, {t})");
                }
            }
        }
    }

    /// The sampled mode returns a subset of the exact list.
    #[test]
    fn sampled_is_a_subset() {
        let p = 65537u64;
        let (s, k, t) = (16usize, 7usize, 10usize);
        let dom = paired_subgroup(p, s);
        let w = rng_word(p, s, 3);
        let full: HashSet<Vec<u64>> = list_paired(&dom, k, &w, t)
            .expect("full")
            .into_iter()
            .collect();
        let sampled = list_paired_sampled(&dom, k, &w, t, 40, 9).expect("sampled");
        for f in sampled {
            assert!(full.contains(&f));
        }
    }

    /// Sharding is a partition: the union of the members over any
    /// split of the core range, deduplicated, equals the full list.
    #[test]
    fn shards_partition_the_sweep() {
        let p = 65537;
        let (s, k, t) = (16, 7, 10);
        let dom = paired_subgroup(p, s);
        // a corrupted codeword, so the list is provably nonempty and
        // the partition check has something to partition
        let rs =
            crate::rs::code::ReedSolomon::on_domain(p, dom.points().to_vec(), k).expect("code");
        let mut w = rs.encode(&rng_word(p, k, 8)).expect("encode");
        for (i, wi) in w.iter_mut().enumerate().take(s - t) {
            *wi = (*wi + 1 + i as u64) % p;
        }
        let mut full = list_paired(&dom, k, &w, t).expect("full");
        assert!(!full.is_empty(), "the plant guarantees a member");
        full.sort();
        let total = core_count(&dom, k, t).expect("cell");
        let mid = total / 3;
        let mut merged: Vec<Vec<u64>> = [0..mid, mid..total]
            .into_iter()
            .flat_map(|r| list_paired_range(&dom, k, &w, t, r).expect("shard"))
            .collect();
        merged.sort();
        merged.dedup();
        assert_eq!(merged, full);
    }

    /// The first cell the information-set engine cannot comfortably
    /// reach: (32, 15) at agreement 21, beyond the Johnson agreement
    /// sqrt(32 * 14) = 21.17, via 4368 residual decodes. Run
    /// explicitly; prints the lists of the standard words.
    #[test]
    #[ignore = "measurement, not a pin yet: run with -- --ignored --nocapture"]
    fn measure_32_15_21() {
        let p = 65537u64;
        let (s, k, t) = (32usize, 15usize, 21usize);
        let dom = paired_subgroup(p, s);
        let pts = dom.points().to_vec();
        // top word x^15 + x^31 and its flip (negated at the real pair)
        let top: Vec<u64> = pts
            .iter()
            .map(|&x| {
                let a = crate::field::powmod(x, 15, p);
                let b = crate::field::powmod(x, 31, p);
                (a + b) % p
            })
            .collect();
        let mut flip = top.clone();
        for (i, &x) in pts.iter().enumerate() {
            if x == 1 || x == p - 1 {
                flip[i] = (p - flip[i]) % p;
            }
        }
        for (name, w) in [("top", &top), ("flip", &flip)] {
            let list = list_paired(&dom, k, w, t).expect("list");
            println!("(32,15,21) {name}: |list| = {}", list.len());
        }
        for seed in 0..3u64 {
            let w = rng_word(p, s, 100 + seed);
            let list = list_paired(&dom, k, &w, t).expect("list");
            println!("(32,15,21) random {seed}: |list| = {}", list.len());
        }
        // planted control: a corrupted codeword at exactly agreement t
        // must be found (rule: no zero believed without a plant)
        let rs = crate::rs::code::ReedSolomon::on_domain(p, pts.clone(), k).expect("code");
        let cw = rs.encode(&rng_word(p, k, 77)).expect("encode");
        let mut w = cw.clone();
        for (i, wi) in w.iter_mut().enumerate().take(s - t) {
            *wi = (*wi + 1 + i as u64) % p;
        }
        let list = list_paired(&dom, k, &w, t).expect("list");
        assert!(
            list.contains(&cw),
            "planted codeword at agreement {t} not found"
        );
        println!(
            "(32,15,21) planted: |list| = {} (contains the plant)",
            list.len()
        );
    }

    /// Timing probe for the production cell: sampled cores of the
    /// top word at (64, 31, 43). Projects the full 129M-core sweep.
    #[test]
    #[ignore = "timing probe: run with --release -- --ignored --nocapture"]
    fn time_64_31_43_sampled() {
        let p = 65537u64;
        let (s, k, t) = (64usize, 31usize, 43usize);
        let dom = paired_subgroup(p, s);
        let pts = dom.points().to_vec();
        let top: Vec<u64> = pts
            .iter()
            .map(|&x| {
                let a = crate::field::powmod(x, k as u64, p);
                let b = crate::field::powmod(x, (s - 1) as u64, p);
                (a + b) % p
            })
            .collect();
        let samples = 20_000u64;
        let start = std::time::Instant::now();
        let found = list_paired_sampled(&dom, k, &top, t, samples, 5).expect("sampled");
        let el = start.elapsed().as_secs_f64();
        let per = el / samples as f64;
        let total = 129_024_480f64;
        println!(
            "(64,31,43) sampled {samples} cores in {el:.2}s ({:.1} us/core); \
             full sweep projection: {:.1} core-hours; found {}",
            per * 1e6,
            per * total / 3600.0,
            found.len()
        );
    }

    #[test]
    fn refuses_undecodable_cells() {
        let p = 65537u64;
        let dom = paired_subgroup(p, 16usize);
        let w = rng_word(p, 16, 1);
        // (16, 7, 9): residual (14, 5, 7) has 49 <= 14 * 4
        assert!(list_paired(&dom, 7, &w, 9).is_err());
        // t <= n
        assert!(list_paired(&dom, 7, &w, 8).is_err());
        // t > s: an error, not an arithmetic panic, whatever the k
        assert!(list_paired(&dom, 20, &w, 17).is_err());
        // sampled mode refuses a word of the wrong length too
        assert!(list_paired_sampled(&dom, 7, &w[..15], 10, 4, 1).is_err());
    }
}
