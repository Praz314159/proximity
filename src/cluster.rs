//! Bottom-up cluster growth: discover large-list words by growing clusters of
//! codewords around a moving center, instead of positing their algebraic form.
//!
//! A word with a large list is the shared near-center of many codewords. In an
//! MDS code that is rigid — distinct codewords agree on at most `k - 1` points —
//! so a *large* cluster forces its members' agreement sets to interlock
//! tightly. Whether that rigidity forces the frozen-symmetric-function
//! structure of a bucket, or admits something else, is exactly the reduction
//! question the data program turns on. This engine lets the answer emerge:
//! grow a cluster, then read its structure with [`crate::classify`].
//!
//! It is a **constructive probe of bucket-extremality**. If every large cluster
//! it grows is algebraic (a bucket), that is evidence for the reduction; a
//! large *non*-bucket cluster is a candidate new attack. And because it is
//! output-sensitive — cost scales with the cluster, not `C(n, k)` — it reaches
//! the `s = 64` accident regime where exact enumeration is infeasible.
//!
//! The engine is `grow-and-center`: alternate **recruitment** (list-decode the
//! current center) and **majority re-centering** to a fixed point. It is a
//! heuristic local search — locally-extremal words, not a certified global
//! maximum — so vary the seed for coverage.

use crate::classify::{classify, WordKind};
use crate::code::ReedSolomon;
use crate::decode::{interp_eval_all, DecodeOracle, Radius};
use crate::error::{Error, Result};
use std::collections::{HashMap, HashSet};

/// A cluster: a center word and the codewords within a fixed radius of it.
#[derive(Debug, Clone)]
pub struct Cluster {
    center: Vec<u64>,
    members: Vec<Vec<u64>>,
    radius: Radius,
}

impl Cluster {
    /// The center word.
    #[must_use]
    pub fn center(&self) -> &[u64] {
        &self.center
    }

    /// The member codewords (as evaluation vectors on the domain).
    #[must_use]
    pub fn members(&self) -> &[Vec<u64>] {
        &self.members
    }

    /// The cluster size = the list size at the center.
    #[must_use]
    pub fn size(&self) -> usize {
        self.members.len()
    }

    /// The radius the cluster lives at.
    #[must_use]
    pub fn radius(&self) -> Radius {
        self.radius
    }

    /// Classify the center by the algebraic structure of the cluster.
    pub fn classify(&self, rs: &ReedSolomon) -> Result<WordKind> {
        classify(rs, &self.center, self.radius)
    }
}

/// Grow a cluster from `seed` by alternating recruitment and majority-vote
/// re-centering, to a fixed point or `max_rounds`. Returns the largest cluster
/// encountered.
///
/// Recruitment is exact when `C(n, k)` is within the cap, otherwise Monte-Carlo
/// over `samples` information sets (`samples` is ignored in the exact case).
/// Heuristic local search: finds locally-extremal words, not a certified global
/// maximum; vary `seed` and `rng_seed` for coverage.
pub fn grow(
    rs: &ReedSolomon,
    seed: &[u64],
    radius: Radius,
    samples: u64,
    max_rounds: usize,
    rng_seed: u64,
) -> Result<Cluster> {
    if seed.len() != rs.n() {
        return Err(Error::OutOfRange("seed length != n".into()));
    }
    let oracle = DecodeOracle::new(rs);
    let n = rs.n();
    let mut center = seed.to_vec();
    let mut best: Option<Cluster> = None;
    let mut rng_seed = rng_seed;
    for _ in 0..max_rounds {
        let members = if oracle.exact_feasible() {
            oracle.list(&center, radius)?
        } else {
            oracle.sample_list(&center, radius, samples, rng_seed)?
        };
        rng_seed = rng_seed.wrapping_add(1);
        if best.as_ref().map_or(true, |b| members.len() > b.members.len()) {
            best = Some(Cluster {
                center: center.clone(),
                members: members.clone(),
                radius,
            });
        }
        if members.is_empty() {
            break;
        }
        let next = majority_center(&members, n, &center);
        if next == center {
            break;
        }
        center = next;
    }
    best.ok_or_else(|| Error::Unsupported("no rounds executed".into()))
}

/// Build the sunflower center word from a pencil, without growing: the core on
/// `core_coords`, and every other coordinate assigned round-robin to a petal
/// codeword. The raw code-first seed shared by [`grow_from_pencil`], [`search`],
/// and direct callers.
pub fn pencil_seed(
    rs: &ReedSolomon,
    core_coords: &[usize],
    core_values: &[u64],
    petal_point: usize,
    petal_values: &[u64],
) -> Result<Vec<u64>> {
    let (n, k, p) = (rs.n(), rs.k(), rs.domain().p());
    let dom = rs.domain().elements();
    if core_coords.len() != k - 1 || core_values.len() != k - 1 {
        return Err(Error::OutOfRange("core must have exactly k-1 points".into()));
    }
    if petal_values.is_empty() || petal_point >= n || core_coords.iter().any(|&c| c >= n) {
        return Err(Error::OutOfRange("need >=1 petal and in-range coords".into()));
    }
    let mut xs: Vec<u64> = core_coords.iter().map(|&i| dom[i]).collect();
    xs.push(dom[petal_point]);
    let codewords: Vec<Vec<u64>> = petal_values
        .iter()
        .map(|&y| {
            let mut ys = core_values.to_vec();
            ys.push(y);
            interp_eval_all(&xs, &ys, dom, p)
        })
        .collect();
    let core: HashSet<usize> = core_coords.iter().copied().collect();
    let mut seed = vec![0u64; n];
    for (i, &c) in core_coords.iter().enumerate() {
        seed[c] = core_values[i];
    }
    let mut petal = 0usize;
    for z in 0..n {
        if core.contains(&z) {
            continue;
        }
        seed[z] = codewords[petal % codewords.len()][z];
        petal += 1;
    }
    Ok(seed)
}

/// A random pencil seed word: random `(k-1)`-core coordinates, core values from
/// a random codeword, and `petals` distinct petal values. The unbiased,
/// code-first start for [`search`]. Deterministic in `seed`.
pub fn random_pencil_seed(rs: &ReedSolomon, petals: usize, seed: u64) -> Result<Vec<u64>> {
    let (n, k, p) = (rs.n(), rs.k(), rs.domain().p());
    let mut s = seed;
    let mut coords: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + (splitmix(&mut s) as usize) % (n - i);
        coords.swap(i, j);
    }
    let core_coords = coords[..k - 1].to_vec();
    let petal_point = coords[k - 1];
    let core_values: Vec<u64> = (0..k - 1).map(|_| splitmix(&mut s) % p).collect();
    let mut pv: HashSet<u64> = HashSet::new();
    let cap = petals.min(p as usize - 1);
    while pv.len() < cap {
        pv.insert(splitmix(&mut s) % p);
    }
    let petal_values: Vec<u64> = pv.into_iter().collect();
    pencil_seed(rs, &core_coords, &core_values, petal_point, &petal_values)
}

/// Build a cluster **from the code**, presupposing no bucket: assemble the
/// [`pencil_seed`] sunflower center from a `(k-1)`-core pencil and grow it.
/// For all `m` petals to reach radius `t`, pick `m <= (n-k+1)/(t-k+1)`.
#[allow(clippy::too_many_arguments)]
pub fn grow_from_pencil(
    rs: &ReedSolomon,
    core_coords: &[usize],
    core_values: &[u64],
    petal_point: usize,
    petal_values: &[u64],
    radius: Radius,
    samples: u64,
    max_rounds: usize,
    rng_seed: u64,
) -> Result<Cluster> {
    let seed = pencil_seed(rs, core_coords, core_values, petal_point, petal_values)?;
    grow(rs, &seed, radius, samples, max_rounds, rng_seed)
}

/// Grow from a *random* pencil (random core coordinates, core values from a
/// random codeword, and `petals` distinct petal values) — the unbiased,
/// code-first seed for exploring whether large clusters are forced to be
/// buckets. Deterministic in `seed`.
pub fn grow_random_pencil(
    rs: &ReedSolomon,
    petals: usize,
    radius: Radius,
    seed: u64,
) -> Result<Cluster> {
    let w = random_pencil_seed(rs, petals, seed)?;
    grow(rs, &w, radius, 0, 12, seed.wrapping_add(1))
}

/// Instrumentation for one optimizer run — the raw material for studying the
/// optimizer itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptTrace {
    /// List-size trajectory: `sizes[0]` the seed, then one entry per accepted
    /// move. Monotone for [`optimize`]; may dip for [`anneal`].
    pub sizes: Vec<usize>,
    /// Best list size seen over the run.
    pub best_size: usize,
    /// Accepted moves.
    pub accepts: usize,
    /// Pool re-decodes performed.
    pub refreshes: usize,
}

/// Maximize **list size** directly (no structural hypothesis) by a greedy
/// hill-climb with boundary-alignment moves: at each step flip the single
/// coordinate that most increases `|List|`, drawing candidate flips from
/// near-member codewords (agreement `>= t - slack`) so the move actively pulls
/// boundary codewords into the list. Bootstraps thin seeds where majority-vote
/// re-centering stalls.
///
/// Because a one-coordinate flip changes any codeword's agreement by at most
/// one, `slack >= 1` guarantees the near-member pool already contains every
/// codeword a flip could promote — so each step's gain is evaluated exactly and
/// incrementally over the pool. Returns the optimized cluster and a performance
/// [`OptTrace`]. A local search: a local maximum of list size, not a certified
/// global one; vary the seed.
pub fn optimize(
    rs: &ReedSolomon,
    seed: &[u64],
    radius: Radius,
    slack: usize,
    max_flips: usize,
) -> Result<(Cluster, OptTrace)> {
    if seed.len() != rs.n() {
        return Err(Error::OutOfRange("seed length != n".into()));
    }
    let (k, t) = (rs.k(), radius.min_agreement());
    let oracle = DecodeOracle::new(rs);
    let relaxed = Radius::agreement(t.saturating_sub(slack.max(1)).max(k));
    let mut w = seed.to_vec();
    let mut trace = OptTrace {
        sizes: Vec::new(),
        best_size: 0,
        accepts: 0,
        refreshes: 0,
    };
    let agree = |c: &[u64], w: &[u64]| c.iter().zip(w).filter(|(a, b)| a == b).count();

    loop {
        let pool = oracle.list(&w, relaxed)?;
        trace.refreshes += 1;
        let ag: Vec<usize> = pool.iter().map(|c| agree(c, &w)).collect();
        let cur = ag.iter().filter(|&&a| a >= t).count();
        trace.sizes.push(cur);

        // Candidate flips: (coord, target value) from any pool codeword that
        // disagrees with w there.
        let mut cands: HashSet<(usize, u64)> = HashSet::new();
        for c in &pool {
            for (x, (&cx, &wx)) in c.iter().zip(&w).enumerate() {
                if cx != wx {
                    cands.insert((x, cx));
                }
            }
        }
        let mut best: Option<(usize, u64, usize)> = None;
        for (x, v) in cands {
            let mut nl = 0usize;
            for (i, c) in pool.iter().enumerate() {
                let d = (c[x] == v) as i64 - (c[x] == w[x]) as i64;
                if ag[i] as i64 + d >= t as i64 {
                    nl += 1;
                }
            }
            if best.map_or(true, |(_, _, bl)| nl > bl) {
                best = Some((x, v, nl));
            }
        }
        match best {
            Some((x, v, nl)) if nl > cur => {
                w[x] = v;
                trace.accepts += 1;
                if trace.accepts >= max_flips {
                    break;
                }
            }
            _ => break, // no strictly improving flip: local maximum
        }
    }

    let members = oracle.list(&w, radius)?;
    trace.best_size = trace.sizes.iter().copied().max().unwrap_or(0).max(members.len());
    Ok((
        Cluster {
            center: w,
            members,
            radius,
        },
        trace,
    ))
}

/// Simulated-annealing maximization of `|List|`: propose boundary-alignment
/// flips (a coordinate → some near-member codeword's value there) and accept a
/// worsening flip with Metropolis probability `exp(dL / T)`, annealing `T` from
/// `t0` by `cooling` each step. Escapes the local maxima that trap the greedy
/// [`optimize`]. The pool is decoded at radius `t-1`, so each step's `dL` is
/// exact for a single flip. Returns the best cluster seen and the trajectory.
/// Deterministic in `rng_seed`.
#[allow(clippy::too_many_arguments)]
pub fn anneal(
    rs: &ReedSolomon,
    seed: &[u64],
    radius: Radius,
    steps: usize,
    t0: f64,
    cooling: f64,
    rng_seed: u64,
) -> Result<(Cluster, OptTrace)> {
    if seed.len() != rs.n() {
        return Err(Error::OutOfRange("seed length != n".into()));
    }
    let (k, t) = (rs.k(), radius.min_agreement());
    let oracle = DecodeOracle::new(rs);
    let relaxed = Radius::agreement(t.saturating_sub(1).max(k));
    let agree = |c: &[u64], w: &[u64]| c.iter().zip(w).filter(|(a, b)| a == b).count();
    let mut w = seed.to_vec();
    let mut rng = rng_seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut trace = OptTrace {
        sizes: Vec::new(),
        best_size: 0,
        accepts: 0,
        refreshes: 0,
    };
    let mut pool = oracle.list(&w, relaxed)?;
    trace.refreshes += 1;
    let mut cur = pool.iter().filter(|c| agree(c, &w) >= t).count();
    trace.sizes.push(cur);
    let (mut best_w, mut best_l) = (w.clone(), cur);
    let mut temp = t0.max(1e-9);

    for _ in 0..steps {
        let cands: Vec<(usize, u64)> = {
            let mut set: HashSet<(usize, u64)> = HashSet::new();
            for c in &pool {
                for (x, (&cx, &wx)) in c.iter().zip(&w).enumerate() {
                    if cx != wx {
                        set.insert((x, cx));
                    }
                }
            }
            set.into_iter().collect()
        };
        if cands.is_empty() {
            break;
        }
        let (x, v) = cands[(splitmix(&mut rng) as usize) % cands.len()];
        let ag: Vec<usize> = pool.iter().map(|c| agree(c, &w)).collect();
        let nl = pool
            .iter()
            .enumerate()
            .filter(|(i, c)| {
                ag[*i] as i64 + (c[x] == v) as i64 - (c[x] == w[x]) as i64 >= t as i64
            })
            .count();
        let dl = nl as i64 - cur as i64;
        let accept = dl >= 0 || {
            let u = (splitmix(&mut rng) >> 11) as f64 / (1u64 << 53) as f64;
            u < (dl as f64 / temp).exp()
        };
        if accept {
            w[x] = v;
            cur = nl;
            trace.accepts += 1;
            trace.sizes.push(cur);
            pool = oracle.list(&w, relaxed)?;
            trace.refreshes += 1;
            if cur > best_l {
                best_l = cur;
                best_w.clone_from(&w);
            }
        }
        temp *= cooling;
    }
    trace.best_size = best_l;
    let members = oracle.list(&best_w, radius)?;
    Ok((
        Cluster {
            center: best_w,
            members,
            radius,
        },
        trace,
    ))
}

/// Multi-restart search: run `restarts` independent [`anneal`] runs from random
/// pencil seeds and return the best cluster together with every run's trace —
/// the raw performance data for studying the optimizer. Maximizes list size
/// from diverse, unbiased code-first starts.
pub fn search(
    rs: &ReedSolomon,
    radius: Radius,
    restarts: usize,
    petals: usize,
    steps: usize,
    rng_seed: u64,
) -> Result<(Cluster, Vec<OptTrace>)> {
    let mut best: Option<Cluster> = None;
    let mut traces = Vec::with_capacity(restarts);
    for i in 0..restarts {
        let seed = random_pencil_seed(rs, petals, rng_seed.wrapping_add(i as u64 * 0x100))?;
        let (cl, tr) = anneal(rs, &seed, radius, steps, 2.0, 0.92, rng_seed.wrapping_add(i as u64))?;
        if best.as_ref().map_or(true, |b| cl.size() > b.size()) {
            best = Some(cl);
        }
        traces.push(tr);
    }
    best.map(|b| (b, traces))
        .ok_or_else(|| Error::OutOfRange("restarts must be >= 1".into()))
}

/// SplitMix64 — the shared deterministic PRNG for seed sampling and annealing.
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Majority-vote center: at each coordinate, the value the most members take
/// (ties broken toward the previous center, else the smallest value). This is
/// the coordinate-wise maximizer of total agreement with the members.
fn majority_center(members: &[Vec<u64>], n: usize, prev: &[u64]) -> Vec<u64> {
    (0..n)
        .map(|x| {
            let mut counts: HashMap<u64, usize> = HashMap::new();
            for c in members {
                *counts.entry(c[x]).or_insert(0) += 1;
            }
            let best = counts.values().copied().max().unwrap_or(0);
            if counts.get(&prev[x]).copied().unwrap_or(0) == best {
                prev[x]
            } else {
                counts
                    .iter()
                    .filter(|(_, &c)| c == best)
                    .map(|(&v, _)| v)
                    .min()
                    .unwrap_or(prev[x])
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Subgroup;

    #[test]
    fn bucket_seed_grows_full_bucket() {
        let sg = Subgroup::new(65537, 16).unwrap();
        let rs = ReedSolomon::new(&sg, 7).unwrap();
        let f = rs.c5_word(8, &[0]).unwrap();
        let cl = grow(&rs, &f, Radius::agreement(8), 0, 8, 1).unwrap();
        assert_eq!(cl.size(), 70, "recovers the full 70-codeword bucket");
        assert!(matches!(
            cl.classify(&rs).unwrap(),
            WordKind::Bucket { .. }
        ));
    }

    #[test]
    fn grows_from_arbitrary_seed() {
        let sg = Subgroup::new(65537, 16).unwrap();
        let rs = ReedSolomon::new(&sg, 7).unwrap();
        let seed: Vec<u64> = sg.elements().iter().map(|&x| (x * 7 + 3) % 65537).collect();
        let cl = grow(&rs, &seed, Radius::agreement(8), 0, 8, 1).unwrap();
        assert!(cl.size() <= 70);
    }

    #[test]
    fn pencil_builds_a_cluster_from_the_code_at_large_p() {
        // At p = 65537 a cold random seed lists 0; a code-first pencil of 5
        // petals constructs a >= 5 cluster with no bucket presupposed.
        let sg = Subgroup::new(65537, 16).unwrap();
        let rs = ReedSolomon::new(&sg, 7).unwrap();
        let rad = Radius::agreement(8);
        let cold: Vec<u64> = (0..16).map(|i| (i as u64 * 31 + 7) % 65537).collect();
        let cold_size = grow(&rs, &cold, rad, 0, 8, 1).unwrap().size();
        let cl = grow_random_pencil(&rs, 5, rad, 42).unwrap();
        assert!(cold_size < 5, "cold seed has only an incidental list ({cold_size})");
        assert!(cl.size() >= 5, "pencil constructs >= 5 members, got {}", cl.size());
    }

    #[test]
    fn optimize_climbs_and_holds_bucket() {
        let sg = Subgroup::new(65537, 16).unwrap();
        let rs = ReedSolomon::new(&sg, 7).unwrap();
        let f = rs.c5_word(8, &[0]).unwrap();
        let (opt, tr) = optimize(&rs, &f, Radius::agreement(8), 1, 6).unwrap();
        assert!(opt.size() >= 70, "should not lose the bucket, got {}", opt.size());
        assert!(tr.sizes.windows(2).all(|w| w[1] >= w[0]), "list size must not decrease");
    }

    #[test]
    fn anneal_reports_best_and_never_below_seed() {
        let sg = Subgroup::new(65537, 16).unwrap();
        let rs = ReedSolomon::new(&sg, 7).unwrap();
        let f = rs.c5_word(8, &[0]).unwrap();
        let (cl, tr) = anneal(&rs, &f, Radius::agreement(8), 4, 2.0, 0.9, 3).unwrap();
        assert!(tr.best_size >= 70, "best must be >= the 70-bucket seed");
        assert_eq!(cl.size(), tr.best_size, "returned cluster is the best seen");
    }

    #[test]
    fn search_returns_best_and_all_traces() {
        let sg = Subgroup::new(65537, 16).unwrap();
        let rs = ReedSolomon::new(&sg, 7).unwrap();
        let (best, traces) = search(&rs, Radius::agreement(8), 2, 5, 2, 1).unwrap();
        assert_eq!(traces.len(), 2, "one trace per restart");
        assert!(best.size() >= 5, "best of the restarts >= a pencil seed");
    }
}
