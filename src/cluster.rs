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

/// Build a cluster **from the code**, presupposing no bucket: take a *pencil*
/// of codewords sharing a `(k-1)`-coordinate core, assemble their sunflower
/// center, and grow it.
///
/// `core_coords` (length `k-1`) and `core_values` fix the shared agreement; each
/// `petal_values[j]` at `petal_point` selects one pencil codeword `c_j` (all
/// `c_j` pass through the core, so pairwise they meet in exactly it). The seed
/// center equals the core on `core_coords` and equals a distinct `c_j` on each
/// of `m` disjoint blocks of the remaining coordinates — so every `c_j` lands in
/// its list at agreement `k-1 + block`. For all petals to reach radius `t`, pick
/// `m <= (n-k+1)/(t-k+1)`. Then grow-and-center accretes and re-centers.
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
    let (n, k, p) = (rs.n(), rs.k(), rs.domain().p());
    let dom = rs.domain().elements();
    if core_coords.len() != k - 1 || core_values.len() != k - 1 {
        return Err(Error::OutOfRange("core must have exactly k-1 points".into()));
    }
    if petal_values.is_empty() || petal_point >= n || core_coords.iter().any(|&c| c >= n) {
        return Err(Error::OutOfRange("need >=1 petal and in-range coords".into()));
    }
    // Interpolation nodes: the k-1 core points plus the petal point.
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
    // Seed center: core from core_values; every other coordinate to a petal,
    // round-robin, taking that petal codeword's value there.
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
    let (n, k, p) = (rs.n(), rs.k(), rs.domain().p());
    let mut s = seed;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        s >> 33
    };
    // k distinct coordinates: the first k-1 are the core, the last the petal pt.
    let mut coords: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + (next() as usize) % (n - i);
        coords.swap(i, j);
    }
    let core_coords = coords[..k - 1].to_vec();
    let petal_point = coords[k - 1];
    let core_values: Vec<u64> = (0..k - 1).map(|_| next() % p).collect();
    let mut pv: HashSet<u64> = HashSet::new();
    let cap = petals.min(p as usize - 1);
    while pv.len() < cap {
        pv.insert(next() % p);
    }
    let petal_values: Vec<u64> = pv.into_iter().collect();
    grow_from_pencil(
        rs,
        &core_coords,
        &core_values,
        petal_point,
        &petal_values,
        radius,
        0,
        12,
        seed.wrapping_add(1),
    )
}

/// Instrumentation for one [`optimize`] run — the raw material for studying the
/// optimizer itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OptTrace {
    /// List size at the true radius after each step (`sizes[0]` = the seed).
    pub sizes: Vec<usize>,
    /// Accepted single-coordinate flips.
    pub flips: usize,
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
        flips: 0,
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
                trace.flips += 1;
                if trace.flips >= max_flips {
                    break;
                }
            }
            _ => break, // no strictly improving flip: local maximum
        }
    }

    let members = oracle.list(&w, radius)?;
    Ok((
        Cluster {
            center: w,
            members,
            radius,
        },
        trace,
    ))
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
}
