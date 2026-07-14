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
use crate::decode::{DecodeOracle, Radius};
use crate::error::{Error, Result};
use std::collections::HashMap;

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
}
