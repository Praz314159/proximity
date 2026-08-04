//! The bucket-vs-entropy diagnostic, graded.
//!
//! For a word and its list, measure how far the list is from a **bucket** —
//! near-codewords sharing frozen top-`q` symmetric functions. Every
//! exhaustively measured cell has resolved this way (extremal words are
//! bucket words: spiked top words, in discrete tiers), and this diagnostic
//! is what checks a new word against that classification. It grades rather
//! than labels, because real objects live on a *spectrum*
//! — a union of a few buckets is still bucket-reducible yet is not perfectly
//! frozen — so a binary label misroutes the fork. The order parameter is the
//! **Shannon entropy of the symmetric-function distribution over the list**: for
//! each `e_i`, form the distribution of `e_i(A(c))` over the codewords `c` and
//! measure its spread. Zero entropy = a frozen bucket; `log2 L` = maximally
//! scattered; in between = the structured-but-not-frozen middle. This is (up to
//! normalization) the characteristic-`p` entropy quantity CS25 conjectures
//! governs the wall.
//!
//! [`structure`] returns that data — per-`e_i` distributions and entropies —
//! for post-processing and visualization. [`classify`] is a thin thresholded
//! label on top of it.

use crate::error::Result;
use crate::rs::code::top_elementary_symmetric;
use crate::rs::code::ReedSolomon;
use crate::rs::decode::{DecodeOracle, Radius};
use std::collections::BTreeMap;

/// The largest number of leading symmetric functions to profile.
const SIG_CAP: usize = 8;

/// Distributional summary of one symmetric function `e_i` over a list's
/// agreement sets.
#[derive(Debug, Clone, PartialEq)]
pub struct SymStat {
    /// Which symmetric function: `1` for `e_1`, etc.
    pub index: usize,
    /// Shannon entropy (bits) of the `e_i` distribution over the list. `0` iff
    /// `e_i` is frozen to a single value (a bucket in this coordinate).
    pub entropy: f64,
    /// Number of distinct `e_i` values.
    pub distinct: usize,
    /// Fraction of the list in the largest `e_i` class.
    pub max_class_fraction: f64,
    /// The most common `e_i` value (the frozen value when entropy is `0`).
    pub mode_value: u64,
    /// The full distribution `(e_i value, count)`, sorted by value — the raw
    /// material for a histogram/plot.
    pub distribution: Vec<(u64, u64)>,
}

/// The graded structure of a word's list: agreement-size spread and the
/// symmetric-function distributions. The post-processable / visualizable output.
#[derive(Debug, Clone, PartialEq)]
pub struct ListStructure {
    /// The list size `L`.
    pub list_size: u64,
    /// Agreement-set sizes `(size, count)`, sorted.
    pub agreement_sizes: Vec<(usize, u64)>,
    /// Shannon entropy (bits) of the agreement-size distribution.
    pub size_entropy: f64,
    /// Per-`e_i` distributional stats, `e_1` first, up to `min(SIG_CAP, r)`.
    pub symmetric: Vec<SymStat>,
    /// Shannon entropy (bits) of the *joint* signature `(e_1, ..., e_cap)`
    /// distribution — the overall algebraic-coherence order parameter.
    pub joint_entropy: f64,
    /// Number of distinct joint signatures.
    pub joint_distinct: usize,
}

/// The thresholded label derived from [`ListStructure`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WordKind {
    /// Same agreement size and a nonempty frozen symmetric prefix (`e_1..e_q`
    /// each at zero entropy): a bucket at radius `1 - r/n`.
    Bucket {
        /// Common agreement-set size.
        r: usize,
        /// Length of the zero-entropy (frozen) symmetric prefix.
        frozen_q: usize,
        /// The frozen values `(e_1, ..., e_{frozen_q})`.
        lambda: Vec<u64>,
        /// The list size.
        list_size: u64,
    },
    /// No frozen symmetric prefix: the list scatters across signatures.
    EntropyTypical {
        /// Number of distinct joint signatures.
        distinct_signatures: usize,
        /// The list size.
        list_size: u64,
    },
    /// Empty or singleton list — nothing to classify.
    Trivial {
        /// The list size (`0` or `1`).
        list_size: u64,
    },
}

/// The graded structure of `word`'s list at `radius`: the primary, post-
/// processable diagnostic.
pub fn structure(rs: &ReedSolomon, word: &[u64], radius: Radius) -> Result<ListStructure> {
    let list = DecodeOracle::new(rs).list(word, radius)?;
    let list_size = list.len() as u64;
    let p = rs.p();
    let dom = rs.points();
    let sets: Vec<Vec<u64>> = list
        .iter()
        .map(|cw| {
            dom.iter()
                .zip(word)
                .zip(cw)
                .filter_map(|((&x, &wi), &ci)| (wi == ci).then_some(x))
                .collect()
        })
        .collect();

    let mut size_map: BTreeMap<usize, u64> = BTreeMap::new();
    for s in &sets {
        *size_map.entry(s.len()).or_insert(0) += 1;
    }
    let agreement_sizes: Vec<(usize, u64)> = size_map.iter().map(|(&k, &v)| (k, v)).collect();
    let size_entropy = shannon_bits(&size_map.values().copied().collect::<Vec<_>>());

    let min_size = sets.iter().map(Vec::len).min().unwrap_or(0);
    let cap = SIG_CAP.min(min_size);
    let sigs: Vec<Vec<u64>> = sets
        .iter()
        .map(|s| top_elementary_symmetric(s, cap, p))
        .collect();

    let mut symmetric = Vec::with_capacity(cap);
    for i in 0..cap {
        let mut dist: BTreeMap<u64, u64> = BTreeMap::new();
        for sig in &sigs {
            *dist.entry(sig[i]).or_insert(0) += 1;
        }
        let counts: Vec<u64> = dist.values().copied().collect();
        let (mode_value, max_count) = dist
            .iter()
            .max_by_key(|(_, &c)| c)
            .map(|(&v, &c)| (v, c))
            .unwrap_or((0, 0));
        symmetric.push(SymStat {
            index: i + 1,
            entropy: shannon_bits(&counts),
            distinct: dist.len(),
            max_class_fraction: if list_size > 0 {
                max_count as f64 / list_size as f64
            } else {
                0.0
            },
            mode_value,
            distribution: dist.into_iter().collect(),
        });
    }

    let mut joint: BTreeMap<Vec<u64>, u64> = BTreeMap::new();
    for sig in &sigs {
        *joint.entry(sig.clone()).or_insert(0) += 1;
    }
    Ok(ListStructure {
        list_size,
        agreement_sizes,
        size_entropy,
        symmetric,
        joint_entropy: shannon_bits(&joint.values().copied().collect::<Vec<_>>()),
        joint_distinct: joint.len(),
    })
}

/// A thresholded label over [`structure`]: `Bucket` when the agreement size is
/// constant and a nonempty symmetric prefix is frozen (zero entropy), else
/// `EntropyTypical`. Prefer [`structure`] for analysis — the label discards the
/// gradation.
pub fn classify(rs: &ReedSolomon, word: &[u64], radius: Radius) -> Result<WordKind> {
    let st = structure(rs, word, radius)?;
    if st.list_size <= 1 {
        return Ok(WordKind::Trivial {
            list_size: st.list_size,
        });
    }
    let same_size = st.agreement_sizes.len() == 1;
    let frozen_q = st.symmetric.iter().take_while(|s| s.entropy == 0.0).count();
    if same_size && frozen_q >= 1 {
        let r = st.agreement_sizes[0].0;
        let lambda = st.symmetric[..frozen_q]
            .iter()
            .map(|s| s.mode_value)
            .collect();
        Ok(WordKind::Bucket {
            r,
            frozen_q,
            lambda,
            list_size: st.list_size,
        })
    } else {
        Ok(WordKind::EntropyTypical {
            distinct_signatures: st.joint_distinct,
            list_size: st.list_size,
        })
    }
}

/// Shannon entropy in bits of a distribution given its class counts.
fn shannon_bits(counts: &[u64]) -> f64 {
    let total: u64 = counts.iter().sum();
    if total == 0 {
        return 0.0;
    }
    let t = total as f64;
    -counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let pr = c as f64 / t;
            pr * pr.log2()
        })
        .sum::<f64>()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::MultiplicativeSubgroup;

    #[test]
    fn shannon_is_zero_for_frozen_and_one_bit_for_even_split() {
        assert_eq!(shannon_bits(&[70]), 0.0);
        assert!((shannon_bits(&[5, 5]) - 1.0).abs() < 1e-12);
        assert!((shannon_bits(&[1, 1, 1, 1]) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn bucket_has_frozen_e1_and_positive_e2_entropy() {
        let sg = MultiplicativeSubgroup::new(65537, 16).unwrap();
        let rs = ReedSolomon::on_subgroup(&sg, 7).unwrap();
        let f = rs.c5_word(8, &[0]).unwrap();
        let st = structure(&rs, &f, Radius::agreement(8)).unwrap();
        assert_eq!(st.list_size, 70);
        assert_eq!(st.symmetric[0].index, 1);
        assert_eq!(st.symmetric[0].entropy, 0.0, "e_1 is frozen");
        assert_eq!(st.symmetric[0].mode_value, 0);
        assert!(st.symmetric[1].entropy > 0.0, "e_2 is not frozen");
    }

    #[test]
    fn c5_word_classifies_as_bucket() {
        let sg = MultiplicativeSubgroup::new(65537, 16).unwrap();
        let rs = ReedSolomon::on_subgroup(&sg, 7).unwrap();
        let f = rs.c5_word(8, &[0]).unwrap();
        match classify(&rs, &f, Radius::agreement(8)).unwrap() {
            WordKind::Bucket {
                r,
                frozen_q,
                lambda,
                list_size,
            } => {
                assert_eq!((r, frozen_q, list_size), (8, 1, 70));
                assert_eq!(lambda, vec![0]);
            }
            other => panic!("expected bucket, got {other:?}"),
        }
    }
}
