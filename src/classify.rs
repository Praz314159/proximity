//! The bucket-vs-entropy diagnostic (D1's classifier).
//!
//! The open question the data program turns on: below the Elias wall, is the
//! *extremal* (max-list) word a **bucket** — its near-codewords sharing frozen
//! top-`q` symmetric functions — or **entropy-typical**, with no such algebraic
//! coherence? The answer decides the defense lane: bucket-reduction if the
//! former, a characteristic-`p` entropy converse if the latter. This module
//! reads that structure straight off a word's decoded list.
//!
//! Mechanism: for each codeword `c` in `List(w)`, its agreement set
//! `A(c) = {x : c(x) = w(x)}` is the algebraic fingerprint. For a C.5 bucket
//! word the `A(c)` are exactly the frozen subsets `{S : e_i(S) = lambda_i}`, so
//! their top symmetric functions are *constant across the list*. For an
//! entropy-typical word they scatter. The classifier measures the longest
//! frozen prefix.

use crate::code::top_elementary_symmetric;
use crate::decode::{DecodeOracle, Radius};
use crate::code::ReedSolomon;
use crate::error::Result;

/// How far the symmetric structure of a bucket is prime-specific.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Accidental {
    /// No accident inflates any bucket at these parameters (certified).
    Structural,
    /// Accidents at this prime merge structural classes (certified nonempty
    /// census).
    Inflated,
    /// Not determined (outside the `q = 1`, `s <= 32` certification range).
    Unknown,
}

/// The structural verdict on a word, read from its decoded list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WordKind {
    /// The list's agreement sets share the top-`frozen_q` symmetric functions:
    /// a bucket at radius `1 - r/n`.
    Bucket {
        /// Common agreement-set size.
        r: usize,
        /// Longest frozen symmetric prefix (`e_1..e_{frozen_q}` constant).
        frozen_q: usize,
        /// The frozen values `(e_1, ..., e_{frozen_q})`.
        lambda: Vec<u64>,
        /// Whether an accident inflates the bucket at this prime (best effort).
        accident: Accidental,
        /// The list size.
        list_size: u64,
    },
    /// The agreement sets do not share a frozen symmetric signature.
    EntropyTypical {
        /// Number of distinct symmetric signatures among the list.
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

/// The largest number of leading symmetric functions to test for freezing.
const SIG_CAP: usize = 8;

/// Classify a word by the algebraic structure of its list at `radius`.
pub fn classify(rs: &ReedSolomon, word: &[u64], radius: Radius) -> Result<WordKind> {
    let list = DecodeOracle::new(rs).list(word, radius)?;
    if list.len() <= 1 {
        return Ok(WordKind::Trivial {
            list_size: list.len() as u64,
        });
    }
    let p = rs.domain().p();
    let dom = rs.domain().elements();
    // Agreement set of each codeword, as the domain elements where it meets w.
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
    let a = analyze_sets(&sets, p);
    let list_size = list.len() as u64;
    if a.same_size && a.frozen_q >= 1 {
        let accident = accident_status(rs, a.r);
        Ok(WordKind::Bucket {
            r: a.r,
            frozen_q: a.frozen_q,
            lambda: a.lambda,
            accident,
            list_size,
        })
    } else {
        Ok(WordKind::EntropyTypical {
            distinct_signatures: a.distinct_signatures,
            list_size,
        })
    }
}

struct SetAnalysis {
    same_size: bool,
    r: usize,
    frozen_q: usize,
    lambda: Vec<u64>,
    distinct_signatures: usize,
}

/// Symmetric analysis of a family of agreement sets: whether they share size,
/// the longest constant symmetric prefix, and how many distinct signatures
/// they span.
fn analyze_sets(sets: &[Vec<u64>], p: u64) -> SetAnalysis {
    let r = sets[0].len();
    let same_size = sets.iter().all(|s| s.len() == r);
    let cap = SIG_CAP.min(r);
    let sigs: Vec<Vec<u64>> = sets
        .iter()
        .map(|s| top_elementary_symmetric(s, cap, p))
        .collect();
    let mut frozen_q = 0;
    for j in 0..cap {
        if sigs.iter().all(|s| s[j] == sigs[0][j]) {
            frozen_q = j + 1;
        } else {
            break;
        }
    }
    let mut distinct = sigs.clone();
    distinct.sort_unstable();
    distinct.dedup();
    SetAnalysis {
        same_size,
        r,
        frozen_q,
        lambda: sigs[0][..frozen_q].to_vec(),
        distinct_signatures: distinct.len(),
    }
}

/// Best-effort structural-vs-accident flag via [`crate::certify`] (only the
/// `q = 1`, power-of-two `s <= 32` range is certifiable; `Unknown` otherwise).
fn accident_status(rs: &ReedSolomon, r: usize) -> Accidental {
    let sg = rs.domain();
    if !sg.is_two_smooth() || sg.order() > 32 || r == 0 || r >= sg.order() {
        return Accidental::Unknown;
    }
    match crate::certify::certify_q1(sg, r) {
        Ok(cert) => match cert.verdict {
            crate::certify::Verdict::AllBucketsStructural => Accidental::Structural,
            _ => Accidental::Inflated,
        },
        Err(_) => Accidental::Unknown,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Subgroup;

    #[test]
    fn c5_word_classifies_as_structural_bucket() {
        let sg = Subgroup::new(65537, 16).unwrap();
        let rs = ReedSolomon::new(&sg, 7).unwrap();
        let f = rs.c5_word(8, &[0]).unwrap();
        match classify(&rs, &f, Radius::agreement(8)).unwrap() {
            WordKind::Bucket {
                r,
                frozen_q,
                lambda,
                accident,
                list_size,
            } => {
                assert_eq!(r, 8);
                assert_eq!(frozen_q, 1, "only e_1 is frozen (q = 1)");
                assert_eq!(lambda, vec![0]);
                assert_eq!(accident, Accidental::Structural);
                assert_eq!(list_size, 70);
            }
            other => panic!("expected structural bucket, got {other:?}"),
        }
    }

    #[test]
    fn frozen_prefix_detects_shared_e1() {
        // Two sets with equal element-sum (e_1) but different e_2.
        let p = 65537;
        let a = analyze_sets(&[vec![1, p - 1, 5], vec![2, p - 2, 5]], p);
        assert!(a.frozen_q >= 1, "e_1 = 5 is shared");
    }

    #[test]
    fn frozen_prefix_rejects_scattered_e1() {
        let p = 65537;
        let a = analyze_sets(&[vec![1, 2, 3], vec![10, 20, 30]], p);
        assert_eq!(a.frozen_q, 0, "e_1 differs (6 vs 60)");
        assert_eq!(a.distinct_signatures, 2);
    }
}
