//! `valuemap` — the value-map layer: a Formula over a Cloud (PR-1).
//!
//! The semantic core and nothing else: a total, compositional
//! `eval`, cloud enumeration under the builder restrictions, and
//! bucket statistics defined once against them. Every later fast
//! path (the MITM half-tables, the bucket engines) is a refinement
//! of the definitions in this file and answers to them; this file
//! answers to no one. Design and law schedule:
//! `design/valuemap_layer.md` (algebra v3, naming v4, staged PRs).

use crate::domain::MultiplicativeSubgroup;
use crate::error::Result;
use crate::field::{mulmod, powmod};
use std::collections::HashMap;

/// A support: a subset of `mu_s`, as sorted indices into the
/// consecutive-powers list of the subgroup.
pub type Support = Vec<usize>;

/// The cloud: all size-`r` supports in the order-`level` subgroup,
/// filtered by the builder restrictions. The program's ch.1 object.
#[derive(Clone, Debug)]
pub struct Cloud {
    pub level: usize,
    pub r: usize,
    pair_free: bool,
    avoid: Vec<usize>,
}

impl Cloud {
    pub fn new(level: usize, r: usize) -> Self {
        Cloud {
            level,
            r,
            pair_free: false,
            avoid: Vec::new(),
        }
    }
    /// Exclude supports containing an antipodal index pair.
    pub fn pair_free(mut self) -> Self {
        self.pair_free = true;
        self
    }
    /// Exclude supports meeting the given indices.
    pub fn avoiding(mut self, idx: &[usize]) -> Self {
        self.avoid.extend_from_slice(idx);
        self
    }

    fn admits(&self, support: &Support) -> bool {
        if support.iter().any(|i| self.avoid.contains(i)) {
            return false;
        }
        if self.pair_free && self.level % 2 == 0 {
            let half = self.level / 2;
            if support
                .iter()
                .any(|&i| support.binary_search(&((i + half) % self.level)).is_ok())
            {
                return false;
            }
        }
        true
    }
}

/// The formula: what is computed of each support (algebra v3;
/// remaining variants land with their consuming PRs).
#[derive(Clone, Debug)]
pub enum Formula {
    /// The elementary-symmetric coordinate `e_j` of the support.
    Coord(usize),
    /// A liftable-coefficient combination of coordinates — the cut
    /// functional of a small-height syndrome (the P0 corner).
    Combine(Vec<(i64, usize)>),
    /// A ratio of formulas, e.g. the gatekeeper `e_5 / e_1`.
    Ratio(Box<Formula>, Box<Formula>),
    /// `prod (x0 - t)` over the support values — unary evaluation.
    EvalAt(u64),
    /// `F(x0)/F(-x0)` — the phi-product presentation.
    EvalRatio(u64),
}

/// A formula over a cloud, with its total semantics.
pub struct ValueMap {
    pub cloud: Cloud,
    pub formula: Formula,
    sg: MultiplicativeSubgroup,
}

impl ValueMap {
    pub fn new(p: u64, cloud: Cloud, formula: Formula) -> Result<Self> {
        let sg = MultiplicativeSubgroup::new(p, cloud.level)?;
        Ok(ValueMap { cloud, formula, sg })
    }

    /// Total semantic evaluation — the definition of the map.
    /// `None` marks a pole (zero denominator), never an error.
    pub fn eval(&self, support: &Support) -> Option<u64> {
        self.eval_formula(&self.formula, support)
    }

    fn e_vec(&self, support: &Support) -> Vec<u64> {
        let p = self.sg.p();
        let mut e = vec![1u64];
        for &i in support {
            let v = self.sg.elements()[i];
            let mut next = vec![0u64; e.len() + 1];
            for (j, &c) in e.iter().enumerate() {
                next[j] = (next[j] + c) % p;
                next[j + 1] = (next[j + 1] + mulmod(v, c, p)) % p;
            }
            e = next;
        }
        e
    }

    fn eval_formula(&self, f: &Formula, support: &Support) -> Option<u64> {
        let p = self.sg.p();
        match f {
            Formula::Coord(j) => self.e_vec(support).get(*j).copied(),
            Formula::Combine(terms) => {
                let e = self.e_vec(support);
                let mut acc: i128 = 0;
                for &(c, j) in terms {
                    acc += c as i128 * *e.get(j)? as i128;
                }
                Some(acc.rem_euclid(p as i128) as u64)
            }
            Formula::Ratio(num, den) => {
                let d = self.eval_formula(den, support)?;
                if d == 0 {
                    return None;
                }
                let n = self.eval_formula(num, support)?;
                Some(mulmod(n, powmod(d, p - 2, p), p))
            }
            Formula::EvalAt(x0) => Some(support.iter().fold(1u64, |acc, &i| {
                mulmod(acc, (x0 + p - self.sg.elements()[i] % p) % p, p)
            })),
            Formula::EvalRatio(x0) => {
                let num = self.eval_formula(&Formula::EvalAt(*x0), support)?;
                let den = self.eval_formula(&Formula::EvalAt(p - *x0), support)?;
                if den == 0 {
                    return None;
                }
                Some(mulmod(num, powmod(den, p - 2, p), p))
            }
        }
    }

    /// Every admitted support, lexicographically.
    pub fn enumerate(&self) -> Vec<Support> {
        let (n, r) = (self.cloud.level, self.cloud.r);
        let mut out = Vec::new();
        if r > n {
            return out;
        }
        let mut cur: Support = (0..r).collect();
        loop {
            if self.cloud.admits(&cur) {
                out.push(cur.clone());
            }
            let mut i = r;
            loop {
                if i == 0 {
                    return out;
                }
                i -= 1;
                if cur[i] != i + n - r {
                    break;
                }
            }
            cur[i] += 1;
            for j in i + 1..r {
                cur[j] = cur[j - 1] + 1;
            }
        }
    }

    /// Buckets: fiber sizes by value, poles excluded. Defined once,
    /// here, against `eval` — the layer's ground truth.
    pub fn buckets(&self) -> HashMap<u64, u64> {
        let mut b = HashMap::new();
        for s in self.enumerate() {
            if let Some(v) = self.eval(&s) {
                *b.entry(v).or_insert(0) += 1;
            }
        }
        b
    }

    /// The exact second moment of the buckets, `sum m (m - 1) / 2`
    /// — the collision count, the open core's currency.
    pub fn collision_count(&self) -> u128 {
        self.buckets()
            .values()
            .map(|&m| m as u128 * (m as u128 - 1) / 2)
            .sum()
    }

    pub fn max_bucket(&self) -> u64 {
        self.buckets().values().copied().max().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const P: u64 = 65537;

    /// Sum-class sizes of 8-subsets of mu_16 (the top-word/GS
    /// grading): classes 0 and 8 hold 809, classes 4 and 12 hold
    /// 810 — the exact counts behind the rung's L = 809 and the
    /// K = 810 record (resolved 2026-08-08).
    #[test]
    fn gs_class_sizes_from_the_core() {
        let vm = ValueMap::new(P, Cloud::new(16, 8), Formula::Coord(8)).unwrap();
        let b = vm.buckets();
        for (class, expect) in [(0usize, 809u64), (4, 810), (8, 809), (12, 810)] {
            let idx = vm.sg.elements()[class];
            assert_eq!(b[&idx], expect, "class {class}");
        }
    }

    /// The master count at s = 64: pair-free 6-subsets avoiding
    /// {1, -1} on the cut e1 + e3 + e5 = 0 — exactly 1,064
    /// (frontier 2026-08-08), pinned here from the semantic core.
    #[test]
    fn master_count_from_the_core() {
        let vm = ValueMap::new(
            P,
            Cloud::new(64, 6).pair_free().avoiding(&[0, 32]),
            Formula::Combine(vec![(1, 1), (1, 3), (1, 5)]),
        )
        .unwrap();
        assert_eq!(vm.buckets().get(&0).copied().unwrap_or(0), 1064);
    }
}
