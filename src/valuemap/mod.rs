//! `valuemap` — the value-map layer: Formula over a Cloud (PR-1).
//!
//! Semantic core only: total `eval`, enumeration, bucket statistics.
//! Design: design/valuemap_layer.md (algebra v3, naming v4).
//! Laws and refinements land in later PRs; nothing here touches
//! existing engines.

use crate::error::Result;

/// A support: a subset of mu_s, canonically a sorted index list.
pub type Support = Vec<usize>;

/// The cloud: supports of size `r` in the order-`level` subgroup,
/// filtered by restrictions (builder verbs).
pub struct Cloud {
    pub level: usize,
    pub r: usize,
    pub pair_free: bool,
    pub avoid: Vec<usize>,
    pub on_cut: Option<(Vec<i64>, u64)>, // (lift coeffs over e_1.., value)
}

/// The formula: what is computed of each support (algebra v3).
pub enum Formula {
    Coord(usize),
    Window { lo: usize, len: usize, stride: usize },
    Combine(Vec<(i64, Formula)>),
    Ratio(Box<Formula>, Box<Formula>),
    EvalAt { point: u64 },
    EvalRatio { point: u64 },
    Dilate(Box<Formula>),
}

pub struct ValueMap { pub cloud: Cloud, pub formula: Formula }

impl ValueMap {
    /// Total semantic evaluation — the DEFINITION. (PR-1 next
    /// commit: e-vector embedding + compositional interpreter.)
    pub fn eval(&self, _support: &Support) -> Result<Option<u64>> {
        unimplemented!("PR-1: semantic core")
    }
}
