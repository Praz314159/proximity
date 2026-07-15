//! The rung / quantized-ladder combinatorics of the smooth-subgroup bucket
//! program: the structural (characteristic-zero) maximum bucket size and the
//! Theorem-A rung construction that attains it. These are properties of the
//! multiplicative-subgroup structure, kept out of the generic [`crate::rs::code`]
//! layer.

use crate::domain::MultiplicativeSubgroup;
use crate::error::{Error, Result};
use crate::field::binom;
use crate::rs::code::top_elementary_symmetric;

/// The rung level `t = ceil(log2(q + 1))`: the smallest `t` with
/// `2^t - 1 >= q`. Pinning `q` symmetric functions buys exactly the level-`t`
/// coset structure (the ladder's quantization).
fn rung_level(q: usize) -> usize {
    (q + 1).next_power_of_two().ilog2() as usize
}

/// The structural (characteristic-zero) maximum bucket size — the quantized
/// ladder value
/// `M_struct(s, r, q) = C(s/2^t - [r0 != 0], floor(r / 2^t))`,
/// `t = ceil(log2(q + 1))`, `r0 = r mod 2^t`. Attained by the rung
/// construction ([`rung_lambda`]) and matched (within one bit) by the ladder
/// upper bound; exact against exhaustive enumeration at `s <= 32`.
#[must_use]
pub fn m_struct(s: usize, r: usize, q: usize) -> u64 {
    let block = 1usize << rung_level(q);
    let (b, r0) = (r / block, r % block);
    let ncos = s / block;
    let avail = ncos - usize::from(r0 != 0);
    if b > avail {
        return 0;
    }
    binom(avail as u64, b as u64)
}

/// Size of the structural class at weight `w`: the number of `r`-subsets whose
/// `e_1`-representation has exactly `w` forced singles,
/// `C(s/2 - w, (r - w)/2)` (zero when the parity or range is infeasible).
#[must_use]
pub fn class_size(s: usize, r: usize, w: usize) -> u64 {
    if w > r || (r - w) % 2 == 1 {
        return 0;
    }
    let z = s / 2 - w.min(s / 2);
    if w > s / 2 {
        return 0;
    }
    binom(z as u64, ((r - w) / 2) as u64)
}

/// The rung lambda: the common top-`q` elementary symmetric values
/// `(e_1, ..., e_q)` of the Theorem-A rung family (fixed `r0`-subset of one
/// `mu_{2^t}` coset plus `b` full cosets), for `q = 2^t - 1`-quantized `q`.
/// Any member subset realizes them; all members share them.
pub fn rung_lambda(sg: &MultiplicativeSubgroup, r: usize, q: usize) -> Result<Vec<u64>> {
    if !sg.is_two_smooth() {
        return Err(Error::Unsupported(
            "rung construction needs power-of-two s".into(),
        ));
    }
    if q == 0 || q >= r || r >= sg.order() {
        return Err(Error::OutOfRange("need 1 <= q < r < s".into()));
    }
    let t = rung_level(q);
    let block = 1usize << t;
    let (b, r0) = (r / block, r % block);
    let cosets = sg.cosets(t)?;
    if b + 1 > cosets.len() && !(r0 == 0 && b <= cosets.len()) {
        return Err(Error::OutOfRange(
            "rung does not fit in the subgroup".into(),
        ));
    }
    let mut subset: Vec<u64> = cosets[0][..r0].to_vec();
    for coset in cosets.iter().skip(1).take(b) {
        subset.extend_from_slice(coset);
    }
    debug_assert_eq!(subset.len(), r);
    Ok(top_elementary_symmetric(&subset, q, sg.p()))
}
