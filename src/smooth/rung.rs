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

/// The proven multiplicative extremal word (Theorem B_mult, 2026-07-21):
/// `w(x) = x^{r-1} - (-1)^{r+1} zeta^c x^{s-1}` on `mu_s`, whose exact list at
/// agreement `r` (code degree `< r - 1`) is the Graham--Sloane class
/// `{ S : prod S = zeta^c }`, at every prime containing `mu_s`.
pub fn top_word(sg: &MultiplicativeSubgroup, r: usize, c: usize) -> Result<Vec<u64>> {
    use crate::field::{mulmod, powmod};
    let (p, s) = (sg.p(), sg.order());
    if r < 2 || r >= s {
        return Err(Error::OutOfRange("need 2 <= r < s".into()));
    }
    let zeta_c = powmod(sg.w(), (c % s) as u64, p);
    // gamma = (-1)^{r+1} zeta^c
    let gamma = if r % 2 == 1 { zeta_c } else { (p - zeta_c) % p };
    Ok(sg
        .elements()
        .iter()
        .map(|&x| {
            let lead = powmod(x, (r - 1) as u64, p);
            let tail = mulmod(gamma, powmod(x, (s - 1) as u64, p), p);
            (lead + p - tail) % p
        })
        .collect())
}

/// The word of a syndrome vector `b` in the moment convention:
/// `w = sum_j (-1)^j b_j x^{r-1+j}` evaluated on `domain` — the inverse of the
/// cut map `Z(b) = { S : <b, e-hat(complement of S)> = 0 }`, with the sign
/// convention pinned so that `D_S(w) = sum_j b_j e_j(Sbar)` exactly.
pub fn word_from_syndrome(p: u64, domain: &[u64], r: usize, b: &[u64]) -> Vec<u64> {
    use crate::field::{mulmod, powmod};
    domain
        .iter()
        .map(|&x| {
            let mut acc = 0u64;
            for (j, &bj) in b.iter().enumerate() {
                let term = mulmod(bj % p, powmod(x, (r - 1 + j) as u64, p), p);
                acc = if j % 2 == 0 {
                    (acc + term) % p
                } else {
                    (acc + p - term) % p
                };
            }
            acc
        })
        .collect()
}

/// Graham--Sloane class counts: `out[c] = #{ T subset Z_s, |T| = t :
/// sum(T) = c mod s }`, by dynamic programming. `out` has length `s`.
pub fn gs_class_counts(s: usize, t: usize) -> Result<Vec<u64>> {
    if t > s {
        return Err(Error::OutOfRange("need t <= s".into()));
    }
    // dp[j][c] = #{ j-subsets of {0..a-1} with sum c mod s }, folded over a.
    let mut dp = vec![vec![0u64; s]; t + 1];
    dp[0][0] = 1;
    for a in 0..s {
        for j in (1..=t.min(a + 1)).rev() {
            for c in 0..s {
                let prev = dp[j - 1][(c + s - (a % s)) % s];
                dp[j][c] = dp[j][c]
                    .checked_add(prev)
                    .ok_or_else(|| Error::OutOfRange("class count overflows u64".into()))?;
            }
        }
    }
    Ok(dp.swap_remove(t))
}
