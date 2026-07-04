//! The linear-code layer: Reed–Solomon codes evaluated on a subgroup domain
//! (`m = 1` in the useful-family framework: domain = subgroup), plus the
//! structural combinatorics of the quantized ladder.
//!
//! The central connection (the exactness theorem): for the extremal
//! "C.5-form" words `f = x^r - lambda_1 x^{r-1} + ... +- lambda_q x^{r-q}`
//! with `(r - q - 1) < k <= r`, the list of `f` at radius `1 - r/n` is
//! *exactly* the bucket `{S : e_i(S) = lambda_i}` — so bucket sizes computed
//! by [`crate::buckets`] are exact list sizes beyond the Johnson radius.

use crate::domain::Subgroup;
use crate::error::{Error, Result};
use crate::field::binom;

/// `RS[F_p, mu_s, k]`: polynomials of degree `< k` evaluated on the subgroup.
#[derive(Debug, Clone)]
pub struct ReedSolomon<'a> {
    domain: &'a Subgroup,
    k: usize,
}

impl<'a> ReedSolomon<'a> {
    pub fn new(domain: &'a Subgroup, k: usize) -> Result<Self> {
        if k == 0 || k > domain.order() {
            return Err(Error::OutOfRange(format!(
                "message length k = {k} outside [1, n = {}]",
                domain.order()
            )));
        }
        Ok(ReedSolomon { domain, k })
    }

    pub fn domain(&self) -> &Subgroup {
        self.domain
    }
    pub fn n(&self) -> usize {
        self.domain.order()
    }
    pub fn k(&self) -> usize {
        self.k
    }
    pub fn rate(&self) -> f64 {
        self.k as f64 / self.n() as f64
    }
    /// Relative minimum distance `1 - (k-1)/n` (the capacity radius).
    pub fn capacity_radius(&self) -> f64 {
        1.0 - (self.k as f64 - 1.0) / self.n() as f64
    }
    /// Johnson radius `1 - sqrt(rate)`.
    pub fn johnson_radius(&self) -> f64 {
        1.0 - self.rate().sqrt()
    }

    /// Whether `(r, q)` satisfies the Lemma-C.5 window `(r - q - 1) < k <= r`
    /// at `m = 1`, i.e. the C.5-form word at these parameters certifies (and,
    /// by exactness, *equals*) the bucket list at radius `1 - r/n`.
    pub fn in_c5_window(&self, r: usize, q: usize) -> bool {
        r >= self.k && self.k + q >= r && r < self.n()
    }

    /// Evaluate the message polynomial (coefficients low-to-high, length `k`)
    /// on the domain.
    pub fn encode(&self, message: &[u64]) -> Result<Vec<u64>> {
        if message.len() != self.k {
            return Err(Error::OutOfRange("message length != k".into()));
        }
        let p = self.domain.p();
        Ok(self
            .domain
            .elements()
            .iter()
            .map(|&x| {
                message
                    .iter()
                    .rev()
                    .fold(0u64, |acc, &c| (crate::field::mulmod(acc, x, p) + c) % p)
            })
            .collect())
    }
}

/// The structural (characteristic-zero) maximum bucket size — the quantized
/// ladder value
/// `M_struct(s, r, q) = C(s/2^t - [r0 != 0], floor(r / 2^t))`,
/// `t = ceil(log2(q + 1))`, `r0 = r mod 2^t`. Attained by the rung
/// construction ([`rung_lambda`]) and matched (within one bit) by the ladder
/// upper bound; exact against exhaustive enumeration at `s <= 32`.
pub fn m_struct(s: usize, r: usize, q: usize) -> u64 {
    let mut t = 0usize;
    while (1usize << t) - 1 < q {
        t += 1;
    }
    let block = 1usize << t;
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
pub fn rung_lambda(sg: &Subgroup, r: usize, q: usize) -> Result<Vec<u64>> {
    if !sg.is_two_smooth() {
        return Err(Error::Unsupported(
            "rung construction needs power-of-two s".into(),
        ));
    }
    if q == 0 || q >= r || r >= sg.order() {
        return Err(Error::OutOfRange("need 1 <= q < r < s".into()));
    }
    let mut t = 0usize;
    while (1usize << t) - 1 < q {
        t += 1;
    }
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

/// The top-`q` elementary symmetric functions `e_1..e_q` of a set of field
/// elements.
pub fn top_elementary_symmetric(els: &[u64], q: usize, p: u64) -> Vec<u64> {
    let mut e = vec![0u64; q + 1];
    e[0] = 1;
    for (cnt, &a) in els.iter().enumerate() {
        let top = q.min(cnt + 1);
        for i in (1..=top).rev() {
            e[i] = (e[i] + crate::field::mulmod(a, e[i - 1], p)) % p;
        }
    }
    e.remove(0);
    e
}
