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
use crate::field::{binom, mulmod, powmod};

/// `RS[F_p, mu_s, k]`: polynomials of degree `< k` evaluated on the subgroup.
#[derive(Debug, Clone)]
pub struct ReedSolomon<'a> {
    domain: &'a Subgroup,
    k: usize,
}

impl<'a> ReedSolomon<'a> {
    /// Construct `RS[F_p, mu_s, k]` on the given domain; validates
    /// `1 <= k <= n`.
    pub fn new(domain: &'a Subgroup, k: usize) -> Result<Self> {
        if k == 0 || k > domain.order() {
            return Err(Error::OutOfRange(format!(
                "message length k = {k} outside [1, n = {}]",
                domain.order()
            )));
        }
        Ok(ReedSolomon { domain, k })
    }

    /// The evaluation domain.
    #[must_use]
    pub fn domain(&self) -> &Subgroup {
        self.domain
    }
    /// Code length `n = s`.
    #[must_use]
    pub fn n(&self) -> usize {
        self.domain.order()
    }
    /// Message length `k`.
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }
    /// Rate `k / n`.
    #[must_use]
    pub fn rate(&self) -> f64 {
        self.k as f64 / self.n() as f64
    }
    /// Relative minimum distance `1 - (k-1)/n` (the capacity radius).
    #[must_use]
    pub fn capacity_radius(&self) -> f64 {
        1.0 - (self.k as f64 - 1.0) / self.n() as f64
    }
    /// Johnson radius `1 - sqrt(rate)`.
    #[must_use]
    pub fn johnson_radius(&self) -> f64 {
        1.0 - self.rate().sqrt()
    }

    /// Whether `(r, q)` satisfies the Lemma-C.5 window `(r - q - 1) < k <= r`
    /// at `m = 1`, i.e. the C.5-form word at these parameters certifies (and,
    /// by exactness, *equals*) the bucket list at radius `1 - r/n`.
    #[must_use]
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

impl ReedSolomon<'_> {
    /// Materialize the C.5-form (payoff) word on the domain: the "frozen head"
    /// `f(x) = sum_{i=0}^q (-1)^i lambda_i x^{r-i}` (with `lambda_0 = 1`, `m = 1`).
    ///
    /// This is the *received word* whose list at radius `1 - r/n` is, by the
    /// exactness theorem, exactly the bucket `{S : e_i(S) = lambda_i}` (see the
    /// module intro and [`crate::buckets`]). It is the bridge between the
    /// count axis ([`crate::buckets`]) and the decode axis
    /// ([`crate::decode`]): decoding this word and counting its bucket must
    /// return the same list size.
    ///
    /// `lambda` holds the `e`-values `(e_1, ..., e_q)`; the alternating signs
    /// are the `c_i = (-1)^i e_i` sign convention institutionalized in
    /// [`crate::buckets::mitm`]. Requires `q <= r < n`.
    pub fn c5_word(&self, r: usize, lambda: &[u64]) -> Result<Vec<u64>> {
        let q = lambda.len();
        if r >= self.n() || q > r {
            return Err(Error::OutOfRange("need q <= r < n".into()));
        }
        let p = self.domain.p();
        Ok(self
            .domain
            .elements()
            .iter()
            .map(|&x| {
                let mut acc = 0u64;
                for i in 0..=q {
                    let lam = if i == 0 { 1 } else { lambda[i - 1] % p };
                    let term = mulmod(lam, powmod(x, (r - i) as u64, p), p);
                    acc = if i % 2 == 0 {
                        (acc + term) % p
                    } else {
                        (acc + p - term) % p
                    };
                }
                acc
            })
            .collect())
    }
}

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
pub fn rung_lambda(sg: &Subgroup, r: usize, q: usize) -> Result<Vec<u64>> {
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

/// The top-`q` elementary symmetric functions `e_1..e_q` of a set of field
/// elements.
#[must_use]
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
