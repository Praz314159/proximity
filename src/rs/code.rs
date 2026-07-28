//! The linear-code layer: Reed–Solomon codes evaluated on a subgroup domain
//! (`m = 1` in the useful-family framework: domain = subgroup), plus the
//! structural combinatorics of the quantized ladder.
//!
//! The central connection (the exactness theorem): for the extremal
//! "C.5-form" words `f = x^r - lambda_1 x^{r-1} + ... +- lambda_q x^{r-q}`
//! with `(r - q - 1) < k <= r`, the list of `f` at radius `1 - r/n` is
//! *exactly* the bucket `{S : e_i(S) = lambda_i}` — so bucket sizes computed
//! by [`crate::census::buckets`] are exact list sizes beyond the Johnson radius.

use crate::domain::{EvalDomain, MultiplicativeSubgroup};
use crate::error::{Error, Result};
use crate::field::{mulmod, powmod};

/// `RS[F_p, D, k]`: polynomials of degree `< k` evaluated on an explicit domain
/// `D` of `n` distinct points in `F_p`. Generic — a multiplicative subgroup
/// ([`Self::on_subgroup`]) is one way to build `D`, not a requirement. The
/// discovery layer (`decode`/`cluster`/`classify`) works on any `D`; the
/// bucket/census/accident machinery stays subgroup-specific.
#[derive(Debug, Clone)]
pub struct ReedSolomon {
    domain: EvalDomain,
    k: usize,
}

impl ReedSolomon {
    /// Construct `RS[F_p, D, k]` on an evaluation domain; validates `1 <= k <= n`.
    pub fn new(domain: EvalDomain, k: usize) -> Result<Self> {
        let n = domain.n();
        if k == 0 || k > n {
            return Err(Error::OutOfRange(format!(
                "message length k = {k} outside [1, n = {n}]"
            )));
        }
        Ok(ReedSolomon { domain, k })
    }

    /// Construct on an explicit distinct-point domain (sugar over
    /// [`EvalDomain::from_points`] + [`Self::new`]).
    pub fn on_domain(p: u64, points: Vec<u64>, k: usize) -> Result<Self> {
        Self::new(EvalDomain::from_points(p, points)?, k)
    }

    /// Construct on a multiplicative-subgroup domain (the smooth SNARK case).
    pub fn on_subgroup(sg: &MultiplicativeSubgroup, k: usize) -> Result<Self> {
        Self::new(sg.eval_domain(), k)
    }

    /// The evaluation domain.
    #[must_use]
    pub fn domain(&self) -> &EvalDomain {
        &self.domain
    }
    /// The field characteristic.
    #[must_use]
    pub fn p(&self) -> u64 {
        self.domain.p()
    }
    /// The evaluation points (`n` of them).
    #[must_use]
    pub fn points(&self) -> &[u64] {
        self.domain.points()
    }
    /// Code length `n = |D|`.
    #[must_use]
    pub fn n(&self) -> usize {
        self.domain.n()
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
        let p = self.p();
        Ok(self
            .points()
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

impl ReedSolomon {
    /// Materialize the C.5-form (payoff) word on the domain: the "frozen head"
    /// `f(x) = sum_{i=0}^q (-1)^i lambda_i x^{r-i}` (with `lambda_0 = 1`, `m = 1`).
    ///
    /// This is the *received word* whose list at radius `1 - r/n` is, by the
    /// exactness theorem, exactly the bucket `{S : e_i(S) = lambda_i}` (see the
    /// module intro and [`crate::census::buckets`]). It is the bridge between the
    /// count axis ([`crate::census::buckets`]) and the decode axis
    /// ([`crate::rs::decode`]): decoding this word and counting its bucket must
    /// return the same list size.
    ///
    /// `lambda` holds the `e`-values `(e_1, ..., e_q)`; the alternating signs
    /// are the `c_i = (-1)^i e_i` sign convention institutionalized in
    /// [`crate::census::buckets::mitm`]. Requires `q <= r < n`.
    pub fn c5_word(&self, r: usize, lambda: &[u64]) -> Result<Vec<u64>> {
        let q = lambda.len();
        if r >= self.n() || q > r {
            return Err(Error::OutOfRange("need q <= r < n".into()));
        }
        let p = self.p();
        Ok(self
            .points()
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

pub use crate::field::top_elementary_symmetric;
