//! The decode axis: list decoding of an **arbitrary** word.
//!
//! Everything else in the crate computes list sizes of the *structured*
//! extremal words by *counting* (the bucket = symmetric-function level set,
//! via the exactness theorem). This module is the dual: given any word
//! `w in F_p^n` and a radius, it *decodes* — enumerates the actual codewords
//! within that radius. It is the computational form of the reduction question
//! (`arbitrary word -> bucket`): the only tool that can compare a generic
//! word's list against the structured buckets.
//!
//! ## The seam
//!
//! [`ListOracle`] is the shared contract — "the size of the list of a word."
//! Two implementations meet here:
//! - [`DecodeOracle`]: brute list decoding, valid for *any* word (the new
//!   axis).
//! - [`ExactC5Oracle`]: the count-via-exactness path for a C.5 word (the
//!   existing axis, wrapping [`crate::buckets`]).
//!
//! The **exactness theorem** is then a checkable equality: the two oracles
//! must return the same size on a C.5 word (`tests`). The **reduction defect**
//! ([`reduction_defect`]) is a subtraction across the seam: a generic word's
//! decoded list size minus the largest bucket the same radius can explain.
//!
//! ## Cost
//!
//! The exact engine enumerates `C(n, k)` information sets, so it is a
//! *small-`s` reference oracle* (a codeword with agreement `>= t >= k` is
//! pinned by any `k` of its agreement points; interpolate and re-check). Above
//! the feasible range use [`DecodeOracle::list_size_atleast`] — a Monte-Carlo
//! *lower bound* good enough to certify "the list exceeds a threshold" for the
//! extremal search. Sub-`C(n,k)` completeness (ISD sampling, branch-and-bound
//! pruning) is the documented scaling path; the exact engine stays the ground
//! truth the rest is validated against.

use crate::buckets::mitm::HalfTables;
use crate::code::ReedSolomon;
use crate::error::{Error, Result};
use crate::field::{binom, mulmod, powmod};
use std::collections::HashSet;

/// Beyond this many information sets the exact engine refuses to run; use
/// [`DecodeOracle::list_size_atleast`] instead.
const EXACT_SUBSET_CAP: u64 = 1_000_000_000;

/// A decoding radius, held as a minimum agreement count `t`: the list is every
/// codeword agreeing with the word on at least `t` of the `n` coordinates,
/// i.e. at relative distance `<= 1 - t/n`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Radius {
    min_agreement: usize,
}

impl Radius {
    /// A radius admitting codewords that agree on at least `t` coordinates.
    #[must_use]
    pub fn agreement(t: usize) -> Self {
        Radius { min_agreement: t }
    }

    /// The radius `delta` (relative distance) for code length `n`: agreement
    /// `t = n - floor(delta * n)` (so `dist <= delta` iff `agree >= t`).
    #[must_use]
    pub fn from_delta(n: usize, delta: f64) -> Self {
        let disagree = (delta * n as f64).floor() as usize;
        Radius {
            min_agreement: n.saturating_sub(disagree),
        }
    }

    /// The minimum agreement count `t`.
    #[must_use]
    pub fn min_agreement(&self) -> usize {
        self.min_agreement
    }
}

/// The shared contract of the two axes: the size of the list of a word.
///
/// Deliberately size-only — that is the common denominator. [`DecodeOracle`]
/// additionally exposes the codewords themselves; [`ExactC5Oracle`] can only
/// count.
pub trait ListOracle {
    /// `|List(C, radius, word)|`.
    fn list_size(&self, word: &[u64], radius: Radius) -> Result<u64>;
}

/// Brute list decoder for an arbitrary word (the decode axis).
#[derive(Debug, Clone, Copy)]
pub struct DecodeOracle<'a> {
    rs: &'a ReedSolomon<'a>,
}

impl<'a> DecodeOracle<'a> {
    /// A decoder for the given code.
    #[must_use]
    pub fn new(rs: &'a ReedSolomon<'a>) -> Self {
        DecodeOracle { rs }
    }

    /// The exact list: every codeword (as its evaluation vector on the domain)
    /// within `radius` of `word`, deduplicated.
    ///
    /// Requires `t >= k` (below `k` a codeword is not pinned by a `k`-subset of
    /// its agreements, i.e. the radius is at or beyond capacity). Refuses to
    /// run past [`EXACT_SUBSET_CAP`] information sets.
    pub fn list(&self, word: &[u64], radius: Radius) -> Result<Vec<Vec<u64>>> {
        let n = self.rs.n();
        let k = self.rs.k();
        let t = radius.min_agreement();
        if word.len() != n {
            return Err(Error::OutOfRange("word length != n".into()));
        }
        if t < k {
            return Err(Error::Unsupported(
                "exact decode needs agreement t >= k (radius within capacity)".into(),
            ));
        }
        if binom(n as u64, k as u64) > EXACT_SUBSET_CAP {
            return Err(Error::Unsupported(
                "C(n, k) exceeds the exact cap; use list_size_atleast".into(),
            ));
        }
        let p = self.rs.domain().p();
        let dom = self.rs.domain().elements();
        let mut seen: HashSet<Vec<u64>> = HashSet::new();
        let mut out: Vec<Vec<u64>> = Vec::new();
        for_each_combination(n, k, |idx| {
            let xs: Vec<u64> = idx.iter().map(|&i| dom[i]).collect();
            let ys: Vec<u64> = idx.iter().map(|&i| word[i]).collect();
            let cw = interp_eval_all(&xs, &ys, dom, p);
            let agree = cw.iter().zip(word).filter(|(a, b)| a == b).count();
            if agree >= t && seen.insert(cw.clone()) {
                out.push(cw);
            }
        });
        Ok(out)
    }

    /// Whether the exact engine will run (`C(n, k)` within the cap); if not,
    /// the sampling methods are the way in.
    #[must_use]
    pub fn exact_feasible(&self) -> bool {
        binom(self.rs.n() as u64, self.rs.k() as u64) <= EXACT_SUBSET_CAP
    }

    /// Sample `samples` information sets and return the *distinct* codewords
    /// found at agreement `>= t` — a subset of the true list, and the
    /// recruitment primitive for cluster growth ([`crate::cluster`]).
    /// Deterministic in `seed`.
    pub fn sample_list(
        &self,
        word: &[u64],
        radius: Radius,
        samples: u64,
        seed: u64,
    ) -> Result<Vec<Vec<u64>>> {
        Ok(self
            .sample_codewords(word, radius, samples, seed, None)?
            .into_iter()
            .collect())
    }

    /// A Monte-Carlo *lower bound* on the list size: distinct codewords found
    /// at agreement `>= t` over `samples` random information sets, capped at
    /// `threshold` (early-stops there). Never overcounts; the true list is at
    /// least the returned value. Deterministic in `seed`.
    pub fn list_size_atleast(
        &self,
        word: &[u64],
        radius: Radius,
        threshold: u64,
        samples: u64,
        seed: u64,
    ) -> Result<u64> {
        let set = self.sample_codewords(word, radius, samples, seed, Some(threshold))?;
        Ok((set.len() as u64).min(threshold))
    }

    /// Shared sampling core: distinct codewords at agreement `>= t` over
    /// `samples` information sets, optionally early-stopping at `stop_at`.
    fn sample_codewords(
        &self,
        word: &[u64],
        radius: Radius,
        samples: u64,
        seed: u64,
        stop_at: Option<u64>,
    ) -> Result<HashSet<Vec<u64>>> {
        let n = self.rs.n();
        let k = self.rs.k();
        let t = radius.min_agreement();
        if word.len() != n {
            return Err(Error::OutOfRange("word length != n".into()));
        }
        if t < k {
            return Err(Error::Unsupported("needs agreement t >= k".into()));
        }
        let p = self.rs.domain().p();
        let dom = self.rs.domain().elements();
        let mut rng = SplitMix64::new(seed);
        let mut seen: HashSet<Vec<u64>> = HashSet::new();
        for _ in 0..samples {
            let idx = rng.combination(n, k);
            let xs: Vec<u64> = idx.iter().map(|&i| dom[i]).collect();
            let ys: Vec<u64> = idx.iter().map(|&i| word[i]).collect();
            let cw = interp_eval_all(&xs, &ys, dom, p);
            let agree = cw.iter().zip(word).filter(|(a, b)| a == b).count();
            if agree >= t {
                seen.insert(cw);
                if let Some(cap) = stop_at {
                    if seen.len() as u64 >= cap {
                        break;
                    }
                }
            }
        }
        Ok(seen)
    }
}

impl ListOracle for DecodeOracle<'_> {
    fn list_size(&self, word: &[u64], radius: Radius) -> Result<u64> {
        Ok(self.list(word, radius)?.len() as u64)
    }
}

/// The count-via-exactness path (the existing axis) as a [`ListOracle`]:
/// answers for exactly one C.5 word, by counting its bucket.
///
/// Constructed from `(rs, r, lambda)`; materializes its own C.5 word via
/// [`ReedSolomon::c5_word`]. Its [`ListOracle::list_size`] answers only for
/// that word at radius `agreement(r)` — the regime where the exactness theorem
/// applies — and returns [`Error::Unsupported`] otherwise. This is what makes
/// the two axes comparable through one interface.
#[derive(Debug, Clone)]
pub struct ExactC5Oracle<'a> {
    rs: &'a ReedSolomon<'a>,
    r: usize,
    lambda: Vec<u64>,
    word: Vec<u64>,
}

impl<'a> ExactC5Oracle<'a> {
    /// The exact oracle for the C.5 word at `(r, lambda)` (`q = lambda.len()`).
    pub fn new(rs: &'a ReedSolomon<'a>, r: usize, lambda: &[u64]) -> Result<Self> {
        let word = rs.c5_word(r, lambda)?;
        Ok(ExactC5Oracle {
            rs,
            r,
            lambda: lambda.to_vec(),
            word,
        })
    }

    /// The materialized C.5 word (feed this to a [`DecodeOracle`] to cross the
    /// seam).
    #[must_use]
    pub fn word(&self) -> &[u64] {
        &self.word
    }
}

impl ListOracle for ExactC5Oracle<'_> {
    fn list_size(&self, word: &[u64], radius: Radius) -> Result<u64> {
        if word != self.word.as_slice() {
            return Err(Error::Unsupported(
                "ExactC5Oracle answers only for its own C.5 word".into(),
            ));
        }
        if radius.min_agreement() != self.r {
            return Err(Error::Unsupported(
                "exactness holds at radius agreement(r) = 1 - r/n".into(),
            ));
        }
        // Equality (not just the >= payoff bound) requires exactly q symmetric
        // constraints on the r-subsets, i.e. k = r - q. Away from it the list
        // is strictly larger (k > r - q) or smaller (k < r - q) than the bucket.
        if self.rs.k() + self.lambda.len() != self.r {
            return Err(Error::Unsupported(
                "exactness (list == bucket) holds only at k = r - q".into(),
            ));
        }
        let tables = HalfTables::build(self.rs.domain(), self.r, self.lambda.len())?;
        tables.bucket(&self.lambda)
    }
}

/// The reduction defect at a generic word (`q = 1`): its decoded list size at
/// radius `1 - r/n`, minus the largest `q = 1` bucket at `(s, r)`.
///
/// `> 0` means the word out-lists *every* structured word at this radius — a
/// violation of bucket-extremality, i.e. a candidate new attack. `<= 0` is
/// consistent with buckets being the worst case. This is D4 of the data
/// program, expressed as one subtraction across the seam.
pub fn reduction_defect(rs: &ReedSolomon, word: &[u64], r: usize) -> Result<i128> {
    let decoded = DecodeOracle::new(rs).list_size(word, Radius::agreement(r))? as i128;
    let (best_bucket, _) = crate::buckets::dp::distribution_q1(rs.domain(), r)?.max();
    Ok(decoded - best_bucket as i128)
}

// ---- interpolation, enumeration, rng ------------------------------------

/// `a^{-1} mod p` via Fermat (`p` prime).
#[inline]
fn inv(a: u64, p: u64) -> u64 {
    powmod(a, p - 2, p)
}

/// Interpolate the unique degree-`< k` polynomial through the `k` nodes
/// `(xs, ys)` and evaluate it at every point of `domain`, via the barycentric
/// form (no monomial-coefficient conversion — the identity
/// `sum_j w_j prod_{m != j}(X - x_m) = 1` keeps the denominator nonzero off the
/// nodes).
fn interp_eval_all(xs: &[u64], ys: &[u64], domain: &[u64], p: u64) -> Vec<u64> {
    let k = xs.len();
    let mut wts = vec![0u64; k];
    for j in 0..k {
        let mut d = 1u64;
        for m in 0..k {
            if m != j {
                d = mulmod(d, (xs[j] + p - xs[m]) % p, p);
            }
        }
        wts[j] = inv(d, p);
    }
    domain
        .iter()
        .map(|&x| {
            if let Some(j) = xs.iter().position(|&xj| xj == x) {
                return ys[j];
            }
            let (mut num, mut den) = (0u64, 0u64);
            for j in 0..k {
                let term = mulmod(wts[j], inv((x + p - xs[j]) % p, p), p);
                num = (num + mulmod(term, ys[j], p)) % p;
                den = (den + term) % p;
            }
            mulmod(num, inv(den, p), p)
        })
        .collect()
}

/// Call `f` on each `k`-subset of `0..n`, as a sorted index slice.
fn for_each_combination(n: usize, k: usize, mut f: impl FnMut(&[usize])) {
    if k > n {
        return;
    }
    let mut c: Vec<usize> = (0..k).collect();
    loop {
        f(&c);
        let mut i = k as isize - 1;
        while i >= 0 && c[i as usize] == i as usize + n - k {
            i -= 1;
        }
        if i < 0 {
            break;
        }
        let i = i as usize;
        c[i] += 1;
        for j in i + 1..k {
            c[j] = c[j - 1] + 1;
        }
    }
}

/// Tiny deterministic PRNG (SplitMix64) — avoids a `rand` dependency and keeps
/// the Monte-Carlo lower bound reproducible.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// A uniform `k`-subset of `0..n` (partial Fisher–Yates), returned sorted.
    fn combination(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut pool: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = i + (self.next_u64() as usize) % (n - i);
            pool.swap(i, j);
        }
        let mut idx = pool[..k].to_vec();
        idx.sort_unstable();
        idx
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::Subgroup;

    /// The exactness bridge: at `p = 65537` (accident-free at `s = 16`, since
    /// any `{-1,0,1}` accident needs `p <= w^{s/4} <= 8^4 = 4096`), the C.5
    /// word `f = x^8` (`r = 8`, `q = 1`, `lambda = 0`) has list size exactly
    /// the zero bucket `C(8, 4) = 70` at radius `1 - 8/16`. The decode axis and
    /// the count axis must agree on it.
    #[test]
    fn exactness_bridge_s16() {
        let sg = Subgroup::new(65537, 16).unwrap();
        let rs = ReedSolomon::new(&sg, 7).unwrap();
        let exact = ExactC5Oracle::new(&rs, 8, &[0]).unwrap();
        let f = exact.word().to_vec();
        let rad = Radius::agreement(8);

        let counted = exact.list_size(&f, rad).unwrap();
        let decoded = DecodeOracle::new(&rs).list_size(&f, rad).unwrap();

        assert_eq!(counted, 70, "structural zero bucket C(8,4)");
        assert_eq!(decoded, counted, "exactness theorem: decode == count");
    }

    /// At the max-bucket word the reduction defect is zero: the C.5 word for
    /// `lambda = 0` realizes the largest `q = 1` bucket, so decode == best
    /// bucket.
    #[test]
    fn defect_zero_at_max_word() {
        let sg = Subgroup::new(65537, 16).unwrap();
        let rs = ReedSolomon::new(&sg, 7).unwrap();
        let f = rs.c5_word(8, &[0]).unwrap();
        assert_eq!(reduction_defect(&rs, &f, 8).unwrap(), 0);
    }

    /// The Monte-Carlo lower bound never exceeds the exact list size and, given
    /// enough samples, reaches it.
    #[test]
    fn sampling_is_a_lower_bound() {
        let sg = Subgroup::new(65537, 16).unwrap();
        let rs = ReedSolomon::new(&sg, 7).unwrap();
        let f = rs.c5_word(8, &[0]).unwrap();
        let rad = Radius::agreement(8);
        let exact = DecodeOracle::new(&rs).list_size(&f, rad).unwrap();
        let lb = DecodeOracle::new(&rs)
            .list_size_atleast(&f, rad, exact + 5, 200_000, 1)
            .unwrap();
        assert!(lb <= exact, "lower bound {lb} exceeded exact {exact}");
        assert_eq!(lb, exact, "200k samples should saturate a 70-codeword list");
    }
}
