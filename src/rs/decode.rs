//! The decode axis: list decoding of an **arbitrary** word.
//!
//! Everything else in the crate computes list sizes of the *structured*
//! extremal words by *counting* (the bucket = symmetric-function level set,
//! via the exactness theorem). This module is the dual: given any word
//! `w in F_p^n` and a radius, it *decodes* — enumerates the actual codewords
//! within that radius. It is the measurement arm of the arbitrary-word ->
//! bucket comparison; the analytic arm (the syndrome dictionary and its
//! descent) lives in the notes, and this decoder is what checks it against
//! ground truth.
//!
//! Generic over the evaluation domain — nothing here depends on the
//! multiplicative-subgroup structure; it decodes `RS[F_p, D, k]` for any domain
//! `D`. The exactness bridge to [`crate::census::buckets`] (decode a C.5 word, check its
//! size equals the counted bucket) is validated in the tests, on a subgroup
//! domain where buckets are defined.
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

use crate::error::{Error, Result};
use crate::field::{batch_inv, checked_binom, mulmod};
use crate::rs::code::ReedSolomon;
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

/// The contract "the size of the list of a word." Deliberately size-only, so a
/// count-only path (e.g. the bucket count in the tests) can meet a full decoder
/// through one interface; [`DecodeOracle`] additionally exposes the codewords.
pub trait ListOracle {
    /// `|List(C, radius, word)|`.
    fn list_size(&self, word: &[u64], radius: Radius) -> Result<u64>;
}

/// Brute list decoder for an arbitrary word (the decode axis).
#[derive(Debug, Clone, Copy)]
pub struct DecodeOracle<'a> {
    rs: &'a ReedSolomon,
}

impl<'a> DecodeOracle<'a> {
    /// A decoder for the given code.
    #[must_use]
    pub fn new(rs: &'a ReedSolomon) -> Self {
        DecodeOracle { rs }
    }

    /// The exact list: every codeword (as its evaluation vector on the domain)
    /// within `radius` of `word`, deduplicated.
    ///
    /// Requires `t >= k` (below `k` a codeword is not pinned by a `k`-subset of
    /// its agreements, i.e. the radius is at or beyond capacity). Refuses to
    /// run past `EXACT_SUBSET_CAP` information sets.
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
        if !checked_binom(n as u64, k as u64).is_some_and(|c| c <= EXACT_SUBSET_CAP) {
            return Err(Error::Unsupported(
                "C(n, k) exceeds the exact cap; use list_size_atleast".into(),
            ));
        }
        let p = self.rs.p();
        let dom = self.rs.points();
        // Parallelize over the smallest element of the information set: the
        // branches partition the combinations, and merging them in `i0` order
        // reproduces the serial lex enumeration exactly, so the output (set
        // AND order) is identical to the sequential version.
        use rayon::prelude::*;
        let branches: Vec<Vec<Vec<u64>>> = (0..=n - k)
            .into_par_iter()
            .map(|i0| {
                let mut xs = vec![0u64; k];
                let mut ys = vec![0u64; k];
                let mut local_seen: HashSet<Vec<u64>> = HashSet::new();
                let mut local_out: Vec<Vec<u64>> = Vec::new();
                for_each_combination(n - i0 - 1, k - 1, |rest| {
                    xs[0] = dom[i0];
                    ys[0] = word[i0];
                    for (slot, &r) in rest.iter().enumerate() {
                        let i = i0 + 1 + r;
                        xs[slot + 1] = dom[i];
                        ys[slot + 1] = word[i];
                    }
                    let cw = interp_eval_all(&xs, &ys, dom, p);
                    let agree = cw.iter().zip(word).filter(|(a, b)| a == b).count();
                    if agree >= t && !local_seen.contains(&cw) {
                        local_seen.insert(cw.clone());
                        local_out.push(cw);
                    }
                });
                local_out
            })
            .collect();
        let mut seen: HashSet<Vec<u64>> = HashSet::new();
        let mut out: Vec<Vec<u64>> = Vec::new();
        for branch in branches {
            for cw in branch {
                if !seen.contains(&cw) {
                    seen.insert(cw.clone());
                    out.push(cw);
                }
            }
        }
        Ok(out)
    }

    /// Whether the exact engine will run (`C(n, k)` within the cap); if not,
    /// the sampling methods are the way in.
    #[must_use]
    pub fn exact_feasible(&self) -> bool {
        checked_binom(self.rs.n() as u64, self.rs.k() as u64).is_some_and(|c| c <= EXACT_SUBSET_CAP)
    }

    /// Sample `samples` information sets and return the *distinct* codewords
    /// found at agreement `>= t` — a subset of the true list, and the
    /// recruitment primitive for cluster growth ([`crate::rs::cluster`]).
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
        let p = self.rs.p();
        let dom = self.rs.points();
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

// The exactness bridge (decode == bucket for a C.5 word) and the reduction
// defect are subgroup/bucket-specific, so they are not part of the generic
// decode API: the bridge is validated in this module's tests, and the defect
// (D4) is computed in the Python discovery layer via `list_decode` + the bucket
// bindings. Keeping them out leaves `decode` generic over the domain.

// ---- interpolation, enumeration, rng ------------------------------------

/// Interpolate the unique degree-`< k` polynomial through the `k` nodes
/// `(xs, ys)` and evaluate it at every point of `domain`, via the barycentric
/// form (no monomial-coefficient conversion — the identity
/// `sum_j w_j prod_{m != j}(X - x_m) = 1` keeps the denominator nonzero off the
/// nodes). All inversions are batched (two Fermat exponentiations per call, not
/// `~(n-k)(k+1)`): this is the hot kernel under every decode/search flip.
/// Shared with [`crate::rs::cluster`] for building codewords from a pencil.
pub(crate) fn interp_eval_all(xs: &[u64], ys: &[u64], domain: &[u64], p: u64) -> Vec<u64> {
    let k = xs.len();
    let n = domain.len();
    // Node index of each domain point (or usize::MAX), and the flattened
    // nonzero differences to invert: k weight denominators, then k diffs
    // `x - x_j` for each non-node point.
    let node_of: Vec<usize> = domain
        .iter()
        .map(|&x| xs.iter().position(|&xj| xj == x).unwrap_or(usize::MAX))
        .collect();
    let n_off = node_of.iter().filter(|&&j| j == usize::MAX).count();
    let mut to_inv = Vec::with_capacity(k + n_off * k);
    for j in 0..k {
        let mut d = 1u64;
        for m in 0..k {
            if m != j {
                d = mulmod(d, (xs[j] + p - xs[m]) % p, p);
            }
        }
        to_inv.push(d);
    }
    for (&x, &nd) in domain.iter().zip(&node_of) {
        if nd == usize::MAX {
            for &xj in xs {
                to_inv.push((x + p - xj) % p);
            }
        }
    }
    batch_inv(&mut to_inv, p);
    let (wts, diffs) = to_inv.split_at(k);
    // First pass: numerators and denominators; collect denominators for the
    // second (tiny) inversion batch.
    let mut nums = vec![0u64; n];
    let mut dens = Vec::with_capacity(n_off);
    let mut off = 0usize;
    for (i, (&_x, &nd)) in domain.iter().zip(&node_of).enumerate() {
        if nd != usize::MAX {
            nums[i] = ys[nd];
        } else {
            let row = &diffs[off * k..(off + 1) * k];
            let (mut num, mut den) = (0u64, 0u64);
            for j in 0..k {
                let term = mulmod(wts[j], row[j], p);
                num = (num + mulmod(term, ys[j], p)) % p;
                den = (den + term) % p;
            }
            nums[i] = num;
            dens.push(den);
            off += 1;
        }
    }
    batch_inv(&mut dens, p);
    let mut out = nums;
    let mut off = 0usize;
    for (i, &nd) in node_of.iter().enumerate() {
        if nd == usize::MAX {
            out[i] = mulmod(out[i], dens[off], p);
            off += 1;
        }
    }
    out
}

/// Call `f` on each `k`-subset of `0..n`, as a sorted index slice.
pub(crate) fn for_each_combination(n: usize, k: usize, mut f: impl FnMut(&[usize])) {
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
    use crate::domain::MultiplicativeSubgroup;

    /// The exactness bridge: at `p = 65537` (accident-free at `s = 16`, since
    /// any `{-1,0,1}` accident needs `p <= w^{s/4} <= 8^4 = 4096`), the C.5
    /// word `f = x^8` (`r = 8`, `q = 1`, `lambda = 0`) has list size exactly
    /// the zero bucket `C(8, 4) = 70` at radius `1 - 8/16`. The decode axis and
    /// the count axis must agree on it.
    #[test]
    fn exactness_bridge_s16() {
        use crate::census::buckets::mitm::HalfTables;
        let sg = MultiplicativeSubgroup::new(65537, 16).unwrap();
        let rs = ReedSolomon::on_subgroup(&sg, 7).unwrap(); // k = r - q = 7
        let f = rs.c5_word(8, &[0]).unwrap();
        let decoded = DecodeOracle::new(&rs)
            .list_size(&f, Radius::agreement(8))
            .unwrap();
        let counted = HalfTables::build(&sg, 8, 1).unwrap().bucket(&[0]).unwrap();
        assert_eq!(counted, 70, "structural zero bucket C(8,4)");
        assert_eq!(decoded, counted, "exactness: decode == counted bucket");
    }

    /// The decoder is generic over the domain: it works on an arbitrary
    /// distinct-point domain that is not a multiplicative subgroup, and rejects
    /// duplicate points.
    #[test]
    fn decodes_on_a_generic_domain() {
        let p = 65537;
        let pts: Vec<u64> = (3u64..15).map(|x| x * x % p).collect(); // 12 non-subgroup points
        let rs = ReedSolomon::on_domain(p, pts, 5).unwrap();
        let cw = rs.encode(&[1, 2, 3, 4, 5]).unwrap();
        let list = DecodeOracle::new(&rs)
            .list(&cw, Radius::agreement(6))
            .unwrap();
        assert_eq!(
            list.len(),
            1,
            "only the codeword itself agrees on >= 6 points"
        );
        assert_eq!(list[0], cw);
        assert!(ReedSolomon::on_domain(p, vec![1, 1, 2, 3, 4], 3).is_err());
    }

    /// An oversized instance must produce a clean `Unsupported`, not a
    /// process-aborting overflow panic in the cap check: C(128, 64) > u64.
    #[test]
    fn oversized_cap_check_errs_instead_of_panicking() {
        let pts: Vec<u64> = (1..=128).collect();
        let rs = ReedSolomon::on_domain(65537, pts, 64).unwrap();
        let oracle = DecodeOracle::new(&rs);
        assert!(!oracle.exact_feasible());
        let w = vec![0u64; 128];
        assert!(oracle.list(&w, Radius::agreement(70)).is_err());
    }

    /// Deduplication and output determinism survive the parallel enumeration:
    /// a distance-1 word's single codeword (agreement 15) is discovered from
    /// information sets in many branches, and must appear exactly once.
    #[test]
    fn dedup_survives_parallel_enumeration() {
        use std::collections::HashSet;
        let sg = MultiplicativeSubgroup::new(65537, 16).unwrap();
        let rs = ReedSolomon::on_subgroup(&sg, 7).unwrap();
        let cw = rs.encode(&[3, 1, 4, 1, 5, 9, 2]).unwrap();
        let mut w = cw.clone();
        w[0] = (w[0] + 1) % 65537;
        let list = DecodeOracle::new(&rs)
            .list(&w, Radius::agreement(8))
            .unwrap();
        assert!(
            list.contains(&cw),
            "the agreement-15 codeword is in the list"
        );
        let uniq: HashSet<_> = list.iter().cloned().collect();
        assert_eq!(uniq.len(), list.len(), "no duplicates across branches");
    }

    /// The Monte-Carlo lower bound never exceeds the exact list size and, given
    /// enough samples, reaches it.
    #[test]
    fn sampling_is_a_lower_bound() {
        let sg = MultiplicativeSubgroup::new(65537, 16).unwrap();
        let rs = ReedSolomon::on_subgroup(&sg, 7).unwrap();
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
