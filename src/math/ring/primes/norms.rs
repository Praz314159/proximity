//! Cyclotomic norms and the accident inventory — the arithmetic half of
//! the shadow view.
//!
//! The program's domains are shadows: at a prime `p = 1 (mod s)` the
//! ring `Z[zeta_s]` splits completely, and reduction projects its unit
//! circle onto the order-`s` subgroup of `F_p^*`. Everything the census
//! measures is fiber statistics of that projection. An *accident* is an
//! extra zero of the shadow — a small vector `v` with `p | N(v)`, hence
//! in the kernel of some embedding `Z[zeta_s] -> F_p` — collapsing
//! values the generic picture keeps apart. Divisibility of bounded-height
//! norms is therefore the entire criterion, and this module runs it in
//! both directions: forward from vectors to the primes they can visit,
//! and inverted to the per-prime inventory. The largest prime in the
//! inventory is the level's die-out scale — above it, no vector of the
//! enumerated height class misbehaves anywhere.
//!
//! Pipeline: enumerate coefficient vectors (weight-capped, entries in
//! `[-cmax, cmax]`) -> exact norms `N(v) = prod_k v(zeta^k)` -> factor the
//! (heavily degenerate) unique norm values -> invert to the *bad set*: every
//! prime `p = 1 mod s` that any weight-`w` kernel vector can ever visit, with
//! Galois-normalized per-weight counts. The submodules carry the
//! inventory beyond counting: [`events`] holds one row per (prime,
//! orbit) incidence with the witness vector in hand, and [`ingest`]
//! streams externally computed tables (the `s = 64` arm) through the
//! same accumulation skeleton.
//!
//! Exactness: norms are computed by CRT over 61-bit split primes (never
//! floating point — at `s = 64` norms exceed the f64 mantissa). The Parseval
//! law caps `N(v) <= (sum v_i^2)^{s/4}`, which sizes the CRT.
//!
//! Ring conventions are owned by [`crate::ring`]: the enumeration stays flat
//! (no per-vector allocation; see the [`crate::ring`] docs), but every
//! embedding-exponent reduction routes through [`crate::ring::fold`], and the
//! glue test pins the flat CRT loop against the independent
//! [`crate::ring::Cyclo::norm_i128`] path entry-for-entry.
//!
//! Galois normalization (two historical pitfalls, now built in): "p | N(v)"
//! means `v` is in the kernel of *some* embedding, so raw divisibility counts
//! overstate the single-embedding census by the factor `s/2`; the fix divides
//! valuation-weighted counts by `s/2` (per-weight censuses are
//! Galois-invariant). When `p^2` divides some norm the equal split can break
//! (multiplicity may sit inside one embedding), so such primes are flagged and
//! their counts replaced by a direct census where available.

pub mod events;
pub mod ingest;

use crate::census::kernel as census;
use crate::domain::MultiplicativeSubgroup;
use crate::error::{Error, Result};
use crate::field::{factor, mulmod, powmod, primes_one_mod};
use crate::ring::fold;
use rayon::prelude::*;
use std::collections::HashMap;

/// Unique norm values with per-weight vector counts.
#[derive(Debug, Clone)]
pub struct NormTable {
    /// MultiplicativeSubgroup order the table was built for.
    pub s: usize,
    /// Weight cap.
    pub wmax: usize,
    /// Coefficient bound.
    pub cmax: i64,
    /// norm value -> counts\[w\] = number of weight-`w` vectors with that norm.
    pub entries: HashMap<u128, Vec<u64>>,
}

impl NormTable {
    /// Largest norm at each weight (the anticorrelation profile).
    #[must_use]
    pub fn n_max_by_weight(&self) -> Vec<u128> {
        let mut out = vec![0u128; self.wmax + 1];
        for (&n, counts) in &self.entries {
            for (w, &c) in counts.iter().enumerate() {
                if c > 0 && n > out[w] {
                    out[w] = n;
                }
            }
        }
        out
    }
}

/// How a bad-set row's per-weight counts were obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provenance {
    /// The equal valuation split was safe (no `p^2 | N(v)`): counts are exact
    /// from the norm table alone.
    ValuationSplit,
    /// The split was unsafe and the counts were replaced by a direct census
    /// ([`crate::census::kernel::mitm`]): exact, by construction.
    CensusCorrected,
    /// The split was unsafe and no census was feasible: counts are
    /// approximate and must be treated as such downstream.
    UnsafeSplit,
}

impl Provenance {
    /// Whether the counts are exact (split-safe or census-corrected).
    #[must_use]
    pub fn is_exact(self) -> bool {
        !matches!(self, Provenance::UnsafeSplit)
    }
}

/// One bad-set row: a prime `p = 1 mod s` with Galois-normalized per-weight
/// kernel-vector counts (index = weight).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BadSetEntry {
    /// The prime.
    pub p: u64,
    /// Per-weight counts.
    pub counts: Vec<u64>,
    /// How the counts were obtained.
    pub provenance: Provenance,
}

/// Factor `n`, group prime multiplicities, and hand each *bad* prime —
/// `p > s`, `p = 1 (mod s)` — with its valuation `e` to the sink. No-op for
/// `n <= 1`. The single accumulation skeleton shared by [`bad_set`] and the
/// GPU-table ingest ([`ingest`]); the two pipelines must factor identically
/// for their equivalence tests to be meaningful.
pub(crate) fn for_each_bad_prime(n: u64, s: u64, mut sink: impl FnMut(u64, u64)) {
    if n <= 1 {
        return;
    }
    let fs = factor(n);
    let mut i = 0;
    while i < fs.len() {
        let p = fs[i];
        let mut e = 0u64;
        while i < fs.len() && fs[i] == p {
            e += 1;
            i += 1;
        }
        if p > s && (p - 1) % s == 0 {
            sink(p, e);
        }
    }
}

fn crt_primes(s: usize, bound_bits: u32) -> Result<Vec<u64>> {
    let n_primes = bound_bits.div_ceil(60) as usize;
    if n_primes > 2 {
        return Err(Error::Unsupported(
            "norms beyond ~2^120 not supported (raise via wider CRT)".into(),
        ));
    }
    Ok(primes_one_mod(s as u64, 1 << 61).take(n_primes).collect())
}

/// The exact-norm kernel shared by [`norm_table`] and the event paths
/// ([`events`], [`ingest`]): CRT tables over 61-bit split primes sized by
/// the Parseval cap, the embedding folds of a support, and the norm of one
/// coefficient pattern on those folds. One kernel, so every consumer's
/// norms agree by construction rather than by parallel maintenance.
pub(crate) struct NormEngine {
    s: usize,
    // per CRT prime: order-s root's half-basis power table (the images of
    // `{1 .. zeta^{s/2-1}}`; exponents outside the half-basis are placed
    // by [`crate::ring::fold`])
    tables: Vec<(u64, Vec<u64>)>,
    // hoisted CRT inverse for the two-prime path
    crt_inv: Option<u64>,
}

impl NormEngine {
    /// Build the CRT schedule for norms of weight-`<= wmax` vectors with
    /// entries in `[-cmax, cmax]`. Parseval caps the norm at
    /// `(cmax^2 * wmax)^{s/4}`, which sizes the schedule.
    pub(crate) fn new(s: usize, wmax: usize, cmax: i64) -> Result<Self> {
        let half = s / 2;
        let bound_bits =
            ((s as f64 / 4.0) * ((cmax * cmax) as f64 * wmax as f64).log2()).ceil() as u32 + 2;
        let qs = crt_primes(s, bound_bits)?;
        let tables: Vec<(u64, Vec<u64>)> = qs
            .iter()
            .map(|&q| {
                let sg = MultiplicativeSubgroup::new(q, s)
                    .expect("schedule primes satisfy q = 1 (mod s) by construction");
                (q, sg.pow_table(half))
            })
            .collect();
        let crt_inv = (tables.len() == 2)
            .then(|| powmod(tables[0].0 % tables[1].0, tables[1].0 - 2, tables[1].0));
        Ok(NormEngine { s, tables, crt_inv })
    }

    /// Embedding-exponent folds of one support, hoisted out of the pattern
    /// loop: `folds[j][i]` = (index, sign) of `zeta^{sup_i k_j}` on the
    /// half-basis, `k_j` the j-th odd exponent — [`crate::ring::fold`] is
    /// the one authority for this reduction (conventions: [`crate::ring`]).
    pub(crate) fn folds(&self, sup: &[u8]) -> Vec<Vec<(usize, i64)>> {
        let half = self.s / 2;
        (1..self.s)
            .step_by(2)
            .map(|k| sup.iter().map(|&si| fold(half, si as usize * k)).collect())
            .collect()
    }

    /// Exact norm of the coefficient pattern `cvec` on pre-folded support
    /// embeddings, by CRT over the schedule.
    pub(crate) fn norm(&self, folds: &[Vec<(usize, i64)>], cvec: &[i64]) -> u128 {
        let mut residues = [0u64; 2];
        for (ti, (q, pw)) in self.tables.iter().enumerate() {
            let mut prod: u64 = 1;
            for fk in folds {
                // Lazy accumulation: terms |c|*pw < 2^64 and at most 32 of
                // them fit a u128 sum, so the mod drops from per-term to
                // per-embedding.
                let mut acc: u128 = 0;
                for (&(idx, sgn), &cv) in fk.iter().zip(cvec.iter()) {
                    let c = cv * sgn;
                    acc += if c >= 0 {
                        (c as u128) * (pw[idx] as u128)
                    } else {
                        ((-c) as u128) * ((q - pw[idx]) as u128)
                    };
                }
                prod = mulmod(prod, (acc % (*q as u128)) as u64, *q);
            }
            residues[ti] = prod;
        }
        if self.tables.len() == 1 {
            residues[0] as u128
        } else {
            // CRT for two primes
            let (q1, q2) = (self.tables[0].0, self.tables[1].0);
            let inv = self
                .crt_inv
                .expect("crt_inv precomputed whenever the schedule has two primes");
            let diff = (residues[1] + q2 - residues[0] % q2) % q2;
            residues[0] as u128 + (q1 as u128) * (mulmod(diff, inv, q2) as u128)
        }
    }
}

/// Decode enumeration pattern `pat` into its `w` nonzero coefficients —
/// the one pattern/coefficient convention shared by [`norm_table`] and
/// [`events::accident_events`], so the occupancy certificate compares
/// two sweeps of the same enumeration, not two conventions.
#[inline]
pub(crate) fn decode_pattern(pat: u64, coefs: &[i64], w: usize, cvec: &mut [i64; 32]) {
    let ncoef = coefs.len() as u64;
    let mut t = pat;
    for slot in cvec.iter_mut().take(w) {
        *slot = coefs[(t % ncoef) as usize];
        t /= ncoef;
    }
}

pub(crate) fn combinations(n: usize, k: usize) -> Vec<Vec<u8>> {
    let mut out = Vec::new();
    let mut idx: Vec<u8> = (0..k as u8).collect();
    if k == 0 || k > n {
        return out;
    }
    loop {
        out.push(idx.clone());
        // next combination
        let mut i = k as i64 - 1;
        while i >= 0 && idx[i as usize] as usize == n - k + i as usize {
            i -= 1;
        }
        if i < 0 {
            break;
        }
        idx[i as usize] += 1;
        for j in (i as usize + 1)..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
    out
}

/// Exact norm table for all vectors of weight `1..=wmax` with entries in
/// `[-cmax, cmax] \ {0}` on the half-basis of `Z[zeta_s]`.
pub fn norm_table(s: usize, wmax: usize, cmax: i64) -> Result<NormTable> {
    if !s.is_power_of_two() || s < 4 {
        return Err(Error::Unsupported(
            "norms require power-of-two s >= 4".into(),
        ));
    }
    let half = s / 2;
    if wmax == 0 || wmax > half || !(1..=4).contains(&cmax) {
        return Err(Error::OutOfRange(
            "need 1 <= wmax <= s/2, cmax in [1,4]".into(),
        ));
    }
    let engine = NormEngine::new(s, wmax, cmax)?;
    let coefs: Vec<i64> = (-cmax..=cmax).filter(|&c| c != 0).collect();
    let ncoef = coefs.len();

    let mut entries: HashMap<u128, Vec<u64>> = HashMap::new();
    for w in 1..=wmax {
        let supports = combinations(half, w);
        let npat: u64 = (ncoef as u64).pow(w as u32);
        let partial: Vec<HashMap<u128, u64>> = supports
            .par_chunks(1.max(supports.len() / 64))
            .map(|chunk| {
                let mut local: HashMap<u128, u64> = HashMap::new();
                for sup in chunk {
                    let folds = engine.folds(sup);
                    for pat in 0..npat {
                        let mut cvec = [0i64; 32];
                        decode_pattern(pat, &coefs, w, &mut cvec);
                        let n = engine.norm(&folds, &cvec);
                        *local.entry(n).or_insert(0) += 1;
                    }
                }
                local
            })
            .collect();
        for m in partial {
            for (n, c) in m {
                let e = entries.entry(n).or_insert_with(|| vec![0; wmax + 1]);
                e[w] += c;
            }
        }
    }
    Ok(NormTable {
        s,
        wmax,
        cmax,
        entries,
    })
}

/// The complete bad set for weights `<= wmax`, coefficients in `[-cmax, cmax]`:
/// every prime `p = 1 mod s`, `p > s`, dividing any enumerated norm, with
/// Galois-normalized per-weight kernel-vector counts.
pub fn bad_set(s: usize, wmax: usize, cmax: i64) -> Result<Vec<BadSetEntry>> {
    bad_set_from_table(&norm_table(s, wmax, cmax)?)
}

/// [`bad_set`] from an already-built table — the campaign shape, where
/// one enumeration serves both the bad set and
/// [`events::accident_events`].
pub fn bad_set_from_table(table: &NormTable) -> Result<Vec<BadSetEntry>> {
    let (s, wmax, cmax) = (table.s, table.wmax, table.cmax);
    let half = (s / 2) as u64;
    let mut raw: HashMap<u64, (Vec<u64>, bool)> = HashMap::new();
    for (&n, counts) in &table.entries {
        if n <= 1 {
            continue;
        }
        let n64 = u64::try_from(n).map_err(|_| {
            Error::Unsupported("factoring norms above 2^64 not yet supported".into())
        })?;
        for_each_bad_prime(n64, s as u64, |p, e| {
            let entry = raw.entry(p).or_insert_with(|| (vec![0; wmax + 1], false));
            for (w, &c) in counts.iter().enumerate() {
                entry.0[w] += e * c;
            }
            if e >= 2 {
                entry.1 = true; // p^2 divides a norm: valuation split unsafe
            }
        });
    }
    let mut out: Vec<BadSetEntry> = raw
        .into_par_iter()
        .map(|(p, (val_counts, pp))| {
            if pp && s <= 32 {
                // direct census: exact per-embedding counts by construction
                let sg = MultiplicativeSubgroup::new(p, s)
                    .expect("bad-set primes satisfy p = 1 (mod s) by construction");
                let c = census::mitm(&sg, cmax)
                    .expect("cmax validated at entry; census fallback cannot reject it");
                let mut counts = vec![0u64; wmax + 1];
                for (w, slot) in counts.iter_mut().enumerate() {
                    *slot = *c.get(w).unwrap_or(&0);
                }
                BadSetEntry {
                    p,
                    counts,
                    provenance: Provenance::CensusCorrected,
                }
            } else {
                let counts = val_counts.iter().map(|&v| v / half).collect();
                BadSetEntry {
                    p,
                    counts,
                    // pp at s > 32: the split is unsafe and no census is
                    // feasible -- honest flagging of the latent state
                    provenance: if pp {
                        Provenance::UnsafeSplit
                    } else {
                        Provenance::ValuationSplit
                    },
                }
            }
        })
        .collect();
    out.sort_by_key(|e| e.p);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::Cyclo;
    use std::collections::HashMap;

    /// Glue (the [`crate::ring`] division of labor): the flat CRT hot loop of
    /// [`norm_table`] equals the ring's exact Bareiss-determinant norm
    /// [`Cyclo::norm_i128`] entry-for-entry on the full `s = 8`
    /// enumeration — two independent `Z[zeta_s]` norm paths, one answer.
    #[test]
    fn norm_table_matches_cyclo_exact_norms() {
        let (s, wmax, cmax) = (8usize, 4usize, 2i64);
        let table = norm_table(s, wmax, cmax).unwrap();
        let mut expected: HashMap<u128, Vec<u64>> = HashMap::new();
        let ncoef = (2 * cmax + 1) as u64;
        for pat in 0..ncoef.pow(4) {
            let mut v = vec![0i64; 4];
            let mut t = pat;
            for slot in v.iter_mut() {
                *slot = (t % ncoef) as i64 - cmax;
                t /= ncoef;
            }
            let w = v.iter().filter(|&&c| c != 0).count();
            if w == 0 {
                continue;
            }
            let n = Cyclo::from_coeffs(v).unwrap().norm_i128().unwrap();
            assert!(n >= 0, "field norm is nonnegative for s >= 8");
            expected
                .entry(n as u128)
                .or_insert_with(|| vec![0; wmax + 1])[w] += 1;
        }
        assert_eq!(table.entries, expected);
    }
}
