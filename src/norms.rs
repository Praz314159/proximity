//! Cyclotomic norms and bad-set enumeration — the machinery that closes
//! middle-regime instances by exhaustion.
//!
//! Pipeline: enumerate coefficient vectors (weight-capped, entries in
//! `[-cmax, cmax]`) -> exact norms `N(v) = prod_k v(zeta^k)` -> factor the
//! (heavily degenerate) unique norm values -> invert to the *bad set*: every
//! prime `p = 1 mod s` that any weight-`w` kernel vector can ever visit, with
//! Galois-normalized per-weight counts.
//!
//! Exactness: norms are computed by CRT over 61-bit split primes (never
//! floating point — at `s = 64` norms exceed the f64 mantissa). The Parseval
//! law caps `N(v) <= (sum v_i^2)^{s/4}`, which sizes the CRT.
//!
//! Galois normalization (two historical pitfalls, now built in): "p | N(v)"
//! means `v` is in the kernel of *some* embedding, so raw divisibility counts
//! overstate the single-embedding census by the factor `s/2`; the fix divides
//! valuation-weighted counts by `s/2` (per-weight censuses are
//! Galois-invariant). When `p^2` divides some norm the equal split can break
//! (multiplicity may sit inside one embedding), so such primes are flagged and
//! their counts replaced by a direct census where available.

use crate::census;
use crate::domain::Subgroup;
use crate::error::{Error, Result};
use crate::field::{factor, is_prime, mulmod, powmod};
use rayon::prelude::*;
use std::collections::HashMap;

/// Unique norm values with per-weight vector counts.
#[derive(Debug, Clone)]
pub struct NormTable {
    /// Subgroup order the table was built for.
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

/// One bad-set row.
#[derive(Debug, Clone)]
pub struct BadSetEntry {
    /// The prime.
    pub p: u64,
    /// Galois-normalized per-weight kernel-vector counts (index = weight).
    pub counts: Vec<u64>,
    /// True when counts came from a direct census (the `p^2`-divisibility
    /// fallback) rather than valuation splitting.
    pub census_fallback: bool,
}

fn crt_primes(s: usize, bound_bits: u32) -> Result<Vec<u64>> {
    let n_primes = bound_bits.div_ceil(60) as usize;
    if n_primes > 2 {
        return Err(Error::Unsupported(
            "norms beyond ~2^120 not supported (raise via wider CRT)".into(),
        ));
    }
    let mut out = Vec::new();
    let mut q = (1u64 << 61) + 1;
    q += (1 + s as u64 - q % s as u64) % s as u64;
    while out.len() < n_primes {
        if is_prime(q) {
            out.push(q);
        }
        q += s as u64;
    }
    Ok(out)
}

fn combinations(n: usize, k: usize) -> Vec<Vec<u8>> {
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
    // Parseval: N <= (cmax^2 * wmax)^{s/4}
    let bound_bits =
        ((s as f64 / 4.0) * ((cmax * cmax) as f64 * wmax as f64).log2()).ceil() as u32 + 2;
    let qs = crt_primes(s, bound_bits)?;
    // per CRT prime: order-s root and its power table
    let tables: Vec<(u64, Vec<u64>)> = qs
        .iter()
        .map(|&q| {
            let sg = Subgroup::new(q, s).expect("split prime");
            (q, sg.elements().to_vec())
        })
        .collect();
    let coefs: Vec<i64> = (-cmax..=cmax).filter(|&c| c != 0).collect();
    let ncoef = coefs.len();
    // hoisted CRT inverse for the two-prime path
    let crt_inv = (tables.len() == 2)
        .then(|| powmod(tables[0].0 % tables[1].0, tables[1].0 - 2, tables[1].0));

    let mut entries: HashMap<u128, Vec<u64>> = HashMap::new();
    for w in 1..=wmax {
        let supports = combinations(half, w);
        let npat: u64 = (ncoef as u64).pow(w as u32);
        let partial: Vec<HashMap<u128, u64>> = supports
            .par_chunks(1.max(supports.len() / 64))
            .map(|chunk| {
                let mut local: HashMap<u128, u64> = HashMap::new();
                for sup in chunk {
                    for pat in 0..npat {
                        // decode coefficient pattern
                        let mut cvec = [0i64; 32];
                        let mut t = pat;
                        for slot in cvec.iter_mut().take(w) {
                            *slot = coefs[(t % ncoef as u64) as usize];
                            t /= ncoef as u64;
                        }
                        // norm via CRT
                        let mut residues = [0u64; 2];
                        for (ti, (q, pw)) in tables.iter().enumerate() {
                            let mut prod: u64 = 1;
                            for k in (1..s).step_by(2) {
                                // Lazy accumulation: terms |c|*pw < 2^64 and at
                                // most 32 of them fit a u128 sum, so the mod
                                // drops from per-term to per-embedding. The
                                // exponent mask is valid because s is a power
                                // of two (validated at entry).
                                let mut acc: u128 = 0;
                                for i in 0..w {
                                    let e = (sup[i] as usize * k) & (s - 1);
                                    let c = cvec[i];
                                    acc += if c >= 0 {
                                        (c as u128) * (pw[e] as u128)
                                    } else {
                                        ((-c) as u128) * ((q - pw[e]) as u128)
                                    };
                                }
                                prod = mulmod(prod, (acc % (*q as u128)) as u64, *q);
                            }
                            residues[ti] = prod;
                        }
                        let n: u128 = if tables.len() == 1 {
                            residues[0] as u128
                        } else {
                            // CRT for two primes
                            let (q1, q2) = (tables[0].0, tables[1].0);
                            let inv = crt_inv.unwrap();
                            let diff = (residues[1] + q2 - residues[0] % q2) % q2;
                            residues[0] as u128 + (q1 as u128) * (mulmod(diff, inv, q2) as u128)
                        };
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
    let table = norm_table(s, wmax, cmax)?;
    let half = (s / 2) as u64;
    let mut raw: HashMap<u64, (Vec<u64>, bool)> = HashMap::new();
    for (&n, counts) in &table.entries {
        if n <= 1 {
            continue;
        }
        let n64 = u64::try_from(n).map_err(|_| {
            Error::Unsupported("factoring norms above 2^64 not yet supported".into())
        })?;
        let fs = factor(n64);
        let mut i = 0;
        while i < fs.len() {
            let p = fs[i];
            let mut e = 0u64;
            while i < fs.len() && fs[i] == p {
                e += 1;
                i += 1;
            }
            if p > s as u64 && (p - 1) % s as u64 == 0 {
                let entry = raw.entry(p).or_insert_with(|| (vec![0; wmax + 1], false));
                for (w, &c) in counts.iter().enumerate() {
                    entry.0[w] += e * c;
                }
                if e >= 2 {
                    entry.1 = true; // p^2 divides a norm: valuation split unsafe
                }
            }
        }
    }
    let mut out: Vec<BadSetEntry> = raw
        .into_par_iter()
        .map(|(p, (val_counts, pp))| {
            if pp && s <= 32 {
                // direct census: exact per-embedding counts by construction
                let sg = Subgroup::new(p, s).expect("bad-set prime");
                let c = census::mitm(&sg, cmax).expect("census fallback");
                let mut counts = vec![0u64; wmax + 1];
                for (w, slot) in counts.iter_mut().enumerate() {
                    *slot = *c.get(w).unwrap_or(&0);
                }
                BadSetEntry {
                    p,
                    counts,
                    census_fallback: true,
                }
            } else {
                let counts = val_counts.iter().map(|&v| v / half).collect();
                BadSetEntry {
                    p,
                    counts,
                    census_fallback: false,
                }
            }
        })
        .collect();
    out.sort_by_key(|e| e.p);
    Ok(out)
}
