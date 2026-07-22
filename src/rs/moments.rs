//! The moment cloud and syndrome-cut kernels (the q = 1 hyperplane layer).
//!
//! Every word of `RS[F_p, D, k]`, modulo the code, acts on candidate
//! agreement sets through one linear functional: `D_S(w) = <b, e-hat(S-bar)>`,
//! where `b` is the word's syndrome (top coefficients) and `e-hat(S-bar)` is
//! the elementary-symmetric moment vector of the *complement* of `S`. The
//! **moment cloud** is the fixed point set `{e-hat(S-bar)}` over all
//! `r`-subsets; a word is a hyperplane; its cut is the incidence set. These
//! kernels materialize the cloud, count cuts in bulk, and run the exhaustive
//! slope enumerations behind the top-cut certifications — audited once here
//! instead of re-implemented per experiment script (where a stratum
//! double-count once produced a wrong maximum).
//!
//! Subsets are enumerated in lexicographic index order, matching Python's
//! `itertools.combinations(range(n), r)`, so cloud rows line up with the
//! reference scripts.

use crate::error::{Error, Result};
use crate::field::{batch_inv, checked_binom, mulmod};
use crate::rs::decode::for_each_combination;
use rayon::prelude::*;

/// Refuse to materialize clouds beyond this many entries (`C(n, r)` rows
/// times `n - r + 1` columns).
const CLOUD_ENTRY_CAP: u64 = 1 << 28;

/// The elementary-symmetric vector `(e_0 = 1, e_1, ..., e_m)` of `vals`.
#[must_use]
pub fn e_vec(vals: &[u64], p: u64) -> Vec<u64> {
    let m = vals.len();
    let mut e = vec![0u64; m + 1];
    e[0] = 1;
    for (idx, &v) in vals.iter().enumerate() {
        let hi = idx + 1;
        for i in (1..=hi).rev() {
            e[i] = (e[i] + mulmod(v, e[i - 1], p)) % p;
        }
    }
    e
}

/// The moment cloud of `(p, domain, r)`: row `S` (lex order over index
/// subsets) holds `(e_0, ..., e_{n-r})` of the *complement* of `S`. Returned
/// flattened with `(rows, cols)`.
pub fn moment_cloud(p: u64, domain: &[u64], r: usize) -> Result<(Vec<u64>, usize, usize)> {
    let n = domain.len();
    if r == 0 || r >= n {
        return Err(Error::OutOfRange("need 1 <= r < n".into()));
    }
    let cols = n - r + 1;
    let rows = checked_binom(n as u64, r as u64)
        .filter(|c| {
            c.checked_mul(cols as u64)
                .is_some_and(|e| e <= CLOUD_ENTRY_CAP)
        })
        .ok_or_else(|| Error::Unsupported("moment cloud exceeds the entry cap".into()))?
        as usize;
    // Parallel over the smallest subset element; branch outputs concatenate
    // in i0 order = lex order.
    let branches: Vec<Vec<u64>> = (0..=n - r)
        .into_par_iter()
        .map(|i0| {
            let mut flat = Vec::new();
            let mut comp = Vec::with_capacity(n - r);
            for_each_combination(n - i0 - 1, r - 1, |rest| {
                comp.clear();
                // complement of {i0} u {i0+1+rest}: all indices < i0, plus the
                // gaps above i0
                comp.extend(domain[..i0].iter().copied());
                let mut prev = 0usize; // offset into (i0+1..n)
                for &rr in rest {
                    for g in prev..rr {
                        comp.push(domain[i0 + 1 + g]);
                    }
                    prev = rr + 1;
                }
                for g in prev..(n - i0 - 1) {
                    comp.push(domain[i0 + 1 + g]);
                }
                flat.extend(e_vec(&comp, p));
            });
            flat
        })
        .collect();
    let mut flat = Vec::with_capacity(rows * cols);
    for b in branches {
        flat.extend(b);
    }
    debug_assert_eq!(flat.len(), rows * cols);
    Ok((flat, rows, cols))
}

/// Cut sizes `|Z(b)| = #{S : <b, e-hat(S-bar)> = 0}` for many syndromes at
/// once, streaming over the subsets (no cloud materialization; any size the
/// enumeration itself can cover).
pub fn cut_counts(p: u64, domain: &[u64], r: usize, bs: &[Vec<u64>]) -> Result<Vec<u64>> {
    let n = domain.len();
    let cols = n - r + 1;
    if r == 0 || r >= n {
        return Err(Error::OutOfRange("need 1 <= r < n".into()));
    }
    if bs.iter().any(|b| b.len() != cols) {
        return Err(Error::OutOfRange(format!(
            "each syndrome must have n - r + 1 = {cols} coordinates"
        )));
    }
    let counts = (0..=n - r)
        .into_par_iter()
        .map(|i0| {
            let mut local = vec![0u64; bs.len()];
            let mut comp = Vec::with_capacity(n - r);
            for_each_combination(n - i0 - 1, r - 1, |rest| {
                comp.clear();
                comp.extend(domain[..i0].iter().copied());
                let mut prev = 0usize;
                for &rr in rest {
                    for g in prev..rr {
                        comp.push(domain[i0 + 1 + g]);
                    }
                    prev = rr + 1;
                }
                for g in prev..(n - i0 - 1) {
                    comp.push(domain[i0 + 1 + g]);
                }
                let ev = e_vec(&comp, p);
                for (bi, b) in bs.iter().enumerate() {
                    let mut acc = 0u64;
                    for (c, &bc) in b.iter().enumerate() {
                        acc = (acc + mulmod(bc % p, ev[c], p)) % p;
                    }
                    if acc == 0 {
                        local[bi] += 1;
                    }
                }
            });
            local
        })
        .reduce(
            || vec![0u64; bs.len()],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                a
            },
        );
    Ok(counts)
}

/// Exhaustive sparse-cut maximum over ALL words supported on `support`
/// (moment-coordinate indices, `1 <= i < ... <= n - r`), by direct counting
/// with the last coefficient normalized to `-1` (words with a zero last
/// coefficient live on a smaller support — certify those separately).
/// `|support|` must be 3, 4, or 5 (5 costs p^3 slope triples). Returns `(max cut, coefficients on support)`.
///
/// This is the audited kernel behind the top-cut certifications: for each
/// slope tuple it bins `(e_last - sum slope_i e_mid_i) / e_first` over the
/// cloud (the intercept read-off), adds the `e_first = 0` stratum to every
/// intercept, and takes the maximum — no stratified bookkeeping, hence no
/// double-count surface.
pub fn cut_max_sparse(
    p: u64,
    domain: &[u64],
    r: usize,
    support: &[usize],
) -> Result<(u64, Vec<u64>)> {
    let n = domain.len();
    let cols = n - r + 1;
    if !(3..=5).contains(&support.len()) {
        return Err(Error::Unsupported(
            "support must have 3, 4, or 5 coordinates".into(),
        ));
    }
    if support.windows(2).any(|w| w[0] >= w[1]) || support.iter().any(|&i| i == 0 || i >= cols) {
        return Err(Error::OutOfRange(
            "support must be strictly increasing in 1..=n-r".into(),
        ));
    }
    // Materialize the support columns once.
    let (flat, rows, ncols) = moment_cloud(p, domain, r)?;
    let col = |s: usize, row: usize| flat[row * ncols + s];
    let pu = p as usize;
    let e0: Vec<u64> = (0..rows).map(|z| col(support[0], z)).collect();
    let mut inv0: Vec<u64> = e0.iter().map(|&x| if x == 0 { 1 } else { x }).collect();
    batch_inv(&mut inv0, p);
    let mids: Vec<Vec<u64>> = support[1..support.len() - 1]
        .iter()
        .map(|&s| (0..rows).map(|z| col(s, z)).collect())
        .collect();
    let elast: Vec<u64> = (0..rows)
        .map(|z| col(*support.last().unwrap(), z))
        .collect();

    let eval_slope = |slopes: &[u64]| -> (u64, u64) {
        // returns (max over intercepts, argmax intercept)
        let mut bins = vec![0u64; pu];
        let mut stratum0 = 0u64;
        for z in 0..rows {
            let mut tv = elast[z];
            for (m, &a) in mids.iter().zip(slopes) {
                tv = (tv + p - mulmod(a, m[z], p)) % p;
            }
            if e0[z] == 0 {
                if tv == 0 {
                    stratum0 += 1;
                }
            } else {
                bins[mulmod(tv, inv0[z], p) as usize] += 1;
            }
        }
        let (mut mx, mut arg) = (0u64, 0u64);
        for (c, &b) in bins.iter().enumerate() {
            if b + stratum0 > mx {
                mx = b + stratum0;
                arg = c as u64;
            }
        }
        (mx, arg)
    };

    let best = if support.len() == 3 {
        (0..p)
            .into_par_iter()
            .map(|a| {
                let (mx, c) = eval_slope(&[a]);
                (mx, vec![c, a, p - 1])
            })
            .max()
            .unwrap_or((0, vec![]))
    } else if support.len() == 5 {
        (0..p)
            .into_par_iter()
            .map(|a| {
                let mut local = (0u64, Vec::new());
                for b in 0..p {
                    for cc in 0..p {
                        let (mx, c0) = eval_slope(&[a, b, cc]);
                        if mx > local.0 {
                            local = (mx, vec![c0, a, b, cc, p - 1]);
                        }
                    }
                }
                local
            })
            .max()
            .unwrap_or((0, vec![]))
    } else {
        (0..p)
            .into_par_iter()
            .map(|a| {
                let mut local = (0u64, Vec::new());
                for b in 0..p {
                    let (mx, c) = eval_slope(&[a, b]);
                    if mx > local.0 {
                        local = (mx, vec![c, a, b, p - 1]);
                    }
                }
                local
            })
            .max()
            .unwrap_or((0, vec![]))
    };
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::MultiplicativeSubgroup;

    /// Cloud pins at (16, 8): the coordinate fibers reproduce the census —
    /// the top class 810, the additive/antipodal 70, and the e_4 family 266.
    #[test]
    fn cloud_coordinate_fibers() {
        let sg = MultiplicativeSubgroup::new(65537, 16).unwrap();
        let (flat, rows, cols) = moment_cloud(65537, sg.elements(), 8).unwrap();
        assert_eq!((rows, cols), (12870, 9));
        let fiber_max = |c: usize| {
            let mut m = std::collections::HashMap::new();
            for z in 0..rows {
                *m.entry(flat[z * cols + c]).or_insert(0u64) += 1;
            }
            (*m.values().max().unwrap(), m.get(&0).copied().unwrap_or(0))
        };
        assert_eq!(fiber_max(8).0, 810, "top class");
        assert_eq!(fiber_max(1).1, 70, "additive zero fiber");
        assert_eq!(fiber_max(4).1, 266, "the e_4 family");
    }

    /// cut_counts reproduces the top word's 810 and the additive 70 from the
    /// syndrome side.
    #[test]
    fn cut_counts_known_families() {
        let sg = MultiplicativeSubgroup::new(65537, 16).unwrap();
        let p = 65537;
        // top word syndrome: b = (-zeta^4, 0, ..., 0, 1) in moment coords
        let v = crate::field::powmod(sg.w(), 4, p);
        let mut top = vec![0u64; 9];
        top[0] = p - v;
        top[8] = 1;
        let mut additive = vec![0u64; 9];
        additive[1] = 1;
        let counts = cut_counts(p, sg.elements(), 8, &[top, additive]).unwrap();
        assert_eq!(counts, vec![810, 70]);
    }

    /// The exhaustive sparse kernel reproduces the certified maxima: the
    /// trinomial 698 at supp (1,2,6) (p = 257) and the 4-sparse 3074 at
    /// supp (1,3,5,7) (p = 97) — the antipodal-pair bump words.
    #[test]
    fn sparse_certification_pins() {
        let sg = MultiplicativeSubgroup::new(257, 16).unwrap();
        let (mx, _) = cut_max_sparse(257, sg.elements(), 8, &[1, 2, 6]).unwrap();
        assert_eq!(mx, 698);
        let sg = MultiplicativeSubgroup::new(97, 16).unwrap();
        let (mx, _) = cut_max_sparse(97, sg.elements(), 8, &[1, 3, 5, 7]).unwrap();
        assert_eq!(mx, 3074);
    }
}
