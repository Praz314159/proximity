//! Guruswami–Sudan list decoding over `F_p` — the reference
//! implementation: bivariate interpolation as one nullspace, then
//! Roth–Ruckenstein root finding, then verification of every
//! candidate against the received word. Complete within its stated
//! radius: every codeword with agreement at least `t` is returned,
//! provided `t` exceeds the Johnson agreement `sqrt(n (k - 1))`
//! (the parameter search refuses otherwise). Built for small
//! instances (the residual decodes of the paired-domain enumeration:
//! `n` in the tens, `k` below ten, lists of a handful); the fast
//! path, when it lands, is gated against this one.

use std::collections::HashSet;

use crate::error::{Error, Result};
use crate::field::{binom_table_mod, mulmod};
use crate::poly;
use crate::rs::linalg::nullspace_mod;

/// Interpolation parameters certifying completeness at agreement `t`:
/// multiplicity `m` and weighted-degree bound `d` such that the
/// `(1, k-1)`-weighted monomials of degree at most `d` outnumber the
/// `n C(m+1, 2)` vanishing conditions (so a nonzero interpolant
/// exists) while `m t > d` (so every codeword at agreement `t`
/// divides it).
#[derive(Debug, Clone, Copy)]
pub struct GsParams {
    /// Vanishing multiplicity at each point.
    pub m: u64,
    /// `(1, k-1)`-weighted degree bound of the interpolant.
    pub d: u64,
}

/// The number of monomials `x^a y^b` with `a + (k-1) b <= d`.
fn monomials(d: u64, k: u64) -> u64 {
    let w = k - 1;
    (0..=d / w).map(|b| d - w * b + 1).sum()
}

/// The smallest workable `(m, d)` at `(n, k, t)`, trying
/// multiplicities up to 16. Errors when `k < 2` or `t < k` (no
/// interpolation problem) and when `t^2 <= n (k - 1)`: at or below
/// the Johnson agreement no multiplicity certifies completeness.
pub fn gs_params(n: u64, k: u64, t: u64) -> Result<GsParams> {
    if k < 2 || t < k {
        return Err(Error::OutOfRange(format!(
            "gs_params needs k >= 2 and t >= k (got n = {n}, k = {k}, t = {t})"
        )));
    }
    if t * t <= n * (k - 1) {
        return Err(Error::Unsupported(format!(
            "agreement t = {t} is at or below the Johnson agreement \
             sqrt({n} * {}) — no multiplicity certifies completeness",
            k - 1
        )));
    }
    for m in 1..=16 {
        let conditions = n * m * (m + 1) / 2;
        // the smallest d with strictly more monomials than conditions
        let d = (k - 1..)
            .find(|&d| monomials(d, k) > conditions)
            .expect("monomial count is unbounded in d");
        if m * t > d {
            return Ok(GsParams { m, d });
        }
    }
    Err(Error::Unsupported(format!(
        "no multiplicity up to 16 certifies (n, k, t) = ({n}, {k}, {t})"
    )))
}

/// A bivariate polynomial as dense `q[b][a]`: the coefficient of
/// `x^a y^b`. Rows may have different lengths.
type Bivariate = Vec<Vec<u64>>;

/// A nonzero `(1, k-1)`-weighted interpolant vanishing to order `m`
/// at every `(xs[i], ys[i])` — one nullspace of the Hasse-derivative
/// system.
fn interpolate(p: u64, xs: &[u64], ys: &[u64], k: u64, prm: GsParams) -> Result<Bivariate> {
    let (m, d) = (prm.m as usize, prm.d as usize);
    let w = (k - 1) as usize;
    let dy = d / w;
    // column layout: (a, b) with a + w b <= d
    let cols: Vec<(usize, usize)> = (0..=dy)
        .flat_map(|b| (0..=d - w * b).map(move |a| (a, b)))
        .collect();
    let table = binom_table_mod(d + 1, p);
    let mut rows = Vec::with_capacity(xs.len() * m * (m + 1) / 2);
    for (&x, &y) in xs.iter().zip(ys) {
        let xp = power_table(x, d, p);
        let yp = power_table(y, dy, p);
        // one Hasse derivative (r, s) per multiplicity pair r + s < m
        for r in 0..m {
            for s in 0..m - r {
                rows.push(
                    cols.iter()
                        .map(|&(a, b)| {
                            if a < r || b < s {
                                return 0;
                            }
                            let c = mulmod(table[a][r], table[b][s], p);
                            mulmod(c, mulmod(xp[a - r], yp[b - s], p), p)
                        })
                        .collect(),
                );
            }
        }
    }
    let null = nullspace_mod(&rows, p)?;
    let v = null
        .first()
        .ok_or_else(|| Error::Unsupported("interpolation system has full rank".into()))?;
    let mut q = vec![vec![0; d + 1]; dy + 1];
    for (&(a, b), &c) in cols.iter().zip(v) {
        q[b][a] = c;
    }
    Ok(q)
}

/// `[1, x, x^2, ..., x^top]` mod `p`.
fn power_table(x: u64, top: usize, p: u64) -> Vec<u64> {
    std::iter::successors(Some(1), |&prev| Some(mulmod(prev, x, p)))
        .take(top + 1)
        .collect()
}

/// Divide `q` by the largest common power of `x`.
fn strip_x(q: &mut [Vec<u64>]) {
    let low = q
        .iter()
        .filter_map(|row| row.iter().position(|&c| c != 0))
        .min()
        .unwrap_or(0);
    if low > 0 {
        for row in q.iter_mut() {
            row.drain(..low.min(row.len()));
        }
    }
}

/// `Q(x, x y + alpha)`.
fn shift_y(q: &[Vec<u64>], alpha: u64, p: u64) -> Bivariate {
    let dy = q.len() - 1;
    let dx = q.iter().map(Vec::len).max().unwrap_or(1);
    let table = binom_table_mod(dy, p);
    let alpha_pow = power_table(alpha, dy, p);
    // new coefficient of y^j: x^j sum_{b >= j} C(b, j) alpha^(b-j) q_b(x)
    let mut out = vec![vec![0; dx + dy + 1]; dy + 1];
    for (j, row) in out.iter_mut().enumerate() {
        for b in j..=dy {
            let c = mulmod(table[b][j], alpha_pow[b - j], p);
            if c == 0 {
                continue;
            }
            for (a, &qa) in q[b].iter().enumerate().filter(|&(_, &qa)| qa != 0) {
                row[a + j] = (row[a + j] + mulmod(c, qa, p)) % p;
            }
        }
    }
    out
}

/// Roth–Ruckenstein: every `y = f(x)` with `deg f < k` such that
/// `(y - f(x)) | Q`, coefficient by coefficient. Soundness needs no
/// care here — every candidate is verified by the caller — so the
/// degenerate all-zero column takes the single branch `alpha = 0`
/// rather than the full root set.
fn roth_ruckenstein(q: &[Vec<u64>], k: usize, p: u64) -> Vec<Vec<u64>> {
    let mut out = Vec::new();
    descend(q, k, &mut Vec::new(), &mut out, p);
    out
}

fn descend(q: &[Vec<u64>], k: usize, prefix: &mut Vec<u64>, out: &mut Vec<Vec<u64>>, p: u64) {
    let mut q = q.to_vec();
    strip_x(&mut q);
    // the candidate branches at this depth: the roots of Q(0, y)
    let mut constant_column: Vec<u64> = q
        .iter()
        .map(|row| row.first().copied().unwrap_or(0))
        .collect();
    poly::trim(&mut constant_column);
    let branches = if constant_column.is_empty() {
        vec![0]
    } else {
        poly::roots(&constant_column, p)
    };
    for alpha in branches {
        prefix.push(alpha);
        if prefix.len() == k {
            out.push(prefix.clone());
        } else {
            descend(&shift_y(&q, alpha, p), k, prefix, out, p);
        }
        prefix.pop();
    }
}

/// Every codeword of `RS[F_p, xs, k]` agreeing with `ys` on at least
/// `t` points, as coefficient vectors (low-to-high, length `k`).
/// Complete — [`gs_params`] certifies the radius or errors — and
/// sound: every candidate is re-verified against `(xs, ys)`
/// pointwise.
pub fn gs_list(p: u64, xs: &[u64], ys: &[u64], k: u64, t: u64) -> Result<Vec<Vec<u64>>> {
    if xs.len() != ys.len() {
        return Err(Error::OutOfRange("xs and ys differ in length".into()));
    }
    if xs.iter().chain(ys).any(|&v| v >= p) {
        return Err(Error::OutOfRange("points outside F_p".into()));
    }
    if xs.iter().collect::<HashSet<_>>().len() != xs.len() {
        return Err(Error::OutOfRange("repeated evaluation points".into()));
    }
    let prm = gs_params(xs.len() as u64, k, t)?;
    let q = interpolate(p, xs, ys, k, prm)?;
    let mut seen = HashSet::new();
    Ok(roth_ruckenstein(&q, k as usize, p)
        .into_iter()
        .filter(|f| {
            let agree = xs
                .iter()
                .zip(ys)
                .filter(|&(&x, &y)| poly::horner(f, x, p) == y)
                .count();
            agree as u64 >= t
        })
        .filter(|f| seen.insert(f.clone()))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::evaluate;
    use crate::rs::code::ReedSolomon;
    use crate::rs::combi::SplitMix64;
    use crate::rs::decode::{DecodeOracle, Radius};

    #[test]
    fn params_refuse_at_johnson() {
        // the Johnson agreement at (42, 9) is sqrt(42 * 8) = 18.33
        assert!(gs_params(42, 9, 18).is_err()); // 18^2 = 324 <= 336
        let prm = gs_params(42, 9, 19).expect("just beyond Johnson");
        assert!(prm.m * 19 > prm.d);
        // the (64, 31, 43) residual: multiplicity 2 suffices
        let prm = gs_params(42, 9, 21).expect("beyond Johnson");
        assert_eq!(prm.m, 2);
        // the (64, 31, 42) residual is decodable too
        assert!(gs_params(44, 11, 22).is_ok()); // 484 > 440
    }

    /// The decoder against the exact information-set engine on random
    /// words over random domains, at both battery primes.
    #[test]
    fn agrees_with_exact_engine() {
        let mut rng = SplitMix64::new(1);
        for p in [97, 65537] {
            for trial in 0..8u64 {
                let n = 14 + (trial % 3) * 2; // 14, 16, 18
                let k = 3 + trial % 2; // 3, 4
                let xs = distinct_points(&mut rng, p, n as usize);
                let t = 1 + (k..).find(|&t| t * t > n * (k - 1)).expect("unbounded");
                // a planted codeword with the low points re-randomized
                let f = rng.word(p, k as usize);
                let mut ys = evaluate(&f, &xs, p);
                for y in ys.iter_mut().take((n - t) as usize) {
                    *y = rng.next_u64() % p;
                }
                let rs = ReedSolomon::on_domain(p, xs.clone(), k as usize).expect("code");
                let mut truth = DecodeOracle::new(&rs)
                    .list(&ys, Radius::agreement(t as usize))
                    .expect("exact list");
                truth.sort();
                let mut got: Vec<Vec<u64>> = gs_list(p, &xs, &ys, k, t)
                    .expect("gs")
                    .iter()
                    .map(|f| evaluate(f, &xs, p))
                    .collect();
                got.sort();
                assert_eq!(got, truth, "p = {p}, n = {n}, k = {k}, t = {t}");
            }
        }
    }

    /// Two planted codewords sharing one word at agreement `t` must
    /// both be found.
    #[test]
    fn planted_codewords_are_found() {
        let p = 65537;
        let mut rng = SplitMix64::new(42);
        let (n, k, t) = (20, 4, 10); // 100 > 20 * 3
        let xs = distinct_points(&mut rng, p, n);
        let ya = evaluate(&rng.word(p, k), &xs, p);
        let yb = evaluate(&rng.word(p, k), &xs, p);
        let ys: Vec<u64> = ya[..t].iter().chain(&yb[t..]).copied().collect();
        let got: Vec<Vec<u64>> = gs_list(p, &xs, &ys, k as u64, t as u64)
            .expect("gs")
            .iter()
            .map(|f| evaluate(f, &xs, p))
            .collect();
        assert!(got.contains(&ya), "planted A missing");
        assert!(got.contains(&yb), "planted B missing");
    }

    fn distinct_points(rng: &mut SplitMix64, p: u64, n: usize) -> Vec<u64> {
        let mut xs = Vec::with_capacity(n);
        while xs.len() < n {
            let x = rng.next_u64() % p;
            if !xs.contains(&x) {
                xs.push(x);
            }
        }
        xs
    }
}
