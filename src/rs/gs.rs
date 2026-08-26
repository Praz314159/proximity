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

use crate::error::{Error, Result};
use crate::field::{binom_table_mod, mulmod};
use crate::rs::linalg::{horner, inv, nullspace_mod};

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
    (0..=d / w.max(1))
        .take_while(|b| w * b <= d)
        .map(|b| d - w * b + 1)
        .sum()
}

/// The smallest workable `(m, d)` at `(n, k, t)`, trying
/// multiplicities up to 16. Errors when `t^2 <= n (k - 1)` (at or
/// below the Johnson agreement no multiplicity certifies
/// completeness) and when `k < 2` or `t < k` (no interpolation
/// problem).
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
    for m in 1..=16u64 {
        let conditions = n * m * (m + 1) / 2;
        // smallest d with strictly more monomials than conditions
        let mut d = k - 1;
        while monomials(d, k) <= conditions {
            d += 1;
        }
        if m * t > d {
            return Ok(GsParams { m, d });
        }
    }
    Err(Error::Unsupported(format!(
        "no multiplicity up to 16 certifies (n, k, t) = ({n}, {k}, {t})"
    )))
}

/// A nonzero `(1, k-1)`-weighted interpolant vanishing to order `m`
/// at every `(xs[i], ys[i])`, as dense `q[b][a]` (coefficient of
/// `x^a y^b`). One nullspace of the Hasse-derivative system.
fn interpolate(p: u64, xs: &[u64], ys: &[u64], k: u64, prm: GsParams) -> Result<Vec<Vec<u64>>> {
    let (m, d) = (prm.m, prm.d);
    let w = (k - 1) as usize;
    let dy = (d as usize) / w;
    // column layout: (a, b) with a + w b <= d
    let mut cols = Vec::new();
    for b in 0..=dy {
        for a in 0..=(d as usize - w * b) {
            cols.push((a, b));
        }
    }
    let table = binom_table_mod(d as usize + 1, p);
    let mut rows = Vec::new();
    for (&x, &y) in xs.iter().zip(ys) {
        // powers of x and y up to the degree caps
        let mut xp = vec![1u64; d as usize + 1];
        for i in 1..xp.len() {
            xp[i] = mulmod(xp[i - 1], x, p);
        }
        let mut yp = vec![1u64; dy + 1];
        for i in 1..yp.len() {
            yp[i] = mulmod(yp[i - 1], y, p);
        }
        for r in 0..m as usize {
            for s in 0..(m as usize - r) {
                let row: Vec<u64> = cols
                    .iter()
                    .map(|&(a, b)| {
                        if a < r || b < s {
                            return 0;
                        }
                        let c = mulmod(table[a][r], table[b][s], p);
                        mulmod(c, mulmod(xp[a - r], yp[b - s], p), p)
                    })
                    .collect();
                rows.push(row);
            }
        }
    }
    let null = nullspace_mod(&rows, p)?;
    let v = null
        .first()
        .ok_or_else(|| Error::Unsupported("interpolation system has full rank".into()))?;
    let mut q = vec![vec![0u64; d as usize + 1]; dy + 1];
    for (&(a, b), &c) in cols.iter().zip(v) {
        q[b][a] = c;
    }
    Ok(q)
}

// ---- univariate polynomials over F_p, dense low-to-high ----

fn poly_trim(f: &mut Vec<u64>) {
    while f.last() == Some(&0) {
        f.pop();
    }
}

fn poly_mulmod(a: &[u64], b: &[u64], modulus: &[u64], p: u64) -> Vec<u64> {
    let mut out = vec![0u64; a.len() + b.len()];
    for (i, &ai) in a.iter().enumerate() {
        if ai == 0 {
            continue;
        }
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] = (out[i + j] + mulmod(ai, bj, p)) % p;
        }
    }
    poly_rem(&mut out, modulus, p);
    out
}

/// In-place remainder of `f` mod the monic-normalized `modulus`.
fn poly_rem(f: &mut Vec<u64>, modulus: &[u64], p: u64) {
    let dm = modulus.len() - 1;
    let lead_inv = inv(modulus[dm], p);
    while f.len() > dm {
        let c = mulmod(*f.last().unwrap(), lead_inv, p);
        let shift = f.len() - 1 - dm;
        if c != 0 {
            for (i, &mi) in modulus.iter().enumerate() {
                let t = mulmod(c, mi, p);
                f[shift + i] = (f[shift + i] + p - t) % p;
            }
        }
        f.pop();
        poly_trim(f);
        if f.is_empty() {
            return;
        }
    }
    poly_trim(f);
}

fn poly_gcd(mut a: Vec<u64>, mut b: Vec<u64>, p: u64) -> Vec<u64> {
    poly_trim(&mut a);
    poly_trim(&mut b);
    while !b.is_empty() {
        poly_rem(&mut a, &b, p);
        std::mem::swap(&mut a, &mut b);
    }
    if let Some(&l) = a.last() {
        let li = inv(l, p);
        for c in &mut a {
            *c = mulmod(*c, li, p);
        }
    }
    a
}

/// `x^e mod f` by square and multiply.
fn poly_xpow(e: u64, f: &[u64], p: u64) -> Vec<u64> {
    let mut result = vec![1u64];
    let mut base = vec![0u64, 1];
    poly_rem(&mut base, f, p);
    let mut e = e;
    let mut b = base;
    while e > 0 {
        if e & 1 == 1 {
            result = poly_mulmod(&result, &b, f, p);
        }
        b = poly_mulmod(&b, &b, f, p);
        e >>= 1;
    }
    result
}

/// All roots in `F_p` of a nonzero univariate `f` (dense,
/// low-to-high), by splitting off the product of linear factors with
/// `gcd(f, x^p - x)` and then Cantor–Zassenhaus on shifted
/// square-root maps. Degrees here are single digits; the loop is a
/// handful of gcds.
fn poly_roots(f: &[u64], p: u64) -> Vec<u64> {
    let mut f = f.to_vec();
    poly_trim(&mut f);
    assert!(!f.is_empty(), "poly_roots of the zero polynomial");
    let mut roots = Vec::new();
    // constant term zero: root at 0
    while f.len() > 1 && f[0] == 0 {
        if !roots.contains(&0) {
            roots.push(0);
        }
        f.remove(0);
    }
    if f.len() == 1 {
        return roots;
    }
    // product of the distinct linear factors: gcd(f, x^p - x)
    let xp = poly_xpow(p, &f, p);
    let mut xp_minus_x = xp;
    if xp_minus_x.len() < 2 {
        xp_minus_x.resize(2, 0);
    }
    xp_minus_x[1] = (xp_minus_x[1] + p - 1) % p;
    let mut g = poly_gcd(f, xp_minus_x, p);
    // deterministic split sequence: shifts 1, 2, 3, ...
    let mut stack = vec![g.clone()];
    let mut shift = 1u64;
    while let Some(h) = stack.pop() {
        let deg = h.len() - 1;
        if deg == 0 {
            continue;
        }
        if deg == 1 {
            // monic x + c: root -c
            let r = (p - h[0] % p) % p;
            if !roots.contains(&r) {
                roots.push(r);
            }
            continue;
        }
        // split h by gcd(h, (x + shift)^((p-1)/2) - 1)
        let mut split = None;
        while split.is_none() {
            let mut xs = vec![shift % p, 1];
            poly_rem(&mut xs, &h, p);
            let mut e = poly_xshift_pow((p - 1) / 2, xs, &h, p);
            if e.is_empty() {
                e = vec![0];
            }
            e[0] = (e[0] + p - 1) % p;
            poly_trim(&mut e);
            shift += 1;
            if e.is_empty() {
                continue; // (x+shift) was a square map on all roots
            }
            let d = poly_gcd(h.clone(), e, p);
            let dd = d.len().saturating_sub(1);
            if dd > 0 && dd < deg {
                split = Some(d);
            }
        }
        let d = split.unwrap();
        // h / d
        let mut quot = poly_div_exact(&h, &d, p);
        poly_trim(&mut quot);
        stack.push(d);
        stack.push(quot);
    }
    g.clear();
    roots
}

/// `(base)^e mod f` for an already-reduced `base`.
fn poly_xshift_pow(e: u64, base: Vec<u64>, f: &[u64], p: u64) -> Vec<u64> {
    let mut result = vec![1u64];
    let mut b = base;
    let mut e = e;
    while e > 0 {
        if e & 1 == 1 {
            result = poly_mulmod(&result, &b, f, p);
        }
        b = poly_mulmod(&b, &b, f, p);
        e >>= 1;
    }
    result
}

/// Exact quotient `h / d` (remainder known zero).
fn poly_div_exact(h: &[u64], d: &[u64], p: u64) -> Vec<u64> {
    let mut rem = h.to_vec();
    let dd = d.len() - 1;
    let li = inv(d[dd], p);
    let mut quot = vec![0u64; rem.len() - dd];
    while rem.len() > dd {
        let c = mulmod(*rem.last().unwrap(), li, p);
        let shift = rem.len() - 1 - dd;
        quot[shift] = c;
        for (i, &di) in d.iter().enumerate() {
            let t = mulmod(c, di, p);
            rem[shift + i] = (rem[shift + i] + p - t) % p;
        }
        rem.pop();
        poly_trim(&mut rem);
        if rem.is_empty() {
            break;
        }
    }
    quot
}

// ---- Roth–Ruckenstein ----

/// Divide `q[b][a]` by the largest common power of `x`.
fn strip_x(q: &mut [Vec<u64>]) {
    let v = q
        .iter()
        .map(|row| row.iter().position(|&c| c != 0).unwrap_or(usize::MAX))
        .min()
        .unwrap_or(0);
    if v == 0 || v == usize::MAX {
        return;
    }
    for row in q.iter_mut() {
        row.drain(..v.min(row.len()));
    }
}

/// `Q(x, x y + alpha)` on dense `q[b][a]`.
fn shift_y(q: &[Vec<u64>], alpha: u64, p: u64) -> Vec<Vec<u64>> {
    let dy = q.len() - 1;
    let dx = q.iter().map(Vec::len).max().unwrap_or(1);
    let table = binom_table_mod(dy, p);
    let mut alpha_pow = vec![1u64; dy + 1];
    for i in 1..=dy {
        alpha_pow[i] = mulmod(alpha_pow[i - 1], alpha, p);
    }
    // new coeff of y^j: x^j * sum_{b >= j} C(b, j) alpha^{b-j} q_b(x)
    let mut out = vec![vec![0u64; dx + dy + 1]; dy + 1];
    for j in 0..=dy {
        for b in j..=dy {
            let c = mulmod(table[b][j], alpha_pow[b - j], p);
            if c == 0 {
                continue;
            }
            for (a, &qa) in q[b].iter().enumerate() {
                if qa != 0 {
                    out[j][a + j] = (out[j][a + j] + mulmod(c, qa, p)) % p;
                }
            }
        }
    }
    out
}

fn rr(q: &[Vec<u64>], k: usize, depth: usize, cur: &mut Vec<u64>, out: &mut Vec<Vec<u64>>, p: u64) {
    let mut q = q.to_vec();
    strip_x(&mut q);
    // roots of Q(0, y)
    let mut r: Vec<u64> = q.iter().map(|row| *row.first().unwrap_or(&0)).collect();
    poly_trim(&mut r);
    if r.is_empty() {
        // Q(0, y) identically zero after stripping x: y | Q handled
        // by the root at 0 below only if present; an all-zero column
        // means every alpha works at this depth — bounded by taking
        // the roots of the next x-coefficient instead is not needed
        // for soundness (candidates are verified), so take 0 alone.
        r = vec![0, 1];
    }
    for alpha in poly_roots(&r, p) {
        cur.push(alpha);
        if depth + 1 == k {
            out.push(cur.clone());
        } else {
            rr(&shift_y(&q, alpha, p), k, depth + 1, cur, out, p);
        }
        cur.pop();
    }
}

/// Every codeword of `RS[F_p, xs, k]` agreeing with `ys` on at least
/// `t` points, as coefficient vectors (low-to-high, length `k`).
/// Complete: `gs_params` certifies the radius or errors. Sound: every
/// returned candidate is re-verified against `(xs, ys)` pointwise.
pub fn gs_list(p: u64, xs: &[u64], ys: &[u64], k: u64, t: u64) -> Result<Vec<Vec<u64>>> {
    if xs.len() != ys.len() {
        return Err(Error::OutOfRange("xs and ys differ in length".into()));
    }
    let n = xs.len() as u64;
    if xs.iter().any(|&x| x >= p) || ys.iter().any(|&y| y >= p) {
        return Err(Error::OutOfRange("points outside F_p".into()));
    }
    {
        let mut sorted: Vec<u64> = xs.to_vec();
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            return Err(Error::OutOfRange("repeated evaluation points".into()));
        }
    }
    let prm = gs_params(n, k, t)?;
    let q = interpolate(p, xs, ys, k, prm)?;
    let mut cand = Vec::new();
    rr(&q, k as usize, 0, &mut Vec::new(), &mut cand, p);
    // verify and dedup
    let mut out: Vec<Vec<u64>> = Vec::new();
    for f in cand {
        let agree = xs
            .iter()
            .zip(ys)
            .filter(|(&x, &y)| horner(&f, x, p) == y)
            .count() as u64;
        if agree >= t && !out.contains(&f) {
            out.push(f);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rs::code::ReedSolomon;
    use crate::rs::combi::SplitMix64;
    use crate::rs::decode::{DecodeOracle, Radius};
    use crate::rs::linalg::{evaluate, interpolate_poly};

    #[test]
    fn params_refuse_at_johnson() {
        // Johnson agreement at (42, 9) is sqrt(42 * 8) = 18.33
        assert!(gs_params(42, 9, 18).is_err()); // 18^2 = 324 <= 336
        let prm = gs_params(42, 9, 19).expect("just beyond Johnson");
        assert!(prm.m * 19 > prm.d);
        // the (64, 31, 43) residual: multiplicity 2 suffices
        let prm = gs_params(42, 9, 21).expect("beyond Johnson");
        assert_eq!(prm.m, 2);
        // the (64, 31, 42) residual is decodable too
        assert!(gs_params(44, 11, 22).is_ok()); // 484 > 440
    }

    #[test]
    fn roots_of_small_polys() {
        for p in [97u64, 65537, 2130706433] {
            // (x - 3)(x - 5)(x^2 + 1)-ish: roots 3 and 5 planted
            let f = {
                // (x + p-3)(x + p-5) = x^2 - 8x + 15
                let a = (p - 8) % p;
                vec![15 % p, a, 1]
            };
            let mut r = poly_roots(&f, p);
            r.sort_unstable();
            assert_eq!(r, vec![3, 5]);
        }
    }

    #[test]
    fn interpolate_poly_roundtrip() {
        let p = 65537;
        let mut rng = SplitMix64::new(7);
        for _ in 0..20 {
            let xs: Vec<u64> = (0..9).map(|i| (i * i + 3 * i + 1) % p).collect();
            let f: Vec<u64> = (0..9).map(|_| rng.next_u64() % p).collect();
            let ys = evaluate(&f, &xs, p);
            let g = interpolate_poly(p, &xs, &ys);
            assert_eq!(evaluate(&g, &xs, p), ys);
        }
    }

    /// The decoder against the exact information-set engine on random
    /// words, random domains, both primes.
    #[test]
    fn agrees_with_exact_engine() {
        let mut rng = SplitMix64::new(1);
        for p in [97u64, 65537] {
            for trial in 0..8 {
                let n = 14 + (trial % 3) * 2; // 14, 16, 18
                let k = 3 + trial % 2; // 3, 4
                                       // distinct random points
                let mut xs: Vec<u64> = Vec::new();
                while xs.len() < n as usize {
                    let x = rng.next_u64() % p;
                    if !xs.contains(&x) {
                        xs.push(x);
                    }
                }
                let t = {
                    let mut t = k;
                    while t * t <= n * (k - 1) {
                        t += 1;
                    }
                    t + 1
                };
                // half-random word seeded with a planted codeword
                let f: Vec<u64> = (0..k).map(|_| rng.next_u64() % p).collect();
                let mut ys = evaluate(&f, &xs, p);
                for y in ys.iter_mut().take(n as usize - t as usize) {
                    *y = rng.next_u64() % p;
                }
                let rs = ReedSolomon::on_domain(p, xs.clone(), k as usize).expect("code");
                let oracle = DecodeOracle::new(&rs);
                let mut truth = oracle
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

    /// Planted lists: several codewords sharing agreement `t` with
    /// one word must all be found.
    #[test]
    fn planted_codewords_are_found() {
        let p = 65537u64;
        let mut rng = SplitMix64::new(42);
        let n = 20u64;
        let k = 4u64;
        let xs: Vec<u64> = {
            let mut v = Vec::new();
            while v.len() < n as usize {
                let x = rng.next_u64() % p;
                if !v.contains(&x) {
                    v.push(x);
                }
            }
            v
        };
        let t = 10u64; // 100 > 20*3
                       // word = codeword A on the first 10 points, codeword B on the rest
        let fa: Vec<u64> = (0..k).map(|_| rng.next_u64() % p).collect();
        let fb: Vec<u64> = (0..k).map(|_| rng.next_u64() % p).collect();
        let ya = evaluate(&fa, &xs, p);
        let yb = evaluate(&fb, &xs, p);
        let ys: Vec<u64> = (0..n as usize)
            .map(|i| if i < 10 { ya[i] } else { yb[i] })
            .collect();
        let got = gs_list(p, &xs, &ys, k, t).expect("gs");
        let got_evals: Vec<Vec<u64>> = got.iter().map(|f| evaluate(f, &xs, p)).collect();
        assert!(got_evals.contains(&ya), "planted A missing");
        assert!(got_evals.contains(&yb), "planted B missing");
    }
}
