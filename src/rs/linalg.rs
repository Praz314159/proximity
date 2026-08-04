//! Dense linear algebra over `F_p` — the small toolbox every syndrome/kernel
//! experiment re-implemented in Python (five separate Gaussian eliminations,
//! ten `modinv` copies): row reduction, nullspaces, residues mod a span,
//! batch inversion, divided-difference rows. Each elimination copy in a
//! proof-verification script is a fresh chance for a pivot bug; this is the
//! audited one.

use crate::error::{Error, Result};
use crate::field::{batch_inv, mulmod, powmod};

fn inv(a: u64, p: u64) -> u64 {
    powmod(a % p, p - 2, p)
}

/// Reduced row-echelon form in place; returns `(rank, pivot columns)`.
pub fn rref_mod(rows: &mut [Vec<u64>], p: u64) -> Result<(usize, Vec<usize>)> {
    let ncols = rows.first().map_or(0, Vec::len);
    if rows.iter().any(|r| r.len() != ncols) {
        return Err(Error::OutOfRange("ragged matrix".into()));
    }
    for row in rows.iter_mut() {
        for v in row.iter_mut() {
            *v %= p;
        }
    }
    let mut pivots = Vec::new();
    let mut rr = 0usize;
    for c in 0..ncols {
        let Some(piv) = (rr..rows.len()).find(|&i| rows[i][c] != 0) else {
            continue;
        };
        rows.swap(rr, piv);
        let iv = inv(rows[rr][c], p);
        for v in rows[rr].iter_mut() {
            *v = mulmod(*v, iv, p);
        }
        let prow = rows[rr].clone();
        for (i, row) in rows.iter_mut().enumerate() {
            if i != rr && row[c] != 0 {
                let f = row[c];
                for (v, &pv) in row.iter_mut().zip(&prow) {
                    let sub = mulmod(f, pv, p);
                    *v = (*v + p - sub) % p;
                }
            }
        }
        pivots.push(c);
        rr += 1;
    }
    Ok((rr, pivots))
}

/// A basis of the right nullspace of the row span: vectors `v` with
/// `A v = 0`, one per free column.
pub fn nullspace_mod(rows: &[Vec<u64>], p: u64) -> Result<Vec<Vec<u64>>> {
    let ncols = rows.first().map_or(0, Vec::len);
    let mut a: Vec<Vec<u64>> = rows.to_vec();
    let (_, pivots) = rref_mod(&mut a, p)?;
    let free: Vec<usize> = (0..ncols).filter(|c| !pivots.contains(c)).collect();
    Ok(free
        .iter()
        .map(|&fc| {
            let mut v = vec![0u64; ncols];
            v[fc] = 1;
            for (ri, &pc) in pivots.iter().enumerate() {
                v[pc] = (p - a[ri][fc]) % p;
            }
            v
        })
        .collect())
}

/// Residues of `vecs` after eliminating the pivots of `span` (RREF): the
/// canonical representatives mod the row span.
pub fn reduce_mod_span(vecs: &[Vec<u64>], span: &[Vec<u64>], p: u64) -> Result<Vec<Vec<u64>>> {
    let mut a: Vec<Vec<u64>> = span.to_vec();
    let (_, pivots) = rref_mod(&mut a, p)?;
    let ncols = a.first().map_or(0, Vec::len);
    vecs.iter()
        .map(|v| {
            if v.len() != ncols {
                return Err(Error::OutOfRange("vector length mismatch".into()));
            }
            let mut w: Vec<u64> = v.iter().map(|&x| x % p).collect();
            for (ri, &pc) in pivots.iter().enumerate() {
                if w[pc] != 0 {
                    let f = w[pc];
                    for (v, &av) in w.iter_mut().zip(&a[ri]) {
                        let sub = mulmod(f, av, p);
                        *v = (*v + p - sub) % p;
                    }
                }
            }
            Ok(w)
        })
        .collect()
}

/// Batch modular inverses (Montgomery's trick). Entries must be nonzero.
pub fn inv_mod(vals: &[u64], p: u64) -> Result<Vec<u64>> {
    if vals.iter().any(|&v| v % p == 0) {
        return Err(Error::OutOfRange("cannot invert 0".into()));
    }
    let mut out: Vec<u64> = vals.iter().map(|&v| v % p).collect();
    batch_inv(&mut out, p);
    Ok(out)
}

/// The divided-difference row of an index subset `T`: `row[x] =
/// 1 / prod_{y in T, y != x}(x - y)` at the domain points of `T`, zero
/// elsewhere — the functional with `D_T(w) = <row, w>`.
/// Leading divided difference of the values `ys` over arbitrary nodes
/// `xs` (generic degree): the coefficient of `x^{|xs|-1}` in the
/// interpolant, `sum_i ys[i] / prod_{j != i} (xs[i] - xs[j])`. The
/// subgroup-indexed form is [`crate::rs::vs::VsSpace::divided_difference`];
/// parity between the two is pinned in tests.
pub fn dd(p: u64, xs: &[u64], ys: &[u64]) -> Result<u64> {
    if xs.len() != ys.len() || xs.is_empty() {
        return Err(Error::OutOfRange("dd: mismatched or empty nodes".into()));
    }
    let mut acc = 0u64;
    for (i, &xi) in xs.iter().enumerate() {
        let mut den = 1u64;
        for (j, &xj) in xs.iter().enumerate() {
            if j != i {
                den = mulmod(den, (xi + p - xj) % p, p);
            }
        }
        if den == 0 {
            return Err(Error::OutOfRange("dd: repeated node".into()));
        }
        acc = (acc + mulmod(ys[i], powmod(den, p - 2, p), p)) % p;
    }
    Ok(acc)
}

/// Lagrange interpolant through `(xs, ys)`, evaluated at `t`.
pub fn interp_eval(p: u64, xs: &[u64], ys: &[u64], t: u64) -> Result<u64> {
    if xs.len() != ys.len() || xs.is_empty() {
        return Err(Error::OutOfRange(
            "interp_eval: mismatched or empty nodes".into(),
        ));
    }
    let mut acc = 0u64;
    for (i, &xi) in xs.iter().enumerate() {
        let mut num = 1u64;
        let mut den = 1u64;
        for (j, &xj) in xs.iter().enumerate() {
            if j != i {
                num = mulmod(num, (t + p - xj) % p, p);
                den = mulmod(den, (xi + p - xj) % p, p);
            }
        }
        if den == 0 {
            return Err(Error::OutOfRange("interp_eval: repeated node".into()));
        }
        acc = (acc + mulmod(ys[i], mulmod(num, powmod(den, p - 2, p), p), p)) % p;
    }
    Ok(acc)
}

pub fn dd_rows(p: u64, domain: &[u64], subsets: &[Vec<usize>]) -> Result<Vec<Vec<u64>>> {
    let n = domain.len();
    subsets
        .iter()
        .map(|t| {
            if t.iter().any(|&i| i >= n) {
                return Err(Error::OutOfRange("subset index out of range".into()));
            }
            let mut denoms: Vec<u64> = t
                .iter()
                .map(|&i| {
                    let mut d = 1u64;
                    for &j in t {
                        if j != i {
                            d = mulmod(d, (domain[i] + p - domain[j]) % p, p);
                        }
                    }
                    d
                })
                .collect();
            if denoms.contains(&0) {
                return Err(Error::OutOfRange("repeated domain point in subset".into()));
            }
            batch_inv(&mut denoms, p);
            let mut row = vec![0u64; n];
            for (&i, &d) in t.iter().zip(&denoms) {
                row[i] = d;
            }
            Ok(row)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dd_and_interp_parity() {
        use crate::domain::MultiplicativeSubgroup;
        use crate::rs::vs::VsSpace;
        let (p, s, k) = (97u64, 16usize, 7usize);
        let sg = MultiplicativeSubgroup::new(p, s).unwrap();
        let vs = VsSpace::new(&sg, k).unwrap();
        let dom = sg.elements().to_vec();
        let w: Vec<u64> = (0..s as u64).map(|i| (i * i * 13 + 5) % p).collect();
        // parity with the subgroup-indexed divided difference on r-subsets
        for sub in [[0usize, 1, 2, 3, 4, 5, 6, 7], [2, 3, 5, 7, 9, 11, 13, 15]] {
            let xs: Vec<u64> = sub.iter().map(|&i| dom[i]).collect();
            let ys: Vec<u64> = sub.iter().map(|&i| w[i]).collect();
            assert_eq!(
                dd(p, &xs, &ys).unwrap(),
                vs.divided_difference(&w, &sub).unwrap()
            );
        }
        // interp_eval reproduces values on the nodes and interpolates a
        // known polynomial off them
        let xs: Vec<u64> = dom[..5].to_vec();
        let ys: Vec<u64> = xs.iter().map(|&x| (x * x + 3 * x + 1) % p).collect();
        for (&x, &y) in xs.iter().zip(&ys) {
            assert_eq!(interp_eval(p, &xs, &ys, x).unwrap(), y);
        }
        let t = dom[9];
        assert_eq!(
            interp_eval(p, &xs, &ys, t).unwrap(),
            (t * t + 3 * t + 1) % p
        );
        // degree-(n-1) leading coefficient of that quadratic on 5 nodes is 0
        assert_eq!(dd(p, &xs, &ys).unwrap(), 0);
    }
    use crate::domain::MultiplicativeSubgroup;
    use crate::smooth::rung::top_word;

    /// Nullspace/rank on a known system, and the divided-difference identity
    /// `D_T(x^{r-1}) = 1` on live data.
    #[test]
    fn linalg_basics() {
        let p = 97u64;
        // rank-2 matrix in F_97^3 with nullspace (1, -2, 1)-ish
        let rows = vec![vec![1, 1, 1], vec![1, 2, 3]];
        let ns = nullspace_mod(&rows, p).unwrap();
        assert_eq!(ns.len(), 1);
        for r in &rows {
            let dot: u64 = r
                .iter()
                .zip(&ns[0])
                .fold(0, |a, (&x, &y)| (a + mulmod(x, y, p)) % p);
            assert_eq!(dot, 0);
        }
        assert_eq!(inv_mod(&[2, 3], 7).unwrap(), vec![4, 5]);

        let sg = MultiplicativeSubgroup::new(65537, 16).unwrap();
        let dom = sg.elements();
        let rows = dd_rows(65537, dom, &[vec![0, 2, 3, 5, 7, 8, 11, 13]]).unwrap();
        // D_T(x^{r-1}) = 1 for |T| = r = 8: pair against the monomial x^7
        let w: Vec<u64> = dom.iter().map(|&x| powmod(x, 7, 65537)).collect();
        let dot = rows[0]
            .iter()
            .zip(&w)
            .fold(0u64, |a, (&x, &y)| (a + mulmod(x, y, 65537)) % 65537);
        assert_eq!(dot, 1);
        // and D_T(top word) = 0 exactly when T's product is in the class:
        // the syndrome of the theorem word vanishes on class members only —
        // spot-check one in-class subset (sum of indices = 4 mod 16)
        let tclass = vec![0usize, 1, 2, 3, 4, 5, 6, 15]; // sum = 36 = 4 mod 16
        let rows = dd_rows(65537, dom, &[tclass]).unwrap();
        let wtop = top_word(&sg, 8, 4).unwrap();
        let dot = rows[0]
            .iter()
            .zip(&wtop)
            .fold(0u64, |a, (&x, &y)| (a + mulmod(x, y, 65537)) % 65537);
        assert_eq!(dot, 0, "D_T(top word) vanishes on the class");
    }
}
