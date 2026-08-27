//! Dense univariate polynomials over `F_p`, as `&[u64]` coefficient
//! slices, low-to-high — evaluation, interpolation, and the factoring
//! toolkit (modular arithmetic, gcd, root finding).
//!
//! Conventions. A polynomial is trimmed when its last coefficient is
//! nonzero; the zero polynomial is the empty slice. Every function
//! takes the prime last, matching [`crate::field`]. The degrees this
//! crate meets are small (root finding sees degree six, not six
//! hundred), and the algorithms are chosen for auditability at that
//! scale: schoolbook multiplication, textbook division.

use crate::field::{batch_inv, inv, mulmod};

/// Evaluate a coefficient vector at one point (Horner).
#[must_use]
pub fn horner(f: &[u64], x: u64, p: u64) -> u64 {
    f.iter()
        .rev()
        .fold(0, |acc, &c| (mulmod(acc, x, p) + c) % p)
}

/// Evaluate a coefficient vector on a domain.
#[must_use]
pub fn evaluate(f: &[u64], xs: &[u64], p: u64) -> Vec<u64> {
    xs.iter().map(|&x| horner(f, x, p)).collect()
}

/// The unique polynomial of degree below `xs.len()` through
/// `(xs[i], ys[i])`, as coefficients: Newton's divided differences,
/// then the Newton form expanded.
#[must_use]
pub fn interpolate(p: u64, xs: &[u64], ys: &[u64]) -> Vec<u64> {
    let n = xs.len();
    let mut dd: Vec<u64> = ys.to_vec();
    let mut coeffs = vec![dd[0]];
    for level in 1..n {
        let mut denoms: Vec<u64> = (level..n)
            .map(|i| (xs[i] + p - xs[i - level]) % p)
            .collect();
        batch_inv(&mut denoms, p);
        for i in (level..n).rev() {
            dd[i] = mulmod((dd[i] + p - dd[i - 1]) % p, denoms[i - level], p);
        }
        coeffs.push(dd[level]);
    }
    let mut f = vec![0; n];
    let mut basis = vec![0; n + 1];
    basis[0] = 1;
    let mut basis_len = 1;
    for (level, &c) in coeffs.iter().enumerate() {
        for (fi, &bi) in f.iter_mut().zip(&basis[..basis_len]) {
            *fi = (*fi + mulmod(c, bi, p)) % p;
        }
        if level + 1 < n {
            // basis *= (x - xs[level])
            let neg = (p - xs[level] % p) % p;
            for i in (0..basis_len).rev() {
                let b = basis[i];
                basis[i + 1] = (basis[i + 1] + b) % p;
                basis[i] = mulmod(b, neg, p);
            }
            basis_len += 1;
        }
    }
    f
}

/// Drop trailing zero coefficients; the zero polynomial trims to
/// empty.
pub fn trim(f: &mut Vec<u64>) {
    while f.last() == Some(&0) {
        f.pop();
    }
}

/// `a * b` modulo a trimmed, nonzero `modulus` — schoolbook, then
/// reduce.
#[must_use]
pub fn mul_rem(a: &[u64], b: &[u64], modulus: &[u64], p: u64) -> Vec<u64> {
    let mut out = vec![0; a.len() + b.len()];
    for (i, &ai) in a.iter().enumerate().filter(|&(_, &ai)| ai != 0) {
        for (j, &bj) in b.iter().enumerate() {
            out[i + j] = (out[i + j] + mulmod(ai, bj, p)) % p;
        }
    }
    rem(&mut out, modulus, p);
    out
}

/// In-place remainder of `f` modulo a trimmed, nonzero `modulus`.
pub fn rem(f: &mut Vec<u64>, modulus: &[u64], p: u64) {
    let dm = modulus.len() - 1;
    let lead_inv = inv(modulus[dm], p);
    trim(f);
    while f.len() > dm {
        let c = mulmod(*f.last().expect("nonempty inside the loop"), lead_inv, p);
        if c != 0 {
            let shift = f.len() - 1 - dm;
            for (fi, &mi) in f[shift..].iter_mut().zip(modulus) {
                *fi = (*fi + p - mulmod(c, mi, p)) % p;
            }
        }
        f.pop();
        trim(f);
    }
}

/// Monic gcd; the zero polynomial (empty vector) is the identity.
#[must_use]
pub fn gcd(mut a: Vec<u64>, mut b: Vec<u64>, p: u64) -> Vec<u64> {
    trim(&mut a);
    trim(&mut b);
    while !b.is_empty() {
        rem(&mut a, &b, p);
        std::mem::swap(&mut a, &mut b);
    }
    if let Some(&lead) = a.last() {
        let li = inv(lead, p);
        for c in &mut a {
            *c = mulmod(*c, li, p);
        }
    }
    a
}

/// `base^e` modulo `f`, by square and multiply; `base` need not be
/// reduced.
#[must_use]
pub fn pow_rem(mut base: Vec<u64>, mut e: u64, f: &[u64], p: u64) -> Vec<u64> {
    rem(&mut base, f, p);
    let mut acc = vec![1];
    while e > 0 {
        if e & 1 == 1 {
            acc = mul_rem(&acc, &base, f, p);
        }
        base = mul_rem(&base, &base, f, p);
        e >>= 1;
    }
    acc
}

/// Exact quotient `h / d`, for a divisor known to divide `h`.
#[must_use]
pub fn div_exact(h: &[u64], d: &[u64], p: u64) -> Vec<u64> {
    let mut rest = h.to_vec();
    let dd = d.len() - 1;
    let lead_inv = inv(d[dd], p);
    let mut quot = vec![0; rest.len().saturating_sub(dd)];
    while rest.len() > dd {
        let c = mulmod(*rest.last().expect("nonempty inside the loop"), lead_inv, p);
        let shift = rest.len() - 1 - dd;
        quot[shift] = c;
        for (ri, &di) in rest[shift..].iter_mut().zip(d) {
            *ri = (*ri + p - mulmod(c, di, p)) % p;
        }
        rest.pop();
        trim(&mut rest);
    }
    trim(&mut quot);
    quot
}

/// All roots of a nonzero `f` in `F_p`, sorted, each reported once.
///
/// Splits off the product of the distinct linear factors with
/// `gcd(f, x^p - x)`, then separates them by Cantor–Zassenhaus with
/// the deterministic shift sequence `1, 2, 3, ...`: each shift keeps
/// the roots whose translate is a nonzero square, and any two
/// distinct roots are separated by some shift, so termination is a
/// theorem, not luck.
///
/// # Panics
///
/// On the zero polynomial, which has no root set.
#[must_use]
pub fn roots(f: &[u64], p: u64) -> Vec<u64> {
    let mut f = f.to_vec();
    trim(&mut f);
    assert!(!f.is_empty(), "roots of the zero polynomial");
    let mut out = Vec::new();
    if f.len() > 1 && f[0] == 0 {
        out.push(0);
        let low = f.iter().position(|&c| c != 0).expect("f is nonzero");
        f.drain(..low);
    }
    if f.len() > 1 {
        // x^p - x (mod f): its gcd with f is the product of f's
        // distinct linear factors
        let mut xp = pow_rem(vec![0, 1], p, &f, p);
        if xp.len() < 2 {
            xp.resize(2, 0);
        }
        xp[1] = (xp[1] + p - 1) % p;
        split_linear(gcd(f, xp, p), p, &mut out);
    }
    out.sort_unstable();
    out
}

/// Cantor–Zassenhaus on a product of distinct monic linear factors:
/// push every root onto `out`.
fn split_linear(h: Vec<u64>, p: u64, out: &mut Vec<u64>) {
    let mut stack = vec![h];
    let mut shift = 1u64;
    while let Some(h) = stack.pop() {
        match h.len() {
            0 | 1 => {}
            2 => out.push((p - h[0] % p) % p), // monic x + c
            _ => loop {
                // gcd with (x + shift)^((p-1)/2) - 1 keeps the roots
                // whose shift is a nonzero square
                let mut half = pow_rem(vec![shift % p, 1], (p - 1) / 2, &h, p);
                shift += 1;
                if half.is_empty() {
                    half = vec![0];
                }
                half[0] = (half[0] + p - 1) % p;
                trim(&mut half);
                if half.is_empty() {
                    continue; // this shift squares on every root
                }
                let d = gcd(h.clone(), half, p);
                if d.len() > 1 && d.len() < h.len() {
                    stack.push(div_exact(&h, &d, p));
                    stack.push(d);
                    break;
                }
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rs::combi::SplitMix64;

    #[test]
    fn interpolate_roundtrip() {
        let p = 65537;
        let mut rng = SplitMix64::new(7);
        for _ in 0..20 {
            let xs: Vec<u64> = (0..9).map(|i| (i * i + 3 * i + 1) % p).collect();
            let f = rng.word(p, 9);
            let ys = evaluate(&f, &xs, p);
            assert_eq!(evaluate(&interpolate(p, &xs, &ys), &xs, p), ys);
        }
    }

    #[test]
    fn roots_of_planted_products() {
        let mut rng = SplitMix64::new(3);
        for p in [97u64, 65537, 2_130_706_433] {
            for trial in 0..8u64 {
                let mut planted: Vec<u64> = Vec::new();
                while (planted.len() as u64) < 2 + trial % 4 {
                    let r = rng.next_u64() % p;
                    if !planted.contains(&r) {
                        planted.push(r);
                    }
                }
                // f = prod (x - r) over the planted roots
                let mut f = vec![1u64];
                for &r in &planted {
                    let mut next = vec![0; f.len() + 1];
                    for (i, &c) in f.iter().enumerate() {
                        next[i + 1] = (next[i + 1] + c) % p;
                        next[i] = (next[i] + mulmod(c, (p - r) % p, p)) % p;
                    }
                    f = next;
                }
                planted.sort_unstable();
                assert_eq!(roots(&f, p), planted, "p = {p}");
            }
        }
    }

    #[test]
    fn root_at_zero_reported_once() {
        let p = 97;
        // x^2 (x - 5): roots {0, 5}
        let f = vec![0, 0, mulmod(1, p - 5, p), 1];
        assert_eq!(roots(&f, p), vec![0, 5]);
    }

    #[test]
    fn div_exact_inverts_multiplication() {
        let p = 65537;
        let a = vec![3, 0, 1, 9];
        let b = vec![5, 1, 2];
        let mut prod = vec![0; a.len() + b.len() - 1];
        for (i, &ai) in a.iter().enumerate() {
            for (j, &bj) in b.iter().enumerate() {
                prod[i + j] = (prod[i + j] + mulmod(ai, bj, p)) % p;
            }
        }
        assert_eq!(div_exact(&prod, &b, p), a);
        assert_eq!(div_exact(&prod, &a, p), b);
    }
}
