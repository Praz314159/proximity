//! The fold-unit lattice of `Z[zeta_s]` — exact constructors and the
//! rank certificate.
//!
//! The fold unit at exponent `e` is `u_e = (1 + zeta^e)/(1 - zeta^e)`:
//! the exchange rate between the two halves of an antipodal slot, and
//! the multiplicative bookkeeping currency of the descent calculus
//! (silence lemma, fiber derivations). Writing `n = ord(zeta^e)`, the
//! quotient has the exact closed form
//!
//! ```text
//!   u_e = 1 + zeta^e + zeta^{2e} + ... + zeta^{(n/2) e}
//! ```
//!
//! (`1 + x = 1 - x^{n/2+1}` for `x^{n/2} = -1`, then the geometric
//! telescope), so every fold unit is an explicit element of the ring —
//! no division needed. The two exact identities `u_e u_{s/2-e} = -1`
//! and `u_{s/4} = zeta^{s/4}` are pinned by tests, as is Hilbert 90
//! (`sigma(u_e) = u_e^{-1}` for odd `e`, `sigma = sigma_{s/2+1}`).
//!
//! The *rank certificate* replaces the bare floating min-singular-value
//! computation (2.49 at `s = 32`, stage 56) with an interval-arithmetic
//! determinant of the log-embedding matrix of `u_1..u_{s/4-1}`: if the
//! certified determinant interval excludes zero, the units are
//! multiplicatively independent modulo torsion (a relation
//! `prod u^{m_e} = zeta^t` has all archimedean absolute values 1, so
//! its log vector is exactly zero; a nonsingular log matrix forces
//! `m = 0`). The error model is explicit and generous: every libm
//! evaluation is widened by `ENTRY_SLACK` absolute, and all
//! elimination arithmetic is outward-rounded interval arithmetic, so
//! the certificate is honest modulo only the widened-entry assumption.

use super::Cyclo;
use crate::error::{Error, Result};

/// The fold unit `u_e` as an exact ring element (closed form).
/// Errors for `e = 0` or `e = s/2` (no slot there).
pub fn fold_unit(s: usize, e: usize) -> Result<Cyclo> {
    if !s.is_power_of_two() || s < 8 {
        return Err(Error::OutOfRange(
            "fold_unit requires s a power of two, s >= 8".into(),
        ));
    }
    let e = e % s;
    if e == 0 || e == s / 2 {
        return Err(Error::OutOfRange(
            "fold_unit undefined at e = 0 and e = s/2".into(),
        ));
    }
    let ord = s / gcd(e, s);
    let mut acc = Cyclo::zero(s)?;
    for j in 0..=(ord / 2) {
        acc = acc.add(&Cyclo::monomial(s, j * e)?)?;
    }
    Ok(acc)
}

fn gcd(a: usize, b: usize) -> usize {
    if a == 0 {
        b
    } else {
        gcd(b % a, a)
    }
}

// ------- interval arithmetic (outward-rounded, minimal) -------

/// Absolute widening applied to every computed matrix entry — generous
/// against libm's ~1-ulp sin/log accuracy at these magnitudes.
const ENTRY_SLACK: f64 = 1e-12;

#[derive(Clone, Copy, Debug)]
struct Iv {
    lo: f64,
    hi: f64,
}

fn next_down(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return x;
    }
    let bits = x.to_bits();
    let next = if x > 0.0 {
        bits - 1
    } else if x < 0.0 {
        bits + 1
    } else {
        (-f64::MIN_POSITIVE).to_bits()
    };
    f64::from_bits(next)
}

fn next_up(x: f64) -> f64 {
    -next_down(-x)
}

impl Iv {
    fn point(x: f64, slack: f64) -> Iv {
        Iv {
            lo: next_down(x - slack),
            hi: next_up(x + slack),
        }
    }
    fn sub(self, o: Iv) -> Iv {
        Iv {
            lo: next_down(self.lo - o.hi),
            hi: next_up(self.hi - o.lo),
        }
    }
    fn mul(self, o: Iv) -> Iv {
        let cands = [
            self.lo * o.lo,
            self.lo * o.hi,
            self.hi * o.lo,
            self.hi * o.hi,
        ];
        let lo = cands.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = cands.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Iv {
            lo: next_down(lo),
            hi: next_up(hi),
        }
    }
    fn div(self, o: Iv) -> Option<Iv> {
        if o.lo <= 0.0 && o.hi >= 0.0 {
            return None;
        }
        let cands = [
            self.lo / o.lo,
            self.lo / o.hi,
            self.hi / o.lo,
            self.hi / o.hi,
        ];
        let lo = cands.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = cands.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Some(Iv {
            lo: next_down(lo),
            hi: next_up(hi),
        })
    }
    fn contains_zero(self) -> bool {
        self.lo <= 0.0 && self.hi >= 0.0
    }
    fn mid(self) -> f64 {
        0.5 * (self.lo + self.hi)
    }
}

/// The certified determinant interval of the fold-unit log-embedding
/// matrix, and the verdict.
#[derive(Debug, Clone, Copy)]
pub struct RankCertificate {
    /// Lower bound of the certified determinant interval.
    pub det_lo: f64,
    /// Upper bound of the certified determinant interval.
    pub det_hi: f64,
    /// True iff the interval excludes zero: the fold units
    /// `u_1..u_{s/4-1}` are multiplicatively independent mod torsion.
    pub independent: bool,
}

/// Certify multiplicative independence (mod torsion) of the free fold
/// units `u_1..u_{s/4-1}` at level `s`: interval-arithmetic determinant
/// of the log-embedding matrix `log|sigma_j(u_e)|`, one embedding per
/// conjugate pair.
pub fn rank_certificate(s: usize) -> Result<RankCertificate> {
    if !s.is_power_of_two() || s < 8 {
        return Err(Error::OutOfRange(
            "rank_certificate requires s a power of two, s >= 8".into(),
        ));
    }
    let k = s / 4 - 1;
    // embeddings: sigma_j, j odd, one per conjugate pair; drop one to
    // match rank (unit theorem: rank = #pairs - 1); entries
    // log|u_e| under sigma_j = log|cot(pi j e / s)|.
    let js: Vec<usize> = (1..s / 2).step_by(2).take(k).collect();
    let mut m = vec![vec![Iv::point(0.0, 0.0); k]; k];
    for (row, &j) in js.iter().enumerate() {
        for (col, e) in (1..=k).enumerate() {
            let theta = std::f64::consts::PI * (j * e) as f64 / s as f64;
            let val = (theta.cos() / theta.sin()).abs().ln();
            m[row][col] = Iv::point(val, ENTRY_SLACK);
        }
    }
    // interval Gaussian elimination, partial pivoting on midpoints
    let mut det = Iv::point(1.0, 0.0);
    for col in 0..k {
        let piv = (col..k)
            .max_by(|&a, &b| {
                m[a][col]
                    .mid()
                    .abs()
                    .partial_cmp(&m[b][col].mid().abs())
                    .unwrap()
            })
            .unwrap();
        if piv != col {
            m.swap(piv, col);
            det = det.mul(Iv::point(-1.0, 0.0));
        }
        let p = m[col][col];
        if p.contains_zero() {
            return Ok(RankCertificate {
                det_lo: f64::NEG_INFINITY,
                det_hi: f64::INFINITY,
                independent: false,
            });
        }
        det = det.mul(p);
        let pivot_row = m[col].clone();
        for row_vec in m.iter_mut().skip(col + 1) {
            let factor = match row_vec[col].div(p) {
                Some(f) => f,
                None => unreachable!("pivot excludes zero"),
            };
            for (dst, &src) in row_vec[col..].iter_mut().zip(&pivot_row[col..]) {
                *dst = dst.sub(factor.mul(src));
            }
        }
    }
    Ok(RankCertificate {
        det_lo: det.lo,
        det_hi: det.hi,
        independent: !det.contains_zero(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_form_is_the_quotient() {
        // u_e * (1 - zeta^e) = 1 + zeta^e, at s = 32, all slots
        for e in 1..32usize {
            if e == 16 {
                continue;
            }
            let u = fold_unit(32, e).unwrap();
            let lhs = u.mul(&Cyclo::one_minus(32, e).unwrap()).unwrap();
            let rhs = Cyclo::monomial(32, 0)
                .unwrap()
                .add(&Cyclo::monomial(32, e).unwrap())
                .unwrap();
            assert_eq!(lhs, rhs, "e={e}");
        }
    }

    #[test]
    fn the_two_identities() {
        // u_e * u_{s/2-e} = -1
        for e in 1..16usize {
            if e == 8 {
                continue;
            }
            let prod = fold_unit(32, e)
                .unwrap()
                .mul(&fold_unit(32, 16 - e).unwrap())
                .unwrap();
            assert!(prod.eq_int(-1), "e={e}");
        }
        // u_{s/4} = zeta^{s/4} (pure torsion)
        assert_eq!(fold_unit(32, 8).unwrap(), Cyclo::monomial(32, 8).unwrap());
    }

    #[test]
    fn hilbert_90_inversion() {
        // sigma_{s/2+1}(u_e) = u_e^{-1} for odd e: the odd fold units
        // are the norm-one (Hilbert-90) units of the rung
        for e in [1usize, 3, 5, 7, 9, 11, 13, 15] {
            let u = fold_unit(32, e).unwrap();
            let su = u.galois(17).unwrap();
            assert!(u.mul(&su).unwrap().eq_int(1), "e={e}");
        }
        // and even fold units are sigma-fixed
        for e in [2usize, 4, 6, 10, 12, 14] {
            let u = fold_unit(32, e).unwrap();
            assert_eq!(u.galois(17).unwrap(), u, "e={e}");
        }
    }

    #[test]
    fn rank_certified_at_working_levels() {
        for s in [16usize, 32, 64, 128] {
            let cert = rank_certificate(s).unwrap();
            assert!(cert.independent, "s={s}: {cert:?}");
            let width = cert.det_hi - cert.det_lo;
            let scale = cert.det_lo.abs().max(cert.det_hi.abs());
            assert!(width < 1e-3 * scale, "s={s}: interval too wide: {cert:?}");
        }
    }
}
