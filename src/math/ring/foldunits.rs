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
//! computation (2.49 at `s = 32`) with an interval-arithmetic
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
use crate::field::{find_generator, mulmod, powmod};

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
    fn abs_hi(self) -> f64 {
        self.lo.abs().max(self.hi.abs())
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

// ------- the certified alpha tables (atom addresses) -------

/// Verification primes for the torsion pin: `p = 1 mod 2^24` (at
/// least), so `mu_{2s}` maps injectively for every working level.
const CAMERA_PRIMES: [u64; 2] = [2_130_706_433, 2_013_265_921];

/// The certified atom-address table at one level: the exact
/// multiplicative identities
///
/// ```text
///   A_j^D = zeta_{2s}^{t_j} * prod_{c=1}^{k} u_c^{alpha[j-1][c]},
///   A_j = (1 - zeta^j)/(1 - zeta)^{v_j},  v_j = gcd(j, s),
/// ```
///
/// for every atom `j = 1..s-1`, with `k = s/4 - 1` free fold units
/// and `D = denom = max(8, s/8)`: by the denominator law
/// `denom(alpha_j) = max(1, ord(zeta^j)/8)`, the vector `D alpha_j`
/// is integral, and `D = 8` exactly at the skeleton levels 32/64.
/// `alpha` rows are `D alpha_j` — the integer tables the skeleton
/// census carries additively (M1), here *derived and certified*
/// rather than embedded.
#[derive(Debug, Clone)]
pub struct AlphaCertificate {
    /// The level `s`.
    pub level: usize,
    /// The universal denominator `D = max(8, s/8)`.
    pub denom: i64,
    /// Row `j - 1` = the integer exponent vector `D alpha_j`.
    pub alpha: Vec<Vec<i64>>,
    /// `t_j`: the torsion part `zeta_{2s}^{t_j}` of atom `j`'s identity.
    pub torsion2s: Vec<usize>,
    /// Max over atoms of the certified archimedean residual
    /// `sum_embeddings max(0, log|w_j|)` of `w_j = A_j^D prod u^{-D alpha}`.
    pub residual_bound: f64,
    /// The height gap the residuals were tested against: any
    /// non-torsion unit of `Z[zeta_s]` has residual at least this
    /// (Voutier's explicit bound per subfield degree; the quadratic
    /// subfield handled by the fundamental unit of `Q(sqrt 2)`).
    pub height_gap: f64,
}

/// `log(2 |sin(pi a / s)|)` = `log|1 - zeta^a|` under `sigma_1`, as a
/// widened interval. Requires `a != 0 mod s`.
fn log_one_minus_iv(s: usize, a: usize) -> Iv {
    let theta = std::f64::consts::PI * (a % s) as f64 / s as f64;
    Iv::point((2.0 * theta.sin().abs()).ln(), ENTRY_SLACK)
}

/// The minimum certified archimedean residual of a non-torsion unit
/// of `Z[zeta_s]`: `min` over subfield degrees `D | phi(s), D >= 2`
/// of `(phi(s)/D) * gap(D)` with `gap(2) = 0.5 <= log(1 + sqrt 2)`
/// (the only quadratic subfield with non-torsion units is
/// `Q(sqrt 2)`) and `gap(D >= 4) = (1/4)(log log D / log D)^3`
/// (Voutier 1996). A non-torsion unit `w` of inner degree `D` has
/// `sum_embeddings max(0, log|w|) = (phi/D) log M(w) >= (phi/D) gap(D)`.
fn height_gap(s: usize) -> f64 {
    let phi = (s / 2) as f64;
    let mut best = f64::INFINITY;
    let mut d = 2usize;
    while d <= s / 2 {
        let gap = if d == 2 {
            0.5
        } else {
            let ld = (d as f64).ln();
            0.25 * (ld.ln() / ld).powi(3)
        };
        best = best.min(phi / d as f64 * gap);
        d *= 2;
    }
    best
}

/// Pin the torsion part of `A_j^8 prod u_c^{-a8_c}` through one
/// camera: the reduction at a split prime maps `mu_{2s}` injectively,
/// so once the unit is known to be torsion, one evaluation determines
/// it exactly.
fn torsion_pin(s: usize, j: usize, vj: usize, d: u64, a: &[i64], p: u64) -> Result<usize> {
    let two_s = 2 * s as u64;
    if (p - 1) % two_s != 0 {
        return Err(Error::OutOfRange("camera prime lacks 2s-th roots".into()));
    }
    let w2 = powmod(find_generator(p), (p - 1) / two_s, p);
    let z = mulmod(w2, w2, p);
    let inv = |x: u64| powmod(x, p - 2, p);
    let one_minus_z = |e: usize| (1 + p - powmod(z, e as u64, p)) % p;
    let mut x = powmod(one_minus_z(j), d, p);
    x = mulmod(x, inv(powmod(one_minus_z(1), d * vj as u64, p)), p);
    for (c, &a) in a.iter().enumerate() {
        let e = c + 1;
        let uc = mulmod((1 + powmod(z, e as u64, p)) % p, inv(one_minus_z(e)), p);
        let factor = if a >= 0 { inv(uc) } else { uc };
        x = mulmod(x, powmod(factor, a.unsigned_abs(), p), p);
    }
    let mut pw = 1u64;
    for t in 0..2 * s {
        if pw == x {
            return Ok(t);
        }
        pw = mulmod(pw, w2, p);
    }
    Err(Error::Verification(format!(
        "torsion pin: atom {j} at level {s}: unit is not in mu_2s under p = {p}"
    )))
}

/// Derive and certify the atom-address table at `level` (a power of
/// two, `16..=8192`).
///
/// The proof carried by the returned certificate:
/// 1. [`rank_certificate`] passes — the fold units are independent
///    mod torsion, so addresses are unique if they exist.
/// 2. For each atom, a floating solve *proposes* `alpha`, snapped to
///    multiples of `1/D` with `D = max(8, s/8)` (the denominator
///    law); outward-rounded interval arithmetic then certifies the
///    archimedean residual of `w_j = A_j^D prod u_c^{-D alpha_c}` at
///    every conjugate-pair embedding, summed as a bound on
///    `sum max(0, log|w_j|)`.
/// 3. The bound is below `height_gap` — by Voutier's explicit
///    height bound (and the fundamental unit of `Q(sqrt 2)` for the
///    quadratic subfield), `w_j` must be torsion.
/// 4. Two independent cameras pin *which* root of unity, exactly
///    (injective on `mu_{2s}`), and must agree.
///
/// The float solve is only a proposer: soundness rests on 1-4 alone.
pub fn alpha_certificate(level: usize) -> Result<AlphaCertificate> {
    let s = level;
    if !s.is_power_of_two() || !(16..=8192).contains(&s) {
        return Err(Error::OutOfRange(
            "alpha_certificate requires a power of two in 16..=8192".into(),
        ));
    }
    let rank = rank_certificate(s)?;
    if !rank.independent {
        return Err(Error::Verification(format!(
            "alpha_certificate: fold-unit rank not certified at level {s}"
        )));
    }
    let k = s / 4 - 1;
    let places: Vec<usize> = (1..s / 2).step_by(2).collect(); // s/4 of them
                                                              // interval log matrix of the fold units at every place
    let u_iv: Vec<Vec<Iv>> = places
        .iter()
        .map(|&m| {
            (1..=k)
                .map(|c| log_one_minus_iv(s, (m * (c + s / 2)) % s).sub(log_one_minus_iv(s, m * c)))
                .collect()
        })
        .collect();
    let gap = height_gap(s);
    let d = (s as i64 / 8).max(8);
    let mut alpha = Vec::with_capacity(s - 1);
    let mut torsion2s = Vec::with_capacity(s - 1);
    let mut residual_bound = 0.0f64;
    for j in 1..s {
        let vj = gcd(j, s);
        // per-place interval logs of A_j
        let a_iv: Vec<Iv> = places
            .iter()
            .map(|&m| {
                let lj = log_one_minus_iv(s, m * j);
                let l1 = log_one_minus_iv(s, m);
                lj.sub(Iv::point(vj as f64, 0.0).mul(l1))
            })
            .collect();
        // float proposal on the first k places (midpoints)
        let mut aug: Vec<Vec<f64>> = (0..k)
            .map(|r| {
                let mut row: Vec<f64> = u_iv[r].iter().map(|iv| iv.mid()).collect();
                row.push(a_iv[r].mid());
                row
            })
            .collect();
        for col in 0..k {
            let piv = (col..k)
                .max_by(|&a, &b| aug[a][col].abs().partial_cmp(&aug[b][col].abs()).unwrap());
            aug.swap(piv.unwrap(), col);
            let pivot_row = aug[col].clone();
            for (r, row) in aug.iter_mut().enumerate() {
                if r != col {
                    let f = row[col] / pivot_row[col];
                    for (dst, &src) in row[col..].iter_mut().zip(&pivot_row[col..]) {
                        *dst -= f * src;
                    }
                }
            }
        }
        let aj: Vec<i64> = (0..k)
            .map(|r| (d as f64 * aug[r][k] / aug[r][r]).round() as i64)
            .collect();
        // certified residual at every place: D log A - sum (D alpha)_c log u_c
        let mut total = 0.0f64;
        for (pi, _) in places.iter().enumerate() {
            let mut r = Iv::point(d as f64, 0.0).mul(a_iv[pi]);
            for (c, &a) in aj.iter().enumerate() {
                r = r.sub(Iv::point(a as f64, 0.0).mul(u_iv[pi][c]));
            }
            total += 2.0 * r.abs_hi();
        }
        if total >= 0.9 * gap {
            return Err(Error::Verification(format!(
                "alpha_certificate: atom {j} at level {s}: residual {total:.3e} \
                 not below the height gap {gap:.3e}"
            )));
        }
        residual_bound = residual_bound.max(total);
        // torsion pin, two cameras, must agree
        let t0 = torsion_pin(s, j, vj, d as u64, &aj, CAMERA_PRIMES[0])?;
        let t1 = torsion_pin(s, j, vj, d as u64, &aj, CAMERA_PRIMES[1])?;
        if t0 != t1 {
            return Err(Error::Verification(format!(
                "alpha_certificate: atom {j} at level {s}: cameras disagree \
                 ({t0} vs {t1})"
            )));
        }
        alpha.push(aj);
        torsion2s.push(t0);
    }
    Ok(AlphaCertificate {
        level: s,
        denom: d,
        alpha,
        torsion2s,
        residual_bound,
        height_gap: gap,
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

    /// The certificate must reproduce the S4-campaign tables exactly
    /// (embedded provenance: exported from the float log-embedding
    /// solve, then validated end-to-end by the exact censuses at
    /// levels 32/64 — 26,084 and N(128) = 3,758,482,820).
    #[test]
    fn alpha_matches_s4_pins() {
        let c32 = alpha_certificate(32).unwrap();
        assert_eq!(c32.denom, 8);
        assert_eq!(
            c32.alpha,
            vec![
                vec![0, 0, 0, 0, 0, 0, 0],
                vec![8, 0, 0, 0, 0, 0, 0],
                vec![4, 2, -4, 2, 0, -2, 0],
                vec![16, 8, 0, 0, 0, 0, 0],
                vec![4, 2, 0, 2, -4, 2, 0],
                vec![8, 4, 0, 4, 0, -4, 0],
                vec![4, 4, 0, 0, 0, 0, -4],
                vec![32, 16, 0, 8, 0, 0, 0],
                vec![4, 4, 0, 0, 0, 0, 4],
                vec![8, 4, 0, 4, 0, 4, 0],
                vec![4, 2, 0, 2, 4, 2, 0],
                vec![16, 8, 0, 8, 0, 0, 0],
                vec![4, 2, 4, 2, 0, -2, 0],
                vec![8, 8, 0, 0, 0, 0, 0],
                vec![8, 0, 0, 0, 0, 0, 0],
                vec![64, 32, 0, 16, 0, 0, 0],
                vec![8, 0, 0, 0, 0, 0, 0],
                vec![8, 8, 0, 0, 0, 0, 0],
                vec![4, 2, 4, 2, 0, -2, 0],
                vec![16, 8, 0, 8, 0, 0, 0],
                vec![4, 2, 0, 2, 4, 2, 0],
                vec![8, 4, 0, 4, 0, 4, 0],
                vec![4, 4, 0, 0, 0, 0, 4],
                vec![32, 16, 0, 8, 0, 0, 0],
                vec![4, 4, 0, 0, 0, 0, -4],
                vec![8, 4, 0, 4, 0, -4, 0],
                vec![4, 2, 0, 2, -4, 2, 0],
                vec![16, 8, 0, 0, 0, 0, 0],
                vec![4, 2, -4, 2, 0, -2, 0],
                vec![8, 0, 0, 0, 0, 0, 0],
                vec![0, 0, 0, 0, 0, 0, 0]
            ]
        );
        let c64 = alpha_certificate(64).unwrap();
        assert_eq!(c64.denom, 8);
        assert_eq!(
            c64.alpha,
            vec![
                vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                vec![8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                vec![4, 2, -4, 1, 0, -2, 0, 1, 0, 0, 0, -1, 0, 0, 0],
                vec![16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                vec![4, 2, 0, 1, -4, 0, 0, 1, 0, -2, 0, 1, 0, 0, 0],
                vec![8, 4, 0, 2, 0, -4, 0, 2, 0, 0, 0, -2, 0, 0, 0],
                vec![4, 2, 0, 2, 0, 0, -4, 0, 0, 0, 0, 0, 0, -2, 0],
                vec![32, 16, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                vec![4, 2, 0, 2, 0, 0, 0, 0, -4, 0, 0, 0, 0, 2, 0],
                vec![8, 4, 0, 2, 0, 0, 0, 2, 0, -4, 0, 2, 0, 0, 0],
                vec![4, 2, 0, 1, 0, 0, 0, 1, 0, 2, -4, 1, 0, 0, 0],
                vec![16, 8, 0, 4, 0, 0, 0, 4, 0, 0, 0, -4, 0, 0, 0],
                vec![4, 2, 0, 1, 0, 2, 0, 1, 0, 0, 0, -1, -4, 0, 0],
                vec![8, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0],
                vec![4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4],
                vec![64, 32, 0, 16, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
                vec![4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
                vec![8, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
                vec![4, 2, 0, 1, 0, 2, 0, 1, 0, 0, 0, -1, 4, 0, 0],
                vec![16, 8, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0],
                vec![4, 2, 0, 1, 0, 0, 0, 1, 0, 2, 4, 1, 0, 0, 0],
                vec![8, 4, 0, 2, 0, 0, 0, 2, 0, 4, 0, 2, 0, 0, 0],
                vec![4, 2, 0, 2, 0, 0, 0, 0, 4, 0, 0, 0, 0, 2, 0],
                vec![32, 16, 0, 8, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
                vec![4, 2, 0, 2, 0, 0, 4, 0, 0, 0, 0, 0, 0, -2, 0],
                vec![8, 4, 0, 2, 0, 4, 0, 2, 0, 0, 0, -2, 0, 0, 0],
                vec![4, 2, 0, 1, 4, 0, 0, 1, 0, -2, 0, 1, 0, 0, 0],
                vec![16, 8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                vec![4, 2, 4, 1, 0, -2, 0, 1, 0, 0, 0, -1, 0, 0, 0],
                vec![8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                vec![8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                vec![128, 64, 0, 32, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0],
                vec![8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                vec![8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                vec![4, 2, 4, 1, 0, -2, 0, 1, 0, 0, 0, -1, 0, 0, 0],
                vec![16, 8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                vec![4, 2, 0, 1, 4, 0, 0, 1, 0, -2, 0, 1, 0, 0, 0],
                vec![8, 4, 0, 2, 0, 4, 0, 2, 0, 0, 0, -2, 0, 0, 0],
                vec![4, 2, 0, 2, 0, 0, 4, 0, 0, 0, 0, 0, 0, -2, 0],
                vec![32, 16, 0, 8, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
                vec![4, 2, 0, 2, 0, 0, 0, 0, 4, 0, 0, 0, 0, 2, 0],
                vec![8, 4, 0, 2, 0, 0, 0, 2, 0, 4, 0, 2, 0, 0, 0],
                vec![4, 2, 0, 1, 0, 0, 0, 1, 0, 2, 4, 1, 0, 0, 0],
                vec![16, 8, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0],
                vec![4, 2, 0, 1, 0, 2, 0, 1, 0, 0, 0, -1, 4, 0, 0],
                vec![8, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
                vec![4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
                vec![64, 32, 0, 16, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
                vec![4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4],
                vec![8, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0],
                vec![4, 2, 0, 1, 0, 2, 0, 1, 0, 0, 0, -1, -4, 0, 0],
                vec![16, 8, 0, 4, 0, 0, 0, 4, 0, 0, 0, -4, 0, 0, 0],
                vec![4, 2, 0, 1, 0, 0, 0, 1, 0, 2, -4, 1, 0, 0, 0],
                vec![8, 4, 0, 2, 0, 0, 0, 2, 0, -4, 0, 2, 0, 0, 0],
                vec![4, 2, 0, 2, 0, 0, 0, 0, -4, 0, 0, 0, 0, 2, 0],
                vec![32, 16, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                vec![4, 2, 0, 2, 0, 0, -4, 0, 0, 0, 0, 0, 0, -2, 0],
                vec![8, 4, 0, 2, 0, -4, 0, 2, 0, 0, 0, -2, 0, 0, 0],
                vec![4, 2, 0, 1, -4, 0, 0, 1, 0, -2, 0, 1, 0, 0, 0],
                vec![16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                vec![4, 2, -4, 1, 0, -2, 0, 1, 0, 0, 0, -1, 0, 0, 0],
                vec![8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]
        );
    }

    #[test]
    fn alpha_certified_at_working_levels() {
        for s in [16usize, 32, 64, 128] {
            let c = alpha_certificate(s).unwrap();
            assert_eq!(c.alpha.len(), s - 1, "s={s}");
            // atom 1 is the trivial unit: zero address, zero torsion
            assert!(c.alpha[0].iter().all(|&x| x == 0), "s={s}");
            assert_eq!(c.torsion2s[0], 0, "s={s}");
            // the margin is astronomical, not marginal
            assert!(
                c.residual_bound < 1e-4 * c.height_gap,
                "s={s}: residual {} vs gap {}",
                c.residual_bound,
                c.height_gap
            );
            // denominator law max(1, ord/8): 8*alpha divisible by
            // 8/denom on each order class
            for (ji, row) in c.alpha.iter().enumerate() {
                let ord = s / super::gcd(ji + 1, s);
                let denom = (ord as i64 / 8).clamp(1, c.denom);
                let step = c.denom / denom;
                assert!(row.iter().all(|&x| x % step == 0), "s={s} atom {}", ji + 1);
            }
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
