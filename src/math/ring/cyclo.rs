//! [`Cyclo`] — the element type of `Z[zeta_s]`.

use super::fold;
use crate::domain::MultiplicativeSubgroup;
use crate::error::{Error, Result};
use crate::field::mulmod;

/// An element of `Z[zeta_s] = Z[x]/(x^{s/2}+1)`, `s = 2 * coeffs.len()`
/// a power of two; coefficients on the half-basis `1..zeta^{s/2-1}`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Cyclo {
    coeffs: Vec<i64>,
}

impl Cyclo {
    /// The zero element of `Z[zeta_s]`.
    pub fn zero(s: usize) -> Result<Self> {
        if !s.is_power_of_two() || s < 4 {
            return Err(Error::OutOfRange(
                "Cyclo requires s a power of two, s >= 4".into(),
            ));
        }
        Ok(Cyclo {
            coeffs: vec![0; s / 2],
        })
    }

    /// `zeta^exp` — the fold made total.
    pub fn monomial(s: usize, exp: usize) -> Result<Self> {
        let mut z = Self::zero(s)?;
        let (i, sg) = fold(s / 2, exp);
        z.coeffs[i] = sg;
        Ok(z)
    }

    /// From half-basis coefficients (length must be a power of two >= 2).
    pub fn from_coeffs(coeffs: Vec<i64>) -> Result<Self> {
        if !coeffs.len().is_power_of_two() || coeffs.len() < 2 {
            return Err(Error::OutOfRange(
                "coefficient length must be a power of two >= 2".into(),
            ));
        }
        Ok(Cyclo { coeffs })
    }

    /// Subgroup order `s` (so the ring is `Z[zeta_s]`).
    #[must_use]
    pub fn s(&self) -> usize {
        2 * self.coeffs.len()
    }
    /// Half-basis coefficients.
    #[must_use]
    pub fn coeffs(&self) -> &[i64] {
        &self.coeffs
    }
    /// Number of nonzero coefficients.
    #[must_use]
    pub fn weight(&self) -> usize {
        self.coeffs.iter().filter(|&&c| c != 0).count()
    }
    /// `sum c_i^2` — the anticorrelation-law quantity.
    #[must_use]
    pub fn sq_sum(&self) -> i128 {
        self.coeffs.iter().map(|&c| (c as i128) * (c as i128)).sum()
    }
    /// `max |c_i|`.
    #[must_use]
    pub fn height(&self) -> i64 {
        self.coeffs.iter().map(|&c| c.abs()).max().unwrap_or(0)
    }
    /// True iff this is the zero element.
    #[must_use]
    pub fn is_zero(&self) -> bool {
        self.coeffs.iter().all(|&c| c == 0)
    }

    fn check_same(&self, o: &Cyclo) -> Result<()> {
        if self.coeffs.len() != o.coeffs.len() {
            return Err(Error::OutOfRange("mismatched ring levels".into()));
        }
        Ok(())
    }

    /// Coefficient-wise sum.
    pub fn add(&self, o: &Cyclo) -> Result<Cyclo> {
        self.check_same(o)?;
        let coeffs = self
            .coeffs
            .iter()
            .zip(&o.coeffs)
            .map(|(a, b)| a.checked_add(*b).ok_or(()))
            .collect::<std::result::Result<Vec<_>, ()>>()
            .map_err(|_| Error::Unsupported("coefficients within i64 range".into()))?;
        Ok(Cyclo { coeffs })
    }

    /// Coefficient-wise difference.
    pub fn sub(&self, o: &Cyclo) -> Result<Cyclo> {
        self.add(&o.neg())
    }

    /// Negation.
    #[must_use]
    pub fn neg(&self) -> Cyclo {
        Cyclo {
            coeffs: self.coeffs.iter().map(|&c| -c).collect(),
        }
    }

    /// Negacyclic product; accumulates in i128, errors if a coefficient
    /// leaves i64.
    pub fn mul(&self, o: &Cyclo) -> Result<Cyclo> {
        self.check_same(o)?;
        let half = self.coeffs.len();
        let mut acc = vec![0i128; half];
        for (i, &a) in self.coeffs.iter().enumerate() {
            if a == 0 {
                continue;
            }
            for (j, &b) in o.coeffs.iter().enumerate() {
                if b == 0 {
                    continue;
                }
                let (idx, sg) = fold(half, i + j);
                acc[idx] += (a as i128) * (b as i128) * (sg as i128);
            }
        }
        let coeffs = acc
            .into_iter()
            .map(|v| i64::try_from(v).map_err(|_| ()))
            .collect::<std::result::Result<Vec<_>, ()>>()
            .map_err(|_| Error::Unsupported("coefficients within i64 range".into()))?;
        Ok(Cyclo { coeffs })
    }

    /// Negacyclic product via two-prime NTT + signed CRT when the
    /// height bound permits (`n * h_a * h_b < 2^123`); otherwise the
    /// schoolbook path. Identical results to [`Cyclo::mul`]; the fast
    /// path for large `s` and batch campaigns.
    pub fn mul_ntt(&self, o: &Cyclo) -> Result<Cyclo> {
        self.check_same(o)?;
        let n = self.coeffs.len() as u128;
        let (ha, hb) = (self.height() as u128, o.height() as u128);
        let bound_ok = ha
            .checked_mul(hb)
            .and_then(|x| x.checked_mul(n))
            .map(|x| x < (1u128 << 123))
            .unwrap_or(false);
        if !bound_ok || self.coeffs.len() < 2 {
            return self.mul(o);
        }
        let prod = super::ntt::negacyclic_mul_exact(&self.coeffs, &o.coeffs)?;
        let coeffs = prod
            .into_iter()
            .map(|v| i64::try_from(v).map_err(|_| ()))
            .collect::<std::result::Result<Vec<_>, ()>>()
            .map_err(|_| Error::Unsupported("coefficients within i64 range".into()))?;
        Ok(Cyclo { coeffs })
    }

    /// The Galois action `sigma_m : zeta -> zeta^m` (`m` odd).
    pub fn galois(&self, m: usize) -> Result<Cyclo> {
        if m % 2 == 0 {
            return Err(Error::OutOfRange("galois requires odd m".into()));
        }
        let half = self.coeffs.len();
        let m = m % (2 * half); // canonical rep; parity survives (2*half even)
        let mut out = vec![0i64; half];
        for (i, &c) in self.coeffs.iter().enumerate() {
            if c != 0 {
                let (idx, sg) = fold(half, (m * i) % (2 * half));
                out[idx] += sg * c;
            }
        }
        Ok(Cyclo { coeffs: out })
    }

    /// Dilation: multiply by `zeta^d`.
    pub fn dilate(&self, d: usize) -> Cyclo {
        let half = self.coeffs.len();
        let mut out = vec![0i64; half];
        for (i, &c) in self.coeffs.iter().enumerate() {
            if c != 0 {
                let (idx, sg) = fold(half, i + d);
                out[idx] += sg * c;
            }
        }
        Cyclo { coeffs: out }
    }

    /// Complex conjugation `sigma_{-1}`.
    #[must_use]
    pub fn conj(&self) -> Cyclo {
        self.galois(2 * self.coeffs.len() - 1)
            .expect("s-1 is odd for s a power of two >= 4")
    }

    // ------- content-map conveniences -------
    //
    // `1 - zeta^e` is the content map: every difference of roots of
    // unity is a unit times one of these, and every census value is a
    // product of them. These constructors are the single exact home
    // for the quantities the landscape verifiers previously hand-rolled
    // (experiments/analysis/toolkit.py mirrors them in Python).

    /// `1 - zeta^exp`.
    pub fn one_minus(s: usize, exp: usize) -> Result<Self> {
        Self::monomial(s, 0)?.sub(&Self::monomial(s, exp)?)
    }

    /// Exact `prod_{e in exps} (1 - zeta^e)` — the A-map value of a
    /// subset; the quantity the censuses count. NTT-accelerated where
    /// the height bound permits.
    pub fn prod_one_minus(s: usize, exps: &[usize]) -> Result<Self> {
        let mut acc = Self::monomial(s, 0)?;
        for &e in exps {
            acc = acc.mul_ntt(&Self::one_minus(s, e)?)?;
        }
        Ok(acc)
    }

    /// Exact elementary symmetric functions `e_0..e_m` of
    /// `{zeta^e : e in exps}` — the subset's embedding coordinates
    /// (the characteristic-zero side of the syndrome alphabet).
    pub fn e_vector(s: usize, exps: &[usize], m: usize) -> Result<Vec<Self>> {
        let mut es = vec![Self::zero(s)?; m + 1];
        es[0] = Self::monomial(s, 0)?;
        for (t, &e) in exps.iter().enumerate() {
            let top = m.min(t + 1);
            for j in (1..=top).rev() {
                let shifted = es[j - 1].dilate(e);
                es[j] = es[j].add(&shifted)?;
            }
        }
        Ok(es)
    }

    /// `Some(c)` iff this element is the rational integer `c`.
    #[must_use]
    pub fn as_int(&self) -> Option<i64> {
        if self.coeffs[1..].iter().all(|&c| c == 0) {
            Some(self.coeffs[0])
        } else {
            None
        }
    }

    /// Equality against a rational integer, without constructing it.
    #[must_use]
    pub fn eq_int(&self, v: i64) -> bool {
        self.as_int() == Some(v)
    }

    /// Evaluate at `x` in `F_p` (Horner; negative coefficients reduced).
    #[must_use]
    pub fn eval_at(&self, x: u64, p: u64) -> u64 {
        // rem_euclid(p as i64) below requires p < 2^63; every good prime
        // in the program is < 2^62.
        assert!(p < (1 << 63), "eval_at requires p < 2^63");
        let mut v = 0u64;
        for &c in self.coeffs.iter().rev() {
            let cu = c.rem_euclid(p as i64) as u64;
            v = (mulmod(v, x, p) + cu) % p;
        }
        v
    }

    /// `N(v) mod p` for a good prime `p = 1 (mod s)`: the product of the
    /// `s/2` Galois-conjugate evaluations at the odd powers of an
    /// order-`s` element. `p | N(v)` iff this is zero — the accident /
    /// cleanliness test, with no big-integer arithmetic.
    pub fn norm_mod(&self, p: u64) -> Result<u64> {
        let sg = MultiplicativeSubgroup::new(p, self.s())?;
        self.norm_mod_in(&sg)
    }

    /// [`Cyclo::norm_mod`] against a caller-held subgroup — the batch
    /// entry point: campaigns evaluating many values at one prime pay
    /// the subgroup construction (primality test, generator search)
    /// once, not per value.
    pub fn norm_mod_in(&self, sg: &MultiplicativeSubgroup) -> Result<u64> {
        let s = self.s();
        if sg.order() != s {
            return Err(Error::OutOfRange(format!(
                "subgroup order {} != ring level s = {s}",
                sg.order()
            )));
        }
        let p = sg.p();
        let els = sg.elements();
        let mut n = 1u64;
        let mut m = 1usize;
        while m < s {
            n = mulmod(n, self.eval_at(els[m], p), p);
            m += 2;
        }
        Ok(n)
    }

    /// Batch [`Cyclo::norm_mod_in`] over many coefficient vectors with a
    /// shared subgroup, in parallel (rayon) — the campaign entry point
    /// for extremal-norm sweeps (millions of values, one prime).
    pub fn norms_mod_batch(coeffs: &[Vec<i64>], sg: &MultiplicativeSubgroup) -> Result<Vec<u64>> {
        use rayon::prelude::*;
        coeffs
            .par_iter()
            .map(|v| Cyclo::from_coeffs(v.clone())?.norm_mod_in(sg))
            .collect()
    }

    /// Exact field norm via a fraction-free (Bareiss) determinant of the
    /// multiplication matrix, in i128; errors if any intermediate
    /// overflows. Norms of larger height are reconstructed by the
    /// caller via CRT over [`Cyclo::norm_mod`].
    pub fn norm_i128(&self) -> Result<i128> {
        let half = self.coeffs.len();
        let mut m = vec![vec![0i128; half]; half];
        for (i, mcol) in (0..half).map(|i| self.dilate(i)).enumerate() {
            for (row, &cv) in m.iter_mut().zip(mcol.coeffs.iter()) {
                row[i] = cv as i128;
            }
        }
        let ovf = || Error::Unsupported("norm overflow (i128); use norm_mod + CRT".into());
        let mut sign = 1i128;
        let mut prev = 1i128;
        for k in 0..half - 1 {
            if m[k][k] == 0 {
                let piv = (k + 1..half).find(|&r| m[r][k] != 0);
                match piv {
                    None => return Ok(0),
                    Some(r) => {
                        m.swap(k, r);
                        sign = -sign;
                    }
                }
            }
            for i in k + 1..half {
                for j in k + 1..half {
                    let a = m[i][j].checked_mul(m[k][k]).ok_or_else(ovf)?;
                    let b = m[i][k].checked_mul(m[k][j]).ok_or_else(ovf)?;
                    m[i][j] = a.checked_sub(b).ok_or_else(ovf)? / prev;
                }
            }
            prev = m[k][k];
        }
        Ok(sign * m[half - 1][half - 1])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::powmod;

    #[test]
    fn fold_is_the_relation() {
        // zeta^exp = sign * zeta^index, verified via eval in F_p
        let (p, s) = (97u64, 16usize);
        let sg = MultiplicativeSubgroup::new(p, s).unwrap();
        let g = sg.elements()[1];
        for exp in 0..4 * s {
            let (idx, sign) = fold(s / 2, exp);
            let lhs = powmod(g, (exp % s) as u64, p);
            let rhs = if sign == 1 {
                powmod(g, idx as u64, p)
            } else {
                (p - powmod(g, idx as u64, p)) % p
            };
            assert_eq!(lhs, rhs, "exp {exp}");
        }
    }

    #[test]
    fn monomial_mul_matches_fold() {
        for a in 0..32 {
            for b in 0..32 {
                let x = Cyclo::monomial(32, a).unwrap();
                let y = Cyclo::monomial(32, b).unwrap();
                assert_eq!(x.mul(&y).unwrap(), Cyclo::monomial(32, a + b).unwrap());
            }
        }
    }

    #[test]
    fn content_map_identities() {
        // prod over the primitive exponents = Phi_s(1) = 2
        let odds: Vec<usize> = (1..16).step_by(2).collect();
        assert!(Cyclo::prod_one_minus(16, &odds).unwrap().eq_int(2));
        // prod over ALL nonzero exponents = s
        let all: Vec<usize> = (1..16).collect();
        assert!(Cyclo::prod_one_minus(16, &all).unwrap().eq_int(16));
        // the fold identity (1-z^e)(1-z^{e+s/2}) = 1-z^{2e} at s = 32
        for e in [1usize, 3, 8, 11, 15] {
            let lhs = Cyclo::one_minus(32, e)
                .unwrap()
                .mul(&Cyclo::one_minus(32, e + 16).unwrap())
                .unwrap();
            assert_eq!(lhs, Cyclo::one_minus(32, 2 * e).unwrap(), "e={e}");
        }
    }

    #[test]
    fn hand_census_at_s8() {
        // the three-subset shell census of the counting chapter's
        // sec:cc-census: values 2, 4 + 2*sqrt2, 4 - 2*sqrt2
        // (sqrt2 = zeta - zeta^3 on the half-basis)
        assert!(Cyclo::prod_one_minus(8, &[1, 3, 5, 7]).unwrap().eq_int(2));
        assert_eq!(
            Cyclo::prod_one_minus(8, &[2, 3, 5, 6]).unwrap(),
            Cyclo::from_coeffs(vec![4, 2, 0, -2]).unwrap()
        );
        assert_eq!(
            Cyclo::prod_one_minus(8, &[1, 2, 6, 7]).unwrap(),
            Cyclo::from_coeffs(vec![4, -2, 0, 2]).unwrap()
        );
    }

    #[test]
    fn e_vector_identities() {
        // all nonzero exponents: prod(x - z^e) = 1 + x + ... + x^{s-1},
        // so e_j = (-1)^j
        let all: Vec<usize> = (1..16).collect();
        let es = Cyclo::e_vector(16, &all, 5).unwrap();
        for (j, ej) in es.iter().enumerate() {
            let want = if j % 2 == 0 { 1 } else { -1 };
            assert!(ej.eq_int(want), "j={j}");
        }
        // Vieta: the top symmetric function is zeta^{sum of exponents}
        let exps = [1usize, 3, 5, 7];
        let es = Cyclo::e_vector(16, &exps, 4).unwrap();
        assert_eq!(es[4], Cyclo::monomial(16, 16).unwrap());
        // the alternating sum of the e-vector IS the content product
        let mut alt = Cyclo::zero(16).unwrap();
        for (j, ej) in es.iter().enumerate() {
            alt = if j % 2 == 0 {
                alt.add(ej).unwrap()
            } else {
                alt.sub(ej).unwrap()
            };
        }
        assert_eq!(alt, Cyclo::prod_one_minus(16, &exps).unwrap());
    }

    #[test]
    fn as_int_discriminates() {
        assert_eq!(Cyclo::monomial(8, 0).unwrap().as_int(), Some(1));
        assert_eq!(Cyclo::one_minus(8, 2).unwrap().as_int(), None);
        // 1 - zeta^{s/2} = 2: the ramified generator's norm-2 witness
        assert_eq!(Cyclo::one_minus(8, 4).unwrap().as_int(), Some(2));
    }

    #[test]
    fn galois_composes_and_conj_involutes() {
        let v = Cyclo::from_coeffs(vec![3, -1, 4, 1, -5, 9, 2, -6]).unwrap();
        let a = v.galois(3).unwrap().galois(5).unwrap();
        let b = v.galois(15).unwrap();
        assert_eq!(a, b);
        assert_eq!(v.conj().conj(), v);
    }

    #[test]
    fn norm_mod_matches_exact() {
        let v = Cyclo::from_coeffs(vec![2, 2, 0, 2, -1, 0, 0, -1]).unwrap();
        // the s=16 die-out witness (2026-07-24): N = 9986 = 2 * 4993
        let n = v.norm_i128().unwrap();
        assert_eq!(n, 9986);
        for p in [97u64, 113, 4993, 65537] {
            let nm = v.norm_mod(p).unwrap();
            assert_eq!(nm as i128, n.rem_euclid(p as i128), "p={p}");
        }
        assert_eq!(v.norm_mod(4993).unwrap(), 0, "4993 is its accident prime");
    }

    #[test]
    fn norm_is_galois_invariant_and_multiplicative() {
        let v = Cyclo::from_coeffs(vec![1, 2, 0, -1, 0, 3, -2, 1]).unwrap();
        let w = Cyclo::from_coeffs(vec![0, 1, 1, 0, -2, 0, 1, 0]).unwrap();
        assert_eq!(
            v.norm_i128().unwrap(),
            v.galois(7).unwrap().norm_i128().unwrap()
        );
        assert_eq!(
            v.norm_i128().unwrap().abs(),
            v.dilate(5).norm_i128().unwrap().abs()
        );
        let prod = v.mul(&w).unwrap();
        assert_eq!(
            prod.norm_i128().unwrap(),
            v.norm_i128().unwrap() * w.norm_i128().unwrap()
        );
    }

    #[test]
    fn eval_of_monomial_is_domain_power() {
        let (p, s) = (65537u64, 16usize);
        let sg = MultiplicativeSubgroup::new(p, s).unwrap();
        let g = sg.elements()[1];
        for e in 0..s {
            let z = Cyclo::monomial(s, e).unwrap();
            assert_eq!(z.eval_at(g, p), powmod(g, e as u64, p));
        }
    }
}
