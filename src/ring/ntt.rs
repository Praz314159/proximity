//! Number-theoretic transforms for the power-of-two rings.
//!
//! Standard, well-known optimizations (cf. the NTT tutorials in
//! `papers/` and M. Albrecht's power-of-two-rings notes): an iterative
//! radix-2 Cooley-Tukey NTT over an NTT-friendly prime `q = i*2n + 1`,
//! and the *negacyclic* (negative-wrapped) convolution for
//! `k[x]/(x^n + 1)` by pre-twisting with a primitive `2n`-th root
//! `psi` — the ring relation `x^n = -1` becomes the twist, exactly as
//! the fold is the relation in coefficient form.
//!
//! Exact `Z[zeta_s]` products ([`negacyclic_mul_exact`]) run the
//! negacyclic NTT modulo two fixed 62-bit primes and reconstruct by
//! signed CRT; inputs whose height bound exceeds the CRT range fall
//! back to the schoolbook path in [`super::Cyclo::mul`]. Scalar
//! butterflies only — the HEXL/rokoko-style preconditioned and
//! vectorized variants are a known upgrade path, not a semantic one.

use crate::error::{Error, Result};
use crate::field::{mulmod, powmod};

/// A forward/inverse negacyclic NTT context of length `n` (a power of
/// two) over the prime `q = 1 (mod 2n)`.
#[derive(Debug, Clone)]
pub struct Ntt {
    n: usize,
    q: u64,
    /// psi^i for i < n, bit-reversed order not applied (natural order);
    /// psi is a primitive 2n-th root of unity: the negacyclic twist.
    psi: Vec<u64>,
    psi_inv: Vec<u64>,
    /// omega = psi^2 powers for the cyclic stages.
    w: Vec<u64>,
    w_inv: Vec<u64>,
    n_inv: u64,
}

/// Smallest prime `q = k * 2n + 1` with `q >= lo` (deterministic).
pub fn ntt_prime(n: usize, lo: u64) -> u64 {
    let step = 2 * n as u64;
    let mut q = lo.div_ceil(step) * step + 1;
    while !crate::field::is_prime(q) {
        q += step;
    }
    q
}

fn primitive_2nth_root(n: usize, q: u64) -> u64 {
    // find a generator-power of order 2n: g^((q-1)/2n) for g with
    // full-order projection; deterministic scan.
    let target = 2 * n as u64;
    let mut g = 2u64;
    loop {
        let cand = powmod(g, (q - 1) / target, q);
        // order exactly 2n: cand^n = -1
        if powmod(cand, n as u64, q) == q - 1 {
            return cand;
        }
        g += 1;
    }
}

impl Ntt {
    /// Build a context for length `n` (power of two) over `q`
    /// (`q = 1 mod 2n`, prime).
    pub fn new(n: usize, q: u64) -> Result<Self> {
        if !n.is_power_of_two() || n < 2 {
            return Err(Error::OutOfRange(
                "NTT length must be a power of two >= 2".into(),
            ));
        }
        if (q - 1) % (2 * n as u64) != 0 || !crate::field::is_prime(q) {
            return Err(Error::OutOfRange(
                "q must be prime with q = 1 (mod 2n)".into(),
            ));
        }
        let psi0 = primitive_2nth_root(n, q);
        let psi0_inv = powmod(psi0, q - 2, q);
        let w0 = mulmod(psi0, psi0, q);
        let w0_inv = powmod(w0, q - 2, q);
        let mut psi = vec![1u64; n];
        let mut psi_inv = vec![1u64; n];
        let mut w = vec![1u64; n / 2];
        let mut w_inv = vec![1u64; n / 2];
        for i in 1..n {
            psi[i] = mulmod(psi[i - 1], psi0, q);
            psi_inv[i] = mulmod(psi_inv[i - 1], psi0_inv, q);
        }
        for i in 1..n / 2 {
            w[i] = mulmod(w[i - 1], w0, q);
            w_inv[i] = mulmod(w_inv[i - 1], w0_inv, q);
        }
        let n_inv = powmod(n as u64, q - 2, q);
        Ok(Ntt {
            n,
            q,
            psi,
            psi_inv,
            w,
            w_inv,
            n_inv,
        })
    }

    /// In-place iterative radix-2 cyclic NTT (decimation in time,
    /// bit-reversal first), twiddles from `tw`.
    fn cyclic(&self, a: &mut [u64], tw: &[u64]) {
        let n = self.n;
        // bit-reversal permutation
        let mut j = 0usize;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j |= bit;
            if i < j {
                a.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let stride = n / len;
            for start in (0..n).step_by(len) {
                for k in 0..len / 2 {
                    let u = a[start + k];
                    let v = mulmod(a[start + k + len / 2], tw[k * stride], self.q);
                    a[start + k] = (u + v) % self.q;
                    a[start + k + len / 2] = (u + self.q - v) % self.q;
                }
            }
            len <<= 1;
        }
    }

    /// Negacyclic product of two length-`n` residue vectors mod `q`.
    pub fn negacyclic_mul(&self, a: &[u64], b: &[u64]) -> Result<Vec<u64>> {
        if a.len() != self.n || b.len() != self.n {
            return Err(Error::OutOfRange("length mismatch".into()));
        }
        let mut fa: Vec<u64> = (0..self.n)
            .map(|i| mulmod(a[i], self.psi[i], self.q))
            .collect();
        let mut fb: Vec<u64> = (0..self.n)
            .map(|i| mulmod(b[i], self.psi[i], self.q))
            .collect();
        let (w, w_inv) = (self.w.clone(), self.w_inv.clone());
        self.cyclic(&mut fa, &w);
        self.cyclic(&mut fb, &w);
        for i in 0..self.n {
            fa[i] = mulmod(fa[i], fb[i], self.q);
        }
        // the same routine with inverse twiddles IS the inverse
        // transform up to the factor n (F_{w^-1} F_w = n I).
        self.cyclic(&mut fa, &w_inv);
        let mut out = vec![0u64; self.n];
        for i in 0..self.n {
            let v = mulmod(fa[i], self.n_inv, self.q);
            out[i] = mulmod(v, self.psi_inv[i], self.q);
        }
        Ok(out)
    }
}

/// Fixed 62-bit CRT primes supporting lengths up to 2^20 (computed
/// once, cached).
pub fn crt_primes() -> [u64; 2] {
    use std::sync::OnceLock;
    static PRIMES: OnceLock<[u64; 2]> = OnceLock::new();
    *PRIMES.get_or_init(|| {
        [
            ntt_prime(1 << 20, 1 << 61),
            ntt_prime(1 << 20, (1 << 61) + (1 << 40)),
        ]
    })
}

/// Exact negacyclic product of signed coefficient vectors via two-prime
/// NTT + signed CRT. Caller guarantees `n * max|a| * max|b| < Q/2`
/// (`Q ~ 2^124`); [`super::Cyclo::mul_ntt`] checks and falls back.
pub fn negacyclic_mul_exact(a: &[i64], b: &[i64]) -> Result<Vec<i128>> {
    let n = a.len();
    let [q1, q2] = crt_primes();
    let n1 = Ntt::new(n, q1)?;
    let n2 = Ntt::new(n, q2)?;
    let lift = |v: &[i64], q: u64| -> Vec<u64> {
        v.iter().map(|&x| x.rem_euclid(q as i64) as u64).collect()
    };
    let r1 = n1.negacyclic_mul(&lift(a, q1), &lift(b, q1))?;
    let r2 = n2.negacyclic_mul(&lift(a, q2), &lift(b, q2))?;
    // CRT: x = r1 + q1 * ((r2 - r1) * q1^{-1} mod q2), centered
    let q1_inv_q2 = powmod(q1 % q2, q2 - 2, q2);
    let big_q = (q1 as u128) * (q2 as u128);
    let half_q = big_q / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let d = (r2[i] + q2 - r1[i] % q2) % q2;
        let t = mulmod(d, q1_inv_q2, q2);
        let x = (r1[i] as u128) + (q1 as u128) * (t as u128);
        out.push(if x > half_q {
            (x as i128) - (big_q as i128)
        } else {
            x as i128
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schoolbook(a: &[i64], b: &[i64]) -> Vec<i128> {
        let n = a.len();
        let mut out = vec![0i128; n];
        for i in 0..n {
            for j in 0..n {
                let (idx, sg) = super::super::fold(n, i + j);
                out[idx] += (a[i] as i128) * (b[j] as i128) * (sg as i128);
            }
        }
        out
    }

    #[test]
    fn ntt_prime_is_1_mod_2n() {
        for n in [8usize, 1 << 10, 1 << 20] {
            let q = ntt_prime(n, 1 << 61);
            assert!(crate::field::is_prime(q));
            assert_eq!((q - 1) % (2 * n as u64), 0);
        }
    }

    #[test]
    fn negacyclic_matches_schoolbook() {
        let mut state = 0x12345u64;
        let mut rnd = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as i64) - (1 << 30)
        };
        for n in [8usize, 16, 64, 256] {
            let a: Vec<i64> = (0..n).map(|_| rnd()).collect();
            let b: Vec<i64> = (0..n).map(|_| rnd()).collect();
            let fast = negacyclic_mul_exact(&a, &b).unwrap();
            assert_eq!(fast, schoolbook(&a, &b), "n = {n}");
        }
    }

    #[test]
    fn cyclo_mul_ntt_matches_mul() {
        use crate::ring::Cyclo;
        let mut state = 0x9e37u64;
        let mut rnd = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 40) as i64) - (1 << 22)
        };
        for half in [8usize, 16, 128] {
            let a = Cyclo::from_coeffs((0..half).map(|_| rnd()).collect()).unwrap();
            let b = Cyclo::from_coeffs((0..half).map(|_| rnd()).collect()).unwrap();
            assert_eq!(a.mul_ntt(&b).unwrap(), a.mul(&b).unwrap(), "half = {half}");
        }
    }
}
