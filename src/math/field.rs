//! Arithmetic over prime fields `F_p` for `p < 2^63`, plus the integer utilities
//! the toolkit needs (deterministic primality, factorization, binomials).
//!
//! Everything here is scalar and allocation-free; the analysis kernels build on
//! these primitives. Values are plain `u64` residues in `[0, p)`.

/// The deterministic Miller–Rabin witness set for `n < 2^64` (Sinclair);
/// shared by the small-prime trial screen and both strong-probable-prime
/// loops.
const MR_WITNESSES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

/// `(a * b) mod p` without overflow, via a `u128` intermediate.
#[inline]
#[must_use]
pub fn mulmod(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128 * b as u128) % p as u128) as u64
}

/// `b^e mod p` by square-and-multiply.
#[must_use]
pub fn powmod(mut b: u64, mut e: u64, p: u64) -> u64 {
    let mut acc: u64 = 1 % p;
    b %= p;
    while e > 0 {
        if e & 1 == 1 {
            acc = mulmod(acc, b, p);
        }
        b = mulmod(b, b, p);
        e >>= 1;
    }
    acc
}

/// Montgomery-form arithmetic for a fixed odd modulus `n < 2^63`.
///
/// Replaces the `u128` division inside [`mulmod`] with two multiplications
/// and a shift (REDC). The payoff is in loops that stay in Montgomery form
/// across many multiplications — Miller–Rabin exponentiation and the rho
/// walk — where the per-step division was the measured cost center of
/// norm-table ingestion.
struct Montgomery {
    n: u64,
    /// `-n^{-1} mod 2^64`.
    ninv: u64,
    /// `2^128 mod n` (converts into Montgomery form via one `mul`).
    r2: u64,
}

impl Montgomery {
    fn new(n: u64) -> Self {
        debug_assert!(n & 1 == 1 && n < (1 << 63));
        // Newton iteration for n^{-1} mod 2^64: each step doubles the
        // number of correct low bits; 6 steps from the 5-bit seed `n`.
        let mut inv = n;
        for _ in 0..6 {
            inv = inv.wrapping_mul(2u64.wrapping_sub(n.wrapping_mul(inv)));
        }
        let r = ((1u128 << 64) % n as u128) as u64;
        Montgomery {
            n,
            ninv: inv.wrapping_neg(),
            r2: mulmod(r, r, n),
        }
    }
    /// REDC: for `t < n * 2^64`, returns `t * 2^{-64} mod n`.
    #[inline]
    fn redc(&self, t: u128) -> u64 {
        let m = (t as u64).wrapping_mul(self.ninv);
        let s = ((t + m as u128 * self.n as u128) >> 64) as u64;
        if s >= self.n {
            s - self.n
        } else {
            s
        }
    }
    /// Product of two Montgomery-form values, in Montgomery form.
    #[inline]
    fn mul(&self, a: u64, b: u64) -> u64 {
        self.redc(a as u128 * b as u128)
    }
    /// Convert `x < n` into Montgomery form (`x * 2^64 mod n`).
    #[inline]
    fn to_mont(&self, x: u64) -> u64 {
        self.mul(x, self.r2)
    }
    /// `1` in Montgomery form (`2^64 mod n`).
    #[inline]
    fn one(&self) -> u64 {
        self.redc(self.r2 as u128)
    }
}

/// Deterministic Miller–Rabin, valid for all `n < 2^64`.
#[must_use]
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for q in MR_WITNESSES {
        if n % q == 0 {
            return n == q;
        }
    }
    if n >= (1 << 63) {
        return is_prime_generic(n);
    }
    let mt = Montgomery::new(n);
    let one = mt.one();
    let minus_one = n - one;
    let mut d = n - 1;
    let mut t = 0u32;
    while d % 2 == 0 {
        d /= 2;
        t += 1;
    }
    'outer: for a in MR_WITNESSES {
        // Montgomery-form modexp a^d
        let mut base = mt.to_mont(a);
        let mut x = one;
        let mut e = d;
        while e > 0 {
            if e & 1 == 1 {
                x = mt.mul(x, base);
            }
            base = mt.mul(base, base);
            e >>= 1;
        }
        if x == one || x == minus_one {
            continue;
        }
        for _ in 0..t - 1 {
            x = mt.mul(x, x);
            if x == minus_one {
                continue 'outer;
            }
        }
        return false;
    }
    true
}

/// Fallback Miller–Rabin via [`powmod`] for `n >= 2^63`, where the REDC
/// carry bound does not hold. Same witness set, same determinism.
fn is_prime_generic(n: u64) -> bool {
    let mut d = n - 1;
    let mut t = 0u32;
    while d % 2 == 0 {
        d /= 2;
        t += 1;
    }
    'outer: for a in MR_WITNESSES {
        let mut x = powmod(a, d, n);
        if x == 1 || x == n - 1 {
            continue;
        }
        for _ in 0..t - 1 {
            x = mulmod(x, x, n);
            if x == n - 1 {
                continue 'outer;
            }
        }
        return false;
    }
    true
}

/// Distinct prime factors by trial division. Suitable for the `p - 1` values
/// that arise when constructing subgroups; for large cyclotomic norms use
/// [`factor`], which falls back to Pollard rho.
#[must_use]
pub fn distinct_prime_factors(mut n: u64) -> Vec<u64> {
    let mut out = Vec::new();
    let mut d = 2u64;
    while d.saturating_mul(d) <= n {
        if n % d == 0 {
            out.push(d);
            while n % d == 0 {
                n /= d;
            }
        }
        d += 1;
    }
    if n > 1 {
        out.push(n);
    }
    out
}

/// Euclidean gcd (shared utility; also the batch step of Brent-rho).
pub fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// One Pollard-rho split of a composite `n`, Brent's variant: powers-of-two
/// cycle detection and gcds batched 128 iterations at a time (one gcd per
/// batch instead of per step — the difference between microseconds and
/// milliseconds on hard semiprimes, which dominate norm-table ingestion).
/// Deterministic parameter schedule; loops until a proper factor is found.
fn pollard_rho(n: u64) -> u64 {
    if n % 2 == 0 {
        return 2;
    }
    let mt = Montgomery::new(n);
    let mut c = 1u64;
    loop {
        // The walk runs entirely in Montgomery form: if X = x*R mod n, then
        // REDC(X^2) + cR = (x^2 + c)*R mod n. Differences carry the same R
        // factor, and gcd(dR, n) = gcd(d, n) because R = 2^64 is coprime to
        // odd n — so factors are detected without ever converting back.
        let cm = mt.to_mont(c % n);
        let f = |x: u64| {
            let s = mt.mul(x, x) + cm;
            if s >= n {
                s - n
            } else {
                s
            }
        };
        let (mut y, mut r, mut q, mut g) = (mt.to_mont(2), 1u64, 1u64, 1u64);
        let (mut x, mut ys) = (0u64, 0u64);
        while g == 1 {
            x = y;
            for _ in 0..r {
                y = f(y);
            }
            let mut k = 0;
            while k < r && g == 1 {
                ys = y;
                let m = 128.min(r - k);
                for _ in 0..m {
                    y = f(y);
                    q = mt.mul(q, x.abs_diff(y));
                }
                g = gcd(q, n);
                k += m;
            }
            r *= 2;
        }
        if g == n {
            // the batched product collapsed to 0 mod n: retrace this batch
            // one step at a time with individual gcds
            g = 1;
            while g == 1 {
                ys = f(ys);
                g = gcd(x.abs_diff(ys), n);
            }
        }
        if g != n {
            return g;
        }
        c += 1;
    }
}

/// Full prime factorization (with multiplicity), trial division to 31 then
/// Pollard rho. Handles the cyclotomic-norm magnitudes (≲ 2^63) that arise in
/// bad-set enumeration.
#[must_use]
pub fn factor(mut n: u64) -> Vec<u64> {
    let mut out = Vec::new();
    for d in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31] {
        while n % d == 0 {
            out.push(d);
            n /= d;
        }
    }
    let mut stack = if n > 1 { vec![n] } else { Vec::new() };
    while let Some(m) = stack.pop() {
        if m == 1 {
            continue;
        }
        if is_prime(m) {
            out.push(m);
        } else {
            let d = pollard_rho(m);
            stack.push(d);
            stack.push(m / d);
        }
    }
    out.sort_unstable();
    out
}

/// Primes `p = 1 (mod s)` with `p > s`, ascending from the first aligned
/// candidate at or after `lo`. Unbounded — combine with `take` or
/// `take_while`. The sweep-population helper shared by the CLI, the Python
/// bindings, and the CRT-prime hunt.
pub fn primes_one_mod(s: u64, lo: u64) -> impl Iterator<Item = u64> {
    let mut p = lo.max(s + 1);
    p += (1 + s - p % s) % s;
    std::iter::successors(Some(p), move |&q| Some(q + s)).filter(|&q| is_prime(q))
}

/// Smallest generator of `F_p^*` (matches the reference Python implementation,
/// so generator-dependent artifacts are reproducible across the two stacks).
#[must_use]
pub fn find_generator(p: u64) -> u64 {
    let fac = distinct_prime_factors(p - 1);
    (2..p)
        .find(|&g| fac.iter().all(|&q| powmod(g, (p - 1) / q, p) != 1))
        .expect("every prime field has a generator")
}

/// Binomial coefficient as exact `u64` (panics on overflow past `u64`).
#[must_use]
pub fn binom(n: u64, k: u64) -> u64 {
    checked_binom(n, k).expect("binomial overflows u64")
}

/// Binomial coefficient as exact `u64`, `None` on overflow. Use this for
/// feasibility/cap comparisons reachable from FFI, where an oversized input
/// must produce a clean `Unsupported` rather than a process abort.
#[must_use]
pub fn checked_binom(n: u64, k: u64) -> Option<u64> {
    if k > n {
        return Some(0);
    }
    let k = k.min(n - k);
    let mut num: u128 = 1;
    for i in 0..k {
        num = num.checked_mul((n - i) as u128)? / (i + 1) as u128;
    }
    u64::try_from(num).ok()
}

/// Invert every entry of `vals` in place with Montgomery's batch trick:
/// prefix products, ONE Fermat inversion, unwind. Entries must be nonzero
/// mod `p` (`p` prime).
pub fn batch_inv(vals: &mut [u64], p: u64) {
    let n = vals.len();
    if n == 0 {
        return;
    }
    let mut pref = vec![0u64; n];
    let mut acc = 1u64;
    for (i, &v) in vals.iter().enumerate() {
        pref[i] = acc;
        acc = mulmod(acc, v, p);
    }
    let mut inv_acc = powmod(acc, p - 2, p);
    for i in (0..n).rev() {
        let orig = vals[i];
        vals[i] = mulmod(inv_acc, pref[i], p);
        inv_acc = mulmod(inv_acc, orig, p);
    }
}

/// Pascal's triangle mod `p` up to row `top` (inclusive) — the
/// binomials as field elements, next to the exact [`binom`].
#[must_use]
pub fn binom_table_mod(top: usize, p: u64) -> Vec<Vec<u64>> {
    let mut c = vec![vec![0u64; top + 1]; top + 1];
    for row in &mut c {
        row[0] = 1;
    }
    for a in 1..=top {
        for b in 1..=a {
            let v = (c[a - 1][b - 1] + c[a - 1][b]) % p;
            c[a][b] = v;
        }
    }
    c
}

/// The top-`q` elementary symmetric functions `e_1..e_q` of a set of field
/// elements.
#[must_use]
pub fn top_elementary_symmetric(els: &[u64], q: usize, p: u64) -> Vec<u64> {
    let mut e = vec![0u64; q + 1];
    e[0] = 1;
    for (cnt, &a) in els.iter().enumerate() {
        let top = q.min(cnt + 1);
        for i in (1..=top).rev() {
            e[i] = (e[i] + crate::field::mulmod(a, e[i - 1], p)) % p;
        }
    }
    e.remove(0);
    e
}

/// Named primes of the deployment landscape — the single source of truth
/// for the concrete fields that tests, gates, and attack reproductions
/// share. Anything that needs "the KoalaBear prime" references these
/// constants; nothing re-declares them.
pub mod named {
    /// KoalaBear: `2^31 - 2^24 + 1`. Smooth through `s = 2^24`
    /// (`2^24 | p - 1`); deployments use the sextic extension
    /// (`|F| = p^6`, `log2 |F| ≈ 185.93`).
    pub const KOALABEAR: u64 = (1 << 31) - (1 << 24) + 1;
    /// BabyBear: `2^31 - 2^27 + 1`. Smooth through `s = 2^27`
    /// (`2^27 | p - 1`); quartic and quintic extensions appear in
    /// deployments.
    pub const BABYBEAR: u64 = (1 << 31) - (1 << 27) + 1;
    /// Goldilocks: `2^64 - 2^32 + 1`. Smooth through `s = 2^32`
    /// (`2^32 | p - 1`); the 64-bit deployment field (Plonky2-era
    /// stacks). The one named prime whose taxonomy is height-split:
    /// provably clean by the Parseval ceiling at every level `<= 32`,
    /// live at level 64 only from weight `p^{1/16} ~ 16` up.
    pub const GOLDILOCKS: u64 = 0xFFFF_FFFF_0000_0001;
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `checked_binom` matches `binom` in range and returns `None` (rather
    /// than panicking) past u64.
    #[test]
    fn named_primes() {
        assert_eq!(named::KOALABEAR, 2_130_706_433);
        assert_eq!(named::BABYBEAR, 2_013_265_921);
        assert_eq!(named::GOLDILOCKS, 18_446_744_069_414_584_321);
        assert!(is_prime(named::KOALABEAR) && is_prime(named::BABYBEAR));
        assert!(is_prime(named::GOLDILOCKS));
        assert_eq!((named::KOALABEAR - 1) % (1 << 24), 0);
        assert_eq!((named::BABYBEAR - 1) % (1 << 27), 0);
        assert_eq!((named::GOLDILOCKS - 1) % (1 << 32), 0);
    }

    #[test]
    fn checked_binom_range_and_overflow() {
        assert_eq!(checked_binom(16, 8), Some(12870));
        assert_eq!(checked_binom(8, 12), Some(0));
        assert_eq!(checked_binom(64, 32), Some(binom(64, 32)));
        assert_eq!(checked_binom(128, 64), None);
    }
}
