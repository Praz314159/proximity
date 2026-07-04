//! Arithmetic over prime fields `F_p` for `p < 2^63`, plus the integer utilities
//! the toolkit needs (deterministic primality, factorization, binomials).
//!
//! Everything here is scalar and allocation-free; the analysis kernels build on
//! these primitives. Values are plain `u64` residues in `[0, p)`.

/// `(a * b) mod p` without overflow, via a `u128` intermediate.
#[inline]
pub fn mulmod(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128 * b as u128) % p as u128) as u64
}

/// `b^e mod p` by square-and-multiply.
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

/// Deterministic Miller–Rabin, valid for all `n < 2^64`.
pub fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    for q in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
        if n % q == 0 {
            return n == q;
        }
    }
    let mut d = n - 1;
    let mut t = 0u32;
    while d % 2 == 0 {
        d /= 2;
        t += 1;
    }
    'outer: for a in [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37] {
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

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// One Pollard-rho split of a composite `n` (Brent-style iteration with a
/// deterministic parameter schedule; loops over parameters until a proper
/// factor is found).
fn pollard_rho(n: u64) -> u64 {
    if n % 2 == 0 {
        return 2;
    }
    let mut c = 1u64;
    loop {
        let f = |x: u64| (mulmod(x, x, n) + c) % n;
        let (mut x, mut y, mut d) = (2u64, 2u64, 1u64);
        while d == 1 {
            x = f(x);
            y = f(f(y));
            d = if x == y { n } else { gcd(x.abs_diff(y), n) };
        }
        if d != n {
            return d;
        }
        c += 1;
    }
}

/// Full prime factorization (with multiplicity), trial division to 31 then
/// Pollard rho. Handles the cyclotomic-norm magnitudes (≲ 2^63) that arise in
/// bad-set enumeration.
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

/// Smallest generator of `F_p^*` (matches the reference Python implementation,
/// so generator-dependent artifacts are reproducible across the two stacks).
pub fn find_generator(p: u64) -> u64 {
    let fac = distinct_prime_factors(p - 1);
    (2..p)
        .find(|&g| fac.iter().all(|&q| powmod(g, (p - 1) / q, p) != 1))
        .expect("every prime field has a generator")
}

/// Binomial coefficient as exact `u64` (panics on overflow past `u64`).
pub fn binom(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut num: u128 = 1;
    for i in 0..k {
        num = num * (n - i) as u128 / (i + 1) as u128;
    }
    u64::try_from(num).expect("binomial overflows u64")
}
