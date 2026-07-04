//! Modular arithmetic and subgroup construction for primes p < 2^63.
//! Conventions match `useful_families.py` (smallest generator, subgroup as
//! consecutive powers) so that cross-language golden values line up where
//! they are generator-dependent; all landscape statistics are provably
//! generator-invariant regardless.

pub fn mulmod(a: u64, b: u64, p: u64) -> u64 {
    ((a as u128 * b as u128) % p as u128) as u64
}

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

/// Deterministic Miller–Rabin for u64.
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

/// Distinct prime factors by trial division (fine for p - 1 < 2^63 with small factors;
/// our primes are ~2^31 or synthetic sweep primes).
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

/// Smallest generator of F_p^* (matches the Python helper).
pub fn find_generator(p: u64) -> u64 {
    let fac = distinct_prime_factors(p - 1);
    for g in 2..p {
        if fac.iter().all(|&q| powmod(g, (p - 1) / q, p) != 1) {
            return g;
        }
    }
    panic!("no generator found for p = {p}");
}

/// Elements of the order-s subgroup as consecutive powers [w^0, ..., w^{s-1}].
pub fn subgroup(p: u64, s: usize) -> Vec<u64> {
    assert!((p - 1) % s as u64 == 0, "s must divide p - 1");
    let g = find_generator(p);
    let w = powmod(g, (p - 1) / s as u64, p);
    let mut els = Vec::with_capacity(s);
    let mut x = 1u64;
    for _ in 0..s {
        els.push(x);
        x = mulmod(x, w, p);
    }
    debug_assert_eq!(x, 1, "w does not have order dividing s");
    debug_assert!(
        s % 2 == 1 || els[s / 2] == p - 1,
        "for even s, w^(s/2) must be -1 (order exactly s)"
    );
    els
}

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
