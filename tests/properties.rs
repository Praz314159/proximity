//! Randomized (seeded, deterministic) property tests and error-path coverage.
//! These complement the golden pins: invariants that must hold at *every*
//! parameter point, checked across a pseudo-random sample of valid ones.

use vanish::buckets::{dp, mitm};
use vanish::census;
use vanish::code::ReedSolomon;
use vanish::domain::Subgroup;
use vanish::field::{binom, is_prime};
use vanish::Error;

/// Tiny deterministic LCG so the sample is reproducible without dependencies.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn primes_1_mod(s: u64, from: u64, count: usize) -> Vec<u64> {
    let mut out = Vec::new();
    let mut p = from + (1 + s - from % s) % s; // first value = 1 mod s above from
    while out.len() < count {
        if is_prime(p) {
            out.push(p);
        }
        p += s;
    }
    out
}

#[test]
fn random_invariants_mass_dilation_mitm() {
    let mut rng = Lcg(0xC0FFEE);
    for s in [8usize, 16, 32] {
        for &p in &primes_1_mod(s as u64, 100 + rng.below(1000), 2) {
            let sg = Subgroup::new(p, s).unwrap();
            let r = 2 + rng.below(s as u64 - 3) as usize;
            let d = dp::distribution_q1(&sg, r).unwrap();
            // mass
            assert_eq!(
                d.total(),
                binom(s as u64, r as u64),
                "mass at p={p}, s={s}, r={r}"
            );
            // dilation invariance at random lambda
            let v = d.values();
            for _ in 0..4 {
                let lam = 1 + rng.below(p - 1);
                let g = sg.elements()[1 + rng.below(s as u64 - 1) as usize];
                let gl = (g as u128 * lam as u128 % p as u128) as usize;
                assert_eq!(v[gl], v[lam as usize], "dilation at p={p}, s={s}, r={r}");
            }
            // DP <-> MitM agreement at random lambda (and at the argmax)
            let t = mitm::HalfTables::build(&sg, r, 1).unwrap();
            let (mx, arg) = d.max();
            assert_eq!(t.bucket(&[arg]).unwrap(), mx);
            for _ in 0..3 {
                let lam = rng.below(p);
                assert_eq!(
                    t.bucket(&[lam]).unwrap(),
                    v[lam as usize],
                    "MitM at p={p}, lam={lam}"
                );
            }
            // decomposition law at the argmax (power-of-two s only)
            if s.is_power_of_two() {
                let (total, _) = mitm::decompose_bucket_q1(&sg, r, arg).unwrap();
                assert_eq!(total, mx, "decomposition law at p={p}, s={s}, r={r}");
            }
        }
    }
}

#[test]
fn error_paths() {
    // core-type validation
    assert!(matches!(Subgroup::new(3458, 32), Err(Error::NotPrime(_))));
    assert!(matches!(
        Subgroup::new(97, 31),
        Err(Error::OrderDoesNotDivide { .. })
    ));
    assert!(Subgroup::new(97, 1).is_err());
    let sg = Subgroup::new(97, 32).unwrap();
    assert!(ReedSolomon::new(&sg, 0).is_err());
    assert!(ReedSolomon::new(&sg, 33).is_err());
    // engine limits are errors, not panics
    let big = Subgroup::new(257, 128).unwrap();
    assert!(matches!(
        dp::distribution_q1(&big, 64),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        mitm::HalfTables::build(&big, 64, 1),
        Err(Error::Unsupported(_))
    ));
    assert!(matches!(
        mitm::decompose_bucket_q1(&big, 64, 0),
        Err(Error::Unsupported(_))
    ));
    assert!(census::direct(&sg, 0, 4).is_err());
    assert!(census::direct(&sg, 2, 17).is_err(), "wmax > s/2 rejected");
    assert!(census::mitm(&sg, 5).is_err(), "cmax cap enforced");
    // q out of range for MitM tables
    assert!(mitm::HalfTables::build(&sg, 16, 0).is_err());
    assert!(mitm::HalfTables::build(&sg, 16, 9).is_err());
    // lambda length mismatch
    let t = mitm::HalfTables::build(&sg, 16, 2).unwrap();
    assert!(t.bucket(&[1]).is_err());
    // non-two-smooth subgroup: construction fine, smooth-only ops fail
    let s6 = Subgroup::new(31, 6).unwrap();
    assert!(!s6.is_two_smooth());
    assert!(s6.cosets(1).is_err());
    assert!(vanish::code::rung_lambda(&s6, 3, 1).is_err());
}
