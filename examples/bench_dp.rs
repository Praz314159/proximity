//! Quick wall-clock check: `cargo run --release --example bench_dp`
//! Compares directly with the numpy DP timings from exp17/exp20b.

use bucketlab::dp::{bucket_dist_q1, max_and_argmax};
use bucketlab::field::binom;
use std::time::Instant;

fn main() {
    for (p, s, r) in [
        (180001u64, 32usize, 16usize),
        (299969, 64, 32),
        (1048577 + 63 * 16, 64, 32), // just a big p = 1 mod 64 candidate region
    ] {
        if (p - 1) % s as u64 != 0 || !bucketlab::field::is_prime(p) {
            println!("skip p={p} (not prime = 1 mod {s})");
            continue;
        }
        let t0 = Instant::now();
        let d = bucket_dist_q1(p, s, r);
        let dt = t0.elapsed();
        let (mx, arg) = max_and_argmax(&d);
        let sum: u64 = d.iter().sum();
        assert_eq!(sum, binom(s as u64, r as u64));
        println!(
            "s={s} r={r} p={p}: maxN={mx} at lam={arg}  [{:.3}s, mass OK]",
            dt.as_secs_f64()
        );
    }
}
