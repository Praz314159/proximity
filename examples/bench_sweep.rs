//! The exp17-scale campaign as a single parallel sweep:
//! all primes p = 1 mod 32 up to 300k, full q=1 DP + max + census, rayon over
//! primes. `cargo run --release --example bench_sweep`

use bucketlab::census::census_direct;
use bucketlab::dp::{bucket_dist_q1, max_and_argmax};
use bucketlab::field::{binom, is_prime};
use rayon::prelude::*;
use std::time::Instant;

fn main() {
    let (s, r) = (32usize, 16usize);
    let m0 = binom(16, 8) as f64;
    let cs = binom(32, 16) as f64;
    let primes: Vec<u64> = (33..300_000u64)
        .step_by(32)
        .filter(|&p| is_prime(p))
        .collect();
    println!("{} primes = 1 mod 32 below 300k", primes.len());
    let t0 = Instant::now();
    let mut rows: Vec<(f64, u64, u64, u64)> = primes
        .par_iter()
        .map(|&p| {
            let d = bucket_dist_q1(p, s, r);
            let (mx, _) = max_and_argmax(&d);
            let cen = census_direct(p, s, 2, 4); // weight <= 4, coeffs [-2,2]
            let low: u64 = cen[2] + cen[3] + cen[4];
            let ratio = mx as f64 / (m0 + cs / p as f64);
            (ratio, p, mx, low)
        })
        .collect();
    let dt = t0.elapsed();
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!(
        "full sweep (DP + max + census<=4 per prime): {:.2}s total, {:.1} ms/prime",
        dt.as_secs_f64(),
        dt.as_secs_f64() * 1000.0 / primes.len() as f64
    );
    println!("worst 5 conjecture ratios:");
    for (ratio, p, mx, low) in rows.iter().take(5) {
        println!("  p={p:>7} ratio={ratio:.3} maxN={mx} lowwt_census={low}");
    }
}
