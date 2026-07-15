//! Full-campaign benchmark: all primes p = 1 mod 32 below 300k, exact q=1 DP +
//! max + low-weight census per prime, rayon over primes.
//! `cargo run --release --example bench_sweep`

use rayon::prelude::*;
use std::time::Instant;
use vanish::domain::MultiplicativeSubgroup;
use vanish::field::{binom, is_prime};
use vanish::smooth::buckets::dp;
use vanish::smooth::census;
use vanish::smooth::rung::m_struct;

fn main() {
    let (s, r) = (32usize, 16usize);
    let m0 = m_struct(s, r, 1) as f64;
    let cs = binom(32, 16) as f64;
    let primes: Vec<u64> = (33..300_000u64)
        .step_by(32)
        .filter(|&p| is_prime(p))
        .collect();
    println!("{} primes = 1 mod 32 below 300k", primes.len());
    let t0 = Instant::now();
    let mut rows: Vec<(f64, u64, u64, u64)> = primes
        .par_iter()
        .filter_map(|&p| {
            let sg = MultiplicativeSubgroup::new(p, s).ok()?;
            let d = dp::distribution_q1(&sg, r).ok()?;
            let (mx, _) = d.max();
            let cen = census::direct(&sg, 2, 4).ok()?;
            let low: u64 = cen[2] + cen[3] + cen[4];
            Some((mx as f64 / (m0 + cs / p as f64), p, mx, low))
        })
        .collect();
    let dt = t0.elapsed();
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    println!(
        "sweep: {:.2}s total, {:.1} ms/prime; worst 5 ratios:",
        dt.as_secs_f64(),
        dt.as_secs_f64() * 1000.0 / primes.len() as f64
    );
    for (ratio, p, mx, low) in rows.iter().take(5) {
        println!("  p={p:>7} ratio={ratio:.3} maxN={mx} lowwt_census={low}");
    }
}
