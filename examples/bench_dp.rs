//! Single-DP wall-clock check. `cargo run --release --example bench_dp`

use std::time::Instant;
use vanish::smooth::buckets::dp;
use vanish::domain::MultiplicativeSubgroup;
use vanish::field::binom;

fn main() {
    for (p, s, r) in [(180001u64, 32usize, 16usize), (299969, 64, 32)] {
        let sg = match MultiplicativeSubgroup::new(p, s) {
            Ok(sg) => sg,
            Err(e) => {
                println!("skip p={p}: {e}");
                continue;
            }
        };
        let t0 = Instant::now();
        let d = dp::distribution_q1(&sg, r).unwrap();
        let dt = t0.elapsed();
        let (mx, arg) = d.max();
        assert_eq!(d.total(), binom(s as u64, r as u64));
        println!(
            "s={s} r={r} p={p}: maxN={mx} at lam={arg}  [{:.3}s, mass OK]",
            dt.as_secs_f64()
        );
    }
}
