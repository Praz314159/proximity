//! Command-line interface: run the standard campaigns without writing code.
//!
//!   vanish info    --p 3457 --s 32
//!   vanish bucket  --p 3457 --s 32 --r 16 --lam 0 [--q 1]
//!   vanish rung    --p 3457 --s 32 --r 16 --q 2
//!   vanish census  --p 89633 --s 32 [--cmax 2] [--wmax 6]
//!   vanish decompose --p 77569 --s 32 --r 16 --lam 0
//!   vanish sweep   --s 32 --r 16 --pmax 300000 [--csv]
//!
//! `sweep` prints, per prime `p = 1 mod s`, the exact max bucket, the
//! conjecture ratio `max / (M_struct + C(s,r)/p)`, and the occupied-bucket
//! count (the exact toy-protocol winning-set size for the canonical pair).

use rayon::prelude::*;
use std::collections::HashMap;
use std::process::exit;
use vanish::buckets::{dp, mitm};
use vanish::code::{m_struct, rung_lambda};
use vanish::domain::Subgroup;
use vanish::field::{binom, is_prime};

fn parse_args(args: &[String]) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        if let Some(key) = args[i].strip_prefix("--") {
            if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                map.insert(key.to_string(), args[i + 1].clone());
                i += 2;
            } else {
                map.insert(key.to_string(), "true".to_string());
                i += 1;
            }
        } else {
            eprintln!("unexpected argument: {}", args[i]);
            exit(2);
        }
    }
    map
}

fn req<T: std::str::FromStr>(m: &HashMap<String, String>, key: &str) -> T {
    m.get(key).and_then(|v| v.parse().ok()).unwrap_or_else(|| {
        eprintln!("missing or invalid --{key}");
        exit(2);
    })
}

fn opt<T: std::str::FromStr>(m: &HashMap<String, String>, key: &str, default: T) -> T {
    m.get(key).and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn die(e: impl std::fmt::Display) -> ! {
    eprintln!("error: {e}");
    exit(1);
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let Some((cmd, rest)) = argv.split_first() else {
        eprintln!("usage: vanish <info|bucket|rung|census|decompose|sweep> --flags ...");
        exit(2);
    };
    let m = parse_args(rest);
    match cmd.as_str() {
        "info" => {
            let (p, s) = (req(&m, "p"), req(&m, "s"));
            let sg = Subgroup::new(p, s).unwrap_or_else(|e| die(e));
            println!(
                "F_{p}, subgroup order {s}, w = {}, two-smooth = {}",
                sg.w(),
                sg.is_two_smooth()
            );
            let r = s / 2;
            println!(
                "r = s/2 = {r}: C(s,r) = {}, M_struct(q=1) = {}, pigeonhole = {:.3}",
                binom(s as u64, r as u64),
                m_struct(s, r, 1),
                binom(s as u64, r as u64) as f64 / p as f64
            );
        }
        "bucket" => {
            let (p, s, r) = (req(&m, "p"), req(&m, "s"), req(&m, "r"));
            let lam: Vec<u64> = m
                .get("lam")
                .map(|v| {
                    v.split(',')
                        .map(|x| x.parse().unwrap_or_else(|_| die("bad --lam")))
                        .collect()
                })
                .unwrap_or_else(|| die("missing --lam (comma-separated e-values)"));
            let sg = Subgroup::new(p, s).unwrap_or_else(|e| die(e));
            let t = mitm::HalfTables::build(&sg, r, lam.len()).unwrap_or_else(|e| die(e));
            println!("{}", t.bucket(&lam).unwrap_or_else(|e| die(e)));
        }
        "rung" => {
            let (p, s, r, q) = (req(&m, "p"), req(&m, "s"), req(&m, "r"), req(&m, "q"));
            let sg = Subgroup::new(p, s).unwrap_or_else(|e| die(e));
            let lam = rung_lambda(&sg, r, q).unwrap_or_else(|e| die(e));
            let t = mitm::HalfTables::build(&sg, r, q).unwrap_or_else(|e| die(e));
            let b = t.bucket(&lam).unwrap_or_else(|e| die(e));
            println!(
                "rung lambda = {lam:?}\nbucket = {b} (M_struct = {})",
                m_struct(s, r, q)
            );
        }
        "census" => {
            let (p, s) = (req(&m, "p"), req(&m, "s"));
            let cmax: i64 = opt(&m, "cmax", 2);
            let sg = Subgroup::new(p, s).unwrap_or_else(|e| die(e));
            let counts = if let Some(w) = m.get("wmax") {
                let wmax: usize = w.parse().unwrap_or_else(|_| die("bad --wmax"));
                vanish::census::direct(&sg, cmax as u64, wmax).unwrap_or_else(|e| die(e))
            } else {
                vanish::census::mitm(&sg, cmax).unwrap_or_else(|e| die(e))
            };
            for (w, c) in counts.iter().enumerate() {
                if *c > 0 {
                    println!("w={w}: {c}");
                }
            }
        }
        "decompose" => {
            let (p, s, r) = (req(&m, "p"), req(&m, "s"), req(&m, "r"));
            let lam: u64 = req(&m, "lam");
            let sg = Subgroup::new(p, s).unwrap_or_else(|e| die(e));
            let (total, per_w) = mitm::decompose_bucket_q1(&sg, r, lam).unwrap_or_else(|e| die(e));
            println!("bucket({lam}) = {total}");
            for (w, c) in per_w.iter().enumerate() {
                if *c > 0 {
                    println!("  weight {w}: {c} classes");
                }
            }
        }
        "sweep" => {
            let (s, r): (usize, usize) = (req(&m, "s"), req(&m, "r"));
            let pmax: u64 = req(&m, "pmax");
            let csv = m.contains_key("csv");
            let m0 = m_struct(s, r, 1) as f64;
            let cs = binom(s as u64, r as u64) as f64;
            let primes: Vec<u64> = (s as u64 + 1..pmax)
                .step_by(s)
                .filter(|&p| is_prime(p))
                .collect();
            let mut rows: Vec<(f64, u64, u64, u64)> = primes
                .par_iter()
                .filter_map(|&p| {
                    let sg = Subgroup::new(p, s).ok()?;
                    let d = dp::distribution_q1(&sg, r).ok()?;
                    let (mx, _) = d.max();
                    let ratio = mx as f64 / (m0 + cs / p as f64);
                    Some((ratio, p, mx, d.occupied()))
                })
                .collect();
            if csv {
                println!("p,max_bucket,conjecture_ratio,occupied");
                for (ratio, p, mx, occ) in &rows {
                    println!("{p},{mx},{ratio:.6},{occ}");
                }
            } else {
                rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                println!("{} primes; worst 10 conjecture ratios:", rows.len());
                for (ratio, p, mx, occ) in rows.iter().take(10) {
                    println!("  p={p:>9} ratio={ratio:.3} maxN={mx} occupied={occ}");
                }
            }
        }
        "attack" => {
            use vanish::attack::*;
            let params = AttackParams {
                n: req(&m, "n"),
                k: req(&m, "k"),
                list_bits: req(&m, "list-bits"),
            };
            let dmin = capacity_radius(params.n, params.k);
            println!(
                "n = {}, k = {} (rate {:.4}), required list bits = {:.2}",
                params.n,
                params.k,
                params.k as f64 / params.n as f64,
                params.list_bits
            );
            println!("capacity radius (dmin)     : {dmin:.6}");
            match antipodal_attack(&params).unwrap_or_else(|e| die(e)) {
                Some(a) => println!("antipodal (Table-5 method) : delta* = {:.6}", a.delta_star),
                None => println!("antipodal (Table-5 method) : unattainable"),
            }
            match best_attack(&params).unwrap_or_else(|e| die(e)) {
                Some(a) => println!(
                    "ladder (best rung)         : delta* = {:.6}  [t={}, s_G={}, r={}, \
                     log2 list = {:.2}, deficit = {:.6}]",
                    a.delta_star, a.t, a.s_g, a.r, a.log2_list, a.deficit
                ),
                None => println!("ladder (best rung)         : unattainable"),
            }
            println!(
                "framework ceiling          : {:.6}",
                hyperbola_ceiling(&params).unwrap_or_else(|e| die(e))
            );
            if let Some(bb) = m.get("base-bits") {
                let bb: f64 = bb.parse().unwrap_or_else(|_| die("bad --base-bits"));
                match elias_delta_star(&params, bb).unwrap_or_else(|e| die(e)) {
                    Some(d) => println!("Elias (base field {bb:.1} bits): delta* = {d:.6}"),
                    None => println!("Elias (base field {bb:.1} bits): unattainable"),
                }
            }
        }
        other => {
            eprintln!("unknown subcommand: {other}");
            eprintln!("usage: vanish <info|bucket|rung|census|decompose|sweep|attack> --flags ...");
            exit(2);
        }
    }
}
