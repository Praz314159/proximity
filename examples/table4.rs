//! Certified reproduction of ABF26 (ePrint 2026/680) Table 4: the Elias
//! attack radius per interleaving width, at the concrete KoalaBear-sextic
//! rate-1/2 parameters (k = 2^20, s*n = 2^21, epsilon* = 2^-128).
//!
//! Differences from the paper's computation: we evaluate the EXACT Elias
//! count (Lemma 3.7, certified interval enclosure of the true ball volume)
//! instead of the MS77 approximation (Corollary 3.8), and every printed
//! value carries a machine-checked bracket.

use vanish::volumes::{elias_row, Alphabet};

fn main() {
    let base = Alphabet::koalabear();
    let ext = Alphabet::koalabear6();
    let total = 1u64 << 21;
    println!("ABF26 Table 4, certified (exact Lemma 3.7 + Lemma 6.12 chain)");
    println!("target: soundness >= 2^-128; base |B| = 2^31 - 2^24 + 1; |F| = |B|^6");
    println!();
    println!(
        "{:>6} {:>9} {:>9} {:>10} {:>10} {:>26} {:>7}",
        "s", "n", "z*", "delta*", "printed", "lg2 soundness at z*", "pinned"
    );
    for e in 0..=12u32 {
        let s = 1u64 << e;
        match elias_row(s, total, &base, &ext, -128.0) {
            Ok(r) => {
                // ABF print delta* to 3 decimals as the smallest grid value
                // at which the bound certifies; that is ceil(delta* * 1000)/1000.
                let printed = (r.delta_star * 1000.0).ceil() / 1000.0;
                println!(
                    "{:>6} {:>9} {:>9} {:>10.6} {:>10.3} [{:>11.4}, {:>11.4}] {:>7}",
                    format!("2^{e}"),
                    r.n,
                    r.z_star,
                    r.delta_star,
                    printed,
                    r.lg_sound_lo,
                    r.lg_sound_hi,
                    r.crossing_pinned
                );
            }
            Err(err) => println!("{:>6}  ERROR: {err}", format!("2^{e}")),
        }
    }
}
