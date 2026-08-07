// scratch: the loss map — the tower assembled level by level at the
// box, printed where the challenge trajectory reads it. For each
// level: the dimension window, and at the trajectory dimension the
// envelope in bits at the curve, at the level's quarter-radius, and
// at full agreement. Where the bits blow past the budget marks the
// input (base, D_b, D_c, or the deep-strata tail) the compilation
// chapter must sharpen.
use rug::ops::Pow;
use rug::Integer;
use vanish::soundness::envelope::{assemble_levels, TrivialInterface, DEFAULT_RESOLUTION};
use vanish::soundness::lg_list_threshold;

fn main() {
    let total = 1u64 << 21;
    let k = total / 2 - 1;
    let ext = Integer::from(vanish::field::named::KOALABEAR).pow(6);
    let thr = lg_list_threshold(&ext, -128.0).unwrap();
    println!(
        "budget eps*|F|: {:.2} bits;  wall z* = 981106 (the floor)\n",
        thr.hi.to_f64()
    );
    let t0 = std::time::Instant::now();
    let levels = assemble_levels(total, k, 64, &TrivialInterface, DEFAULT_RESOLUTION).unwrap();
    println!("assembled {} levels in {:?}\n", levels.len(), t0.elapsed());
    println!(
        "{:>9} {:>9} {:>7}   {:>14} {:>14} {:>14}",
        "level", "dim", "curve", "E(curve)", "E(3n/4)", "E(n)"
    );
    for prof in &levels {
        let dim = prof.dims().max().expect("nonempty window");
        let curve = prof.t_min(dim).expect("in window");
        let bits = |t: u64| -> String {
            match prof.eval(dim, t) {
                Ok(v) => format!("{:.1}", v.hi.to_f64()),
                Err(_) => "-".into(),
            }
        };
        println!(
            "{:>9} {:>9} {:>7}   {:>14} {:>14} {:>14}",
            prof.n,
            dim,
            curve,
            bits(curve),
            bits((3 * prof.n) / 4),
            bits(prof.n)
        );
    }
    let top = levels.last().unwrap();
    let z_wall = 981106u64;
    let at_wall = top.lg_at_disagreement(k, z_wall).unwrap();
    println!(
        "\ntop envelope at the wall cell (z = {z_wall}): [{:.1}, {:.1}] bits \
         against the {:.1}-bit budget",
        at_wall.lo.to_f64(),
        at_wall.hi.to_f64(),
        thr.hi.to_f64()
    );
}
