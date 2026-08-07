// scratch: the pinch at the box, in the challenge's own currency —
// the floor row against the assembled tower envelope (trivial
// interface data; the numbers are the loss map's starting point)
use rug::ops::Pow;
use rug::Integer;
use vanish::soundness::envelope::{assemble, TrivialInterface, DEFAULT_RESOLUTION};
use vanish::soundness::{elias_list_row, list_ceiling_row};

fn main() {
    let base = Integer::from(vanish::field::named::KOALABEAR);
    let ext = base.clone().pow(6);
    let total = 1u64 << 21;
    let k = total / 2 - 1;
    // the challenge: largest delta with |Lambda| <= eps* |F|
    let t0 = std::time::Instant::now();
    let floor = elias_list_row(1, total, &base, &ext, -128.0).unwrap();
    println!(
        "floor   (Elias count crosses eps*|F|): delta = {:.5}  z = {}  [{:.1}, {:.1}] bits  in {:?}",
        floor.delta_star, floor.z_star, floor.lg_list_lo, floor.lg_list_hi, t0.elapsed()
    );
    let t1 = std::time::Instant::now();
    let prof = assemble(total, k, 64, &TrivialInterface, DEFAULT_RESOLUTION).unwrap();
    println!(
        "tower assembled (n0 = 64, trivial data) in {:?}",
        t1.elapsed()
    );
    let t2 = std::time::Instant::now();
    match list_ceiling_row(1, total, total - k - 1, &ext, -128.0, |z| {
        prof.lg_at_disagreement(k, z)
    }) {
        Ok(ceil) => println!(
            "ceiling (tower envelope under eps*|F|): delta = {:.5}  z = {}  [{:.1}, {:.1}] bits  in {:?}",
            ceil.delta_star, ceil.z_star, ceil.lg_list_lo, ceil.lg_list_hi, t2.elapsed()
        ),
        Err(e) => println!(
            "ceiling: none — {e}\n\
             (trivial interface data holds no radius under the budget: even at\n\
             full agreement the tower reads the base's {:.1} bits against the\n\
             budget; the loss map (examples/tower.rs) locates where sharpness\n\
             must come from — D_b and D_c, the engine supply and the per-prime\n\
             envelope, not the tower plumbing, which is loss-free at t = n)",
            prof.lg_at_disagreement(k, 0).map(|v| v.hi.to_f64()).unwrap_or(f64::NAN)
        ),
    }
}
