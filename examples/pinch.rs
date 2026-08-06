// scratch: the pinch at the box, in the challenge's own currency
use rug::ops::Pow;
use rug::Integer;
use vanish::soundness::{elias_list_row, lg_cut_envelope, list_ceiling_row};

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
    let ceil = list_ceiling_row(1, total, total - k - 1, &ext, -128.0, |z| {
        lg_cut_envelope(total, k, z)
    })
    .unwrap();
    println!(
        "ceiling (SCAFFOLD envelope under eps*|F|): delta = {:.5}  z = {}  [{:.1}, {:.1}] bits  in {:?}",
        ceil.delta_star, ceil.z_star, ceil.lg_list_lo, ceil.lg_list_hi, t1.elapsed()
    );
    println!(
        "gap: {:.5} in delta — the ceiling is a plumbing reading (the scaffold\n\
         uses the trivial stratum bound); real rows need interface data + the recursion",
        floor.delta_star - ceil.delta_star
    );
}
