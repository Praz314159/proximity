// scratch: the first pinch at the box (not committed)
use rug::ops::Pow;
use rug::Integer;
use vanish::soundness::{elias_row, envelope_row, lg_cut_envelope};

fn main() {
    let base = Integer::from(vanish::field::named::KOALABEAR);
    let ext = base.clone().pow(6);
    let total = 1u64 << 21;
    let k = total / 2 - 1;
    let t0 = std::time::Instant::now();
    let floor = elias_row(1, total, &base, &ext, -128.0).unwrap();
    println!("floor:   z* = {} (delta {:.5}) in {:?}",
             floor.z_star, floor.delta_star, t0.elapsed());
    let t1 = std::time::Instant::now();
    let ceil = envelope_row(1, total, total - k - 1, &ext, -128.0, |z| {
        lg_cut_envelope(total, k, z)
    })
    .unwrap();
    println!("ceiling: z* = {} (delta {:.5}, avg-form cut term) in {:?}",
             ceil.z_star, ceil.delta_star, t1.elapsed());
    println!("gap: {} z-steps ({:.5} in delta)",
             floor.z_star as i64 - ceil.z_star as i64,
             floor.delta_star - ceil.delta_star);
}
