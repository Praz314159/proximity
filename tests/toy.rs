//! Golden tests for the toy-protocol module, pinned to the exhaustive
//! exp23-style reference sweep (independent Python computation).

use vanish::domain::Subgroup;
use vanish::toy::{classes_count, exact_soundness};

#[test]
fn golden_toy_soundness() {
    // s = 16, r = 8: class count 3281 (the F2 value); soundness = 1 below the
    // transition, then exactly 3281/p once collisions vanish.
    assert_eq!(classes_count(16, 8), 3281);
    let t = exact_soundness(&Subgroup::new(17, 16).unwrap(), 8).unwrap();
    assert_eq!(t.winning, 17);
    assert!(
        (t.soundness - 1.0).abs() < 1e-12,
        "fully broken below transition"
    );
    let t = exact_soundness(&Subgroup::new(35521, 16).unwrap(), 8).unwrap();
    assert_eq!(
        t.winning, 3281,
        "saturates at the class count, zero collisions"
    );
    // s = 32, r = 16: every challenge wins at p = 97
    assert_eq!(classes_count(32, 16), 21523361);
    let t = exact_soundness(&Subgroup::new(97, 32).unwrap(), 16).unwrap();
    assert_eq!(t.winning, 97);
}
