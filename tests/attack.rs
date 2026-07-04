//! Golden tests for the attack calculator, pinned to the exhaustively
//! cross-checked threshold tables (independent Python optimizer over the same
//! constructions, itself validated against the survey's Table 5: the
//! antipodal method reproduces 0.492188 at rate 1/2 before any improvement is
//! claimed).

use vanish::attack::{
    antipodal_attack, best_attack, elias_delta_star, hyperbola_ceiling, AttackParams,
};

fn koala_list_bits() -> f64 {
    // sextic extension of KoalaBear, grand-challenge target 2^-128 |F|
    6.0 * (2130706433f64).log2() - 128.0
}

#[test]
fn golden_rate_half_koalabear() {
    let p = AttackParams {
        n: 1 << 21,
        k: 1 << 20,
        list_bits: koala_list_bits(),
    };
    let anti = antipodal_attack(&p).unwrap().unwrap();
    assert!((anti.delta_star - 0.492188).abs() < 1e-5, "survey Table 5");
    let best = best_attack(&p).unwrap().unwrap();
    assert_eq!((best.t, best.s_g), (15, 1 << 21));
    assert_eq!(best.r, (1 << 20) + (1 << 15) - 1);
    // exact: delta* = 1 - r/s_G = 1/2 - (2^15 - 1)/2^21 (prints as 0.484375)
    assert_eq!(best.delta_star, 1.0 - best.r as f64 / (1u64 << 21) as f64);
    assert!(
        (best.delta_star - 0.4843755).abs() < 1e-6,
        "ladder improvement"
    );
    assert!((best.log2_list - 59.67).abs() < 0.1); // C(63, 32)
    let ceil = hyperbola_ceiling(&p).unwrap();
    assert!((ceil - 0.482739).abs() < 1e-4);
    assert!(best.delta_star > ceil, "attack cannot beat the ceiling");
}

#[test]
fn golden_other_rates_and_worstcase_field() {
    for (k_shift, lam, want_anti, want_best) in [
        (19u32, koala_list_bits(), 0.746094, 0.742188), // rate 1/4
        (18, koala_list_bits(), 0.871094, 0.867188),    // rate 1/8
        (17, koala_list_bits(), 0.935547, 0.933594),    // rate 1/16
        (20, 128.0, 0.498047, 0.496094),                // rate 1/2, |F| = 2^256
    ] {
        let p = AttackParams {
            n: 1 << 21,
            k: 1u64 << k_shift,
            list_bits: lam,
        };
        let anti = antipodal_attack(&p).unwrap().unwrap().delta_star;
        let best = best_attack(&p).unwrap().unwrap().delta_star;
        assert!(
            (anti - want_anti).abs() < 1e-5,
            "antipodal at k=2^{k_shift}"
        );
        assert!((best - want_best).abs() < 1e-5, "ladder at k=2^{k_shift}");
    }
}

#[test]
fn golden_elias_koalabear_base() {
    // survey Table 4: delta* = 0.468 for the 31-bit KoalaBear base field
    let p = AttackParams {
        n: 1 << 21,
        k: 1 << 20,
        list_bits: koala_list_bits(),
    };
    let d = elias_delta_star(&p, (2130706433f64).log2())
        .unwrap()
        .unwrap();
    assert!((d - 0.468).abs() < 2e-3, "Table 4 Elias threshold, got {d}");
}

#[test]
fn attack_error_paths() {
    let bad = AttackParams {
        n: 100,
        k: 50,
        list_bits: 10.0,
    };
    assert!(best_attack(&bad).is_err(), "non-power-of-two n rejected");
    let bad = AttackParams {
        n: 128,
        k: 0,
        list_bits: 10.0,
    };
    assert!(best_attack(&bad).is_err());
    // unattainable list size -> Ok(None), not an error
    let p = AttackParams {
        n: 128,
        k: 64,
        list_bits: 1e9,
    };
    assert!(best_attack(&p).unwrap().is_none());
}
