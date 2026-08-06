//! The soundness chain — one spine, two faces.
//!
//! Everything here converts certified counts into certified soundness
//! statements at the challenge box, as machine-checked interval
//! brackets ([`Lg`](crate::math::enclosure::Lg) enclosures). The
//! layering: [`volumes`] counts (balls, expected lists, the exact
//! Elias count); [`chain`] converts (the Lemma 6.12 soundness map and
//! the z-lattice crossing reports); [`floor`] consumes counts on the
//! attack side — what adversaries certifiably achieve. The forthcoming
//! `ceiling` consumes the master theorem's list envelope through the
//! identical chain, and the prize's pinch is one testable assertion:
//! the floor's certified crossing equals the ceiling's.
//!
//! The flat namespace is preserved: every item re-exports here, and
//! `attack::certified` remains an alias of this module.

pub mod chain;
pub mod floor;
pub mod volumes;

pub use chain::*;
pub use floor::*;
pub use volumes::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::enclosure::Lg;
    use rug::float::Round;
    use rug::ops::Pow;
    use rug::Integer;

    fn assert_encloses(lg: &Lg, exact: &Integer) {
        let e = Lg::from_integer(exact);
        assert!(
            lg.lo <= e.lo && e.hi <= lg.hi,
            "enclosure violated: [{}, {}] vs exact log2 ~ {}",
            lg.lo.to_f64_round(Round::Down),
            lg.hi.to_f64_round(Round::Up),
            e.lo.to_f64_round(Round::Down),
        );
    }

    #[test]
    fn ball_encloses_exact() {
        let cases: &[(u64, u64, u64)] = &[
            (10, 4, 97),
            (24, 11, 257),
            (64, 31, 65537),
            (200, 99, 1_000_003),
        ];
        for &(n, z, q) in cases {
            let q = Integer::from(q);
            let exact = ball_exact(n, z, &q);
            let lg = lg_ball(n, z, &q).unwrap();
            assert_encloses(&lg, &exact);
            // and the bracket is tight: within 0.01 bits at these sizes
            let width = lg.hi.to_f64_round(Round::Up) - lg.lo.to_f64_round(Round::Down);
            assert!(width < 0.01, "bracket too wide: {width}");
        }
    }

    #[test]
    fn expected_list_is_exact_identity_small() {
        // E\[list\] = |C| V / q^n for ANY code (linearity). Check the interval
        // against the exact rational at small parameters.
        let (n, k, z, q) = (16u64, 8u64, 7u64, 97u64);
        let q = Integer::from(q);
        let v = ball_exact(n, z, &q);
        let num = q.clone().pow(k as u32) * &v;
        let den = q.clone().pow(n as u32);
        // log2(num/den) must lie inside the enclosure
        let lg = lg_expected_list(n, k, z, &q).unwrap();
        let ln = Lg::from_integer(&num);
        let ld = Lg::from_integer(&den);
        let exact = ln.div(&ld);
        assert!(lg.lo <= exact.lo && exact.hi <= lg.hi);
    }

    #[test]
    fn koalabear_sextic_constant() {
        let ext = Integer::from(crate::field::named::KOALABEAR).pow(6);
        let l = Lg::from_integer(&ext).lo.to_f64_round(Round::Down);
        assert!((l - 185.93196).abs() < 0.001, "log2|F| = {l}");
    }

    #[test]
    fn table4_certified_profile() {
        // Golden pins for the certified ABF26 Table 4 reproduction (exact
        // Lemma 3.7 Elias count + Lemma 6.12 soundness map, target 2^-128).
        // Every crossing is pinned to a single z (one z-step moves the count
        // by ~31 bits over the KoalaBear base alphabet).
        let base = Integer::from(crate::field::named::KOALABEAR);
        let ext = base.clone().pow(6);
        let expect: &[(u64, u64)] = &[
            (1, 981_106),
            (1 << 1, 490_554),
            (1 << 2, 245_279),
            (1 << 3, 122_641),
            (1 << 4, 61_322),
            (1 << 5, 30_662),
            (1 << 6, 15_332),
            (1 << 7, 7_667),
            (1 << 8, 3_835),
            (1 << 9, 1_919),
            (1 << 10, 961),
            (1 << 11, 482),
            (1 << 12, 242),
        ];
        for &(s, z) in expect {
            let r = elias_row(s, 1 << 21, &base, &ext, -128.0).unwrap();
            assert_eq!(r.z_star, z, "row s = {s}");
            assert!(r.crossing_pinned, "row s = {s} not pinned");
            if s == 1 {
                // the certified wall floor: delta* = 981106 / 2^21,
                // i.e. 0.46783 to five places
                assert!((r.delta_star * 1e5).round() / 1e5 == 0.46783);
            }
        }
    }
}
