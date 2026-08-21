//! The counting layer of the soundness chain: certified Hamming-ball
//! volumes, the exact expected-list identity, and the original Elias
//! count (ABF26 Lemma 3.7) — the attack side's data layer, consumed
//! by [`super::floor`]; the defense side's counterpart is
//! [`super::envelope`]. Upgrades to the estimates land here once
//! (the Diamond--Gruen Thm 3.10 ball estimate).

use crate::error::{Error, Result};
use crate::math::enclosure::{lg_binom, Lg};
use rug::float::Round;
use rug::ops::SubAssignRound;
use rug::{Float, Integer};

/// Validate an alphabet size (sizes come from the caller — named
/// fields live in [`crate::field::named`], extension sizes are
/// `Integer::pow` at the call site; this module models none of them).
fn check_q(q: &Integer) -> Result<()> {
    if *q < 3 {
        return Err(Error::OutOfRange("alphabet size must be >= 3".into()));
    }
    Ok(())
}

/// Certified log2 enclosure of the alphabet size `Q`.
pub(crate) fn lg_q(q: &Integer) -> Result<Lg> {
    check_q(q)?;
    Ok(Lg::from_integer(q))
}

/// Certified log2 enclosure of `Q - 1` (the ball volumes' base).
pub(crate) fn lg_q1(q: &Integer) -> Result<Lg> {
    check_q(q)?;
    Ok(Lg::from_integer(&Integer::from(q - 1u32)))
}

/// Certified enclosure of log2 V(n, z) with V the Hamming ball volume
/// `sum_{j<=z} C(n,j) (Q-1)^j`, valid in the term-growth regime
/// (Q - 1) > z / (n - z + 1), i.e. terms strictly increase in j so the top
/// term dominates and the rest is a certified geometric tail.
pub fn lg_ball(n: u64, z: u64, q: &Integer) -> Result<Lg> {
    let lg_q1 = lg_q1(q)?;
    if z == 0 {
        return Ok(Lg::zero());
    }
    if z >= n {
        return Err(Error::OutOfRange("ball radius must satisfy z < n".into()));
    }
    let top = lg_binom(n, z).mul(&lg_q1.pow(z));
    // per-step down-ratio r = (j / (n - j + 1)) / (Q - 1) maximised at j = z
    let r_log = Lg::from_u64(z).div(&Lg::from_u64(n - z + 1)).div(&lg_q1);
    let mut r = r_log.hi.clone().exp2();
    r.next_up();
    if r >= 0.5 {
        return Err(Error::Unsupported(
            "ball tail ratio >= 1/2: outside the Q >> n regime".into(),
        ));
    }
    // sum <= top / (1 - r): widen hi by -log2(1 - r)
    let mut one_minus = Float::with_val(r.prec(), 1.0);
    one_minus.sub_assign_round(&r, Round::Down);
    let mut tail_bits = one_minus;
    tail_bits.log2_round(Round::Down);
    tail_bits = -tail_bits; // >= 0, rounded up as a widening
    Ok(top.widen_hi(&tail_bits))
}

/// Certified enclosure of log2 E\[list\] = k log2 Q + log2 V(n, z) - n log2 Q:
/// the exact expected number of codewords of ANY [n, k] code over the
/// alphabet within radius z of a uniformly random word.
pub fn lg_expected_list(n: u64, k: u64, z: u64, q: &Integer) -> Result<Lg> {
    let lg_q = lg_q(q)?;
    if k >= n {
        return Err(Error::OutOfRange("need k < n".into()));
    }
    Ok(lg_q.pow(k).mul(&lg_ball(n, z, q)?).div(&lg_q.pow(n)))
}

/// Certified enclosure of log2 |Λ|-lower-bound from the exact Elias count
/// (ABF26 Lemma 3.7, no MS77 approximation): |Λ(C, z/n)| >= V(n, z) / Q^(n-k)
/// for ANY code C: Σ^k -> Σ^n over an alphabet of size Q.
pub fn lg_elias_list(n: u64, k: u64, z: u64, q: &Integer) -> Result<Lg> {
    let lg_q = lg_q(q)?;
    if k >= n {
        return Err(Error::OutOfRange("need k < n".into()));
    }
    Ok(lg_ball(n, z, q)?.div(&lg_q.pow(n - k)))
}

/// Exact ball volume by bignum summation (small parameters; test oracle).
pub fn ball_exact(n: u64, z: u64, q: &Integer) -> Integer {
    let mut v = Integer::from(1);
    let q1 = Integer::from(q - 1u32);
    let mut term = Integer::from(1);
    for j in 0..z {
        // term_{j+1} = term_j * (n - j) / (j + 1) * (q - 1)
        term *= n - j;
        term /= j + 1;
        term *= &q1;
        v += &term;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::float::Round;
    use rug::ops::Pow;

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
            let width = lg.hi.to_f64_round(Round::Up) - lg.lo.to_f64_round(Round::Down);
            assert!(width < 0.01, "bracket too wide: {width}");
        }
    }

    #[test]
    fn expected_list_is_exact_identity_small() {
        let (n, k, z, q) = (16u64, 8u64, 7u64, 97u64);
        let q = Integer::from(q);
        let v = ball_exact(n, z, &q);
        let num = q.clone().pow(k as u32) * &v;
        let den = q.clone().pow(n as u32);
        let lg = lg_expected_list(n, k, z, &q).unwrap();
        let ln = Lg::from_integer(&num);
        let ld = Lg::from_integer(&den);
        let exact = ln.div(&ld);
        assert!(lg.lo <= exact.lo && exact.hi <= lg.hi);
    }
}
