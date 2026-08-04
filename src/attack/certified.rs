//! The certified attack tier: counting bounds and attack radii as
//! machine-checked brackets ([`Lg`] enclosures from
//! [`enclosure`](crate::enclosure)).
//!
//! The counting layer is generic over any `[n, k]` code on an alphabet
//! of size `Q` (passed as a plain `Integer`; named fields live in
//! [`crate::field::named`], extension sizes are `Integer::pow` at the
//! call site): certified Hamming-ball volumes ([`lg_ball`], with
//! Theta(n)-term sums never enumerated — the dominant term exact, the
//! tail capped by a certified geometric series, which is what makes
//! `n = 2^41` tractable), the exact expected-list identity
//! ([`lg_expected_list`]), and the *original* Elias count of ABF26
//! Lemma 3.7 ([`lg_elias_list`]) — not the MS77 approximation of their
//! Corollary 3.8.
//!
//! The attack layer chains these into certified radii: the Lemma 6.12
//! soundness map ([`lg_soundness`]), Table-4-style rows ([`elias_row`] —
//! because attack radii live on the lattice `{z/n}`, the certified
//! crossing is an integer `z*`, which corrects printed continuous-delta
//! tables at small `n`), and first-moment crossings
//! ([`first_moment_crossing`], the Diamond–Gruen random-words attack).
//! Reproductions of published tables are configuration at the edges
//! (`examples/table4.rs`); [`ball_exact`] is the small-parameter bignum
//! oracle the tests cross-check against.

use crate::error::{Error, Result};
use crate::math::enclosure::{lg_binom, Lg};
use rug::float::Round;
use rug::ops::SubAssignRound;
use rug::{Float, Integer};

/// Validate an alphabet size and return the log2 enclosures of `Q` and
/// `Q - 1`. Sizes come from the caller — named fields live in
/// [`crate::field::named`], extension sizes are `Integer::pow` at the
/// call site; this module models none of them.
fn lg_q_pair(q: &Integer) -> Result<(Lg, Lg)> {
    if *q < 3 {
        return Err(Error::OutOfRange("alphabet size must be >= 3".into()));
    }
    Ok((
        Lg::from_integer(q),
        Lg::from_integer(&Integer::from(q - 1u32)),
    ))
}

/// Certified enclosure of log2 V(n, z) with V the Hamming ball volume
/// `sum_{j<=z} C(n,j) (Q-1)^j`, valid in the term-growth regime
/// (Q - 1) > z / (n - z + 1), i.e. terms strictly increase in j so the top
/// term dominates and the rest is a certified geometric tail.
pub fn lg_ball(n: u64, z: u64, q: &Integer) -> Result<Lg> {
    let (_, lg_q1) = lg_q_pair(q)?;
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

/// Certified enclosure of log2 E[list] = k log2 Q + log2 V(n, z) - n log2 Q:
/// the exact expected number of codewords of ANY [n, k] code over the
/// alphabet within radius z of a uniformly random word.
pub fn lg_expected_list(n: u64, k: u64, z: u64, q: &Integer) -> Result<Lg> {
    let (lg_q, _) = lg_q_pair(q)?;
    if k >= n {
        return Err(Error::OutOfRange("need k < n".into()));
    }
    Ok(lg_q.pow(k).mul(&lg_ball(n, z, q)?).div(&lg_q.pow(n)))
}

/// Report row for one radius in a crossing sweep.
#[derive(Debug, Clone)]
pub struct CrossingRow {
    /// Radius (absolute number of errors).
    pub z: u64,
    /// Fractional radius z / n.
    pub delta: f64,
    /// Certified lower endpoint of log2 E[list].
    pub lg_e_lo: f64,
    /// Certified upper endpoint of log2 E[list].
    pub lg_e_hi: f64,
}

/// The certified first-moment crossing: the largest radius z such that
/// log2 E[list] >= target is *certified* (lower endpoint clears the target),
/// plus the smallest z where it is certified NOT to (upper endpoint below).
/// Between the two the bracket straddles the target (a gap of at most a few
/// z-values at challenge scale, where one z step moves the log by ~log2 Q).
pub struct Crossing {
    /// Largest z whose lower endpoint clears the target.
    pub certified_at_or_above: Option<CrossingRow>,
    /// Smallest z whose upper endpoint is under the target.
    pub certified_below: Option<CrossingRow>,
}

/// Binary-search the crossing of log2 E[list] against `target` over
/// z in [z_min, z_max]. E[list] is strictly increasing in z, so the
/// certified predicates are monotone and binary search is sound.
pub fn first_moment_crossing(
    n: u64,
    k: u64,
    target: f64,
    z_min: u64,
    z_max: u64,
    q: &Integer,
) -> Result<Crossing> {
    let row = |z: u64| -> Result<CrossingRow> {
        let e = lg_expected_list(n, k, z, q)?;
        Ok(CrossingRow {
            z,
            delta: z as f64 / n as f64,
            lg_e_lo: e.lo.to_f64_round(Round::Down),
            lg_e_hi: e.hi.to_f64_round(Round::Up),
        })
    };
    // largest z with certified lo >= target
    let mut above: Option<CrossingRow> = None;
    let (mut a, mut b) = (z_min, z_max);
    while a <= b {
        let mid = a + (b - a) / 2;
        let r = row(mid)?;
        if r.lg_e_lo >= target {
            above = Some(r);
            a = mid + 1;
        } else {
            if mid == 0 {
                break;
            }
            b = mid - 1;
        }
    }
    // smallest z with certified hi < target
    let mut below: Option<CrossingRow> = None;
    let (mut a, mut b) = (z_min, z_max);
    while a <= b {
        let mid = a + (b - a) / 2;
        let r = row(mid)?;
        if r.lg_e_hi < target {
            below = Some(r);
            b = mid.saturating_sub(1);
            if mid == 0 {
                break;
            }
        } else {
            a = mid + 1;
        }
    }
    Ok(Crossing {
        certified_at_or_above: above,
        certified_below: below,
    })
}

/// Certified enclosure of log2 |Λ|-lower-bound from the exact Elias count
/// (ABF26 Lemma 3.7, no MS77 approximation): |Λ(C, z/n)| >= V(n, z) / Q^(n-k)
/// for ANY code C: Σ^k -> Σ^n over an alphabet of size Q.
pub fn lg_elias_list(n: u64, k: u64, z: u64, q: &Integer) -> Result<Lg> {
    let (lg_q, _) = lg_q_pair(q)?;
    if k >= n {
        return Err(Error::OutOfRange("need k < n".into()));
    }
    Ok(lg_ball(n, z, q)?.div(&lg_q.pow(n - k)))
}

/// Certified enclosure of log2 of the ABF26 Lemma 6.12 soundness floor
/// x / (|F| + 2x), given an enclosure of log2 x and the extension field F.
/// Monotone increasing in x, so endpoints map to endpoints.
pub fn lg_soundness(lg_list: &Lg, ext_q: &Integer) -> Result<Lg> {
    let (lg_q, _) = lg_q_pair(ext_q)?;
    let two_x = lg_list.mul(&Lg::from_u64(2));
    Ok(lg_list.div(&lg_q.add(&two_x)))
}

/// One certified Table-4-style row: the attack radius for interleaved RS at
/// rate 1/2 over the KoalaBear stack, per ABF26 Section 6.4's recipe with the
/// exact Elias count in place of Corollary 3.8.
#[derive(Debug, Clone)]
pub struct EliasRow {
    /// Interleaving width s (the table index).
    pub s: u64,
    /// Base-code block length n = 2^21 / s.
    pub n: u64,
    /// Smallest z certified to push soundness >= 2^-128 (lower endpoint).
    pub z_star: u64,
    /// delta* = z_star / n.
    pub delta_star: f64,
    /// Certified lower endpoint of log2 soundness at z_star.
    pub lg_sound_lo: f64,
    /// Certified upper endpoint of log2 soundness at z_star.
    pub lg_sound_hi: f64,
    /// True if at z_star - 1 the soundness is certified BELOW the target
    /// (upper endpoint under target), i.e. the crossing is pinned to one z.
    pub crossing_pinned: bool,
}

/// Certified reproduction of ABF26 Table 4: for interleaving width `s`
/// (block length n = total_len / s, message len n/2), find the smallest
/// radius z such that the Lemma 3.7 + Lemma 6.12 chain certifies soundness
/// error >= 2^target_bits. Soundness is monotone increasing in z, so binary
/// search on the certified predicate is sound.
pub fn elias_row(
    s: u64,
    total_len: u64,
    base_q: &Integer,
    ext_q: &Integer,
    target_bits: f64,
) -> Result<EliasRow> {
    let n = total_len / s;
    let k = n / 2;
    let sound_at = |z: u64| -> Result<Lg> { lg_soundness(&lg_elias_list(n, k, z, base_q)?, ext_q) };
    // smallest z with certified lo >= target
    let (mut a, mut b) = (1u64, n - 1);
    let mut best: Option<(u64, Lg)> = None;
    while a <= b {
        let mid = a + (b - a) / 2;
        let sd = sound_at(mid)?;
        if sd.lo.to_f64_round(Round::Down) >= target_bits {
            best = Some((mid, sd));
            if mid == 1 {
                break;
            }
            b = mid - 1;
        } else {
            a = mid + 1;
        }
    }
    let (z_star, sd) =
        best.ok_or_else(|| Error::Unsupported("no radius certifies the target soundness".into()))?;
    let pinned = if z_star > 1 {
        let prev = sound_at(z_star - 1)?;
        prev.hi.to_f64_round(Round::Up) < target_bits
    } else {
        true
    };
    Ok(EliasRow {
        s,
        n,
        z_star,
        delta_star: z_star as f64 / n as f64,
        lg_sound_lo: sd.lo.to_f64_round(Round::Down),
        lg_sound_hi: sd.hi.to_f64_round(Round::Up),
        crossing_pinned: pinned,
    })
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
            // and the bracket is tight: within 0.01 bits at these sizes
            let width = lg.hi.to_f64_round(Round::Up) - lg.lo.to_f64_round(Round::Down);
            assert!(width < 0.01, "bracket too wide: {width}");
        }
    }

    #[test]
    fn expected_list_is_exact_identity_small() {
        // E[list] = |C| V / q^n for ANY code (linearity). Check the interval
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
