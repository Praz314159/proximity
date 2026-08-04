//! Certified volume bounds — rigorous interval evaluation of Hamming-ball
//! counting attacks at challenge scale.
//!
//! The headline use is a certified reproduction of ABF26 (ePrint 2026/680)
//! Table 4: the Elias attack radius per interleaving width. The chain is
//! exact throughout — Lemma 3.7's counting identity
//!
//! ```text
//!   |Λ(C, z/n)| >= V(n, z) / Q^(n-k),   V(n, z) = sum_{j<=z} C(n,j) (Q-1)^j
//! ```
//!
//! (the *original* Elias statement, not the MS77 approximation of ABF's
//! Corollary 3.8), fed through the Lemma 6.12 soundness map x / (|F| + 2x).
//! Because attack radii are distances on the lattice {z/n}, the certified
//! crossing is an integer z*, not a continuous delta — at small n this
//! corrects the printed Table 4 tail rows upward by up to ~10^-3.
//!
//! Same machinery, secondary use: the expected-list identity
//! `E[list] = Q^k V(n,z) / Q^n` (exact, linearity of expectation) gives the
//! Diamond–Gruen (ePrint 2025/2010) random-words attack, and their Thm 2.5
//! converts `P = Pr[d(u, C) <= z]` into a proximity-gap error floor
//! `err >= (1 - 1/Q) P` (P needs a Bonferroni pairwise correction that is
//! not implemented here; E[list] does not).
//!
//! Numbers at challenge scale (n = 2^41, log2 Q ~ 186) have ~10^14-bit
//! exponents: exact integers are unrepresentable and f64 is unsound (the MDS
//! weight enumerator alternates). Everything here is therefore computed as an
//! *interval enclosure of the base-2 logarithm*, in MPFR floats with directed
//! rounding: lower endpoints round down, upper endpoints round up, every
//! operation, no exceptions. A returned `[lo, hi]` is a machine-checked
//! bracket; certified claims use `lo` (for ">=") or `hi` (for "<=") only.
//!
//! Sums with Theta(n) terms are never enumerated: in the challenge regime
//! `Q >> n` every sum here is dominated by its extreme term with an explicit
//! per-step ratio, so we take the dominant term exactly and cap the rest by a
//! certified geometric series. Exact-bignum cross-checks at small parameters
//! pin the enclosures in tests.

use crate::error::{Error, Result};
use rug::float::{Constant, Round};
use rug::ops::{AddAssignRound, DivAssignRound, MulAssignRound, Pow, SubAssignRound};
use rug::{Float, Integer};

/// Working precision (bits). Final log2 values are ~2^48 in magnitude and we
/// want ~30 certified fractional bits; 192 leaves two orders of margin over
/// accumulated rounding across ~10^4 operations.
const PREC: u32 = 192;

/// Interval `[lo, hi]` enclosing the base-2 logarithm of a positive quantity.
#[derive(Clone, Debug)]
pub struct Lg {
    /// Lower endpoint (rounded down at every step).
    pub lo: Float,
    /// Upper endpoint (rounded up at every step).
    pub hi: Float,
}

fn f(v: f64) -> Float {
    Float::with_val(PREC, v)
}

/// ln(2) rounded in both directions (shared divisor for lngamma -> log2).
fn ln2(round: Round) -> Float {
    let mut x = Float::with_val(PREC, Constant::Log2);
    // Constant::Log2 is correctly rounded to nearest; nudge one ulp outward.
    match round {
        Round::Down => x.next_down(),
        Round::Up => x.next_up(),
        _ => unreachable!(),
    }
    x
}

impl Lg {
    /// Exact zero (the quantity 1).
    pub fn zero() -> Self {
        Lg {
            lo: f(0.0),
            hi: f(0.0),
        }
    }

    /// log2 of an exact nonzero integer.
    pub fn from_integer(x: &Integer) -> Self {
        assert!(*x > 0);
        let mut lo = Float::with_val_round(PREC, x, Round::Down).0;
        let mut hi = Float::with_val_round(PREC, x, Round::Up).0;
        lo.log2_round(Round::Down);
        hi.log2_round(Round::Up);
        Lg { lo, hi }
    }

    /// log2 of an exact nonzero u64.
    pub fn from_u64(x: u64) -> Self {
        Self::from_integer(&Integer::from(x))
    }

    /// Product of quantities = sum of logs.
    pub fn mul(&self, o: &Lg) -> Lg {
        let mut lo = self.lo.clone();
        lo.add_assign_round(&o.lo, Round::Down);
        let mut hi = self.hi.clone();
        hi.add_assign_round(&o.hi, Round::Up);
        Lg { lo, hi }
    }

    /// Quotient of quantities = difference of logs.
    pub fn div(&self, o: &Lg) -> Lg {
        let mut lo = self.lo.clone();
        lo.sub_assign_round(&o.hi, Round::Down);
        let mut hi = self.hi.clone();
        hi.sub_assign_round(&o.lo, Round::Up);
        Lg { lo, hi }
    }

    /// Integer power of the quantity = scale the log by `e >= 0`.
    pub fn pow(&self, e: u64) -> Lg {
        let ef = Float::with_val(PREC, e);
        let mut lo = self.lo.clone();
        lo.mul_assign_round(&ef, Round::Down);
        let mut hi = self.hi.clone();
        hi.mul_assign_round(&ef, Round::Up);
        // negative logs scale the other way; swap if needed
        if lo > hi {
            std::mem::swap(&mut lo, &mut hi);
        }
        Lg { lo, hi }
    }

    /// Sum of quantities: log2(2^a + 2^b), each endpoint with its own
    /// rounding direction.
    pub fn add(&self, o: &Lg) -> Lg {
        Lg {
            lo: lg_add_dir(&self.lo, &o.lo, Round::Down),
            hi: lg_add_dir(&self.hi, &o.hi, Round::Up),
        }
    }

    /// Widen the upper endpoint by `bits` (multiplicative slack 2^bits).
    pub fn widen_hi(&self, bits: &Float) -> Lg {
        let mut hi = self.hi.clone();
        hi.add_assign_round(bits, Round::Up);
        Lg {
            lo: self.lo.clone(),
            hi,
        }
    }
}

/// Directed log2(2^a + 2^b): m + log2(1 + 2^(s - m)) with m = max, s = min.
fn lg_add_dir(a: &Float, b: &Float, round: Round) -> Float {
    let (m, s) = if a >= b { (a, b) } else { (b, a) };
    let mut d = s.clone();
    d.sub_assign_round(m, round); // <= 0
    let mut t = d.exp2();
    t.add_assign_round(&f(1.0), round);
    t.log2_round(round);
    t.add_assign_round(m, round);
    t
}

/// Certified lower bound on log2(2^a - 2^b) given a lower bound `a_lo` on the
/// first log and an upper bound `b_hi` on the second. None if the bracket
/// cannot certify positivity.
pub fn lg_sub_lower(a_lo: &Float, b_hi: &Float) -> Option<Float> {
    if a_lo <= b_hi {
        return None;
    }
    // a_lo + log2(1 - 2^(b_hi - a_lo)), rounding down throughout.
    let mut d = b_hi.clone();
    d.sub_assign_round(a_lo, Round::Up); // d < 0; round the exponent up so 2^d is over-estimated
    let mut t = d.exp2(); // rounding of exp2: use next_up to stay safe
    t.next_up();
    let mut one_minus = f(1.0);
    one_minus.sub_assign_round(&t, Round::Down);
    if one_minus <= 0 {
        return None;
    }
    one_minus.log2_round(Round::Down);
    one_minus.add_assign_round(a_lo, Round::Down);
    Some(one_minus)
}

/// log2 Gamma(x) as an interval, for integer x >= 1.
fn lgamma2(x: u64) -> Lg {
    if x <= 2 {
        return Lg::zero(); // Gamma(1) = Gamma(2) = 1
    }
    let xf = Float::with_val(PREC, x);
    let mut lo = xf.clone();
    lo.ln_gamma_round(Round::Down);
    let mut hi = xf;
    hi.ln_gamma_round(Round::Up);
    // divide by ln 2 (positive), directed
    lo.div_assign_round(&ln2(Round::Up), Round::Down);
    hi.div_assign_round(&ln2(Round::Down), Round::Up);
    Lg { lo, hi }
}

/// log2 of the binomial coefficient C(n, k).
pub fn lg_binom(n: u64, k: u64) -> Lg {
    assert!(k <= n);
    lgamma2(n + 1).div(&lgamma2(k + 1)).div(&lgamma2(n - k + 1))
}

/// Alphabet parameters: Q as an exact integer with cached log2 enclosures of
/// Q and Q - 1.
pub struct Alphabet {
    /// Alphabet size as an exact integer.
    pub q: Integer,
    /// Enclosure of log2 Q.
    pub lg_q: Lg,
    /// Enclosure of log2 (Q - 1).
    pub lg_q1: Lg,
}

impl Alphabet {
    /// Build from an exact alphabet size (must be >= 3).
    pub fn new(q: Integer) -> Result<Self> {
        if q < 3 {
            return Err(Error::OutOfRange("alphabet size must be >= 3".into()));
        }
        let lg_q = Lg::from_integer(&q);
        let lg_q1 = Lg::from_integer(&Integer::from(&q - 1u32));
        Ok(Alphabet { q, lg_q, lg_q1 })
    }

    /// KoalaBear base field: 2^31 - 2^24 + 1.
    pub fn koalabear() -> Self {
        Self::new(Integer::from((1u64 << 31) - (1u64 << 24) + 1)).expect("koalabear")
    }

    /// KoalaBear sextic: (2^31 - 2^24 + 1)^6, |F| ~ 2^185.93.
    pub fn koalabear6() -> Self {
        let q0 = Integer::from((1u64 << 31) - (1u64 << 24) + 1);
        Self::new(q0.pow(6)).expect("koalabear")
    }
}

/// Certified enclosure of log2 V(n, z) with V the Hamming ball volume
/// `sum_{j<=z} C(n,j) (Q-1)^j`, valid in the term-growth regime
/// (Q - 1) > z / (n - z + 1), i.e. terms strictly increase in j so the top
/// term dominates and the rest is a certified geometric tail.
pub fn lg_ball(n: u64, z: u64, ab: &Alphabet) -> Result<Lg> {
    if z == 0 {
        return Ok(Lg::zero());
    }
    if z >= n {
        return Err(Error::OutOfRange("ball radius must satisfy z < n".into()));
    }
    let top = lg_binom(n, z).mul(&ab.lg_q1.pow(z));
    // per-step down-ratio r = (j / (n - j + 1)) / (Q - 1) maximised at j = z
    let r_log = Lg::from_u64(z).div(&Lg::from_u64(n - z + 1)).div(&ab.lg_q1);
    let mut r = r_log.hi.clone().exp2();
    r.next_up();
    if r >= 0.5 {
        return Err(Error::Unsupported(
            "ball tail ratio >= 1/2: outside the Q >> n regime".into(),
        ));
    }
    // sum <= top / (1 - r): widen hi by -log2(1 - r)
    let mut one_minus = f(1.0);
    one_minus.sub_assign_round(&r, Round::Down);
    let mut tail_bits = one_minus;
    tail_bits.log2_round(Round::Down);
    tail_bits = -tail_bits; // >= 0, rounded up as a widening
    Ok(top.widen_hi(&tail_bits))
}

/// Certified enclosure of log2 E[list] = k log2 Q + log2 V(n, z) - n log2 Q:
/// the exact expected number of codewords of ANY [n, k] code over the
/// alphabet within radius z of a uniformly random word.
pub fn lg_expected_list(n: u64, k: u64, z: u64, ab: &Alphabet) -> Result<Lg> {
    if k >= n {
        return Err(Error::OutOfRange("need k < n".into()));
    }
    Ok(ab.lg_q.pow(k).mul(&lg_ball(n, z, ab)?).div(&ab.lg_q.pow(n)))
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
    ab: &Alphabet,
) -> Result<Crossing> {
    let row = |z: u64| -> Result<CrossingRow> {
        let e = lg_expected_list(n, k, z, ab)?;
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
pub fn lg_elias_list(n: u64, k: u64, z: u64, ab: &Alphabet) -> Result<Lg> {
    if k >= n {
        return Err(Error::OutOfRange("need k < n".into()));
    }
    Ok(lg_ball(n, z, ab)?.div(&ab.lg_q.pow(n - k)))
}

/// Certified enclosure of log2 of the ABF26 Lemma 6.12 soundness floor
/// x / (|F| + 2x), given an enclosure of log2 x and the extension field F.
/// Monotone increasing in x, so endpoints map to endpoints.
pub fn lg_soundness(lg_list: &Lg, ext: &Alphabet) -> Lg {
    let two_x = lg_list.mul(&Lg::from_u64(2));
    let den = Lg {
        lo: lg_add_dir(&ext.lg_q.lo, &two_x.lo, Round::Down),
        hi: lg_add_dir(&ext.lg_q.hi, &two_x.hi, Round::Up),
    };
    lg_list.div(&den)
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
    base: &Alphabet,
    ext: &Alphabet,
    target_bits: f64,
) -> Result<EliasRow> {
    let n = total_len / s;
    let k = n / 2;
    let sound_at = |z: u64| -> Result<Lg> { Ok(lg_soundness(&lg_elias_list(n, k, z, base)?, ext)) };
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
    fn binom_encloses_exact() {
        for &(n, k) in &[
            (10u64, 3u64),
            (24, 12),
            (100, 47),
            (1000, 400),
            (5000, 2500),
        ] {
            let exact = Integer::from(n).binomial(k as u32);
            assert_encloses(&lg_binom(n, k), &exact);
        }
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
            let ab = Alphabet::new(Integer::from(q)).unwrap();
            let exact = ball_exact(n, z, &ab.q);
            let lg = lg_ball(n, z, &ab).unwrap();
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
        let ab = Alphabet::new(Integer::from(q)).unwrap();
        let v = ball_exact(n, z, &ab.q);
        let num = ab.q.clone().pow(k as u32) * &v;
        let den = ab.q.clone().pow(n as u32);
        // log2(num/den) must lie inside the enclosure
        let lg = lg_expected_list(n, k, z, &ab).unwrap();
        let ln = Lg::from_integer(&num);
        let ld = Lg::from_integer(&den);
        let exact = ln.div(&ld);
        assert!(lg.lo <= exact.lo && exact.hi <= lg.hi);
    }

    #[test]
    fn koalabear_constant() {
        let ab = Alphabet::koalabear6();
        let l = ab.lg_q.lo.to_f64_round(Round::Down);
        assert!((l - 185.93196).abs() < 0.001, "log2|F| = {l}");
    }

    #[test]
    fn table4_certified_profile() {
        // Golden pins for the certified ABF26 Table 4 reproduction (exact
        // Lemma 3.7 Elias count + Lemma 6.12 soundness map, target 2^-128).
        // Every crossing is pinned to a single z (one z-step moves the count
        // by ~31 bits over the KoalaBear base alphabet).
        let base = Alphabet::koalabear();
        let ext = Alphabet::koalabear6();
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

    #[test]
    fn lg_sub_lower_sound() {
        // log2(2^10 - 2^8) = log2(768) exactly
        let a = f(10.0);
        let b = f(8.0);
        let lo = lg_sub_lower(&a, &b).unwrap();
        let exact = Lg::from_u64(768);
        assert!(lo <= exact.lo);
        // and not absurdly loose
        assert!(lo.to_f64_round(Round::Down) > 9.58);
        // refuses to certify when bracket overlaps
        assert!(lg_sub_lower(&f(8.0), &f(10.0)).is_none());
    }
}
