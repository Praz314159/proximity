//! The ceiling face: what the defense certifiably bounds. An envelope
//! row runs a certified list envelope through the same soundness map
//! and lattice search as the floor, and reports the largest radius at
//! which soundness is certified under the target — the prize's pinch
//! is this row's `z_star` landing one step under the floor's.
//!
//! Two things are parameters by design, stated rather than assumed.
//! The envelope itself is a closure, because which envelope is
//! admissible is the compilation chapter's decision; the built-in
//! [`lg_cut_envelope`] is the master theorem's dominant (cut-strata)
//! term in certified arithmetic — the average-form workhorse that
//! carries ~93% of the measured lists — and rows built on it alone are
//! labeled average-form: the sup-form supplement (the classified
//! families' tail mass) must be added before a row is a worst-case
//! theorem. And the soundness map is the floor's Lemma 6.12 form,
//! pending the compilation's defense-side derivation; the pinch
//! comparison is meaningful under a common map, which is exactly how
//! it is consumed.

use crate::error::{Error, Result};
use crate::math::enclosure::{lg_binom, Lg};
use rug::float::Round;
use rug::Integer;

use super::chain::{certified_first, lg_soundness};

/// One certified ceiling row: the largest radius at which the envelope,
/// pushed through the soundness map, is certified under the target.
#[derive(Debug, Clone)]
pub struct CeilingRow {
    /// Interleaving width s (the table index).
    pub s: u64,
    /// Base-code block length n = total_len / s.
    pub n: u64,
    /// Largest z whose soundness upper endpoint is under the target.
    pub z_star: u64,
    /// delta* = z_star / n.
    pub delta_star: f64,
    /// Certified lower endpoint of log2 soundness at z_star.
    pub lg_sound_lo: f64,
    /// Certified upper endpoint of log2 soundness at z_star.
    pub lg_sound_hi: f64,
    /// True if at z_star + 1 the soundness lower endpoint clears the
    /// target: the crossing is pinned to one z.
    pub crossing_pinned: bool,
}

/// The largest radius z in `[1, n - 1]` at which the envelope's
/// soundness is certified under `2^target_bits`. The envelope maps a
/// radius to a certified log2 list bound; soundness is monotone
/// increasing in the list, and the envelope is required monotone in z,
/// so the lattice search is sound.
pub fn envelope_row(
    s: u64,
    total_len: u64,
    z_max: u64,
    ext_q: &Integer,
    target_bits: f64,
    mut envelope: impl FnMut(u64) -> Result<Lg>,
) -> Result<CeilingRow> {
    if s == 0 || total_len % s != 0 {
        return Err(Error::OutOfRange(
            "interleaving width must divide the total length".into(),
        ));
    }
    let n = total_len / s;
    if z_max >= n {
        return Err(Error::OutOfRange("need z_max < n".into()));
    }
    let mut sound_at = |z: u64| -> Result<Lg> { lg_soundness(&envelope(z)?, ext_q) };
    // smallest z NOT certified sound within the envelope's domain; the
    // ceiling sits one below (beyond z_max the envelope says nothing,
    // so the row never claims past it)
    let first_unsound = certified_first(1, z_max, |z| {
        Ok(sound_at(z)?.hi.to_f64_round(Round::Up) >= target_bits)
    })?;
    let z_star = match first_unsound {
        Some(1) => {
            return Err(Error::Unsupported(
                "no radius certifies soundness under the target".into(),
            ))
        }
        Some(z) => z - 1,
        None => z_max,
    };
    let sd = sound_at(z_star)?;
    let pinned = match first_unsound {
        Some(z) => sound_at(z)?.lo.to_f64_round(Round::Down) >= target_bits,
        None => false,
    };
    Ok(CeilingRow {
        s,
        n,
        z_star,
        delta_star: z_star as f64 / n as f64,
        lg_sound_lo: sd.lo.to_f64_round(Round::Down),
        lg_sound_hi: sd.hi.to_f64_round(Round::Up),
        crossing_pinned: pinned,
    })
}

/// A SCAFFOLD envelope: the master theorem's cut-strata sum with the
/// *trivial* stratum bound `D_c(l) = C(n/2, l)` — the whole stratum —
/// in place of interface data, `sum over l < (k-1)/2 of
/// C(n/2, l) / C(t - 2l, k + 1 - 2l)` at `t = n - z`, enclosed term
/// by term.
///
/// What it is and is not. At the base cell `(32, 15, 17)` the trivial
/// choice lands within 7% of the measured record (2489 vs 2674), which
/// is a small-parameter coincidence, not a validation. At challenge
/// scale it is binary: the stratum range is empty for `z/n <= 1/4`
/// (the envelope returns the `L <= 1` bound) and the first nonempty
/// range carries terms of order `2^(n/2)`, so no radius above a
/// quarter is priced at all. Real rows need real interface data —
/// `D_c` from the per-prime envelope and `D_b` from the engine, and
/// the recursion down the tower rather than one application at the
/// top. That instantiation is the compilation chapter's arithmetic;
/// this function exists so the plumbing above it can be tested and
/// timed against a known input.
pub fn lg_cut_envelope(n: u64, k: u64, z: u64) -> Result<Lg> {
    if z >= n || k + 1 > n {
        return Err(Error::OutOfRange("need z < n and k < n".into()));
    }
    let t = n - z;
    let (r, kap, half) = (k + 1, (k - 1) / 2, n / 2);
    if t < r {
        return Err(Error::OutOfRange(
            "agreement below the interpolation threshold".into(),
        ));
    }
    let l_min = t.saturating_sub(half);
    let mut total: Option<Lg> = None;
    for l in l_min..kap {
        if t < 2 * l + (r - 2 * l) {
            break;
        }
        let term = lg_binom(half, l).div(&lg_binom(t - 2 * l, r - 2 * l));
        total = Some(match total {
            Some(acc) => acc.add(&term),
            None => term,
        });
    }
    // an empty stratum range is the high-agreement regime where the
    // cut term contributes nothing: the sum bounds the class by 1
    Ok(total.unwrap_or_else(Lg::zero))
}

/// The generic list-to-MCA conversion (GCXK25, ABF26 Theorem 5.1):
/// a certified list bound `L` at radius `z/n` yields certified mutual
/// correlated agreement at the square-root-loss radius
/// `1 - sqrt(1 - z/n + eta)` with error `(L^2 (z/n) n + 1/eta)/|F|`
/// — i.e. `(L^2 z + 1/eta)/|F|` on the lattice. The proximity loss is
/// intrinsic to the generic conversion (ABF26 Theorem 5.4's
/// counterexample); closing it for smooth-domain RS is the program's
/// open target, so rows through this map are the certified GENERIC
/// ceiling.
pub fn lg_mca_error(lg_list: &Lg, z: u64, inv_eta: u64, ext_q: &Integer) -> Result<Lg> {
    let numerator = lg_list
        .pow(2)
        .mul(&Lg::from_u64(z))
        .add(&Lg::from_u64(inv_eta));
    Ok(numerator.div(&super::volumes::lg_q(ext_q)?))
}

/// The CA radius of the conversion at list radius `z`, on the target
/// lattice, rounded DOWN (a conservative claim): the largest `z_ca`
/// with `z_ca/n <= 1 - sqrt(1 - z/n + eta)`, computed with directed
/// rounding (the sqrt argument and the sqrt both rounded up).
pub fn ca_radius(n: u64, z: u64, inv_eta: u64) -> u64 {
    use rug::Float;
    let prec = 128;
    let mut arg = Float::with_val(prec, n - z);
    arg /= n as f64;
    let mut eta = Float::with_val(prec, 1.0);
    eta /= inv_eta as f64;
    let mut s = Float::with_val_round(prec, arg + eta, Round::Up).0;
    s.sqrt_round(Round::Up);
    let mut frac = Float::with_val(prec, 1.0) - s;
    frac *= n as f64;
    frac.floor_mut();
    frac.to_f64().max(0.0) as u64
}

/// The generic-map ceiling row: the largest certified MCA radius at
/// the target, obtained by pushing the list envelope through the
/// Theorem 5.1 conversion and searching the list-radius lattice. Both
/// the CA radius and the error grow with the list radius, so the
/// certified predicate is monotone.
#[allow(clippy::too_many_arguments)] // the row's parameter list
pub fn ca_ceiling_row(
    s: u64,
    total_len: u64,
    z_max: u64,
    inv_eta: u64,
    ext_q: &Integer,
    target_bits: f64,
    mut envelope: impl FnMut(u64) -> Result<Lg>,
) -> Result<CeilingRow> {
    if s == 0 || total_len % s != 0 {
        return Err(Error::OutOfRange(
            "interleaving width must divide the total length".into(),
        ));
    }
    let n = total_len / s;
    if z_max >= n {
        return Err(Error::OutOfRange("need z_max < n".into()));
    }
    let mut err_at = |z: u64| -> Result<Lg> { lg_mca_error(&envelope(z)?, z, inv_eta, ext_q) };
    let first_bad = certified_first(1, z_max, |z| {
        Ok(err_at(z)?.hi.to_f64_round(Round::Up) >= target_bits)
    })?;
    let z_list = match first_bad {
        Some(1) => {
            return Err(Error::Unsupported(
                "no list radius certifies the target through the conversion".into(),
            ))
        }
        Some(z) => z - 1,
        None => z_max,
    };
    let sd = err_at(z_list)?;
    let z_ca = ca_radius(n, z_list, inv_eta);
    let pinned = match first_bad {
        Some(z) => err_at(z)?.lo.to_f64_round(Round::Down) >= target_bits,
        None => false,
    };
    Ok(CeilingRow {
        s,
        n,
        z_star: z_ca,
        delta_star: z_ca as f64 / n as f64,
        lg_sound_lo: sd.lo.to_f64_round(Round::Down),
        lg_sound_hi: sd.hi.to_f64_round(Round::Up),
        crossing_pinned: pinned,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rug::ops::Pow;
    use rug::Rational;

    /// The certified cut envelope encloses the exact rational sum at
    /// the record cell (the 2489.03 of the first envelope evaluation).
    #[test]
    fn cut_envelope_encloses_exact_rational() {
        let (n, k, t) = (32u64, 15u64, 17u64);
        let binom = |a: u64, b: u64| -> Integer {
            let mut v = Integer::from(1);
            for i in 0..b {
                v *= a - i;
                v /= i + 1;
            }
            v
        };
        let mut exact = Rational::new();
        for l in (t - n / 2)..(k - 1) / 2 {
            exact += Rational::from((binom(n / 2, l), binom(t - 2 * l, k + 1 - 2 * l)));
        }
        let lg = lg_cut_envelope(n, k, n - t).unwrap();
        let approx = exact.to_f64().log2();
        let lo = lg.lo.to_f64_round(Round::Down);
        let hi = lg.hi.to_f64_round(Round::Up);
        assert!(lo <= approx && approx <= hi, "[{lo}, {hi}] vs {approx}");
        assert!(hi - lo < 0.01, "bracket too wide");
        assert!((exact.to_f64() - 2489.03).abs() < 0.5);
    }

    /// End-to-end at small parameters: the ceiling row exists, is
    /// pinned, and moves monotonically with the target.
    #[test]
    fn ceiling_row_is_pinned_and_monotone() {
        let ext = Integer::from(65537u64).pow(4);
        let row = |target: f64| {
            envelope_row(1, 32, 16, &ext, target, |z| lg_cut_envelope(32, 15, z)).unwrap()
        };
        // targets chosen inside the range's soundness span (the
        // in-domain maximum is ~ -49.7 bits at z_max), so both rows
        // have genuine crossings
        let strict = row(-55.0);
        let loose = row(-52.0);
        assert!(strict.crossing_pinned && loose.crossing_pinned);
        assert!(strict.z_star <= loose.z_star, "monotone in the target");
        assert!(strict.lg_sound_hi < -55.0);
        // and a target below the whole span saturates at z_max, unpinned
        let saturated = row(-40.0);
        assert!(saturated.z_star == 16 && !saturated.crossing_pinned);
    }

    /// The consistency law of the two faces: under a common soundness
    /// map, the certified-sound ceiling must sit strictly below the
    /// certified-broken floor. Run at a reduced box (total length
    /// 2^12, KoalaBear sextic) so the test stays fast; the full-box
    /// pinch is the compilation's number.
    #[test]
    fn ceiling_sits_below_floor_at_reduced_box() {
        let base = Integer::from(crate::field::named::KOALABEAR);
        let ext = base.clone().pow(6);
        let total = 1u64 << 12;
        let floor = super::super::floor::elias_row(1, total, &base, &ext, -128.0).unwrap();
        let k = total / 2 - 1;
        let ceil = envelope_row(1, total, total - k - 1, &ext, -128.0, |z| {
            lg_cut_envelope(total, k, z)
        })
        .unwrap();
        assert!(
            ceil.z_star < floor.z_star,
            "faces overlap: ceiling {} vs floor {}",
            ceil.z_star,
            floor.z_star
        );
    }
}

#[cfg(test)]
mod ca_tests {
    use super::*;
    use rug::ops::Pow;

    /// The conversion's radius is conservative and its error formula
    /// encloses the exact integer arithmetic at small parameters.
    #[test]
    fn ca_conversion_is_conservative_and_encloses() {
        let (n, z, inv_eta) = (1u64 << 12, 1u64 << 10, 1u64 << 10);
        let zc = ca_radius(n, z, inv_eta);
        let lhs = (zc as f64) / (n as f64);
        let rhs = 1.0 - ((n - z) as f64 / n as f64 + 1.0 / inv_eta as f64).sqrt();
        assert!(lhs <= rhs + 1e-12, "{lhs} vs {rhs}");
        assert!(rhs - lhs < 2.0 / n as f64, "not tight on the lattice");
        let ext = Integer::from(65537u64).pow(4);
        let lg_list = Lg::from_u64(1000);
        let e = lg_mca_error(&lg_list, z, inv_eta, &ext).unwrap();
        let exact = Integer::from(1000u64 * 1000 * z + inv_eta);
        let ex = Lg::from_integer(&exact).div(&Lg::from_integer(&ext));
        assert!(e.lo <= ex.lo && ex.hi <= e.hi);
    }

    /// End to end at a reduced box: the generic-map ceiling exists and
    /// sits below the floor.
    #[test]
    fn ca_ceiling_below_floor_at_reduced_box() {
        let base = Integer::from(crate::field::named::KOALABEAR);
        let ext = base.clone().pow(6);
        let total = 1u64 << 12;
        let k = total / 2 - 1;
        let floor = super::super::floor::elias_row(1, total, &base, &ext, -128.0).unwrap();
        let ceil = ca_ceiling_row(1, total, total - k - 1, 1 << 20, &ext, -128.0, |z| {
            lg_cut_envelope(total, k, z)
        })
        .unwrap();
        assert!(ceil.z_star < floor.z_star);
        assert!(ceil.z_star > 0, "nontrivial CA radius");
    }
}
