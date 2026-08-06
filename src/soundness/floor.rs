//! The floor face: what attacks certifiably achieve. Table-4-style
//! Elias rows (the certified wall floor delta* = 0.46783 at the box)
//! and the first-moment (Diamond--Gruen random words) crossing. The
//! ceiling face — envelope rows consuming the master theorem's list
//! envelope through the same chain — is the module's forthcoming
//! second half; the prize's pinch is floor z* == ceiling z*. The
//! uncertified float twin of this file is `attack::ladder`: explore
//! with the ladder, cite from here.

use crate::error::{Error, Result};
use crate::math::enclosure::Lg;
use rug::float::Round;
use rug::Integer;

use super::chain::{
    certified_first, lg_list_threshold, lg_soundness, Crossing, CrossingRow, ListRow,
};
use super::volumes::{lg_elias_list, lg_expected_list};

/// The certified first-moment crossing of log2 E\[list\] against
/// `target` over `z` in `[z_min, z_max]`: E\[list\] is strictly
/// increasing in z, so the smallest certified-at-or-above and largest
/// certified-below radii bracket the true crossing.
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
    // smallest z certified at or above: lower endpoint clears target
    let above = certified_first(z_min, z_max, |z| Ok(row(z)?.lg_e_lo >= target))?
        .map(row)
        .transpose()?;
    // largest z certified below: upper endpoint under target — the
    // complement predicate is monotone in the same orientation
    let below = match certified_first(z_min, z_max, |z| Ok(row(z)?.lg_e_hi >= target))? {
        Some(z) if z > z_min => Some(row(z - 1)?),
        Some(_) => None,
        None => Some(row(z_max)?),
    };
    Ok(Crossing {
        certified_at_or_above: above,
        certified_below: below,
    })
}

/// One certified Table-4-style row: interleaved RS at rate 1/2, block
/// length `n = total_len / s`, over any base/extension alphabet pair,
/// per ABF26 Section 6.4's recipe with the exact Lemma 3.7 Elias count
/// in place of Corollary 3.8.
#[derive(Debug, Clone)]
pub struct EliasRow {
    /// Interleaving width s (the table index).
    pub s: u64,
    /// Base-code block length n = total_len / s.
    pub n: u64,
    /// Smallest z certified to push soundness >= 2^target (lower endpoint).
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

/// The smallest radius z whose Lemma 3.7 + Lemma 6.12 chain certifies
/// soundness error >= 2^target_bits. Soundness is monotone increasing
/// in z, so the certified predicate is monotone and the lattice search
/// is sound.
pub fn elias_row(
    s: u64,
    total_len: u64,
    base_q: &Integer,
    ext_q: &Integer,
    target_bits: f64,
) -> Result<EliasRow> {
    if s == 0 || total_len % s != 0 {
        return Err(Error::OutOfRange(
            "interleaving width must divide the total length".into(),
        ));
    }
    let n = total_len / s;
    let k = n / 2;
    let sound_at = |z: u64| -> Result<Lg> { lg_soundness(&lg_elias_list(n, k, z, base_q)?, ext_q) };
    let z_star = certified_first(1, n - 1, |z| {
        Ok(sound_at(z)?.lo.to_f64_round(Round::Down) >= target_bits)
    })?
    .ok_or_else(|| Error::Unsupported("no radius certifies the target soundness".into()))?;
    let sd = sound_at(z_star)?;
    let pinned = if z_star > 1 {
        sound_at(z_star - 1)?.hi.to_f64_round(Round::Up) < target_bits
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

/// The floor in the challenge's own currency: the smallest radius at
/// which the exact Elias count (ABF26 Lemma 3.7) is certified to
/// EXCEED `eps* |F|` — above it no list bound can meet the challenge,
/// so it bounds `delta*` from above. Equivalent to the soundness route
/// of [`elias_row`] up to the negligible `2x` term of the Lemma 6.12
/// map; the two are cross-pinned in the tests.
pub fn elias_list_row(
    s: u64,
    total_len: u64,
    base_q: &Integer,
    ext_q: &Integer,
    eps_bits: f64,
) -> Result<ListRow> {
    if s == 0 || total_len % s != 0 {
        return Err(Error::OutOfRange(
            "interleaving width must divide the total length".into(),
        ));
    }
    let (n, thr) = (total_len / s, lg_list_threshold(ext_q, eps_bits)?);
    let k = n / 2;
    let list_at = |z: u64| lg_elias_list(n, k, z, base_q);
    // certified above: the whole list bracket clears the whole threshold
    let z_star = certified_first(1, n - 1, |z| Ok(list_at(z)?.lo > thr.hi))?.ok_or_else(|| {
        Error::Unsupported("no radius certifies a list above the threshold".into())
    })?;
    let at = list_at(z_star)?;
    let pinned = z_star == 1 || list_at(z_star - 1)?.hi <= thr.lo;
    Ok(ListRow {
        s,
        n,
        z_star,
        delta_star: z_star as f64 / n as f64,
        lg_list_lo: at.lo.to_f64_round(Round::Down),
        lg_list_hi: at.hi.to_f64_round(Round::Up),
        crossing_pinned: pinned,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::enclosure::Lg;
    use rug::ops::Pow;

    #[test]
    fn koalabear_sextic_constant() {
        let ext = Integer::from(crate::field::named::KOALABEAR).pow(6);
        let l = Lg::from_integer(&ext)
            .lo
            .to_f64_round(rug::float::Round::Down);
        assert!((l - 185.93196).abs() < 0.001, "log2|F| = {l}");
    }

    /// The crossing brackets the exact rational threshold — the test
    /// the original search direction lacked (it returned the range
    /// endpoints; caught in the 2026-08-06 review).
    #[test]
    fn first_moment_crossing_brackets_the_exact_threshold() {
        use super::super::volumes::ball_exact;
        let (n, k, q) = (16u64, 8u64, Integer::from(97u64));
        // exact smallest z with E[list] >= 1: q^k V(z) >= q^n
        let den = q.clone().pow(n as u32);
        let num = |z: u64| q.clone().pow(k as u32) * ball_exact(n, z, &q);
        let exact = (1..n).find(|&z| num(z) >= den).unwrap();
        let c = first_moment_crossing(n, k, 0.0, 1, n - 1, &q).unwrap();
        let above = c.certified_at_or_above.unwrap();
        let below = c.certified_below.unwrap();
        assert_eq!(above.z, exact, "smallest certified-above");
        assert_eq!(below.z, exact - 1, "largest certified-below");
        assert!(above.lg_e_lo >= 0.0 && below.lg_e_hi < 0.0);
    }

    /// The two routes to the floor agree on every Table-4 row: the
    /// soundness crossing of [`elias_row`] and the list crossing of
    /// [`elias_list_row`] pick the same lattice point. The equivalence
    /// is algebraic (x/(|F|+2x) >= eps* iff x >= eps*|F|/(1-2eps*)),
    /// which is exactly the kind of "should" worth testing.
    #[test]
    fn list_and_soundness_floors_agree() {
        let base = Integer::from(crate::field::named::KOALABEAR);
        let ext = base.clone().pow(6);
        for s in [1u64, 1 << 4, 1 << 8, 1 << 12] {
            let sound = elias_row(s, 1 << 21, &base, &ext, -128.0).unwrap();
            let list = elias_list_row(s, 1 << 21, &base, &ext, -128.0).unwrap();
            assert_eq!(list.z_star, sound.z_star, "row s = {s}");
            assert!(list.crossing_pinned);
            // and the list bracket at the crossing straddles eps*|F|
            let thr = super::super::chain::lg_list_threshold(&ext, -128.0).unwrap();
            assert!(list.lg_list_lo > thr.hi.to_f64_round(Round::Up));
        }
    }

    #[test]
    fn table4_certified_profile() {
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
                assert!((r.delta_star * 1e5).round() / 1e5 == 0.46783);
            }
        }
    }
}
