//! The soundness map and the lattice-crossing report types: the ABF26
//! Lemma 6.12 conversion from a certified list count to a certified
//! soundness bracket, the challenge threshold, and the row/crossing
//! structures every consumer reports in. The soundness map is the
//! floor's own chain; the ceiling never touches it — both faces meet
//! only in the list currency of [`lg_list_threshold`].

use crate::error::Result;
use crate::math::enclosure::Lg;
use rug::float::Round;
use rug::ops::AddAssignRound;
use rug::{Float, Integer};

use super::volumes::lg_q;

/// Certified enclosure of log2 of the ABF26 Lemma 6.12 soundness floor
/// x / (|F| + 2x), given an enclosure of log2 x and the extension field F.
/// Monotone increasing in x, so endpoints map to endpoints.
pub fn lg_soundness(lg_list: &Lg, ext_q: &Integer) -> Result<Lg> {
    let lg_q = lg_q(ext_q)?;
    let two_x = lg_list.mul(&Lg::from_u64(2));
    Ok(lg_list.div(&lg_q.add(&two_x)))
}

/// Report row for one radius in a crossing sweep.
#[derive(Debug, Clone)]
pub struct CrossingRow {
    /// Radius (absolute number of errors).
    pub z: u64,
    /// Fractional radius z / n.
    pub delta: f64,
    /// Certified lower endpoint of log2 E\[list\].
    pub lg_e_lo: f64,
    /// Certified upper endpoint of log2 E\[list\].
    pub lg_e_hi: f64,
}

/// A certified crossing of a monotone-increasing quantity against a
/// target: the smallest radius certified at-or-above (lower endpoint
/// clears the target) and the largest radius certified below (upper
/// endpoint under it). Between the two the brackets straddle the
/// target — a gap of at most a few z-values at challenge scale, where
/// one z step moves the log by ~log2 Q.
#[derive(Debug)]
pub struct Crossing {
    /// Smallest z whose lower endpoint clears the target.
    pub certified_at_or_above: Option<CrossingRow>,
    /// Largest z whose upper endpoint is under the target.
    pub certified_below: Option<CrossingRow>,
}

/// The smallest `z` in `[lo, hi]` satisfying a monotone certified
/// predicate (false below the threshold, true at and above it), or
/// `None` if none does. The one binary search of the module: every
/// crossing on the z-lattice — floor rows, first-moment rows, and the
/// forthcoming ceiling rows — runs through it, in this orientation or
/// reversed via `!pred`.
pub fn certified_first(
    lo: u64,
    hi: u64,
    mut pred: impl FnMut(u64) -> Result<bool>,
) -> Result<Option<u64>> {
    let (mut a, mut b) = (lo, hi + 1); // half-open [a, b)
    while a < b {
        let mid = a + (b - a) / 2;
        if pred(mid)? {
            b = mid;
        } else {
            a = mid + 1;
        }
    }
    Ok((a <= hi).then_some(a))
}

/// The block length of an interleaved row: `s` must divide the total.
pub(crate) fn interleaved_block_len(s: u64, total_len: u64) -> Result<u64> {
    if s == 0 || total_len % s != 0 {
        return Err(crate::error::Error::OutOfRange(
            "interleaving width must divide the total length".into(),
        ));
    }
    Ok(total_len / s)
}

/// The largest radius certified under a monotone "first over"
/// crossing: `first_over` from [`certified_first`], capped at
/// `z_max`; an error when even the smallest radius is over.
pub(crate) fn largest_under(first_over: Option<u64>, z_max: u64, what: &str) -> Result<u64> {
    match first_over {
        Some(1) => Err(crate::error::Error::Unsupported(format!(
            "no radius certifies {what}"
        ))),
        Some(z) => Ok(z - 1),
        None => Ok(z_max),
    }
}

/// The list-decoding challenge's threshold `eps* |F|` as a certified
/// log2 bracket, for `eps* = 2^eps_bits`. This is the quantity both
/// faces are tested against: the challenge asks for the largest radius
/// with `|Lambda(C, delta)| <= eps* |F|`, so no soundness conversion
/// enters — the comparison is in list size throughout. (The soundness
/// map above is the floor's own chain, and the reason the threshold
/// takes this value; the MCA conversion in `ceiling` serves the
/// sibling challenge.)
pub fn lg_list_threshold(ext_q: &Integer, eps_bits: f64) -> Result<Lg> {
    let q = super::volumes::lg_q(ext_q)?;
    let eps = |r: Round, x: &Float| {
        let mut v = x.clone();
        v.add_assign_round(&Float::with_val(x.prec(), eps_bits), r);
        v
    };
    Ok(Lg {
        lo: eps(Round::Down, &q.lo),
        hi: eps(Round::Up, &q.hi),
    })
}

/// One row in the challenge's own currency: a radius on the z-lattice
/// together with the certified list bracket there.
#[derive(Debug, Clone)]
pub struct ListRow {
    /// Interleaving width (the challenge's `m`).
    pub s: u64,
    /// Base-code block length.
    pub n: u64,
    /// The crossing radius.
    pub z_star: u64,
    /// `z_star / n`.
    pub delta_star: f64,
    /// Certified lower endpoint of log2 of the list bound at `z_star`.
    pub lg_list_lo: f64,
    /// Certified upper endpoint of log2 of the list bound at `z_star`.
    pub lg_list_hi: f64,
    /// Whether the neighbouring radius is certified on the other side:
    /// the crossing is pinned to a single lattice step.
    pub crossing_pinned: bool,
}
