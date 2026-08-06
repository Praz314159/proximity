//! The soundness map and the lattice-crossing report types: the ABF26
//! Lemma 6.12 conversion from a certified list count to a certified
//! soundness bracket, and the row/crossing structures every consumer
//! reports in. Both faces of the chain run through this file: the
//! floor feeds it attack counts, the ceiling will feed it envelope
//! rates.

use crate::error::Result;
use crate::math::enclosure::Lg;
use rug::Integer;

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
