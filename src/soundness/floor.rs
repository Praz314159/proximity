//! The floor face: what attacks certifiably achieve. Table-4-style
//! Elias rows (the certified wall floor delta* = 0.46783 at the box)
//! and the first-moment (Diamond--Gruen random words) crossing. The
//! ceiling face — envelope rows consuming the master theorem's list
//! envelope through the same chain — is the module's forthcoming
//! second half; the prize's pinch is floor z* == ceiling z*.

use crate::error::{Error, Result};
use crate::math::enclosure::Lg;
use rug::float::Round;
use rug::Integer;

use super::chain::{lg_soundness, Crossing, CrossingRow};
use super::volumes::{lg_elias_list, lg_expected_list};

/// Binary-search the crossing of log2 E\[list\] against `target` over
/// z in [z_min, z_max]. E\[list\] is strictly increasing in z, so the
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
