//! Attack-threshold calculator: the best known list-decoding attack radius
//! delta* for smooth-domain Reed–Solomon parameters.
//!
//! Given a domain size `n` (power of two), message length `k`, and a required
//! list size in bits (`list_bits = log2(eps * |F|)` in the grand-challenge
//! normalization), this module optimizes over the quantized-ladder rung
//! constructions — `(G, r, 2^t - 1)` ladder families of size
//! `C(s_G/2^t - [r0 != 0], floor(r/2^t))` on subgroups `s_G | n` — and returns
//! the smallest radius `delta* = 1 - r/s_G` at which a certified list of the
//! required size exists. Restricting to `t = 1` reproduces the antipodal
//! (survey Table-5) method; the full ladder strictly improves it.
//!
//! Also provided: the structural-framework ceiling
//! `delta_min - H2(rate)/list_bits` (no ladder-family construction can beat
//! it), and the Elias/volume threshold for codes defined over a small base
//! field.

use crate::error::{Error, Result};

/// Input parameters for the attack optimization.
#[derive(Debug, Clone, Copy)]
pub struct AttackParams {
    /// Domain size `n = |L|` (must be a power of two).
    pub n: u64,
    /// Message length `k` (rate = k / n).
    pub k: u64,
    /// Required `log2` of the list size (e.g. `log2(|F|) - 128` for the
    /// grand-challenge target `|Lambda| > 2^-128 |F|`).
    pub list_bits: f64,
}

/// A rung attack: the certified construction achieving `delta_star`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RungAttack {
    /// The attack radius `1 - r / s_g` (soundness provably broken beyond it).
    pub delta_star: f64,
    /// Deficit from capacity `delta_min - delta_star`.
    pub deficit: f64,
    /// Rung level (`q = 2^t - 1` symmetric functions pinned).
    pub t: u32,
    /// MultiplicativeSubgroup order used.
    pub s_g: u64,
    /// Subset size (agreement `r/s_g` fraction of the subgroup).
    pub r: u64,
    /// `log2` of the certified (exact) list size `C(avail, b)`.
    pub log2_list: f64,
}

fn validate(p: &AttackParams) -> Result<()> {
    if !p.n.is_power_of_two() || p.n < 8 {
        return Err(Error::OutOfRange("n must be a power of two >= 8".into()));
    }
    if p.k == 0 || p.k >= p.n {
        return Err(Error::OutOfRange("need 1 <= k < n".into()));
    }
    if !p.list_bits.is_finite() || p.list_bits <= 0.0 {
        return Err(Error::OutOfRange("list_bits must be positive".into()));
    }
    Ok(())
}

/// Relative minimum distance `1 - (k - 1)/n` (capacity radius).
#[must_use]
pub fn capacity_radius(n: u64, k: u64) -> f64 {
    1.0 - (k as f64 - 1.0) / n as f64
}

/// Binary entropy `H2(x)` in bits.
#[must_use]
pub fn h2(x: f64) -> f64 {
    if x <= 0.0 || x >= 1.0 {
        return 0.0;
    }
    -x * x.log2() - (1.0 - x) * (1.0 - x).log2()
}

/// `ln(i!)` table for `i in 0..=n`, for O(1) `log2` binomials.
fn ln_fact_table(n: usize) -> Vec<f64> {
    let mut t = vec![0.0f64; n + 1];
    for i in 1..=n {
        t[i] = t[i - 1] + (i as f64).ln();
    }
    t
}

fn log2_binom(lf: &[f64], a: u64, b: u64) -> f64 {
    if b > a {
        return f64::NEG_INFINITY;
    }
    (lf[a as usize] - lf[b as usize] - lf[(a - b) as usize]) / std::f64::consts::LN_2
}

fn optimize(p: &AttackParams, t_max: u32) -> Result<Option<RungAttack>> {
    validate(p)?;
    let lf = ln_fact_table(p.n as usize + 1);
    let dmin = capacity_radius(p.n, p.k);
    let mut best: Option<RungAttack> = None;
    let mut s_g = 4u64;
    while s_g <= p.n {
        let m = p.n / s_g;
        let mut t = 1u32;
        while (1u64 << t) <= s_g && t <= t_max {
            let q = (1u64 << t) - 1;
            let r_lo = p.k.div_ceil(m);
            let r_hi = ((p.k - 1) / m + q + 1).min(s_g - 1);
            for r in r_lo..=r_hi {
                // the exactness window: (r - q - 1) m < k <= r m
                if !(r.saturating_sub(q + 1) * m < p.k && p.k <= r * m) {
                    continue;
                }
                let block = 1u64 << t;
                let (b, r0) = (r / block, r % block);
                let avail = s_g / block - u64::from(r0 != 0);
                if b > avail {
                    continue;
                }
                let log2_list = log2_binom(&lf, avail, b);
                if log2_list < p.list_bits {
                    continue;
                }
                let delta = 1.0 - r as f64 / s_g as f64;
                let improves = match best {
                    None => true,
                    Some(cur) => delta < cur.delta_star,
                };
                if improves {
                    best = Some(RungAttack {
                        delta_star: delta,
                        deficit: dmin - delta,
                        t,
                        s_g,
                        r,
                        log2_list,
                    });
                }
            }
            t += 1;
        }
        s_g *= 2;
    }
    Ok(best)
}

/// Best attack over the full quantized ladder (all rung levels `t`).
pub fn best_attack(p: &AttackParams) -> Result<Option<RungAttack>> {
    optimize(p, u32::MAX)
}

/// The antipodal-only baseline (`t = 1`; the survey's Table-5 method).
pub fn antipodal_attack(p: &AttackParams) -> Result<Option<RungAttack>> {
    optimize(p, 1)
}

/// The structural-framework ceiling `delta_min - H2(rate)/list_bits`: by the
/// ladder upper bound, no characteristic-zero ladder construction can
/// certify the required list size below this radius.
pub fn hyperbola_ceiling(p: &AttackParams) -> Result<f64> {
    validate(p)?;
    Ok(capacity_radius(p.n, p.k) - h2(p.k as f64 / p.n as f64) / p.list_bits)
}

/// Elias/volume threshold for codes defined over a base field of
/// `base_bits = log2 |B|`: the smallest `delta` (on the grid `i/n`) at which a
/// random-word argument already guarantees `list_bits` bits of list, i.e.
/// `n * ((rate - 1) * base_bits + delta * base_bits + H2(delta)) >= list_bits`
/// (using `log2(|B| - 1) ~ base_bits`). Returns `None` if no such `delta < 1`.
pub fn elias_delta_star(p: &AttackParams, base_bits: f64) -> Result<Option<f64>> {
    validate(p)?;
    let (n, rate) = (p.n as f64, p.k as f64 / p.n as f64);
    for i in 1..p.n {
        let d = i as f64 / n;
        let bits = n * ((rate - 1.0) * base_bits + d * base_bits + h2(d));
        if bits >= p.list_bits {
            return Ok(Some(d));
        }
    }
    Ok(None)
}
