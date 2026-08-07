//! The explore tier: fast float scans of the attack landscape.
//!
//! Everything here answers the same kind of question as the certified
//! rows next door — parameters in, threshold radius out — but in `f64`
//! with unchecked rounding, built for millisecond sweeps over the whole
//! parameter space rather than citable brackets on one cell. Explore
//! here; cite from `floor` and `ceiling`.
//!
//! Given a domain size `n` (power of two), message length `k`, and a
//! required list size in bits (`list_bits = log2(eps * |F|)` in the
//! grand-challenge normalization), [`best_attack`] optimizes over the
//! quantized-ladder rung constructions — `(G, r, 2^t - 1)` ladder
//! families of size `C(s_G/2^t - [r0 != 0], floor(r/2^t))` on subgroups
//! `s_G | n` — and returns the smallest radius `delta* = 1 - r/s_G` at
//! which a construction of the required list size exists. Restricting
//! to `t = 1` reproduces the antipodal (survey Table-5) method; the
//! full ladder strictly improves it. The certified twin is
//! `floor::rung_floor_row`. Also provided:
//! the structural-framework ceiling `delta_min - H2(rate)/list_bits`
//! (no ladder-family construction can beat it), and the Elias/volume
//! threshold — the float twin of
//! `floor::elias_row`.

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

/// A rung attack: the best construction found, achieving `delta_star`.
/// Float-tier report; the citable twin is
/// `floor::RungFloorRow`.
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
    /// `log2` of the exact list size `C(avail, b)`, evaluated in floats.
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

/// One admissible rung family at `(n, k)`: subgroup order `s_g`, rung
/// level `t` (so `q = 2^t - 1` symmetric functions pinned), agreement
/// size `r` in the exactness window `(r - q - 1) m < k <= r m`,
/// carrying its count parameters — the family has exactly
/// `C(avail, b)` members.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RungFamily {
    /// MultiplicativeSubgroup order (`s_g | n`).
    pub s_g: u64,
    /// Rung level.
    pub t: u32,
    /// Agreement-set size (agreement fraction `r / s_g`).
    pub r: u64,
    /// Cosets available after the boundary deduction.
    pub avail: u64,
    /// Cosets chosen: the family size is `C(avail, b)`.
    pub b: u64,
}

impl RungFamily {
    /// The attack radius `1 - r / s_g`.
    #[must_use]
    pub fn delta(&self) -> f64 {
        1.0 - self.r as f64 / self.s_g as f64
    }
}

/// Enumerates every admissible rung family at `(n, k)` with rung level
/// `t <= t_max`, in exact integer arithmetic — the shared candidate
/// space beneath both tiers. The float optimizer scores it with
/// log-factorial tables here; the certified construction row
/// (`floor::rung_floor_row`) scores it with
/// bracketed binomials.
pub fn rung_families(n: u64, k: u64, t_max: u32, mut visit: impl FnMut(RungFamily)) -> Result<()> {
    if !n.is_power_of_two() || n < 8 {
        return Err(Error::OutOfRange("n must be a power of two >= 8".into()));
    }
    if k == 0 || k >= n {
        return Err(Error::OutOfRange("need 1 <= k < n".into()));
    }
    let mut s_g = 4u64;
    while s_g <= n {
        let m = n / s_g;
        let mut t = 1u32;
        while (1u64 << t) <= s_g && t <= t_max {
            let q = (1u64 << t) - 1;
            let r_lo = k.div_ceil(m);
            let r_hi = ((k - 1) / m + q + 1).min(s_g - 1);
            for r in r_lo..=r_hi {
                // the exactness window: (r - q - 1) m < k <= r m
                if !(r.saturating_sub(q + 1) * m < k && k <= r * m) {
                    continue;
                }
                let block = 1u64 << t;
                let (b, r0) = (r / block, r % block);
                let avail = s_g / block - u64::from(r0 != 0);
                if b > avail {
                    continue;
                }
                visit(RungFamily {
                    s_g,
                    t,
                    r,
                    avail,
                    b,
                });
            }
            t += 1;
        }
        s_g *= 2;
    }
    Ok(())
}

fn optimize(p: &AttackParams, t_max: u32) -> Result<Option<RungAttack>> {
    validate(p)?;
    let lf = ln_fact_table(p.n as usize + 1);
    let dmin = capacity_radius(p.n, p.k);
    let mut best: Option<RungAttack> = None;
    rung_families(p.n, p.k, t_max, |f| {
        let log2_list = log2_binom(&lf, f.avail, f.b);
        if log2_list < p.list_bits {
            return;
        }
        let delta = f.delta();
        if best.map_or(true, |cur| delta < cur.delta_star) {
            best = Some(RungAttack {
                delta_star: delta,
                deficit: dmin - delta,
                t: f.t,
                s_g: f.s_g,
                r: f.r,
                log2_list,
            });
        }
    })?;
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
