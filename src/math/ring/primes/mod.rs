//! The prime axis of `Z[zeta_s]`: how rational primes behave under the
//! ring — which `p = 1 (mod s)` admit small kernel vectors, and hence
//! which create bucket coincidences beyond the structural floor.
//!
//! The criterion chain is arithmetic end to end: a bounded-coefficient
//! vector `v` lies in the kernel of `Z[zeta_s] -> F_p` iff `p` divides
//! its norm; norms of bounded-height vectors are confined by the height
//! law `N(v) <= (sum v_i^2)^{s/4}`; enumerating and factoring them
//! inverts to the complete per-prime accident inventory. The modules:
//!
//! - [`kernel`]: the kernel-vector engines (direct and MitM counting) —
//!   moved here from `census` because they count *ring elements*, not
//!   subsets; `census::kernel` remains as a re-export.
//! - [`norms`]: exact norm tables and bad-set enumeration — the
//!   enumerate -> norm -> factor -> invert pipeline producing complete
//!   per-prime accident inventories; [`norms::events`] carries the
//!   inventory beyond counting (one row per (prime, orbit) incidence,
//!   witness vector in hand) and [`norms::ingest`] streams
//!   GPU-computed norm tables. Moved from `smooth::norms` (its
//!   interface never involved a subgroup), which remains as a
//!   re-export.
//!
//! The bucket-level consequences of these facts (class merges, ladder
//! values, certificates with bucket semantics) live above, in
//! `smooth::certify`, which delegates its criterion here.

pub mod kernel;
pub mod norms;

use crate::domain::MultiplicativeSubgroup;
use crate::error::Result;

/// Cleanliness of one level at a named prime, with the method and its
/// coverage stated — the audit's tier vocabulary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LevelCleanliness {
    /// Full kernel census empty at coefficients `[-2, 2]`: no bucket
    /// coincidence beyond the structural floor, provably (the
    /// `s <= 32` full-census route; `p`-independent cost).
    Certified,
    /// No kernel vector with coefficients `[-2, 2]` up to `wmax`
    /// nonzero entries; weights above `wmax` unchecked (the large-`s`
    /// route — never a silent claim).
    CertifiedToWeight {
        /// The inclusive weight bound the enumeration covered.
        wmax: usize,
    },
    /// No `{-1, 0, 1}` kernel vector (the zero class is exact), but
    /// `[-2, 2]` vectors exist: coincidences confined away from the
    /// zero class. Counts by weight (dilation orbits of size `s`).
    /// On the weight-capped route (`s > 32`) both claims hold to the
    /// certificate's `wmax_large`, not absolutely.
    CertifiedAtUnit {
        /// Nonzero `[-2, 2]` census counts, indexed by weight.
        census2_by_weight: Vec<u64>,
    },
    /// `{-1, 0, 1}` kernel vectors exist: the prime creates
    /// zero-class coincidences. Counts by weight (weight-capped on the
    /// `s > 32` route).
    Dirty {
        /// Nonzero `{-1, 0, 1}` census counts, indexed by weight.
        census_by_weight: Vec<u64>,
    },
}

/// One level of a tower certificate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LevelEntry {
    /// The level (domain size) this entry certifies.
    pub s: usize,
    /// Its verdict.
    pub verdict: LevelCleanliness,
}

/// A tower cleanliness certificate for a named prime: one entry per
/// level `s_top, s_top/2, ..., 8`. The named-prime instrument of the
/// program's per-prime hypothesis — the interface data's arithmetic
/// half is discharged by this object at the deployment prime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CleanCertificate {
    /// The certified prime.
    pub p: u64,
    /// The tower top.
    pub s_top: usize,
    /// Per-level verdicts, top first.
    pub levels: Vec<LevelEntry>,
    /// The weight budget used at levels above the full-census range.
    pub wmax_large: usize,
}

impl CleanCertificate {
    /// Every level fully certified (no `CertifiedToWeight`, no
    /// `Dirty`).
    #[must_use]
    pub fn fully_certified(&self) -> bool {
        self.levels
            .iter()
            .all(|l| l.verdict == LevelCleanliness::Certified)
    }
    /// No level dirty (full certification or bounded-weight
    /// certification everywhere).
    #[must_use]
    pub fn clean_as_checked(&self) -> bool {
        !self
            .levels
            .iter()
            .any(|l| matches!(l.verdict, LevelCleanliness::Dirty { .. }))
    }
}

/// Cleanliness of a single level: the full census for `s <= 32`, the
/// weight-bounded census above (`wmax_large` caps the enumeration; the
/// verdict says so). Above the MitM range the census runs through
/// [`kernel::sort_join`], whose memory stays at the held join side —
/// `wmax_large` up to ~10 is live at `s = 64` where the direct
/// recursion stopped near 6.
pub fn clean_at_level(p: u64, s: usize, wmax_large: usize) -> Result<LevelCleanliness> {
    let sg = MultiplicativeSubgroup::new(p, s)?;
    if s <= 32 {
        let counts2 = kernel::mitm(&sg, 2)?;
        if counts2.iter().all(|&c| c == 0) {
            return Ok(LevelCleanliness::Certified);
        }
        let counts1 = kernel::mitm(&sg, 1)?;
        return Ok(if counts1.iter().all(|&c| c == 0) {
            LevelCleanliness::CertifiedAtUnit {
                census2_by_weight: counts2,
            }
        } else {
            LevelCleanliness::Dirty {
                census_by_weight: counts1,
            }
        });
    }
    let counts2 = kernel::sort_join(&sg, 2, wmax_large)?;
    if counts2.iter().all(|&c| c == 0) {
        return Ok(LevelCleanliness::CertifiedToWeight { wmax: wmax_large });
    }
    let counts1 = kernel::sort_join(&sg, 1, wmax_large)?;
    Ok(if counts1.iter().all(|&c| c == 0) {
        LevelCleanliness::CertifiedAtUnit {
            census2_by_weight: counts2,
        }
    } else {
        LevelCleanliness::Dirty {
            census_by_weight: counts1,
        }
    })
}

/// The tower certificate: [`clean_at_level`] at every level from
/// `s_top` down to `8`, one prime, one call — the audit's
/// "all levels at one `p`" requirement as an object.
pub fn certify_clean(p: u64, s_top: usize, wmax_large: usize) -> Result<CleanCertificate> {
    let mut levels = Vec::new();
    let mut s = s_top;
    while s >= 8 {
        levels.push(LevelEntry {
            s,
            verdict: clean_at_level(p, s, wmax_large)?,
        });
        s /= 2;
    }
    Ok(CleanCertificate {
        p,
        s_top,
        levels,
        wmax_large,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::named;

    #[test]
    fn tower_certificates_at_named_and_bad_primes() {
        // KoalaBear at s = 32: zero class exact, but one weight-14
        // dilation orbit of [-2, 2] kernel vectors — CertifiedAtUnit,
        // not Certified (found by this test; refines the campaign-era
        // "KB clean at 32", which was the zero-class sense); the dirt
        // is confined to the top level
        let kb = certify_clean(named::KOALABEAR, 32, 4).unwrap();
        assert!(!kb.fully_certified() && kb.clean_as_checked());
        assert_eq!(kb.levels.len(), 3); // 32, 16, 8
        match &kb.levels[0].verdict {
            LevelCleanliness::CertifiedAtUnit { census2_by_weight } => {
                assert_eq!(census2_by_weight[14], 32); // one orbit
                assert_eq!(census2_by_weight.iter().sum::<u64>(), 32);
            }
            v => panic!("KB at 32: {v:?}"),
        }
        assert!(kb.levels[1..]
            .iter()
            .all(|l| l.verdict == LevelCleanliness::Certified));
        // a strictly clean prime just above 2^31: fully certified
        let clean = certify_clean(2_147_489_441, 32, 4).unwrap();
        assert!(clean.fully_certified(), "{clean:?}");
        // Goldilocks sits above the Parseval ceiling at every level
        // <= 32 ((cmax^2 * s/2)^{s/4} <= 2^48 < p), so the height law
        // says fully certified — and the census, running live at a
        // 64-bit prime, must confirm it
        let g = certify_clean(named::GOLDILOCKS, 32, 4).unwrap();
        assert!(g.fully_certified(), "{g:?}");
        // 77569: the bad-set poster prime — zero-class dirty at 32
        let bad = certify_clean(77_569, 32, 4).unwrap();
        assert!(!bad.clean_as_checked());
        assert!(matches!(
            bad.levels[0].verdict,
            LevelCleanliness::Dirty { .. }
        ));
        // the large-s route reports its coverage honestly; the join
        // engine reaches past the direct recursion's range
        let kb64 = clean_at_level(named::KOALABEAR, 64, 6).unwrap();
        assert_eq!(kb64, LevelCleanliness::CertifiedToWeight { wmax: 6 });
    }
}
