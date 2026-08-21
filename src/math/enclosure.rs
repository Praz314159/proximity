//! Certified bracket arithmetic — the crate's third number system.
//!
//! [`field`](crate::field) computes exactly mod `p`; [`ring`](crate::ring)
//! computes exactly in characteristic zero; this module computes with
//! quantities too large for either (challenge-scale counts have
//! `~10^14`-bit exponents) by holding an *interval enclosure of the
//! base-2 logarithm* in MPFR floats with directed rounding: lower
//! endpoints round down, upper endpoints round up, every operation, no
//! exceptions. A returned [`Lg`] is a machine-checked bracket; certified
//! claims use `lo` (for ">=") or `hi` (for "<=") only.
//!
//! Provided on top of the type: [`lg_binom`] (the beyond-u64 counterpart
//! of [`field::binom`](crate::field::binom), via directed log-gamma) and
//! [`lg_sub_lower`] (a certified lower bound on a difference, `None`
//! when the bracket cannot certify positivity). Consumed by
//! the certified [`soundness`](crate::soundness) rows; exact-bignum
//! cross-checks at small parameters pin the enclosures in tests.

use rug::float::{Constant, Round};
use rug::ops::{AddAssignRound, DivAssignRound, MulAssignRound, SubAssignRound};
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

/// ln(2) rounded in both directions (shared divisor for lngamma ->
/// log2), computed once — profiling showed the constant being rebuilt
/// twice per `lgamma2` call.
fn ln2(round: Round) -> &'static Float {
    static LN2: std::sync::OnceLock<(Float, Float)> = std::sync::OnceLock::new();
    let (down, up) = LN2.get_or_init(|| {
        let x = Float::with_val(PREC, Constant::Log2);
        // Constant::Log2 is correctly rounded to nearest; nudge one
        // ulp outward.
        let mut d = x.clone();
        d.next_down();
        let mut u = x;
        u.next_up();
        (d, u)
    });
    match round {
        Round::Down => down,
        Round::Up => up,
        _ => unreachable!(),
    }
}

impl Lg {
    /// Exact zero (the quantity 1).
    #[must_use]
    pub fn zero() -> Self {
        Lg {
            lo: f(0.0),
            hi: f(0.0),
        }
    }

    /// log2 of an exact nonzero integer.
    #[must_use]
    pub fn from_integer(x: &Integer) -> Self {
        assert!(*x > 0);
        let mut lo = Float::with_val_round(PREC, x, Round::Down).0;
        let mut hi = Float::with_val_round(PREC, x, Round::Up).0;
        lo.log2_round(Round::Down);
        hi.log2_round(Round::Up);
        Lg { lo, hi }
    }

    /// log2 of an exact nonzero u64.
    #[must_use]
    pub fn from_u64(x: u64) -> Self {
        Self::from_integer(&Integer::from(x))
    }

    /// Product of quantities = sum of logs.
    #[must_use]
    pub fn mul(&self, o: &Lg) -> Lg {
        let mut lo = self.lo.clone();
        lo.add_assign_round(&o.lo, Round::Down);
        let mut hi = self.hi.clone();
        hi.add_assign_round(&o.hi, Round::Up);
        Lg { lo, hi }
    }

    /// Quotient of quantities = difference of logs.
    #[must_use]
    pub fn div(&self, o: &Lg) -> Lg {
        let mut lo = self.lo.clone();
        lo.sub_assign_round(&o.hi, Round::Down);
        let mut hi = self.hi.clone();
        hi.sub_assign_round(&o.lo, Round::Up);
        Lg { lo, hi }
    }

    /// Integer power of the quantity = scale the log by `e >= 0`.
    /// Scaling by a nonnegative factor with outward rounding preserves
    /// `lo <= hi` for either sign of the log — no case split.
    #[must_use]
    pub fn pow(&self, e: u64) -> Lg {
        let ef = Float::with_val(PREC, e);
        let mut lo = self.lo.clone();
        lo.mul_assign_round(&ef, Round::Down);
        let mut hi = self.hi.clone();
        hi.mul_assign_round(&ef, Round::Up);
        Lg { lo, hi }
    }

    /// Sum of quantities: log2(2^a + 2^b), each endpoint with its own
    /// rounding direction.
    #[must_use]
    pub fn add(&self, o: &Lg) -> Lg {
        Lg {
            lo: lg_add_dir(&self.lo, &o.lo, Round::Down),
            hi: lg_add_dir(&self.hi, &o.hi, Round::Up),
        }
    }

    /// Rebuild a bracket from stored `f64` endpoints (the profile
    /// store's round trip): `f64 -> Float` at the module's working
    /// precision is exact, so this is a widening-free restore. The
    /// single home for the store/restore precision — callers must
    /// not copy `PREC`.
    #[must_use]
    pub fn from_f64_bracket(lo: f64, hi: f64) -> Lg {
        Lg {
            lo: Float::with_val(PREC, lo),
            hi: Float::with_val(PREC, hi),
        }
    }

    /// A bracket hull from two sources: the `lo` of one and the `hi`
    /// of the other. The named form of the `[first term, closed
    /// form]` pattern — hand-rolling it reads as a typo.
    #[must_use]
    pub fn hull(lo_of: &Lg, hi_of: &Lg) -> Lg {
        Lg {
            lo: lo_of.lo.clone(),
            hi: hi_of.hi.clone(),
        }
    }

    /// Enclosure of the smaller of the two enclosed quantities:
    /// elementwise min of endpoints. Valid because
    /// `min(x, y) >= min(lo_x, lo_y)` and `min(x, y) <= min(hi_x,
    /// hi_y)` for any `x` in `[lo_x, hi_x]`, `y` in `[lo_y, hi_y]`.
    #[must_use]
    pub fn min(&self, o: &Lg) -> Lg {
        Lg {
            lo: if o.lo < self.lo {
                o.lo.clone()
            } else {
                self.lo.clone()
            },
            hi: if o.hi < self.hi {
                o.hi.clone()
            } else {
                self.hi.clone()
            },
        }
    }

    /// Enclosure of the larger of the two enclosed quantities:
    /// elementwise max of endpoints (same argument as [`Lg::min`]).
    #[must_use]
    pub fn max(&self, o: &Lg) -> Lg {
        Lg {
            lo: if o.lo > self.lo {
                o.lo.clone()
            } else {
                self.lo.clone()
            },
            hi: if o.hi > self.hi {
                o.hi.clone()
            } else {
                self.hi.clone()
            },
        }
    }

    /// Widen the upper endpoint by `bits` (multiplicative slack 2^bits).
    #[must_use]
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
    d.exp2_round(round); // directed: the no-suffix exp2 rounds to nearest
    let mut t = d;
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
    lo.div_assign_round(ln2(Round::Up), Round::Down);
    hi.div_assign_round(ln2(Round::Down), Round::Up);
    Lg { lo, hi }
}

/// log2 of Gamma(x + 1) = log2 of x! as an interval — the table
/// entry for table-driven binomials (the tower's prefix scans build
/// one factorial table per level instead of three `lgamma2` calls
/// per binomial).
#[must_use]
pub fn lg_factorial(x: u64) -> Lg {
    lgamma2(x + 1)
}

/// A parallel-built factorial table for repeated binomials at one
/// scale: `binom` is bit-for-bit [`lg_binom`] (both reduce to
/// `lgamma2` at the same precision), amortized to three lookups —
/// the tower's prefix scans make ~10^6 binomial queries per level.
pub struct LgFactorials {
    table: Vec<Lg>,
}

impl LgFactorials {
    /// The table of `log2(x!)` for `x <= n_max`.
    #[must_use]
    pub fn new(n_max: u64) -> Self {
        use rayon::prelude::*;
        LgFactorials {
            table: (0..=n_max).into_par_iter().map(lg_factorial).collect(),
        }
    }

    /// `log2 C(n, k)` from the table; panics beyond `n_max` (same
    /// contract as [`lg_binom`]'s `k <= n` assert).
    #[must_use]
    pub fn binom(&self, n: u64, k: u64) -> Lg {
        assert!(k <= n, "binom needs k <= n (got n = {n}, k = {k})");
        self.table[n as usize]
            .div(&self.table[k as usize])
            .div(&self.table[(n - k) as usize])
    }
}

/// log2 of the binomial coefficient C(n, k).
#[must_use]
pub fn lg_binom(n: u64, k: u64) -> Lg {
    assert!(k <= n, "lg_binom needs k <= n (got n = {n}, k = {k})");
    lgamma2(n + 1).div(&lgamma2(k + 1)).div(&lgamma2(n - k + 1))
}

/// [`lg_binom`] behind a global memo — for call sites that query the
/// same fixed arguments many times (per-threshold loops re-deriving
/// per-level constants). Bit-for-bit the same bracket as
/// [`lg_binom`]; the memo stores the full-precision endpoints, so
/// this is a pure amortization, not a widening. Reads dominate after
/// warm-up; the lock is per-process and the key space stays small
/// (level constants, not per-threshold values).
#[must_use]
pub fn lg_binom_memo(n: u64, k: u64) -> Lg {
    use std::collections::HashMap;
    use std::sync::{OnceLock, RwLock};
    static CACHE: OnceLock<RwLock<HashMap<(u64, u64), Lg>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(v) = cache
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&(n, k))
    {
        return v.clone();
    }
    let v = lg_binom(n, k);
    cache
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert((n, k), v.clone());
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
