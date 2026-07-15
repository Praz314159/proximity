//! The core domain object: a multiplicative subgroup `mu_s <= F_p^*`.
//!
//! Everything in this toolkit is an analysis of subsets of such a subgroup —
//! bucket distributions, kernel censuses, winning sets. Constructing a
//! [`MultiplicativeSubgroup`] validates the arithmetic once (`p` prime, `s | p - 1`), after
//! which the analysis kernels can assume well-formed inputs.
//!
//! Conventions: the subgroup is stored as consecutive powers
//! `[w^0, w^1, ..., w^{s-1}]` of a canonical element `w` of order exactly `s`,
//! derived from the *smallest* generator of `F_p^*`. All landscape statistics
//! (bucket maxima, census counts by weight, orbit counts) are provably
//! independent of this choice; per-`lambda` artifacts are reproducible against
//! the reference Python implementation, which uses the same convention.

use crate::error::{Error, Result};
use crate::field::{find_generator, is_prime, mulmod, powmod};

/// A multiplicative subgroup of order `s` in `F_p^*`.
#[derive(Debug, Clone)]
pub struct MultiplicativeSubgroup {
    p: u64,
    s: usize,
    w: u64,
    elements: Vec<u64>,
}

impl MultiplicativeSubgroup {
    /// Construct the order-`s` subgroup of `F_p^*`.
    ///
    /// Validates that `p` is prime, `s >= 2`, and `s | p - 1`.
    pub fn new(p: u64, s: usize) -> Result<Self> {
        if !is_prime(p) {
            return Err(Error::NotPrime(p));
        }
        if s < 2 {
            return Err(Error::OutOfRange(format!("subgroup order {s} < 2")));
        }
        if (p - 1) % s as u64 != 0 {
            return Err(Error::OrderDoesNotDivide {
                s: s as u64,
                pm1: p - 1,
            });
        }
        let g = find_generator(p);
        let w = powmod(g, (p - 1) / s as u64, p);
        let mut elements = Vec::with_capacity(s);
        let mut x = 1u64;
        for _ in 0..s {
            elements.push(x);
            x = mulmod(x, w, p);
        }
        debug_assert_eq!(x, 1);
        Ok(MultiplicativeSubgroup { p, s, w, elements })
    }

    /// The field characteristic.
    #[must_use]
    pub fn p(&self) -> u64 {
        self.p
    }

    /// The subgroup order.
    #[must_use]
    pub fn order(&self) -> usize {
        self.s
    }

    /// The canonical order-`s` element `w`.
    #[must_use]
    pub fn w(&self) -> u64 {
        self.w
    }

    /// Elements as consecutive powers `[w^0, ..., w^{s-1}]`.
    #[must_use]
    pub fn elements(&self) -> &[u64] {
        &self.elements
    }

    /// Whether `s` is a power of two (the SNARK-relevant smooth case; required
    /// by the ladder/rung machinery and the negation-pairing arguments).
    #[must_use]
    pub fn is_two_smooth(&self) -> bool {
        self.s.is_power_of_two()
    }

    /// Powers `[w^0, ..., w^{len-1}]` — the half-basis table (`len = s/2`) used
    /// by censuses and decompositions.
    #[must_use]
    pub fn pow_table(&self, len: usize) -> Vec<u64> {
        let mut t = Vec::with_capacity(len);
        let mut x = 1u64;
        for _ in 0..len {
            t.push(x);
            x = mulmod(x, self.w, self.p);
        }
        t
    }

    /// The cosets of `mu_{2^t}` inside the subgroup, each listed in the
    /// power-order convention `C_i = { w^{i + j * (s / 2^t)} : j }`.
    ///
    /// Requires `s` a power of two and `2^t <= s`.
    pub fn cosets(&self, t: usize) -> Result<Vec<Vec<u64>>> {
        if !self.is_two_smooth() {
            return Err(Error::Unsupported(
                "mu_{2^t} cosets require a power-of-two subgroup".into(),
            ));
        }
        let block = 1usize << t;
        if block > self.s {
            return Err(Error::OutOfRange(format!("2^{t} exceeds subgroup order")));
        }
        let ncos = self.s / block;
        Ok((0..ncos)
            .map(|i| {
                (0..block)
                    .map(|j| self.elements[(i + j * ncos) % self.s])
                    .collect()
            })
            .collect())
    }

    /// This subgroup as an [`EvalDomain`] — the `m = 1` case where the RS
    /// evaluation domain *is* the subgroup. (For `m > 1` towers, build the
    /// domain from the fibers instead.)
    #[must_use]
    pub fn eval_domain(&self) -> EvalDomain {
        EvalDomain::subgroup(self)
    }
}

/// An evaluation domain for a Reed–Solomon code: `n` distinct points in `F_p`.
/// The generic object a [`crate::rs::code::ReedSolomon`] sits on — build one from a
/// subgroup ([`Self::subgroup`]), an explicit set ([`Self::from_points`]), or
/// (later) a coset or an `m > 1` tower. This is the seam between the algebraic
/// [`MultiplicativeSubgroup`] (cosets, dilation, cyclotomic structure) and the
/// bare point list the decoder needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EvalDomain {
    p: u64,
    points: Vec<u64>,
}

impl EvalDomain {
    /// From an explicit list of distinct points in `F_p` (`n >= 2`).
    pub fn from_points(p: u64, points: Vec<u64>) -> Result<Self> {
        if points.len() < 2 {
            return Err(Error::OutOfRange("need at least 2 evaluation points".into()));
        }
        let mut sorted = points.clone();
        sorted.sort_unstable();
        sorted.dedup();
        if sorted.len() != points.len() {
            return Err(Error::OutOfRange("evaluation points must be distinct".into()));
        }
        Ok(EvalDomain { p, points })
    }

    /// The whole multiplicative subgroup as the domain (`m = 1`).
    #[must_use]
    pub fn subgroup(sg: &MultiplicativeSubgroup) -> Self {
        EvalDomain {
            p: sg.p(),
            points: sg.elements().to_vec(),
        }
    }

    /// The field characteristic.
    #[must_use]
    pub fn p(&self) -> u64 {
        self.p
    }
    /// The `n` evaluation points.
    #[must_use]
    pub fn points(&self) -> &[u64] {
        &self.points
    }
    /// The number of points `n`.
    #[must_use]
    pub fn n(&self) -> usize {
        self.points.len()
    }
}
