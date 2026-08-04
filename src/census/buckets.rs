//! Bucket distributions and single-bucket queries.
//!
//! For the order-`s` subgroup `mu_s <= F_p^*`, the *bucket* at
//! `lambda = (lambda_1, ..., lambda_q)` is the number of `r`-subsets `S` of
//! `mu_s` whose top elementary symmetric functions satisfy
//! `e_i(S) = lambda_i` for `i <= q`. By the exactness theorem these are exact
//! list sizes of the extremal frozen-head words beyond the Johnson radius
//! (see [`crate::rs::code`]), which makes them the central computational object of
//! the proximity-gaps landscape.
//!
//! Two engines with complementary cost profiles:
//! - [`dp`]: full distributions over every `lambda` — cost and memory scale
//!   with `p` (use when you need the max over all `lambda`);
//! - [`mitm`]: exact single buckets at arbitrary `q` and bucket
//!   decompositions — cost `2^{s/2}`-type, **independent of `p`** (use to
//!   interrogate primes of any size at chosen `lambda`).

pub mod dp;
pub mod mitm;

/// A full `q = 1` bucket distribution: `values[lambda]` = bucket size.
#[derive(Debug, Clone)]
pub struct BucketDistribution {
    pub(crate) values: Vec<u64>,
}

impl BucketDistribution {
    /// Bucket sizes indexed by `lambda in [0, p)`.
    #[must_use]
    pub fn values(&self) -> &[u64] {
        &self.values
    }
    /// Consume the distribution, returning the raw bucket-size vector.
    #[must_use]
    pub fn into_values(self) -> Vec<u64> {
        self.values
    }
    /// Largest bucket and its `lambda`.
    #[must_use]
    pub fn max(&self) -> (u64, u64) {
        let (mut best, mut arg) = (0u64, 0usize);
        for (i, &v) in self.values.iter().enumerate() {
            if v > best {
                best = v;
                arg = i;
            }
        }
        (best, arg as u64)
    }
    /// Number of occupied buckets — for the canonical toy-protocol attack pair
    /// this is exactly `|Omega| = soundness * p` (winning-set identity).
    #[must_use]
    pub fn occupied(&self) -> u64 {
        self.values.iter().filter(|&&v| v > 0).count() as u64
    }
    /// Total mass; equals `C(s, r)` (checked in the test suite).
    #[must_use]
    pub fn total(&self) -> u64 {
        self.values.iter().sum()
    }
    /// Second moment `sum_lambda N(lambda)^2` (the L^2 / pair-collision count),
    /// as u128 to avoid overflow.
    #[must_use]
    pub fn second_moment(&self) -> u128 {
        self.values.iter().map(|&v| (v as u128) * (v as u128)).sum()
    }
}
