//! The master's three charges — deep strata (the descent, as the
//! smaller of a suffix sum and a single-term form), the middle band
//! (cores against `D_b`), and the small strata (the cut, min'd with
//! the graded route) — with their precomputed structures.

use crate::math::enclosure::{lg_binom, Lg};
use rug::float::Round;
use rug::ops::{AddAssignRound, SubAssignRound};
use rug::{Float, Integer};

use super::interface::Interface;
use super::profile::{add_to, Profile};

/// The channel dimensions of `k` — a `u64` shim over the fold's
/// single implementation in [`crate::rs::descent::channel_dims`].
#[must_use]
pub fn channel_dims(k: u64) -> (u64, u64) {
    let (e, o) = crate::rs::descent::channel_dims(k as usize);
    (e as u64, o as u64)
}

pub(super) struct Cell {
    pub(super) n: u64,
    pub(super) s: u64,
    pub(super) k: u64,
    pub(super) kev: u64,
    pub(super) kod: u64,
    pub(super) r: u64,
    pub(super) lstar: u64,
}

impl Cell {
    /// The largest pair count any member can carry at threshold `t`
    /// on the far branch of the near/far split.
    ///
    /// At threshold `t` a word with `t_max >= s + k - t` has list
    /// exactly one (its best codeword and any member agree with
    /// each other on `>= t + t_max - s >= k` points, and distinct
    /// polynomials of degree `< k` agree at most `k - 1` times), so
    /// that branch consumes no charge and `Charges::rhs` floors the
    /// envelope at one. Every remaining word has
    /// `t_max <= s + k - t - 1`, hence every member has agreement at
    /// most that, hence at most `floor((s + k - t - 1)/2)` pairs.
    /// Strata above this limit are empty and the sums may stop
    /// there.
    pub(super) fn far_pair_limit(&self, t: u64) -> u64 {
        (self.s + self.k).saturating_sub(t + 1) / 2
    }

    pub(super) fn new(n: u64, k: u64) -> Self {
        let (kev, kod) = channel_dims(k);
        Cell {
            n,
            s: 2 * n,
            k,
            kev,
            kod,
            r: k + 1,
            lstar: (n + k - 1).div_ceil(3),
        }
    }
}

/// Charge 1 — deep strata (`l >= l*`): the descent injection places
/// the class in the joint list, anti-squaring collapses it to the
/// larger channel list, and the level-below profile prices it. The
/// profile is block-constant on its grid, so the suffix over
/// `[l*, n]` compresses to runs — stretches where both channel
/// brackets are constant — each contributing width times its max,
/// with the sum strictly above each run precomputed so any lower
/// limit is answered with one partial-width multiply.
pub(super) struct DeepCharge {
    /// Runs in descending `l` order (built from `l = n` down).
    runs: Vec<DeepRun>,
}

struct DeepRun {
    lo: u64,
    hi: u64,
    max: Lg,
    sum_above: Option<Lg>,
}

impl DeepRun {
    /// The channel-max bracket at a threshold inside the run — the
    /// blocks are constant over `[lo, hi]`, so it is the run's max.
    fn max_at(&self, l: u64) -> Lg {
        debug_assert!(self.lo <= l && l <= self.hi);
        self.max.clone()
    }
}

impl DeepCharge {
    pub(super) fn build(prev: &Profile, cell: &Cell) -> Self {
        let mut runs = Vec::new();
        if cell.lstar > cell.n {
            return DeepCharge { runs };
        }
        let row_e = &prev.rows[&cell.kev];
        let row_o = &prev.rows[&cell.kod];
        let mut hi = cell.n;
        let mut sum_above: Option<Lg> = None;
        loop {
            // each channel's bracket comes with the lowest threshold
            // it stays valid at; the run is their meet
            let e = row_e.bracket(hi);
            let o = row_o.bracket(hi);
            let (max, meet) = e.max(&o);
            let lo = meet.max(cell.lstar);
            let contribution = Lg::from_u64(hi - lo + 1).mul(&max);
            let next_above = Some(add_to(sum_above.clone(), contribution));
            runs.push(DeepRun {
                lo,
                hi,
                max,
                sum_above,
            });
            sum_above = next_above;
            if lo == cell.lstar {
                break;
            }
            hi = lo - 1;
        }
        DeepCharge { runs }
    }

    /// The single-term charge at split `l0` (which must lie in
    /// `[l*, n]`): `2 max(E(n, kev, l0), E(n, kod, l0))`.
    pub(super) fn single_at(&self, l0: u64) -> Option<Lg> {
        if self.runs.is_empty() {
            return None;
        }
        let j = self.runs.partition_point(|run| run.lo > l0);
        let run = &self.runs[j];
        Some(Lg::from_u64(2).mul(&run.max_at(l0)))
    }

    /// The charge at threshold `t`: the smaller of the suffix sum
    /// from `l0 = max(l*, t - n)` and the single-term charge
    /// `2 max(E(n, kev, l0), E(n, kod, l0))`. The single term is
    /// valid because every deep class injects into the joint list at
    /// its own threshold, the joint lists nest down to `l0`, and the
    /// fold's tiling condition `3 l0 >= n + k - 1` (guaranteed by the
    /// coverage threshold) collapses the joint list to twice the
    /// larger channel list with no further loss. Both forms are
    /// non-increasing in `t`, so the pointwise min preserves the
    /// grid's enclosure contract. `None` when the stratum range is
    /// empty.
    pub(super) fn at(&self, cell: &Cell, t: u64) -> Option<Lg> {
        let l0 = t.saturating_sub(cell.n).max(cell.lstar);
        if l0 > cell.n || self.runs.is_empty() {
            return None;
        }
        let j = self.runs.partition_point(|run| run.lo > l0);
        let run = &self.runs[j];
        debug_assert!(run.lo <= l0 && l0 <= run.hi);
        let partial = Lg::from_u64(run.hi - l0 + 1).mul(&run.max);
        let sum_form = match &run.sum_above {
            Some(above) => above.add(&partial),
            None => partial,
        };
        let single = Lg::from_u64(2).mul(&run.max_at(l0));
        Some(single.min(&sum_form))
    }
}

/// Charge 2 — the middle band (`kod <= l < l*`): the core charge
/// prices the class by `D_b` against the suffix of `1/C(l, kod)`.
/// Small ranges get the exact term-by-term array; large ones the
/// telescoping identity `sum over l >= l0 of 1/C(l, m) =
/// m/((m - 1) C(l0 - 1, m - 1))`, enclosed as [first term, closed
/// form] — width `lg(l0/(m - 1))`, a fraction of a bit at rate 1/2,
/// for two binomials per query instead of a level-length
/// precomputation. The identity needs `m >= 2`, so `kod = 1` always
/// takes the exact route (the closed form would divide by zero).
pub(super) enum MidSuffix {
    Exact(Vec<Lg>),
    Telescoped,
}

impl MidSuffix {
    const EXACT_LIMIT: usize = 1 << 12;

    pub(super) fn build(cell: &Cell) -> Self {
        let len = cell.lstar.saturating_sub(cell.kod) as usize;
        if len > Self::EXACT_LIMIT && cell.kod >= 2 {
            return MidSuffix::Telescoped;
        }
        let mut suffix: Vec<Lg> = Vec::with_capacity(len);
        for i in 0..len {
            let l = cell.lstar - 1 - i as u64;
            let inv = Lg::zero().div(&lg_binom(l, cell.kod));
            suffix.push(match suffix.last() {
                Some(acc) => acc.add(&inv),
                None => inv,
            });
        }
        MidSuffix::Exact(suffix)
    }

    /// The suffix bracket for an extended split `lambda > l*`: the
    /// range `[l0, lambda)` is
    /// enclosed by [first term, infinite telescope] — loose by a
    /// fraction of a bit, sound for every `lambda`, and free of any
    /// `lambda`-indexed storage. Caller guarantees `kod >= 2`.
    pub(super) fn range_bracket(&self, cell: &Cell, l0: u64) -> Lg {
        let hi = Lg::from_u64(cell.kod)
            .div(&Lg::from_u64(cell.kod - 1))
            .div(&lg_binom(l0 - 1, cell.kod - 1));
        let lo = Lg::zero().div(&lg_binom(l0, cell.kod));
        Lg {
            lo: lo.lo,
            hi: hi.hi,
        }
    }

    pub(super) fn suffix_from(&self, cell: &Cell, l0: u64) -> Lg {
        match self {
            MidSuffix::Exact(suffix) => suffix[(cell.lstar - 1 - l0) as usize].clone(),
            MidSuffix::Telescoped => {
                let hi = Lg::from_u64(cell.kod)
                    .div(&Lg::from_u64(cell.kod - 1))
                    .div(&lg_binom(l0 - 1, cell.kod - 1));
                let lo = Lg::zero().div(&lg_binom(l0, cell.kod));
                Lg {
                    lo: lo.lo,
                    hi: hi.hi,
                }
            }
        }
    }
}

/// The derived-Johnson per-core multiplicity: members through a
/// partial core of size `l` agree
/// pairwise on at most `k' - 1` of the `N = s - 2l` available
/// points, so at derived agreement `m` the classical Johnson count
/// `N (m - k' + 1) / (m^2 - N (k' - 1))` bounds the per-core class.
/// Returned only where both the quadratic condition holds (true on
/// the whole band below the coverage curve) and the expression is
/// non-increasing in `m`
/// (`m >= 2 (k' - 1)`), which the step's off-grid block enclosure
/// requires of every summand.
pub(super) fn derived_johnson(s: u64, k: u64, l: u64, m: u64) -> Option<Lg> {
    let kp = k.checked_sub(2 * l)?;
    let n_av = s - 2 * l;
    // the extra gates are monotone-safety in `t` (the off-grid block
    // enclosure needs every summand non-increasing), not validity —
    // they stay here, outside the shared kernel
    if kp == 0 || m < 2 * (kp - 1) || m < kp {
        return None;
    }
    let (num, den) = super::base::johnson_agreement(n_av, kp, m)?;
    Some(Lg::from_integer(&Integer::from(num)).div(&Lg::from_integer(&Integer::from(den))))
}

/// Charge 3 — small strata (`l < kod`): canonical subsets to the cut,
/// priced by `D_c` over divisors `C(t - 2l, r - 2l)`, summed downward
/// with a certified tail bound: count times the worst remaining
/// numerator over the smallest remaining divisor (divisors grow as
/// `l` falls). The sum closes either when the tail is provably
/// negligible or at the term cap — near `t = r` the divisors grow
/// slowly and the cap bites, but there the remaining terms are
/// near-equal and the bound is tight; everywhere else the terms decay
/// fast and the tail is dust. Cost per threshold stays O(cap). The
/// `D_c` values of the capped window are threshold-independent, so
/// they are fetched once.
pub(super) struct CutCharge {
    /// `(d_c, d_c_sup)` for `l` in `[window_lo, kod)`; `None` = the
    /// provider certifies the stratum (resp. every stratum up to
    /// here) empty, and the charge skips it.
    window: Vec<(Option<Lg>, Option<Lg>)>,
    window_lo: u64,
}

impl CutCharge {
    const CAP: u64 = 16;
    const NEGLIGIBLE_BITS: u32 = 80;

    pub(super) fn build(cell: &Cell, data: &dyn Interface) -> Self {
        let window_lo = cell.kod.saturating_sub(Self::CAP + 2);
        let window = (window_lo..cell.kod)
            .map(|l| (data.d_c(cell.s, cell.k, l), data.d_c_sup(cell.s, cell.k, l)))
            .collect();
        CutCharge { window, window_lo }
    }

    fn dc(&self, cell: &Cell, data: &dyn Interface, l: u64) -> Option<Lg> {
        if l >= self.window_lo {
            self.window[(l - self.window_lo) as usize].0.clone()
        } else {
            data.d_c(cell.s, cell.k, l)
        }
    }

    fn dc_sup(&self, cell: &Cell, data: &dyn Interface, l: u64) -> Option<Lg> {
        if l >= self.window_lo {
            self.window[(l - self.window_lo) as usize].1.clone()
        } else {
            data.d_c_sup(cell.s, cell.k, l)
        }
    }

    pub(super) fn at(&self, cell: &Cell, data: &dyn Interface, t: u64) -> Option<Lg> {
        let lmin = t.saturating_sub(cell.n);
        if lmin >= cell.kod {
            return None;
        }
        let (r, kod) = (cell.r, cell.kod);
        let mut acc: Option<Lg> = None;
        let mut l = kod - 1;
        loop {
            let cut_term = self
                .dc(cell, data, l)
                .map(|dc| dc.div(&lg_binom(t - 2 * l, r - 2 * l)));
            // the graded route: every class member realizes its own
            // fiber set, so zero realized cores (`d_r = None`) empty
            // the class outright; a positive count is multiplied by
            // the derived-Johnson factor where that theorem applies.
            // A pointwise min of valid bounds is valid, and every
            // branch is non-increasing in t.
            let term = match data.d_r(cell.s, cell.k, l, t - 2 * l) {
                // zero realized cores: the class at (l, t) is empty
                None => None,
                Some(rr) => {
                    let graded_term =
                        derived_johnson(cell.s, cell.k, l, t - 2 * l).map(|j| rr.mul(&j));
                    // an empty stratum (cut face None) means the
                    // class is empty outright — the graded term must
                    // not resurrect it
                    match (cut_term, graded_term) {
                        (Some(c), Some(g)) => Some(c.min(&g)),
                        (Some(c), None) => Some(c),
                        (None, _) => None,
                    }
                }
            };
            if let Some(term) = term {
                acc = Some(add_to(acc, term));
            }
            if l == lmin {
                break;
            }
            // a provider that certifies everything below `l` empty
            // closes the sum exactly — no tail term at all: either
            // through the cut face (d_c_sup = None) or through the
            // graded face (d_r_sup = None: zero realized cores at
            // every remaining stratum, so every remaining class is
            // empty by the unconditional decomposition)
            if data.d_r_sup(cell.s, cell.k, l - 1, t).is_none() {
                break;
            }
            let Some(sup) = self.dc_sup(cell, data, l - 1) else {
                break;
            };
            let count = l - lmin;
            let mut tail_hi = Lg::from_u64(count).hi;
            tail_hi.add_assign_round(&sup.hi, Round::Up);
            tail_hi.sub_assign_round(&lg_binom(t - 2 * (l - 1), r - 2 * (l - 1)).lo, Round::Up);
            // the graded tail majorant: on the band (`phi` increasing
            // in `l`, i.e. `t <= (s + k - 1)/2`) the derived-Johnson
            // multiplicity at the binding stratum dominates every
            // remaining one — `N` falls, `m - k'` is constant, `phi`
            // rises — so count x d_r_sup x J(lmin) bounds the whole
            // remainder; non-increasing in `t` (numerator-derivative
            // sign reduces to `k > s - 1`, impossible)
            if 2 * t < cell.s + cell.k {
                if let (Some(j), Some(rsup)) = (
                    derived_johnson(cell.s, cell.k, lmin, t - 2 * lmin),
                    data.d_r_sup(cell.s, cell.k, l - 1, t),
                ) {
                    let mut graded_hi = Lg::from_u64(count).hi;
                    graded_hi.add_assign_round(&rsup.hi, Round::Up);
                    graded_hi.add_assign_round(&j.hi, Round::Up);
                    if graded_hi < tail_hi {
                        tail_hi = graded_hi;
                    }
                }
            }
            let closable = match &acc {
                Some(cur) => {
                    let mut margin = cur.hi.clone();
                    margin -= Self::NEGLIGIBLE_BITS;
                    tail_hi <= margin
                }
                None => false,
            };
            if closable || kod - 1 - l >= Self::CAP {
                let tail = Lg {
                    lo: Float::with_val(tail_hi.prec(), f64::NEG_INFINITY),
                    hi: tail_hi,
                };
                acc = Some(match &acc {
                    Some(cur) => Lg {
                        lo: cur.lo.clone(),
                        hi: cur.add(&tail).hi,
                    },
                    // every visited stratum was empty: the tail
                    // alone bounds the rest, with no mass below
                    None => tail,
                });
                break;
            }
            l -= 1;
        }
        acc
    }
}
