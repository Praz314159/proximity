//! The descent: the level-halving operation `s -> s/2` on words and
//! syndromes, and the objects it produces — channel words, channel
//! syndromes, cores, and the derived words `psi_Y`.
//!
//! A word on `mu_s` splits into two channel words on `mu_{s/2}`
//! (`w(x) = w_even(x^2) + x * w_odd(x^2)`); its interpolant's top
//! coefficients, read in interleaved slices, are the channel syndromes;
//! and for a core `Y` of `k_odd` half-domain points, the derived word
//! `psi_Y` is the pencil coordinate whose value collisions are exactly
//! the top cut stratum (the stratum identity, pinned in tests). The
//! effective syndrome `b_eff` of a point pair packages the level-drop:
//! `<b_eff, e(complement)> = D_S(w)` for the member `S` assembled from
//! a core and the pair.
//!
//! One operation, one implementation: families and campaigns declare
//! behavior *under* this module (the top word's `b_eff` is again
//! two-spike — pinned below); conventions (fold signs, slice offsets,
//! `psi_Y` normalization) are pinned against [`VsSpace`], the
//! convention authority.
//!
//! Two handles, matching the cost tiers: [`Descent`] holds the
//! per-`(p, s, k)` domain tables (build once per campaign);
//! [`Descent::word`] builds a [`WordView`] holding the per-word data —
//! the interpolant transform and the channel split — once, so that
//! per-pair and per-core queries pay only their own tier (the
//! `HalfTables` build-once/query-many pattern).
//!
//! Cost notes: [`Descent::new`] is `O(s)`; [`Descent::word`] is
//! `O(s^2)` (the transform — the only quadratic step);
//! [`WordView::effective_syndrome`] is `O(s - k)` per pair;
//! [`WordView::psi_y`] is `O(avail * k_odd)` per core (batched
//! barycentric + one batched inversion); the stratum sweep runs over
//! all `C(s/2, k_odd)` cores, rayon-parallel over the leading core
//! element (~11k cores at `(32, 15)`: well under a second).

use crate::domain::MultiplicativeSubgroup;
use crate::error::{Error, Result};
use crate::field::{batch_inv, mulmod, powmod};
use crate::rs::decode::interp_eval_all;
use crate::rs::linalg::dd;
use crate::rs::vs::VsSpace;
use rayon::prelude::*;

/// Fiber statistics of a derived word over its available points.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PsiStats {
    /// Available points (both lifts of every half point outside the core).
    pub total: usize,
    /// Distinct `psi_Y` values.
    pub distinct: usize,
    /// Largest fiber.
    pub max_fiber: usize,
    /// Non-antipodal collision pairs — the stratum-identity currency.
    pub collisions: u64,
}

/// The level-halving handle for `(p, s, k)`: domain tables and channel
/// dimensions. Per-word data lives on [`WordView`].
pub struct Descent {
    vs: VsSpace,
    dom: Vec<u64>,
    /// `half_points[j] = dom[2j]` — the half domain `mu_{s/2}`, at which
    /// the channel words take their values (`w_even[j]` at
    /// `half_points[j]`, whose lifts are `dom[j]` and `dom[j + s/2]`).
    half_points: Vec<u64>,
    inv_dom: Vec<u64>,
    inv2: u64,
}

impl Descent {
    /// Construct the handle; requires even `s` (the half domain must
    /// exist) and `1 <= k <= s - 2` (via [`VsSpace`]).
    pub fn new(sg: &MultiplicativeSubgroup, k: usize) -> Result<Self> {
        let s = sg.order();
        if s % 2 != 0 {
            return Err(Error::OutOfRange("descent requires even s".into()));
        }
        let vs = VsSpace::new(sg, k)?;
        let p = sg.p();
        let dom = sg.elements().to_vec();
        let half_points: Vec<u64> = (0..s / 2).map(|j| dom[(2 * j) % s]).collect();
        let inv_dom: Vec<u64> = (0..s).map(|i| dom[(s - i) % s]).collect();
        let inv2 = powmod(2, p - 2, p);
        Ok(Descent {
            vs,
            dom,
            half_points,
            inv_dom,
            inv2,
        })
    }

    /// Field characteristic.
    #[must_use]
    pub fn p(&self) -> u64 {
        self.vs.p()
    }
    /// Level (= domain size) `s`.
    #[must_use]
    pub fn s(&self) -> usize {
        self.vs.s()
    }
    /// Degree bound `k`.
    #[must_use]
    pub fn k(&self) -> usize {
        self.vs.k()
    }
    /// Even channel dimension `ceil(k / 2)`.
    #[must_use]
    pub fn k_even(&self) -> usize {
        self.vs.k().div_ceil(2)
    }
    /// Odd channel dimension `floor(k / 2)` — the core size.
    #[must_use]
    pub fn k_odd(&self) -> usize {
        self.vs.k() / 2
    }
    /// The half domain, in the channel-word convention
    /// (`half_points[j] = dom[2j]`).
    #[must_use]
    pub fn half_points(&self) -> &[u64] {
        &self.half_points
    }
    /// The dual view this handle descends.
    #[must_use]
    pub fn vs(&self) -> &VsSpace {
        &self.vs
    }

    fn check_word(&self, word: &[u64]) -> Result<()> {
        if word.len() != self.s() {
            return Err(Error::OutOfRange(format!(
                "word length {} != s = {}",
                word.len(),
                self.s()
            )));
        }
        Ok(())
    }

    /// The channel split: `w_even[j] = (w[j] + w[j + s/2]) / 2`,
    /// `w_odd[j] = (w[j] - w[j + s/2]) / (2 dom[j])`, so that
    /// `w(x) = w_even(x^2) + x * w_odd(x^2)`.
    pub fn channels(&self, word: &[u64]) -> Result<(Vec<u64>, Vec<u64>)> {
        self.check_word(word)?;
        let (p, s) = (self.p(), self.s());
        let mut wev = Vec::with_capacity(s / 2);
        let mut wod = Vec::with_capacity(s / 2);
        for j in 0..s / 2 {
            let (a, b) = (word[j], word[j + s / 2]);
            wev.push(mulmod((a + b) % p, self.inv2, p));
            let d = mulmod((a + p - b) % p, self.inv2, p);
            wod.push(mulmod(d, self.inv_dom[j], p));
        }
        Ok((wev, wod))
    }

    /// The inverse of [`Descent::channels`].
    pub fn unfold(&self, wev: &[u64], wod: &[u64]) -> Result<Vec<u64>> {
        let (p, s) = (self.p(), self.s());
        if wev.len() != s / 2 || wod.len() != s / 2 {
            return Err(Error::OutOfRange("channel length != s/2".into()));
        }
        let mut w = vec![0u64; s];
        for j in 0..s / 2 {
            let t = mulmod(self.dom[j], wod[j], p);
            w[j] = (wev[j] + t) % p;
            w[j + s / 2] = (wev[j] + p - t) % p;
        }
        Ok(w)
    }

    /// Monomial coefficients of the full interpolant (degree `< s`):
    /// `c[m] = s^{-1} sum_i w[i] dom[i]^{-m}`. The `O(s^2)` step; pay it
    /// once per word through [`Descent::word`].
    pub fn monomial_coeffs(&self, word: &[u64]) -> Result<Vec<u64>> {
        self.check_word(word)?;
        let (p, s) = (self.p(), self.s());
        let sinv = powmod(s as u64 % p, p - 2, p);
        let mut c = vec![0u64; s];
        for (m, cm) in c.iter_mut().enumerate() {
            let mut acc = 0u64;
            for (i, &wi) in word.iter().enumerate() {
                acc = (acc + mulmod(wi, self.dom[(s - (i * m) % s) % s], p)) % p;
            }
            *cm = mulmod(acc, sinv, p);
        }
        Ok(c)
    }

    /// Build the per-word view: the interpolant transform and the
    /// channel split, computed once; every per-pair and per-core query
    /// runs off the cached data.
    pub fn word(&self, word: &[u64]) -> Result<WordView<'_>> {
        let coeffs = self.monomial_coeffs(word)?;
        let (wev, wod) = self.channels(word)?;
        Ok(WordView {
            d: self,
            word: word.to_vec(),
            coeffs,
            wev,
            wod,
        })
    }
}

/// Per-word descent data: the word, its interpolant coefficients, and
/// its channel words — built once by [`Descent::word`], queried many
/// times.
pub struct WordView<'a> {
    d: &'a Descent,
    word: Vec<u64>,
    coeffs: Vec<u64>,
    wev: Vec<u64>,
    wod: Vec<u64>,
}

impl WordView<'_> {
    /// The cell handle this view descends under.
    #[must_use]
    pub fn descent(&self) -> &Descent {
        self.d
    }
    /// The interpolant's monomial coefficients (degree `< s`).
    #[must_use]
    pub fn coeffs(&self) -> &[u64] {
        &self.coeffs
    }
    /// The channel words `(w_even, w_odd)`.
    #[must_use]
    pub fn channel_words(&self) -> (&[u64], &[u64]) {
        (&self.wev, &self.wod)
    }

    /// The channel syndromes: the interleaved slices
    /// `B_t[j] = c[k + 2j + t]`, `t = 0, 1, 2`, of the interpolant's top
    /// coefficients (`0` past degree `s - 1`). `B_0, B_2` read the odd
    /// channel, `B_1` the even — the level-drop's currency.
    #[must_use]
    pub fn channel_syndromes(&self) -> [Vec<u64>; 3] {
        let (s, k) = (self.d.s(), self.d.k());
        let len = (s / 2).saturating_sub(k / 2);
        let slice = |t: usize| -> Vec<u64> {
            (0..len)
                .map(|j| {
                    if k + 2 * j + t < s {
                        self.coeffs[k + 2 * j + t]
                    } else {
                        0
                    }
                })
                .collect()
        };
        [slice(0), slice(1), slice(2)]
    }

    /// The effective syndrome of a point pair `(x, x')`:
    /// `b_eff[i] = c[k + 2i] + s1 c[k + 2i + 1] + s2 c[k + 2i + 2]` with
    /// `s1 = x + x'`, `s2 = x x'` — the level-`s/2` syndrome whose cut
    /// carries the pair's collision condition. A slice combination over
    /// the cached coefficients. Requires odd `k` (the even case is the
    /// parity variant of the species layer).
    pub fn effective_syndrome(&self, x: u64, xp: u64) -> Result<Vec<u64>> {
        let (p, s, k) = (self.d.p(), self.d.s(), self.d.k());
        if k % 2 == 0 {
            return Err(Error::Unsupported(
                "effective_syndrome requires odd k".into(),
            ));
        }
        let s1 = (x + xp) % p;
        let s2 = mulmod(x, xp, p);
        let kp = k.div_ceil(2);
        let at = |i: usize| if i < s { self.coeffs[i] } else { 0 };
        Ok((0..s / 2 - kp)
            .map(|i| {
                let t = (at(k + 2 * i) + mulmod(s1, at(k + 2 * i + 1), p)) % p;
                (t + mulmod(s2, at(k + 2 * i + 2), p)) % p
            })
            .collect())
    }

    /// The derived word at a core: `Y` is a set of `k_odd` half-domain
    /// indices; for every available `x` (both lifts of every half point
    /// outside `Y`),
    /// `psi_Y(x) = (w(x) - g(x^2) - x h(x^2)) / V(x^2)`,
    /// with `g, h` the interpolants of the channel words on `Y` and
    /// `V(u) = prod_{y in Y} (u - u_y)`. Returns `(domain index, value)`
    /// pairs. One batched-barycentric evaluation per channel and one
    /// batched inversion; no per-point work beyond `O(k_odd)`.
    pub fn psi_y(&self, core: &[usize]) -> Result<Vec<(usize, u64)>> {
        let (p, s) = (self.d.p(), self.d.s());
        if core.len() != self.d.k_odd() {
            return Err(Error::OutOfRange(format!(
                "core size {} != k_odd = {}",
                core.len(),
                self.d.k_odd()
            )));
        }
        if core.iter().any(|&y| y >= s / 2) {
            return Err(Error::OutOfRange(
                "core index out of the half domain".into(),
            ));
        }
        let nodes: Vec<u64> = core.iter().map(|&y| self.d.half_points[y]).collect();
        let gy: Vec<u64> = core.iter().map(|&y| self.wev[y]).collect();
        let hy: Vec<u64> = core.iter().map(|&y| self.wod[y]).collect();
        let avail_half: Vec<usize> = (0..s / 2).filter(|j| !core.contains(j)).collect();
        let us: Vec<u64> = avail_half.iter().map(|&j| self.d.half_points[j]).collect();
        let (gs, hs) = if core.is_empty() {
            (vec![0u64; us.len()], vec![0u64; us.len()])
        } else {
            (
                interp_eval_all(&nodes, &gy, &us, p),
                interp_eval_all(&nodes, &hy, &us, p),
            )
        };
        let mut v_inv: Vec<u64> = us
            .iter()
            .map(|&u| {
                nodes
                    .iter()
                    .fold(1u64, |v, &n| mulmod(v, (u + p - n) % p, p))
            })
            .collect();
        batch_inv(&mut v_inv, p);
        let mut out = Vec::with_capacity(2 * us.len());
        for (a, &j) in avail_half.iter().enumerate() {
            for i in [j, j + s / 2] {
                let x = self.d.dom[i];
                let num = ((self.word[i] + p - gs[a]) % p + p - mulmod(x, hs[a], p)) % p;
                out.push((i, mulmod(num, v_inv[a], p)));
            }
        }
        out.sort_unstable();
        Ok(out)
    }

    /// Fiber statistics of `psi_Y`; `collisions` counts unordered
    /// non-antipodal pairs with equal value — the stratum-identity
    /// currency.
    pub fn psi_y_stats(&self, core: &[usize]) -> Result<PsiStats> {
        let vals = self.psi_y(core)?;
        let s = self.d.s();
        let mut sorted: Vec<u64> = vals.iter().map(|&(_, v)| v).collect();
        sorted.sort_unstable();
        let mut distinct = 0usize;
        let mut max_fiber = 0usize;
        let mut run = 0usize;
        let mut prev = None;
        for &v in &sorted {
            if prev == Some(v) {
                run += 1;
            } else {
                distinct += 1;
                run = 1;
                prev = Some(v);
            }
            max_fiber = max_fiber.max(run);
        }
        let mut collisions = 0u64;
        for (a, &(i, vi)) in vals.iter().enumerate() {
            for &(j, vj) in vals.iter().skip(a + 1) {
                if vi == vj && j != (i + s / 2) % s {
                    collisions += 1;
                }
            }
        }
        Ok(PsiStats {
            total: vals.len(),
            distinct,
            max_fiber,
            collisions,
        })
    }

    /// The stratum identity, both sides: `(sum over all cores of
    /// psi_Y collisions, |Z^(k_odd)(b)|)` — equal by the identity the
    /// descent chapter proves; callers assert equality. Rayon-parallel
    /// over the leading core element.
    pub fn stratum_identity_check(&self) -> Result<(u64, u64)> {
        let (s, kod) = (self.d.s(), self.d.k_odd());
        let half = s / 2;
        let total: u64 = (0..=half.saturating_sub(kod))
            .into_par_iter()
            .map(|first| -> Result<u64> {
                if kod == 0 {
                    return Ok(if first == 0 {
                        self.psi_y_stats(&[])?.collisions
                    } else {
                        0
                    });
                }
                let mut core: Vec<usize> = (0..kod).collect();
                core[0] = first;
                for (t, c) in core.iter_mut().enumerate().skip(1) {
                    *c = first + t;
                }
                if *core.last().unwrap() >= half {
                    return Ok(0);
                }
                let mut sum = 0u64;
                loop {
                    sum += self.psi_y_stats(&core)?.collisions;
                    // next lex combination with the first element fixed
                    let mut i = kod;
                    loop {
                        if i <= 1 {
                            return Ok(sum);
                        }
                        i -= 1;
                        if core[i] != i + half - kod {
                            break;
                        }
                    }
                    core[i] += 1;
                    for j in i + 1..kod {
                        core[j] = core[j - 1] + 1;
                    }
                }
            })
            .try_reduce(|| 0, |a, b| Ok(a + b))?;
        let b = self.d.vs.syndrome(&self.word)?;
        let strata = self.d.vs.strata_counts(&b)?;
        let z = strata.get(kod).copied().unwrap_or(0);
        Ok((total, z))
    }

    /// Direct level-drop check for one assembled member: the cut
    /// functional of `S = fibers(Y) u {x, x'}` — compared by tests
    /// against `<b_eff, prod (1 - u z) over the complement>`; exposed
    /// for campaign spot checks.
    pub fn member_functional(&self, core: &[usize], i1: usize, i2: usize) -> Result<u64> {
        let s = self.d.s();
        let mut subset: Vec<usize> = Vec::with_capacity(2 * core.len() + 2);
        for &y in core {
            subset.push(y);
            subset.push(y + s / 2);
        }
        subset.push(i1);
        subset.push(i2);
        subset.sort_unstable();
        subset.dedup();
        if subset.len() != 2 * core.len() + 2 {
            return Err(Error::OutOfRange("member overlaps its core fibers".into()));
        }
        let xs: Vec<u64> = subset.iter().map(|&i| self.d.dom[i]).collect();
        let ys: Vec<u64> = subset.iter().map(|&i| self.word[i]).collect();
        dd(self.d.p(), &xs, &ys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smooth::rung;

    fn top_word(p: u64, dom: &[u64], k: u32, s: u32) -> Vec<u64> {
        dom.iter()
            .map(|&x| (powmod(x, k as u64, p) + powmod(x, (s - 1) as u64, p)) % p)
            .collect()
    }

    #[test]
    fn fold_roundtrip_and_coeffs() {
        let sg = MultiplicativeSubgroup::new(97, 16).unwrap();
        let d = Descent::new(&sg, 7).unwrap();
        let w: Vec<u64> = (0..16u64).map(|i| (i * i * 31 + 7) % 97).collect();
        let (wev, wod) = d.channels(&w).unwrap();
        assert_eq!(d.unfold(&wev, &wod).unwrap(), w);
        // the interpolant reconstructs the word
        let wv = d.word(&w).unwrap();
        let c = wv.coeffs();
        for (i, &x) in sg.elements().iter().enumerate() {
            let mut v = 0u64;
            for m in (0..16).rev() {
                v = (mulmod(v, x, 97) + c[m]) % 97;
            }
            assert_eq!(v, w[i]);
        }
        // channel-syndrome slices agree with the dual view's syndrome
        // through the sign convention b_j = (-1)^j c_{k+j}
        let b = d.vs().syndrome(&w).unwrap();
        let [b0, b1, b2] = wv.channel_syndromes();
        for j in 0..b.len() {
            let expect = if j % 2 == 0 { b[j] } else { (97 - b[j]) % 97 };
            let got = match j % 2 {
                0 => b0[j / 2],
                _ => b1[j / 2],
            };
            assert_eq!(got, expect, "slice bridge at j = {j}");
            if j >= 2 && j % 2 == 0 {
                assert_eq!(b2[j / 2 - 1], expect, "b2 overlap at j = {j}");
            }
        }
    }

    #[test]
    fn level_drop_identity_and_dictionary() {
        let (p, s, k) = (97u64, 16usize, 7usize);
        let sg = MultiplicativeSubgroup::new(p, s).unwrap();
        let d = Descent::new(&sg, k).unwrap();
        let w: Vec<u64> = (0..s as u64).map(|i| (i * i * i + 5 * i + 3) % p).collect();
        let wv = d.word(&w).unwrap();
        let core = [1usize, 4, 6];
        let psi: Vec<(usize, u64)> = wv.psi_y(&core).unwrap();
        let mut checked = 0;
        for (a, &(i1, v1)) in psi.iter().enumerate() {
            for &(i2, v2) in psi.iter().skip(a + 1) {
                if i2 == (i1 + s / 2) % s {
                    continue;
                }
                let delta = wv.member_functional(&core, i1, i2).unwrap();
                // the dictionary: psi collision <=> vanishing functional
                assert_eq!(v1 == v2, delta == 0, "dictionary at ({i1}, {i2})");
                // the level drop: delta = <b_eff, prod (1 - u z) over W>
                let beff = wv
                    .effective_syndrome(sg.elements()[i1], sg.elements()[i2])
                    .unwrap();
                let mut wt = vec![1u64];
                for j in 0..s / 2 {
                    if core.contains(&j) || j == i1 % (s / 2) || j == i2 % (s / 2) {
                        continue;
                    }
                    let u = d.half_points()[j];
                    let mut next = vec![0u64; wt.len() + 1];
                    for (m, &cm) in wt.iter().enumerate() {
                        next[m] = (next[m] + cm) % p;
                        next[m + 1] = (next[m + 1] + mulmod(p - u, cm, p)) % p;
                    }
                    wt = next;
                }
                let pair: u64 = wt
                    .iter()
                    .zip(&beff)
                    .map(|(&a, &b)| mulmod(a, b, p))
                    .fold(0, |acc, t| (acc + t) % p);
                assert_eq!(pair, delta, "level drop at ({i1}, {i2})");
                checked += 1;
            }
        }
        assert!(checked >= 40, "work counter: {checked}");
    }

    #[test]
    fn stratum_identity_pins() {
        // the maximal-class top word at (16, 7), p = 65537: top stratum
        // 96 (the companion cut gate's strata {0:128, 2:576, 3:96,
        // 4:10}); its plain-class form: 128; the W18 extremal: 180; a
        // random word: both sides agree.
        let sg = MultiplicativeSubgroup::new(65537, 16).unwrap();
        let d = Descent::new(&sg, 7).unwrap();
        let top = rung::top_word(&sg, 8, 12).unwrap();
        assert_eq!(
            d.word(&top).unwrap().stratum_identity_check().unwrap(),
            (96, 96)
        );
        let plain = top_word(65537, sg.elements(), 7, 16);
        assert_eq!(
            d.word(&plain).unwrap().stratum_identity_check().unwrap(),
            (128, 128)
        );
        let w18: Vec<u64> = vec![
            14274, 45571, 60798, 30803, 16774, 53622, 23957, 63873, 57198, 44950, 44028, 28126,
            25267, 3166, 17634, 55356,
        ];
        assert_eq!(
            d.word(&w18).unwrap().stratum_identity_check().unwrap(),
            (180, 180)
        );
        let wr: Vec<u64> = (0..16u64).map(|i| (i * 40961 + 11) % 65537).collect();
        let (lhs, rhs) = d.word(&wr).unwrap().stratum_identity_check().unwrap();
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn top_word_beff_is_two_spike() {
        // the descent maps the top word to a two-spike syndrome:
        // b_eff = delta_0 + sigma2 * delta_last — extremal structure
        // transports (the species-closure finding, as a library pin)
        let (p, s, k) = (65537u64, 32usize, 15usize);
        let sg = MultiplicativeSubgroup::new(p, s).unwrap();
        let d = Descent::new(&sg, k).unwrap();
        let w = top_word(p, sg.elements(), 15, 32);
        let wv = d.word(&w).unwrap();
        let (x, xp) = (sg.elements()[3], sg.elements()[9]);
        let beff = wv.effective_syndrome(x, xp).unwrap();
        let s2 = mulmod(x, xp, p);
        assert_eq!(beff[0], 1);
        assert_eq!(*beff.last().unwrap(), s2);
        assert!(beff[1..beff.len() - 1].iter().all(|&v| v == 0));
    }
}
