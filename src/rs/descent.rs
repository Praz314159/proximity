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
//! convention authority. [`Descent`] is a handle holding the
//! precomputed domain tables; construct once per `(p, s, k)` and reuse
//! across a campaign.
//!
//! Cost notes: constructors are `O(s^2)` (the interpolant transform);
//! `psi_y` is `O(avail * k)` per core; `stratum_identity_check`
//! enumerates all `C(s/2, k_odd)` cores (~11k at `(32, 15)`, serial,
//! sub-second).

use crate::domain::MultiplicativeSubgroup;
use crate::error::{Error, Result};
use crate::field::{batch_inv, mulmod, powmod};
use crate::rs::decode::interp_eval_all;
use crate::rs::linalg::dd;
use crate::rs::vs::VsSpace;

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

/// The level-halving handle for `(p, s, k)`: domain tables, channel
/// dimensions, and the descent kernels.
pub struct Descent {
    vs: VsSpace,
    dom: Vec<u64>,
    /// `half_points[j] = dom[2j]` — the half domain `mu_{s/2}`, at which
    /// the channel words take their values (`w_even[j]` at
    /// `half_points[j]`, whose lifts are `dom[j]` and `dom[j + s/2]`).
    half_points: Vec<u64>,
    /// Full-interpolant monomial-coefficient transform rows are built on
    /// demand; the handle caches inverses of the domain instead.
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
    /// `c[m] = s^{-1} sum_i w[i] dom[i]^{-m}`.
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

    /// The channel syndromes: the interleaved slices
    /// `B_t[j] = c[k + 2j + t]`, `t = 0, 1, 2`, of the interpolant's top
    /// coefficients (`0` past degree `s - 1`). `B_0, B_2` read the odd
    /// channel, `B_1` the even — the level-drop's currency.
    pub fn channel_syndromes(&self, word: &[u64]) -> Result<[Vec<u64>; 3]> {
        let c = self.monomial_coeffs(word)?;
        let (s, k) = (self.s(), self.k());
        let len = (s / 2).saturating_sub(self.k() / 2);
        let slice = |t: usize| -> Vec<u64> {
            (0..len)
                .map(|j| {
                    if k + 2 * j + t < s {
                        c[k + 2 * j + t]
                    } else {
                        0
                    }
                })
                .collect()
        };
        Ok([slice(0), slice(1), slice(2)])
    }

    /// The effective syndrome of a point pair `(x, x')`:
    /// `b_eff[i] = c[k + 2i] + s1 c[k + 2i + 1] + s2 c[k + 2i + 2]` with
    /// `s1 = x + x'`, `s2 = x x'` — the level-`s/2` syndrome whose cut
    /// carries the pair's collision condition. Requires odd `k` (the
    /// even case is the parity variant of the species layer).
    pub fn effective_syndrome(&self, word: &[u64], x: u64, xp: u64) -> Result<Vec<u64>> {
        if self.k() % 2 == 0 {
            return Err(Error::Unsupported(
                "effective_syndrome requires odd k".into(),
            ));
        }
        let c = self.monomial_coeffs(word)?;
        let (p, s, k) = (self.p(), self.s(), self.k());
        let s1 = (x + xp) % p;
        let s2 = mulmod(x, xp, p);
        let kp = k.div_ceil(2);
        let at = |i: usize| if i < s { c[i] } else { 0 };
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
    /// pairs.
    pub fn psi_y(&self, word: &[u64], core: &[usize]) -> Result<Vec<(usize, u64)>> {
        self.check_word(word)?;
        if core.len() != self.k_odd() {
            return Err(Error::OutOfRange(format!(
                "core size {} != k_odd = {}",
                core.len(),
                self.k_odd()
            )));
        }
        let (p, s) = (self.p(), self.s());
        if core.iter().any(|&y| y >= s / 2) {
            return Err(Error::OutOfRange(
                "core index out of the half domain".into(),
            ));
        }
        let (wev, wod) = self.channels(word)?;
        let nodes: Vec<u64> = core.iter().map(|&y| self.half_points[y]).collect();
        let gy: Vec<u64> = core.iter().map(|&y| wev[y]).collect();
        let hy: Vec<u64> = core.iter().map(|&y| wod[y]).collect();
        // the available half points, each carrying two lifts
        let avail_half: Vec<usize> = (0..s / 2).filter(|j| !core.contains(j)).collect();
        let us: Vec<u64> = avail_half.iter().map(|&j| self.half_points[j]).collect();
        // one batched-barycentric evaluation per channel over the distinct
        // u-values (the decode hot kernel), plus one batched inversion of
        // the V-products — no per-point interpolation or Fermat inverses
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
                let x = self.dom[i];
                let num = ((word[i] + p - gs[a]) % p + p - mulmod(x, hs[a], p)) % p;
                out.push((i, mulmod(num, v_inv[a], p)));
            }
        }
        out.sort_unstable();
        Ok(out)
    }

    /// Fiber statistics of `psi_Y`; `collisions` counts unordered
    /// non-antipodal pairs with equal value — the stratum-identity
    /// currency.
    pub fn psi_y_stats(&self, word: &[u64], core: &[usize]) -> Result<PsiStats> {
        let vals = self.psi_y(word, core)?;
        let s = self.s();
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
    /// descent chapter proves; callers assert equality.
    pub fn stratum_identity_check(&self, word: &[u64]) -> Result<(u64, u64)> {
        self.check_word(word)?;
        let (s, kod) = (self.s(), self.k_odd());
        let mut core: Vec<usize> = (0..kod).collect();
        let mut total = 0u64;
        loop {
            total += self.psi_y_stats(word, &core)?.collisions;
            // next lex combination of kod from s/2
            let mut i = kod;
            loop {
                if i == 0 {
                    let b = self.vs.syndrome(word)?;
                    let strata = self.vs.strata_counts(&b)?;
                    let z = strata.get(kod).copied().unwrap_or(0);
                    return Ok((total, z));
                }
                i -= 1;
                if core[i] != i + s / 2 - kod {
                    break;
                }
            }
            core[i] += 1;
            for j in i + 1..kod {
                core[j] = core[j - 1] + 1;
            }
        }
    }

    /// Direct level-drop check for one assembled member: the cut
    /// functional of `S = fibers(Y) u {x, x'}` equals
    /// `<b_eff(x, x'), coefficients of prod_{u in W}(1 - u z)>` — used
    /// by the identity tests; exposed for campaign spot checks.
    pub fn member_functional(
        &self,
        word: &[u64],
        core: &[usize],
        i1: usize,
        i2: usize,
    ) -> Result<u64> {
        let s = self.s();
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
        let dom = &self.dom;
        let xs: Vec<u64> = subset.iter().map(|&i| dom[i]).collect();
        let ys: Vec<u64> = subset.iter().map(|&i| word[i]).collect();
        dd(self.p(), &xs, &ys)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let c = d.monomial_coeffs(&w).unwrap();
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
        let [b0, b1, b2] = d.channel_syndromes(&w).unwrap();
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
        let core = [1usize, 4, 6];
        let psi: Vec<(usize, u64)> = d.psi_y(&w, &core).unwrap();
        let mut checked = 0;
        for (a, &(i1, v1)) in psi.iter().enumerate() {
            for &(i2, v2) in psi.iter().skip(a + 1) {
                if i2 == (i1 + s / 2) % s {
                    continue;
                }
                let delta = d.member_functional(&w, &core, i1, i2).unwrap();
                // the dictionary: psi collision <=> vanishing functional
                assert_eq!(v1 == v2, delta == 0, "dictionary at ({i1}, {i2})");
                // the level drop: delta = <b_eff, prod (1 - u z) over W>
                let beff = d
                    .effective_syndrome(&w, sg.elements()[i1], sg.elements()[i2])
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
        let top = crate::smooth::rung::top_word(&sg, 8, 12).unwrap();
        assert_eq!(d.stratum_identity_check(&top).unwrap(), (96, 96));
        let plain = top_word(65537, sg.elements(), 7, 16);
        assert_eq!(d.stratum_identity_check(&plain).unwrap(), (128, 128));
        let w18: Vec<u64> = vec![
            14274, 45571, 60798, 30803, 16774, 53622, 23957, 63873, 57198, 44950, 44028, 28126,
            25267, 3166, 17634, 55356,
        ];
        assert_eq!(d.stratum_identity_check(&w18).unwrap(), (180, 180));
        let wr: Vec<u64> = (0..16u64).map(|i| (i * 40961 + 11) % 65537).collect();
        let (lhs, rhs) = d.stratum_identity_check(&wr).unwrap();
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
        let (x, xp) = (sg.elements()[3], sg.elements()[9]);
        let beff = d.effective_syndrome(&w, x, xp).unwrap();
        let s2 = mulmod(x, xp, p);
        assert_eq!(beff[0], 1);
        assert_eq!(*beff.last().unwrap(), s2);
        assert!(beff[1..beff.len() - 1].iter().all(|&v| v == 0));
    }
}
