//! The vanishing-syndrome geometry `VS(s, k)` — the dual (quotient) view of
//! Reed–Solomon on a multiplicative subgroup.
//!
//! [`crate::rs::code::ReedSolomon`] is the primal view: words as functions,
//! codewords, agreement, decoding. [`VsSpace`] is the same information seen
//! from the quotient `F_p^s / RS_k`: a word matters only through its syndrome
//! `b`, subsets pair with syndromes through elementary symmetric functions of
//! complements, and list questions become incidence questions. The two views
//! are glued by identities that this module's tests enforce permanently
//! (see `syndrome_identity_*` and `ownership_*` below).
//!
//! # Conventions (the authority — everything downstream gates on these)
//!
//! 1. **Domain order**: `mu_s` is enumerated as powers of the subgroup
//!    generator, `elements[i] = w^i`. Dilation (twist) is index `+1 mod s`,
//!    inversion is index `i -> (s - i) mod s`, the antipode (even `s`) is
//!    index `+ s/2 mod s`.
//! 2. **Subset order**: subsets of the domain are ascending index tuples,
//!    ranked lexicographically ([`VsSpace::subset_rank`]). This matches the
//!    row order of [`crate::rs::moments::moment_cloud`] and the GPU kernels'
//!    unrank; [`VsSpace::certificate`] exports test vectors so accelerated
//!    views can prove they agree.
//! 3. **Syndrome coordinates**: for a word `w` with full interpolant
//!    `sum_j c_j x^j` (degree `< s`), the syndrome is
//!    `b_j = (-1)^j c_{k+j}` for `j = 0..s-k-1`. With this sign choice the
//!    master identity reads `D_S(w) = sum_j b_j e_j(complement of S)` — the
//!    pairing used by [`crate::rs::moments::cut_counts`] with **raw**
//!    (unsigned) elementary symmetric vectors, and inverted by
//!    [`crate::smooth::rung::word_from_syndrome`].
//!
//! # Cost notes
//!
//! Constructors and per-subset methods are `O(s)`–`O(s^2)`. Streaming
//! counters ([`VsSpace::cut_counts`], [`VsSpace::strata_counts`]) enumerate
//! all `C(s, r)` subsets on the CPU (rayon): fine through `s = 24`, ~601M
//! subsets at `s = 32` (minutes); the GPU cloud engine is the accelerated
//! view for larger sweeps and must validate against
//! [`VsSpace::certificate`] before its numbers are believed.

use crate::domain::MultiplicativeSubgroup;
use crate::error::{Error, Result};
use crate::field::{checked_binom, mulmod, powmod};
use crate::rs::code::ReedSolomon;
use crate::rs::decode::for_each_combination;
use crate::rs::moments;
use crate::smooth::rung;
use rayon::prelude::*;

/// The vanishing-syndrome geometry `VS(s, k)` on `mu_s subset F_p`.
///
/// Constructed from the *subgroup*, not a generic domain: the syndrome
/// coordinates exist because `prod_{x in mu_s}(z - x) = z^s - 1`, so the
/// smoothness hypothesis of the theory is a type-level fact here.
#[derive(Debug, Clone)]
pub struct VsSpace {
    sg: MultiplicativeSubgroup,
    k: usize,
}

/// Pinned convention vectors for accelerated views (GPU cloud builders) to
/// verify against before their output is believed. All fields are exact.
#[derive(Debug, Clone)]
pub struct VsCertificate {
    /// Convention revision — bump only on a deliberate convention change.
    pub version: u32,
    /// `(p, s, k)` of the issuing space.
    pub p: u64,
    /// Subgroup order.
    pub s: usize,
    /// Code degree bound.
    pub k: usize,
    /// `(rank, subset)` pairs pinning the lex ranking convention.
    pub ranking: Vec<(u64, Vec<usize>)>,
    /// `(rank, moment row)` pairs pinning the complement/e-vector convention.
    pub moment_rows: Vec<(u64, Vec<u64>)>,
    /// First domain elements (pins the generator-power enumeration).
    pub domain_head: Vec<u64>,
    /// Cut sizes of the coordinate syndromes `e_j`, `j = 1..min(4, s-k-1)`
    /// (pins the pairing end to end).
    pub coordinate_cuts: Vec<u64>,
}

impl VsSpace {
    /// Construct `VS(s, k)`; requires `1 <= k <= s - 2` (so the rung
    /// `r = k + 1` is a proper subset size).
    pub fn new(sg: &MultiplicativeSubgroup, k: usize) -> Result<Self> {
        let s = sg.order();
        if k == 0 || k + 2 > s {
            return Err(Error::OutOfRange(format!(
                "need 1 <= k <= s - 2 (k = {k}, s = {s})"
            )));
        }
        Ok(VsSpace { sg: sg.clone(), k })
    }

    /// The primal view: `RS[F_p, mu_s, k]`.
    pub fn code(&self) -> ReedSolomon {
        ReedSolomon::on_subgroup(&self.sg, self.k).expect("k validated at construction")
    }

    /// The underlying subgroup.
    #[must_use]
    pub fn subgroup(&self) -> &MultiplicativeSubgroup {
        &self.sg
    }
    /// Field characteristic.
    #[must_use]
    pub fn p(&self) -> u64 {
        self.sg.p()
    }
    /// Subgroup order (= code length).
    #[must_use]
    pub fn s(&self) -> usize {
        self.sg.order()
    }
    /// Code degree bound.
    #[must_use]
    pub fn k(&self) -> usize {
        self.k
    }
    /// The rung: `r = k + 1`, the subset size at which incidence lives.
    #[must_use]
    pub fn r(&self) -> usize {
        self.k + 1
    }
    /// Syndrome dimension `s - k` (= moment-vector length `s - r + 1`).
    #[must_use]
    pub fn syndrome_dim(&self) -> usize {
        self.s() - self.k
    }

    // ---------------------------------------------------------------------
    // conventions: syndrome <-> word, moments, incidence
    // ---------------------------------------------------------------------

    /// The syndrome of a word: interpolate on the full group (inverse DFT,
    /// `c_j = s^{-1} sum_x w(x) x^{-j}`), keep coefficients `k..s-1`, apply
    /// the sign convention `b_j = (-1)^j c_{k+j}`. `O(s^2)`.
    pub fn syndrome(&self, word: &[u64]) -> Result<Vec<u64>> {
        let (p, s) = (self.p(), self.s());
        if word.len() != s {
            return Err(Error::OutOfRange(format!(
                "word length {} != s = {s}",
                word.len()
            )));
        }
        let elements = self.sg.elements();
        let s_inv = powmod(s as u64 % p, p - 2, p);
        let mut b = Vec::with_capacity(self.syndrome_dim());
        for j in self.k..s {
            let mut acc = 0u64;
            for (i, &wx) in word.iter().enumerate() {
                // x^{-j} for x = w^i is element[(s - (i*j mod s)) mod s]
                let e = (i * j) % s;
                let xinv_j = elements[(s - e) % s];
                acc = (acc + mulmod(wx % p, xinv_j, p)) % p;
            }
            let c = mulmod(acc, s_inv, p);
            let jj = j - self.k;
            b.push(if jj % 2 == 0 { c } else { (p - c) % p });
        }
        Ok(b)
    }

    /// The canonical word of a syndrome (inverse of [`Self::syndrome`] mod
    /// the code): `w = sum_j (-1)^j b_j x^{k+j}`.
    pub fn word(&self, b: &[u64]) -> Result<Vec<u64>> {
        if b.len() != self.syndrome_dim() {
            return Err(Error::OutOfRange(format!(
                "syndrome length {} != s - k = {}",
                b.len(),
                self.syndrome_dim()
            )));
        }
        Ok(rung::word_from_syndrome(
            self.p(),
            self.sg.elements(),
            self.r(),
            b,
        ))
    }

    /// The moment row of a rung subset (given as ascending domain indices):
    /// the **raw** elementary symmetric vector `(e_0 = 1, e_1, ..., e_{s-r})`
    /// of the complement. `O(s^2)`.
    pub fn moment_row(&self, subset: &[usize]) -> Result<Vec<u64>> {
        self.check_subset(subset)?;
        let s = self.s();
        let in_set: Vec<bool> = {
            let mut v = vec![false; s];
            for &i in subset {
                v[i] = true;
            }
            v
        };
        let comp: Vec<u64> = (0..s)
            .filter(|&i| !in_set[i])
            .map(|i| self.sg.elements()[i])
            .collect();
        Ok(moments::e_vec(&comp, self.p()))
    }

    /// Incidence: `<b, e(complement of S)> == 0`, equivalently
    /// `D_S(word(b)) == 0` — the two sides are the module's central identity
    /// and are tested against each other.
    pub fn incident(&self, b: &[u64], subset: &[usize]) -> Result<bool> {
        let row = self.moment_row(subset)?;
        if b.len() != row.len() {
            return Err(Error::OutOfRange(format!(
                "syndrome length {} != s - k = {}",
                b.len(),
                self.syndrome_dim()
            )));
        }
        let p = self.p();
        let mut acc = 0u64;
        for (&bc, &ec) in b.iter().zip(&row) {
            acc = (acc + mulmod(bc % p, ec, p)) % p;
        }
        Ok(acc == 0)
    }

    /// The order-`r` divided difference `D_S(w) = sum_{x in S} w(x) /
    /// prod_{y in S, y != x} (x - y)` — the primal-side incidence functional,
    /// kept for the cross-view tests. `O(r^2)`.
    pub fn divided_difference(&self, word: &[u64], subset: &[usize]) -> Result<u64> {
        self.check_subset(subset)?;
        let p = self.p();
        if word.len() != self.s() {
            return Err(Error::OutOfRange("word length != s".into()));
        }
        let xs: Vec<u64> = subset.iter().map(|&i| self.sg.elements()[i]).collect();
        let mut acc = 0u64;
        for (a, &xa) in xs.iter().enumerate() {
            let mut den = 1u64;
            for (bi, &xb) in xs.iter().enumerate() {
                if bi != a {
                    den = mulmod(den, (xa + p - xb) % p, p);
                }
            }
            let inv = powmod(den, p - 2, p);
            acc = (acc + mulmod(word[subset[a]] % p, inv, p)) % p;
        }
        Ok(acc)
    }

    // ---------------------------------------------------------------------
    // the subset-ranking convention
    // ---------------------------------------------------------------------

    /// Lexicographic rank of an ascending index subset — THE ranking
    /// convention (matches [`moments::moment_cloud`] row order and the GPU
    /// kernels' unrank).
    pub fn subset_rank(&self, subset: &[usize]) -> Result<u64> {
        self.check_subset(subset)?;
        let (s, r) = (self.s(), subset.len());
        let mut rank = 0u64;
        let mut prev = 0usize;
        for (pos, &x) in subset.iter().enumerate() {
            for y in prev..x {
                rank += checked_binom((s - y - 1) as u64, (r - pos - 1) as u64)
                    .ok_or_else(|| Error::Unsupported("rank overflows u64".into()))?;
            }
            prev = x + 1;
        }
        Ok(rank)
    }

    /// Inverse of [`Self::subset_rank`] for rung subsets (`|S| = r`).
    pub fn subset_unrank(&self, mut rank: u64) -> Result<Vec<usize>> {
        let (s, r) = (self.s(), self.r());
        let total = checked_binom(s as u64, r as u64)
            .ok_or_else(|| Error::Unsupported("C(s, r) overflows u64".into()))?;
        if rank >= total {
            return Err(Error::OutOfRange(format!(
                "rank {rank} >= C(s, r) = {total}"
            )));
        }
        let mut out = Vec::with_capacity(r);
        let mut x = 0usize;
        for pos in 0..r {
            loop {
                let cnt = checked_binom((s - x - 1) as u64, (r - pos - 1) as u64)
                    .expect("bounded by C(s, r), checked above");
                if rank < cnt {
                    out.push(x);
                    x += 1;
                    break;
                }
                rank -= cnt;
                x += 1;
            }
        }
        Ok(out)
    }

    // ---------------------------------------------------------------------
    // symmetry
    // ---------------------------------------------------------------------

    /// Dilation on subsets: index `+1 mod s`, re-sorted.
    #[must_use]
    pub fn twist_subset(&self, subset: &[usize]) -> Vec<usize> {
        let s = self.s();
        let mut out: Vec<usize> = subset.iter().map(|&i| (i + 1) % s).collect();
        out.sort_unstable();
        out
    }

    /// Inversion on subsets: index `i -> (s - i) mod s`, re-sorted.
    #[must_use]
    pub fn invert_subset(&self, subset: &[usize]) -> Vec<usize> {
        let s = self.s();
        let mut out: Vec<usize> = subset.iter().map(|&i| (s - i) % s).collect();
        out.sort_unstable();
        out
    }

    /// Canonical representative of the dilation orbit: the lex-least
    /// rotation.
    #[must_use]
    pub fn subset_orbit_canon(&self, subset: &[usize]) -> Vec<usize> {
        let s = self.s();
        let mut best: Option<Vec<usize>> = None;
        let mut cur: Vec<usize> = subset.to_vec();
        cur.sort_unstable();
        for _ in 0..s {
            if !best.as_ref().is_some_and(|b| cur >= *b) {
                best = Some(cur.clone());
            }
            cur = self.twist_subset(&cur);
        }
        best.expect("s >= 2 rotations examined")
    }

    // ---------------------------------------------------------------------
    // 2-power structure: cores and strata
    // ---------------------------------------------------------------------

    /// The antipode-free core and pair count of a subset (`s` even): the
    /// unique split `S = core ⊔ (antipodal pairs)`, with `e_1(S) =
    /// e_1(core)` (the height–multiplicity machinery's basic object).
    pub fn core(&self, subset: &[usize]) -> Result<(Vec<usize>, usize)> {
        let s = self.s();
        if s % 2 != 0 {
            return Err(Error::Unsupported("antipodal cores require even s".into()));
        }
        self.check_subset(subset)?;
        let half = s / 2;
        let in_set: Vec<bool> = {
            let mut v = vec![false; s];
            for &i in subset {
                v[i] = true;
            }
            v
        };
        let core: Vec<usize> = subset
            .iter()
            .copied()
            .filter(|&i| !in_set[(i + half) % s])
            .collect();
        let pairs = (subset.len() - core.len()) / 2;
        Ok((core, pairs))
    }

    /// Antipodal strata of a cut: `out[f] = #{S in Z(b) : S contains exactly
    /// f antipodal pairs}` for `f = 0..=r/2` (`s` even). Streams all
    /// `C(s, r)` subsets (rayon); cost note in the module docs.
    pub fn strata_counts(&self, b: &[u64]) -> Result<Vec<u64>> {
        let (p, s, r) = (self.p(), self.s(), self.r());
        if s % 2 != 0 {
            return Err(Error::Unsupported("antipodal strata require even s".into()));
        }
        if b.len() != self.syndrome_dim() {
            return Err(Error::OutOfRange(format!(
                "syndrome length {} != s - k = {}",
                b.len(),
                self.syndrome_dim()
            )));
        }
        let half = s / 2;
        let elements = self.sg.elements().to_vec();
        let nf = r / 2 + 1;
        let counts = (0..=s - r)
            .into_par_iter()
            .map(|i0| {
                let mut local = vec![0u64; nf];
                let mut idx = Vec::with_capacity(r);
                for_each_combination(s - i0 - 1, r - 1, |rest| {
                    idx.clear();
                    idx.push(i0);
                    for &rr in rest {
                        idx.push(i0 + 1 + rr);
                    }
                    // incidence via the complement moment row
                    let mut in_set = [false; 64];
                    for &i in &idx {
                        in_set[i] = true;
                    }
                    let comp: Vec<u64> = (0..s)
                        .filter(|&i| !in_set[i])
                        .map(|i| elements[i])
                        .collect();
                    let ev = moments::e_vec(&comp, p);
                    let mut acc = 0u64;
                    for (&bc, &ec) in b.iter().zip(&ev) {
                        acc = (acc + mulmod(bc % p, ec, p)) % p;
                    }
                    if acc == 0 {
                        let f = idx
                            .iter()
                            .filter(|&&i| i < half && in_set[(i + half) % s])
                            .count();
                        local[f] += 1;
                    }
                });
                local
            })
            .reduce(
                || vec![0u64; nf],
                |mut a, bvec| {
                    for (x, y) in a.iter_mut().zip(bvec) {
                        *x += y;
                    }
                    a
                },
            );
        Ok(counts)
    }

    // ---------------------------------------------------------------------
    // theorems as constructors
    // ---------------------------------------------------------------------

    /// The Theorem-B_mult extremal word `x^k - (-1)^{r+1} zeta^c x^{s-1}`;
    /// exact list = the Graham–Sloane class `{S : prod S = zeta^c}`.
    pub fn top_word(&self, c: usize) -> Result<Vec<u64>> {
        rung::top_word(&self.sg, self.r(), c)
    }

    /// The coordinate word `x^{k+j}` (syndrome supported on `e_j`),
    /// `1 <= j <= s - k - 1`.
    pub fn coordinate_word(&self, j: usize) -> Result<Vec<u64>> {
        if j == 0 || j >= self.syndrome_dim() {
            return Err(Error::OutOfRange(format!(
                "need 1 <= j <= s - k - 1 = {}",
                self.syndrome_dim() - 1
            )));
        }
        let (p, e) = (self.p(), (self.k + j) as u64);
        Ok(self
            .sg
            .elements()
            .iter()
            .map(|&x| powmod(x, e, p))
            .collect())
    }

    /// The composed fold-ladder word `W(x) = wh(x^2)`, where `wh` is the
    /// max-class top word of the folded space `VS(s/2, ceil(k/2))`. Closed
    /// form throughout — no numeric solve (`s` even).
    pub fn fold_ladder_word(&self) -> Result<Vec<u64>> {
        let (p, s) = (self.p(), self.s());
        if s % 2 != 0 {
            return Err(Error::Unsupported("fold requires even s".into()));
        }
        let s2 = s / 2;
        let r2 = (self.k + 3) / 2;
        let sg2 = MultiplicativeSubgroup::new(p, s2)?;
        let classes = rung::gs_class_counts(s2, r2)?;
        let c_star = (0..s2)
            .max_by_key(|&c| classes[c])
            .expect("s2 >= 2 classes");
        let wh = rung::top_word(&sg2, r2, c_star)?;
        let mut val = std::collections::HashMap::new();
        for (y, v) in sg2.elements().iter().zip(&wh) {
            val.insert(*y, *v);
        }
        Ok(self
            .sg
            .elements()
            .iter()
            .map(|&x| val[&mulmod(x, x, p)])
            .collect())
    }

    /// Graham–Sloane class counts at the rung: `out[c] = #{T : sum T = c}`.
    pub fn gs_class_counts(&self) -> Result<Vec<u64>> {
        rung::gs_class_counts(self.s(), self.r())
    }

    // ---------------------------------------------------------------------
    // streaming counters (delegate to the audited kernels)
    // ---------------------------------------------------------------------

    /// Cut sizes for many syndromes at once (streaming; no cloud).
    pub fn cut_counts(&self, bs: &[Vec<u64>]) -> Result<Vec<u64>> {
        moments::cut_counts(self.p(), self.sg.elements(), self.r(), bs)
    }

    /// Exhaustive sparse-cut maximum on a moment-coordinate support.
    pub fn cut_max_sparse(&self, support: &[usize]) -> Result<(u64, Vec<u64>)> {
        moments::cut_max_sparse(self.p(), self.sg.elements(), self.r(), support)
    }

    /// The materialized moment cloud (capped; see [`moments::moment_cloud`]).
    pub fn cloud(&self) -> Result<(Vec<u64>, usize, usize)> {
        moments::moment_cloud(self.p(), self.sg.elements(), self.r())
    }

    // ---------------------------------------------------------------------
    // the certificate
    // ---------------------------------------------------------------------

    /// Pinned convention vectors for accelerated views. Deterministic for a
    /// given `(p, s, k)` and convention version.
    pub fn certificate(&self) -> Result<VsCertificate> {
        let (s, r) = (self.s(), self.r());
        let total = checked_binom(s as u64, r as u64)
            .ok_or_else(|| Error::Unsupported("C(s, r) overflows u64".into()))?;
        // rank pins: first, last, and a spread of interior ranks
        let mut ranks = vec![0, total - 1];
        for d in 1..=5u64 {
            ranks.push((total / 7) * d + d);
        }
        ranks.retain(|&x| x < total);
        ranks.sort_unstable();
        ranks.dedup();
        let mut ranking = Vec::new();
        let mut moment_rows = Vec::new();
        for &rk in &ranks {
            let sub = self.subset_unrank(rk)?;
            moment_rows.push((rk, self.moment_row(&sub)?));
            ranking.push((rk, sub));
        }
        let jmax = (self.syndrome_dim() - 1).min(4);
        let bs: Vec<Vec<u64>> = (1..=jmax)
            .map(|j| {
                let mut b = vec![0u64; self.syndrome_dim()];
                b[j] = 1;
                b
            })
            .collect();
        let coordinate_cuts = self.cut_counts(&bs)?;
        Ok(VsCertificate {
            version: 1,
            p: self.p(),
            s,
            k: self.k,
            ranking,
            moment_rows,
            domain_head: self.sg.elements().iter().take(8).copied().collect(),
            coordinate_cuts,
        })
    }

    // ---------------------------------------------------------------------

    fn check_subset(&self, subset: &[usize]) -> Result<()> {
        let s = self.s();
        if s > 64 {
            return Err(Error::Unsupported(
                "per-subset methods assume s <= 64".into(),
            ));
        }
        if subset.is_empty() || subset.len() >= s {
            return Err(Error::OutOfRange("subset size outside [1, s)".into()));
        }
        if subset.windows(2).any(|w| w[0] >= w[1]) || subset[subset.len() - 1] >= s {
            return Err(Error::OutOfRange(
                "subset must be ascending indices < s".into(),
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rs::decode::{DecodeOracle, Radius};

    fn space(p: u64, s: usize, k: usize) -> VsSpace {
        let sg = MultiplicativeSubgroup::new(p, s).unwrap();
        VsSpace::new(&sg, k).unwrap()
    }

    /// Rank/unrank roundtrip, and agreement with the moment-cloud row order.
    #[test]
    fn ranking_convention() {
        let vs = space(17, 8, 2); // r = 3
        let (flat, rows, cols) = vs.cloud().unwrap();
        for rk in 0..rows as u64 {
            let sub = vs.subset_unrank(rk).unwrap();
            assert_eq!(vs.subset_rank(&sub).unwrap(), rk, "roundtrip at {rk}");
            let row = vs.moment_row(&sub).unwrap();
            assert_eq!(
                row,
                flat[rk as usize * cols..(rk as usize + 1) * cols].to_vec(),
                "cloud row order must equal lex unrank at {rk}"
            );
        }
    }

    /// The master identity, as *values*: `D_S(w) = <b, e(complement)>` for
    /// random words and every rung subset — the glue between the primal and
    /// dual views (lem:chs + lem:complement).
    #[test]
    fn syndrome_identity_exact() {
        let vs = space(97, 16, 7);
        let p = vs.p();
        let mut state = 5u64;
        let mut word = Vec::new();
        for _ in 0..16 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            word.push(state % p);
        }
        let b = vs.syndrome(&word).unwrap();
        for rk in (0..12870u64).step_by(97) {
            let sub = vs.subset_unrank(rk).unwrap();
            let dd = vs.divided_difference(&word, &sub).unwrap();
            let row = vs.moment_row(&sub).unwrap();
            let mut acc = 0u64;
            for (&bc, &ec) in b.iter().zip(&row) {
                acc = (acc + mulmod(bc, ec, p)) % p;
            }
            assert_eq!(dd, acc, "D_S(w) != <b, e(comp)> at rank {rk}");
        }
        // round trip: the canonical lift has the same syndrome
        let w2 = vs.word(&b).unwrap();
        assert_eq!(vs.syndrome(&w2).unwrap(), b);
    }

    /// Ownership: `|Z(syndrome(w))| = sum_f C(a_f, r)` over the decoded
    /// list — the clique decomposition binding the decoder to the cut.
    #[test]
    fn ownership_binds_decoder_to_cut() {
        let vs = space(65537, 16, 7);
        for (word, expect_cut) in [
            (vs.top_word(4).unwrap(), 810u64), // GS class, all agreements = r
            (vs.coordinate_word(1).unwrap(), 70), // additive zero-fiber
            (vs.coordinate_word(4).unwrap(), 266), // quarter-fold family
        ] {
            let b = vs.syndrome(&word).unwrap();
            let cut = vs.cut_counts(&[b]).unwrap()[0];
            assert_eq!(cut, expect_cut);
            let rs = vs.code();
            let list = DecodeOracle::new(&rs)
                .list(&word, Radius::agreement(vs.r()))
                .unwrap();
            let owned: u64 = list
                .iter()
                .map(|cw| {
                    let a = cw.iter().zip(&word).filter(|(x, y)| x == y).count() as u64;
                    checked_binom(a, vs.r() as u64).unwrap()
                })
                .sum();
            assert_eq!(owned, cut, "ownership violated");
        }
    }

    /// Strata of the pinned odd-cell extremal (the 18-word at p = 65537):
    /// the SH-reduction numbers, now a permanent library gate.
    #[test]
    fn strata_of_the_18_word() {
        let vs = space(65537, 16, 7);
        let w18 = [
            14274u64, 45571, 60798, 30803, 16774, 53622, 23957, 63873, 57198, 44950, 44028, 28126,
            25267, 3166, 17634, 55356,
        ];
        let b = vs.syndrome(&w18).unwrap();
        let strata = vs.strata_counts(&b).unwrap();
        assert_eq!(strata, vec![0, 48, 164, 180, 12]);
        assert_eq!(strata.iter().sum::<u64>(), 404);
    }

    /// Fold-ladder constructor: even word, and the (16, 7) instance decodes
    /// to exactly the folded top class at the aligned cell (L = 7 at t = 10).
    #[test]
    fn fold_ladder_closed_form() {
        let vs = space(65537, 16, 7);
        let w = vs.fold_ladder_word().unwrap();
        for i in 0..8 {
            assert_eq!(w[i], w[i + 8], "composed word must be even");
        }
        let rs = vs.code();
        let list = DecodeOracle::new(&rs)
            .list(&w, Radius::agreement(10))
            .unwrap();
        assert_eq!(list.len(), 7, "aligned-cell law L = N_max(8, 5) = 7");
    }

    /// Core decomposition: e_1 invariance and pair accounting.
    #[test]
    fn core_decomposition() {
        let vs = space(97, 16, 7);
        let sub = vec![0usize, 2, 3, 8, 10, 11, 12, 15];
        let (core, pairs) = vs.core(&sub).unwrap();
        assert_eq!(core.len() + 2 * pairs, sub.len());
        let p = vs.p();
        let e1 = |ix: &[usize]| {
            ix.iter()
                .fold(0u64, |a, &i| (a + vs.subgroup().elements()[i]) % p)
        };
        assert_eq!(e1(&sub), e1(&core), "pairs must cancel in e_1");
    }

    /// Certificate is deterministic and internally consistent.
    #[test]
    fn certificate_consistent() {
        let vs = space(65537, 16, 7);
        let cert = vs.certificate().unwrap();
        assert_eq!(cert.version, 1);
        assert_eq!(cert.coordinate_cuts[0], 70);
        assert_eq!(cert.coordinate_cuts[3], 266);
        for (rk, sub) in &cert.ranking {
            assert_eq!(vs.subset_rank(sub).unwrap(), *rk);
        }
    }
}
