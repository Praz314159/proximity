//! `vs::skeleton` — the G1 skeleton-census kernel (stages 65–67).
//!
//! Counts the unit-equation solution census at level `L` through the
//! proven criterion stack rather than brute enumeration: a solution
//! subset `E` (size `L/2`) is described by its *skeleton* — for each
//! slot `e in 1..L/2` either both of `{e, e+L/2}` (a pair), exactly
//! one (a half-slot), or neither — plus the torsion flag `sigma`
//! (`L/2 in E`) and the side parity `pi`. The stack:
//!
//! 1. **Budget** (valuation filter, exact and necessary): the
//!    `(1 - zeta)`-adic valuations must cancel; per skeleton this is
//!    a closed test on `(T mod L, |P| mod 2, vsum)`.
//! 2. **M1** (sublattice congruence): budget makes the target a unit
//!    with a *linear* fold-lattice address `m = sum n_j alpha_j`; the
//!    `alpha_j` are exactly rational with denominator dividing 8
//!    (the `max(1, ord/8)` law), so `8 alpha` is an integer table
//!    (embedded below) and M1 is a congruence mod 8, carried
//!    additively.
//! 3. **M2** (box): integral addresses must land in the sign box
//!    determined by the half-slot pattern.
//! 4. **M3** (realization): the surviving side vectors are checked
//!    against the unit equation modulo three independent primes with
//!    `L`-th roots of unity; unanimity is enforced.
//!
//! The census is assembled by a meet-in-the-middle join: each half of
//! the slot range is enumerated once, keyed on
//! `(T mod L, |P| mod 2, vsum, used, 8m mod 8)`, and joined per
//! passing final state through a sorted-array lookup. Counting all
//! realizations per budget pair gives the exact census.
//!
//! Provenance: the criterion is stage-67-certified (reproduces the
//! complete solvable sets at levels 16/32); this kernel reproduced
//! the level-32 census (26,084) exactly and predicted the level-64
//! census `N(128) = 3,758,482,820` in exact agreement with the
//! pod-measured ground truth (S4 campaign, 2026-07-28, dataset
//! `s4-n128-census`). The embedded `8 alpha` tables were exported
//! from the certified-rank log-embedding solve (snap error
//! `< 1.5e-13`) and are validated end-to-end by those golden counts.
//! The Python mirror is `experiments/landscape/probes/`
//! (`probe_membership.py` / `probe_n128_sample.py`).

use crate::domain::MultiplicativeSubgroup;
use crate::error::{Error, Result};
use crate::field::mulmod;
use rayon::prelude::*;

const MAXK: usize = 15;

/// The three verification primes (`p = 1 mod 64`); the order-`L`
/// roots come from [`MultiplicativeSubgroup`].
const PRIMES: [u64; 3] = [2_130_706_433, 2_013_265_921, 2_281_701_377];

/// `usize` shim over [`crate::field::gcd`] for slot arithmetic.
fn gcd(a: usize, b: usize) -> usize {
    crate::field::gcd(a as u64, b as u64) as usize
}

/// `8 alpha_j` at level 32 (row `j - 1`), padded to MAXK
/// columns; exported from the certified log-embedding solve.
const A8_32_PAD: [[i16; MAXK]; 31] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 2, -4, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 2, 0, 2, -4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 4, 0, 4, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 4, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0],
    [32, 16, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 4, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 2, 0, 2, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [16, 8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 2, 4, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [64, 32, 0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 2, 4, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [16, 8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 2, 0, 2, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 4, 0, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0],
    [32, 16, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 4, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 4, 0, 4, 0, -4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 2, 0, 2, -4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 2, -4, 2, 0, -2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
];

/// `8 alpha_j` at level 64 (row `j - 1`), padded to MAXK
/// columns; exported from the certified log-embedding solve.
const A8_64_PAD: [[i16; MAXK]; 63] = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 2, -4, 1, 0, -2, 0, 1, 0, 0, 0, -1, 0, 0, 0],
    [16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 2, 0, 1, -4, 0, 0, 1, 0, -2, 0, 1, 0, 0, 0],
    [8, 4, 0, 2, 0, -4, 0, 2, 0, 0, 0, -2, 0, 0, 0],
    [4, 2, 0, 2, 0, 0, -4, 0, 0, 0, 0, 0, 0, -2, 0],
    [32, 16, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 2, 0, 2, 0, 0, 0, 0, -4, 0, 0, 0, 0, 2, 0],
    [8, 4, 0, 2, 0, 0, 0, 2, 0, -4, 0, 2, 0, 0, 0],
    [4, 2, 0, 1, 0, 0, 0, 1, 0, 2, -4, 1, 0, 0, 0],
    [16, 8, 0, 4, 0, 0, 0, 4, 0, 0, 0, -4, 0, 0, 0],
    [4, 2, 0, 1, 0, 2, 0, 1, 0, 0, 0, -1, -4, 0, 0],
    [8, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0],
    [4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4],
    [64, 32, 0, 16, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
    [8, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
    [4, 2, 0, 1, 0, 2, 0, 1, 0, 0, 0, -1, 4, 0, 0],
    [16, 8, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0],
    [4, 2, 0, 1, 0, 0, 0, 1, 0, 2, 4, 1, 0, 0, 0],
    [8, 4, 0, 2, 0, 0, 0, 2, 0, 4, 0, 2, 0, 0, 0],
    [4, 2, 0, 2, 0, 0, 0, 0, 4, 0, 0, 0, 0, 2, 0],
    [32, 16, 0, 8, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [4, 2, 0, 2, 0, 0, 4, 0, 0, 0, 0, 0, 0, -2, 0],
    [8, 4, 0, 2, 0, 4, 0, 2, 0, 0, 0, -2, 0, 0, 0],
    [4, 2, 0, 1, 4, 0, 0, 1, 0, -2, 0, 1, 0, 0, 0],
    [16, 8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 2, 4, 1, 0, -2, 0, 1, 0, 0, 0, -1, 0, 0, 0],
    [8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [128, 64, 0, 32, 0, 0, 0, 16, 0, 0, 0, 0, 0, 0, 0],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 2, 4, 1, 0, -2, 0, 1, 0, 0, 0, -1, 0, 0, 0],
    [16, 8, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 2, 0, 1, 4, 0, 0, 1, 0, -2, 0, 1, 0, 0, 0],
    [8, 4, 0, 2, 0, 4, 0, 2, 0, 0, 0, -2, 0, 0, 0],
    [4, 2, 0, 2, 0, 0, 4, 0, 0, 0, 0, 0, 0, -2, 0],
    [32, 16, 0, 8, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [4, 2, 0, 2, 0, 0, 0, 0, 4, 0, 0, 0, 0, 2, 0],
    [8, 4, 0, 2, 0, 0, 0, 2, 0, 4, 0, 2, 0, 0, 0],
    [4, 2, 0, 1, 0, 0, 0, 1, 0, 2, 4, 1, 0, 0, 0],
    [16, 8, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0],
    [4, 2, 0, 1, 0, 2, 0, 1, 0, 0, 0, -1, 4, 0, 0],
    [8, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0],
    [4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
    [64, 32, 0, 16, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
    [4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4],
    [8, 4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, -4, 0],
    [4, 2, 0, 1, 0, 2, 0, 1, 0, 0, 0, -1, -4, 0, 0],
    [16, 8, 0, 4, 0, 0, 0, 4, 0, 0, 0, -4, 0, 0, 0],
    [4, 2, 0, 1, 0, 0, 0, 1, 0, 2, -4, 1, 0, 0, 0],
    [8, 4, 0, 2, 0, 0, 0, 2, 0, -4, 0, 2, 0, 0, 0],
    [4, 2, 0, 2, 0, 0, 0, 0, -4, 0, 0, 0, 0, 2, 0],
    [32, 16, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 2, 0, 2, 0, 0, -4, 0, 0, 0, 0, 0, 0, -2, 0],
    [8, 4, 0, 2, 0, -4, 0, 2, 0, 0, 0, -2, 0, 0, 0],
    [4, 2, 0, 1, -4, 0, 0, 1, 0, -2, 0, 1, 0, 0, 0],
    [16, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 2, -4, 1, 0, -2, 0, 1, 0, 0, 0, -1, 0, 0, 0],
    [8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
];

/// Exact join statistics of the criterion census at one level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SkeletonCensus {
    /// Budget-passing `(skeleton, parity)` pairs that also pass the
    /// M1 sublattice congruence.
    pub m1_pairs: u64,
    /// ... and land in the M2 box.
    pub m2_pairs: u64,
    /// ... and carry at least one realized solution.
    pub solvable_pairs: u64,
    /// The census: total realized solutions (all realizations
    /// counted per pair).
    pub solutions: u64,
}

struct Par {
    l: usize,
    half: usize,
    quarter: usize,
    lg: i32,
    k: usize,
    a8: &'static [[i16; MAXK]],
    pow_w: [Vec<u64>; 3],
    one_minus: [Vec<u64>; 3],
}

fn params(l: usize) -> Result<Par> {
    let (k, a8) = match l {
        32 => (7, &A8_32_PAD[..]),
        64 => (15, &A8_64_PAD[..]),
        _ => {
            return Err(Error::Unsupported(
                "skeleton census: level must be 32 or 64 (embedded alpha tables)".into(),
            ))
        }
    };
    let lg = l.trailing_zeros() as i32;
    let mut pow_w: [Vec<u64>; 3] = Default::default();
    let mut one_minus: [Vec<u64>; 3] = Default::default();
    for (i, &p) in PRIMES.iter().enumerate() {
        // consecutive powers w^0..w^{l-1} of an exact-order-l root
        let pw = MultiplicativeSubgroup::new(p, l)?.elements().to_vec();
        one_minus[i] = pw.iter().map(|&x| (1 + p - x) % p).collect();
        pow_w[i] = pw;
    }
    Ok(Par {
        l,
        half: l / 2,
        quarter: l / 4,
        lg,
        k,
        a8,
        pow_w,
        one_minus,
    })
}

// ------------------------------------------------------------------ DP

/// Exact skeleton-DP totals at `level`:
/// `(window, budget_pairs, budget_skeletons)` — all skeletons in the
/// valuation window, budget-passing `(skeleton, parity)` pairs, and
/// budget-passing skeletons (at least one parity).
pub fn skeleton_totals(level: usize) -> Result<(u128, u128, u128)> {
    let par = params(level)?;
    let (l, half) = (par.l, par.half);
    let vmax: usize = (1..half).map(|e| 2 * gcd(e, l)).sum();
    let nu = half + 1;
    let idx = |t: usize, pb: usize, vs: usize, u: usize| ((t * 2 + pb) * (vmax + 1) + vs) * nu + u;
    let mut f = vec![0u64; l * 2 * (vmax + 1) * nu];
    f[idx(0, 0, 0, 0)] = 1;
    for e in 1..half {
        let v = gcd(e, l);
        let mut g = f.clone();
        for t in 0..l {
            for pb in 0..2 {
                for vs in 0..=vmax {
                    for u in 0..=half {
                        let w = f[idx(t, pb, vs, u)];
                        if w == 0 {
                            continue;
                        }
                        if vs + 2 * v <= vmax && u + 2 <= half {
                            g[idx((t + 2 * e) % l, 1 - pb, vs + 2 * v, u + 2)] += w;
                        }
                        if vs + v <= vmax && u < half {
                            g[idx((t + e) % l, pb, vs + v, u + 1)] += w;
                        }
                    }
                }
            }
        }
        f = g;
    }
    let mut window = 0u128;
    for sp in 0..2usize {
        for t in 0..l {
            for pb in 0..2 {
                for vs in 0..=vmax {
                    window += f[idx(t, pb, vs, half - sp)] as u128;
                }
            }
        }
    }
    let mut pairs = 0u128;
    let mut skels = 0u128;
    for c in build_combos(&par) {
        if c.vs < 0 || c.vs as usize > vmax {
            continue;
        }
        let w = f[idx(c.t, c.pbar as usize, c.vs as usize, c.used as usize)] as u128;
        pairs += w;
        // both-parity states share (sp, t, pbar, vs); count each
        // skeleton group once, at pi = 0's combo
        if c.pi == 0 || !both_parities_pass(&par, &c) {
            skels += w;
        }
    }
    Ok((window, pairs, skels))
}

/// Whether the state addressed by `c` also passes with the opposite
/// parity (same `vs` requirement).
fn both_parities_pass(par: &Par, c: &Combo) -> bool {
    let other_t2 = other_parity_t2(par, c);
    match other_t2 {
        None => false,
        Some(t2) => gcd(t2, par.l) == gcd(current_t2(par, c), par.l),
    }
}

fn current_t2(par: &Par, c: &Combo) -> usize {
    let eps_neg = (c.pbar + c.sp + c.pi) % 2 == 1;
    if eps_neg {
        c.t
    } else {
        (c.t + par.half) % par.l
    }
}

fn other_parity_t2(par: &Par, c: &Combo) -> Option<usize> {
    let eps_neg = (c.pbar + c.sp + (1 - c.pi)) % 2 == 1;
    let t2 = if eps_neg {
        c.t
    } else {
        (c.t + par.half) % par.l
    };
    if t2 == 0 {
        None
    } else {
        Some(t2)
    }
}

// -------------------------------------------------------------- combos
#[derive(Clone)]
struct Combo {
    sp: u32,
    pi: u32,
    t: usize,
    pbar: u32,
    vs: i64,
    used: i64,
    leaf: [i16; MAXK],
}

fn build_combos(par: &Par) -> Vec<Combo> {
    let (l, half) = (par.l, par.half);
    let mut out = Vec::new();
    for sp in 0..2u32 {
        let used = (half - sp as usize) as i64;
        for pi in 0..2u32 {
            for t in 0..l {
                for pbar in 0..2u32 {
                    let eps_neg = (pbar + sp + pi) % 2 == 1;
                    let t2 = if eps_neg { t } else { (t + half) % l };
                    if t2 == 0 {
                        continue;
                    }
                    let vs = (half as i64) * (par.lg as i64 - 2) + gcd(t2, l) as i64
                        - (half as i64) * sp as i64;
                    if vs < 0 {
                        continue;
                    }
                    let mut leaf = [0i16; MAXK];
                    let scale = par.lg as i16 - 2 - sp as i16;
                    for (c, lv) in leaf.iter_mut().enumerate().take(par.k) {
                        *lv = scale * par.a8[half - 1][c] + par.a8[t2 - 1][c];
                    }
                    out.push(Combo {
                        sp,
                        pi,
                        t,
                        pbar,
                        vs,
                        used,
                        leaf,
                    });
                }
            }
        }
    }
    out
}

// ------------------------------------------------------- side machinery
fn pack_key(t: usize, pbar: u32, vs: i64, used: i64, m8mod: &[u8]) -> u128 {
    let mut key = (t as u128)
        | ((pbar as u128) << 6)
        | (((vs as u128) & 0x1ff) << 7)
        | (((used as u128) & 0x3f) << 16);
    let mut sh = 22;
    for &m in m8mod {
        key |= (m as u128) << sh;
        sh += 3;
    }
    key
}

struct AEntry {
    key: u128,
    m8: [i16; MAXK],
    hmask: u64,
    pmask: u64,
}

struct SideState {
    t: usize,
    pbar: u32,
    vs: i64,
    used: i64,
    m8: [i16; MAXK],
    hmask: u64,
    pmask: u64,
}

fn eval_side(par: &Par, slots: &[usize], code: usize) -> SideState {
    let l = par.l;
    let mut s = SideState {
        t: 0,
        pbar: 0,
        vs: 0,
        used: 0,
        m8: [0i16; MAXK],
        hmask: 0,
        pmask: 0,
    };
    let mut c = code;
    for &e in slots {
        let ch = c % 3;
        c /= 3;
        match ch {
            1 => {
                s.t = (s.t + 2 * e) % l;
                s.pbar ^= 1;
                s.vs += 2 * gcd(e, l) as i64;
                s.used += 2;
                s.pmask |= 1 << e;
                let j = (2 * e) % l;
                for i in 0..par.k {
                    s.m8[i] -= par.a8[j - 1][i];
                }
            }
            2 => {
                s.t = (s.t + e) % l;
                s.vs += gcd(e, l) as i64;
                s.used += 1;
                s.hmask |= 1 << e;
                for i in 0..par.k {
                    s.m8[i] -= par.a8[e - 1][i];
                }
            }
            _ => {}
        }
    }
    s
}

fn build_a_table(par: &Par, slots: &[usize]) -> Vec<AEntry> {
    let n: usize = 3usize.pow(slots.len() as u32);
    let mut tab = Vec::with_capacity(n);
    for code in 0..n {
        let s = eval_side(par, slots, code);
        let mut m8mod = [0u8; MAXK];
        for (i, mm) in m8mod.iter_mut().enumerate().take(par.k) {
            *mm = s.m8[i].rem_euclid(8) as u8;
        }
        tab.push(AEntry {
            key: pack_key(s.t, s.pbar, s.vs, s.used, &m8mod[..par.k]),
            m8: s.m8,
            hmask: s.hmask,
            pmask: s.pmask,
        });
    }
    tab.sort_unstable_by_key(|e| e.key);
    tab
}

// -------------------------------------------------------- realizations
struct Opt {
    f: [u64; 3],
    sum: usize,
    par: u32,
}

/// Count realized solutions of one budget+M1+M2 pair; `mismatch` is
/// bumped on any non-unanimous three-prime verdict.
fn count_real(
    par: &Par,
    combo: &Combo,
    hmask: u64,
    pmask: u64,
    m: &[i32],
    mismatch: &mut u64,
) -> u64 {
    let (l, half, quarter) = (par.l, par.half, par.quarter);
    let mut base_prod = [1u64; 3];
    let mut base_sum = 0usize;
    for e in 1..half {
        if pmask >> e & 1 == 1 {
            for (i, &p) in PRIMES.iter().enumerate() {
                base_prod[i] = mulmod(base_prod[i], par.one_minus[i][e], p);
                base_prod[i] = mulmod(base_prod[i], par.one_minus[i][e + half], p);
            }
            base_sum += 2 * e + half;
        }
    }
    if combo.sp == 1 {
        for (i, &p) in PRIMES.iter().enumerate() {
            base_prod[i] = mulmod(base_prod[i], par.one_minus[i][half], p);
        }
        base_sum += half;
    }
    let mut groups: Vec<Vec<Opt>> = Vec::new();
    for c in 1..quarter {
        let inc = hmask >> c & 1 == 1;
        let incc = hmask >> (half - c) & 1 == 1;
        if !inc && !incc {
            continue;
        }
        let mv = m[c - 1];
        let mut opts = Vec::new();
        let xcs: &[i32] = if inc { &[0, 1] } else { &[-1] };
        let xccs: &[i32] = if incc { &[0, 1] } else { &[-1] };
        for &xc in xcs {
            for &xcc in xccs {
                if xc.max(0) - xcc.max(0) != mv {
                    continue;
                }
                let mut o = Opt {
                    f: [1; 3],
                    sum: 0,
                    par: 0,
                };
                if inc {
                    let e = c + half * xc as usize;
                    for (i, &p) in PRIMES.iter().enumerate() {
                        o.f[i] = mulmod(o.f[i], par.one_minus[i][e], p);
                    }
                    o.sum += e;
                    o.par ^= xc as u32 & 1;
                }
                if incc {
                    let e = (half - c) + half * xcc as usize;
                    for (i, &p) in PRIMES.iter().enumerate() {
                        o.f[i] = mulmod(o.f[i], par.one_minus[i][e], p);
                    }
                    o.sum += e;
                    o.par ^= xcc as u32 & 1;
                }
                opts.push(o);
            }
        }
        if opts.is_empty() {
            return 0;
        }
        groups.push(opts);
    }
    if hmask >> quarter & 1 == 1 {
        let mut opts = Vec::new();
        for x in 0..2usize {
            let e = quarter + half * x;
            let mut o = Opt {
                f: [1; 3],
                sum: e,
                par: x as u32,
            };
            for (i, &p) in PRIMES.iter().enumerate() {
                o.f[i] = mulmod(o.f[i], par.one_minus[i][e], p);
            }
            opts.push(o);
        }
        groups.push(opts);
    }
    let lq = (l / 4) as u64;
    let mut count = 0u64;
    dfs(
        par, combo, &groups, 0, base_prod, base_sum, 0, &mut count, mismatch, lq,
    );
    count
}

#[allow(clippy::too_many_arguments)]
fn dfs(
    par: &Par,
    combo: &Combo,
    groups: &[Vec<Opt>],
    depth: usize,
    prod: [u64; 3],
    sum: usize,
    parity: u32,
    count: &mut u64,
    mismatch: &mut u64,
    lq: u64,
) {
    if depth == groups.len() {
        if parity != combo.pi {
            return;
        }
        let s = sum % par.l;
        let mut eq = 0;
        for (i, &p) in PRIMES.iter().enumerate() {
            let rhs = mulmod(lq % p, (1 + par.pow_w[i][s]) % p, p);
            if prod[i] == rhs {
                eq += 1;
            }
        }
        if eq == 3 {
            *count += 1;
        } else if eq > 0 {
            *mismatch += 1;
        }
        return;
    }
    for o in &groups[depth] {
        let mut np = prod;
        for (i, &p) in PRIMES.iter().enumerate() {
            np[i] = mulmod(np[i], o.f[i], p);
        }
        dfs(
            par,
            combo,
            groups,
            depth + 1,
            np,
            sum + o.sum,
            parity ^ o.par,
            count,
            mismatch,
            lq,
        );
    }
}

// ----------------------------------------------------------------- join

/// The exact criterion census at `level` (32 or 64): MITM join over
/// the slot halves, box, and three-prime-verified realization counts.
/// Errors if the three-prime verification is ever non-unanimous.
///
/// Cost: level 32 is interactive (< 1s); level 64 visits ~2e10 keyed
/// lookups and ~3e9 joined pairs — minutes on a many-core machine
/// (the S4 pod run took 262s on 252 threads).
pub fn skeleton_census(level: usize) -> Result<SkeletonCensus> {
    let par = params(level)?;
    let combos = build_combos(&par);
    let na = (par.half - 1) / 2;
    let a_slots: Vec<usize> = (1..=na).collect();
    let b_slots: Vec<usize> = (na + 1..par.half).collect();
    let atab = build_a_table(&par, &a_slots);
    let max_vs_a: i64 = a_slots.iter().map(|&e| 2 * gcd(e, par.l) as i64).sum();
    let max_used_a: i64 = 2 * na as i64;
    let nb_codes: usize = 3usize.pow(b_slots.len() as u32);
    let chunk = 8192usize;
    let nchunks = nb_codes.div_ceil(chunk);

    let reduced = (0..nchunks)
        .into_par_iter()
        .map(|ci| {
            let mut m1 = 0u64;
            let mut m2 = 0u64;
            let mut solvable = 0u64;
            let mut sols = 0u64;
            let mut mismatch = 0u64;
            let lo = ci * chunk;
            let hi = (lo + chunk).min(nb_codes);
            for code in lo..hi {
                let b = eval_side(&par, &b_slots, code);
                for c in combos.iter() {
                    let vs_a = c.vs - b.vs;
                    if vs_a < 0 || vs_a > max_vs_a {
                        continue;
                    }
                    let used_a = c.used - b.used;
                    if used_a < 0 || used_a > max_used_a {
                        continue;
                    }
                    let t_a = (c.t + par.l - b.t) % par.l;
                    let pbar_a = c.pbar ^ b.pbar;
                    let mut req = [0u8; MAXK];
                    for (i, rq) in req.iter_mut().enumerate().take(par.k) {
                        *rq = (-(b.m8[i] as i32 + c.leaf[i] as i32)).rem_euclid(8) as u8;
                    }
                    let key = pack_key(t_a, pbar_a, vs_a, used_a, &req[..par.k]);
                    let mut ix = atab.partition_point(|e| e.key < key);
                    while ix < atab.len() && atab[ix].key == key {
                        let a = &atab[ix];
                        ix += 1;
                        m1 += 1;
                        let mut m = [0i32; MAXK];
                        for (i, mv) in m.iter_mut().enumerate().take(par.k) {
                            *mv = (a.m8[i] as i32 + b.m8[i] as i32 + c.leaf[i] as i32) / 8;
                        }
                        let hmask = a.hmask | b.hmask;
                        let pmask = a.pmask | b.pmask;
                        let inbox = (1..par.quarter).all(|cc| {
                            let lo_b = -((hmask >> (par.half - cc) & 1) as i32);
                            let hi_b = (hmask >> cc & 1) as i32;
                            (lo_b..=hi_b).contains(&m[cc - 1])
                        });
                        if !inbox {
                            continue;
                        }
                        m2 += 1;
                        let x = count_real(&par, c, hmask, pmask, &m[..par.k], &mut mismatch);
                        if x > 0 {
                            solvable += 1;
                            sols += x;
                        }
                    }
                }
            }
            (m1, m2, solvable, sols, mismatch)
        })
        .reduce(
            || (0, 0, 0, 0, 0),
            |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3, a.4 + b.4),
        );

    let (m1_pairs, m2_pairs, solvable_pairs, solutions, mismatch) = reduced;
    if mismatch > 0 {
        return Err(Error::MalformedInput(format!(
            "skeleton census: {mismatch} non-unanimous three-prime verdicts"
        )));
    }
    Ok(SkeletonCensus {
        m1_pairs,
        m2_pairs,
        solvable_pairs,
        solutions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Level-32 DP totals: S3-certified (probe_n128_sample gate A).
    #[test]
    fn totals_32_golden() {
        assert_eq!(skeleton_totals(32).unwrap(), (3_492_117, 356_588, 178_304));
    }

    /// Level-64 DP totals: S2/S3/S4-certified.
    #[test]
    fn totals_64_golden() {
        assert_eq!(
            skeleton_totals(64).unwrap(),
            (106_495_542_464_222, 3_049_510_275_016, 1_524_755_137_544)
        );
    }

    /// Level-32 census: the stage-67-certified counting instrument
    /// (26,084 = |solutions(32)| exactly), with the S4 join tallies.
    #[test]
    fn census_32_golden() {
        let c = skeleton_census(32).unwrap();
        assert_eq!(
            c,
            SkeletonCensus {
                m1_pairs: 31_788,
                m2_pairs: 20_288,
                solvable_pairs: 15_564,
                solutions: 26_084,
            }
        );
    }

    /// The level-64 census (S4: solutions = N(128) = 3,758,482,820,
    /// 262s on a 252-core pod) is too heavy for CI; run explicitly:
    /// `cargo test --release census_64 -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn census_64_golden() {
        let c = skeleton_census(64).unwrap();
        assert_eq!(
            c,
            SkeletonCensus {
                m1_pairs: 3_013_418_648,
                m2_pairs: 1_299_060_528,
                solvable_pairs: 695_974_452,
                solutions: 3_758_482_820,
            }
        );
    }
}
