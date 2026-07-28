//! `census::skeleton` — the G1 skeleton-census kernel (stages 65–67).
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

use crate::census::join::SortedMultiMap;
use crate::domain::MultiplicativeSubgroup;
use crate::error::{Error, Result};
use crate::field::mulmod;
use rayon::prelude::*;

const MAXK: usize = 15;

/// The three verification primes (`p = 1 mod 64`); the order-`L`
/// roots come from [`MultiplicativeSubgroup`].
const PRIMES: [u64; 3] = [2_130_706_433, 2_013_265_921, 2_281_701_377];

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
    /// `gcd(e, l)` per exponent — the `(1 - zeta)`-adic valuation of
    /// `(1 - zeta^e)`; precomputed once, hot in the join loop.
    vgcd: Vec<usize>,
    tables: [PrimeTable; 3],
}

/// Per-prime verification tables: `pow_w[e] = w^e` for an
/// exact-order-`l` root `w`, and `one_minus[e] = (1 - w^e) mod p`.
struct PrimeTable {
    p: u64,
    pow_w: Vec<u64>,
    one_minus: Vec<u64>,
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
    let vgcd = (0..l)
        .map(|e| crate::field::gcd(e as u64, l as u64) as usize)
        .collect();
    let mut tables = Vec::with_capacity(3);
    for &p in PRIMES.iter() {
        // consecutive powers w^0..w^{l-1} of an exact-order-l root
        let pow_w = MultiplicativeSubgroup::new(p, l)?.elements().to_vec();
        let one_minus = pow_w.iter().map(|&x| (1 + p - x) % p).collect();
        tables.push(PrimeTable {
            p,
            pow_w,
            one_minus,
        });
    }
    let tables = tables
        .try_into()
        .unwrap_or_else(|_| unreachable!("three primes"));
    Ok(Par {
        l,
        half: l / 2,
        quarter: l / 4,
        lg,
        k,
        a8,
        vgcd,
        tables,
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
    let vmax: usize = (1..half).map(|e| 2 * par.vgcd[e]).sum();
    let nu = half + 1;
    let idx = |t: usize, pb: usize, vs: usize, u: usize| ((t * 2 + pb) * (vmax + 1) + vs) * nu + u;
    let mut f = vec![0u64; l * 2 * (vmax + 1) * nu];
    f[idx(0, 0, 0, 0)] = 1;
    for e in 1..half {
        let v = par.vgcd[e];
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

/// The target exponent `t2` of `(L/4)(1 - zeta^{t2})` for one
/// `(sigma, pi, T, |P| mod 2)` state — the eps-sign rule of the
/// budget test. `None` when the target degenerates (`t2 = 0`).
fn t2_of(par: &Par, sp: u32, pi: u32, t: usize, pbar: u32) -> Option<usize> {
    let eps_neg = (pbar + sp + pi) % 2 == 1;
    let t2 = if eps_neg { t } else { (t + par.half) % par.l };
    (t2 != 0).then_some(t2)
}

/// Whether the state addressed by `c` also passes with the opposite
/// parity (same `vs` requirement, i.e. equal `t2` valuations).
fn both_parities_pass(par: &Par, c: &Combo) -> bool {
    match (
        t2_of(par, c.sp, c.pi, c.t, c.pbar),
        t2_of(par, c.sp, 1 - c.pi, c.t, c.pbar),
    ) {
        (Some(a), Some(b)) => par.vgcd[a] == par.vgcd[b],
        _ => false,
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
                    let Some(t2) = t2_of(par, sp, pi, t, pbar) else {
                        continue;
                    };
                    let vs = (half as i64) * (par.lg as i64 - 2) + par.vgcd[t2] as i64
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

/// Join key, low to high: `t` (6 bits) | `pbar` (1) | `vs` (9) |
/// `used` (6) | `k` 3-bit residues of `8m mod 8` — 67 bits at
/// `k = 15`. Equal keys = equal side state and complementary M1
/// congruence class.
fn pack_key(t: usize, pbar: u32, vs: i64, used: i64, m8mod: &[u8]) -> u128 {
    debug_assert!(t < 64 && (0..512).contains(&vs) && (0..64).contains(&used));
    let mut key =
        (t as u128) | ((pbar as u128) << 6) | ((vs as u128) << 7) | ((used as u128) << 16);
    let mut sh = 22;
    for &m in m8mod {
        key |= (m as u128) << sh;
        sh += 3;
    }
    key
}

/// Payload of one A-side entry in the join table.
struct ASide {
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
                s.vs += 2 * par.vgcd[e] as i64;
                s.used += 2;
                s.pmask |= 1 << e;
                let j = (2 * e) % l;
                for i in 0..par.k {
                    s.m8[i] -= par.a8[j - 1][i];
                }
            }
            2 => {
                s.t = (s.t + e) % l;
                s.vs += par.vgcd[e] as i64;
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

fn build_a_table(par: &Par, slots: &[usize]) -> SortedMultiMap<u128, ASide> {
    let n: usize = 3usize.pow(slots.len() as u32);
    let mut rows = Vec::with_capacity(n);
    for code in 0..n {
        let s = eval_side(par, slots, code);
        let mut m8mod = [0u8; MAXK];
        for (i, mm) in m8mod.iter_mut().enumerate().take(par.k) {
            *mm = s.m8[i].rem_euclid(8) as u8;
        }
        rows.push((
            pack_key(s.t, s.pbar, s.vs, s.used, &m8mod[..par.k]),
            ASide {
                m8: s.m8,
                hmask: s.hmask,
                pmask: s.pmask,
            },
        ));
    }
    SortedMultiMap::new(rows)
}

// -------------------------------------------------------- realizations

/// One side-assignment option of a half-slot group: its factor per
/// verification prime, its exponent contribution, and its top-side
/// parity contribution.
#[derive(Clone, Copy)]
struct Opt {
    f: [u64; 3],
    sum: usize,
    parity: u32,
}

const OPT_ONE: Opt = Opt {
    f: [1; 3],
    sum: 0,
    parity: 0,
};

impl Opt {
    /// Fold the factor `(1 - w^e)` into every prime and record the
    /// exponent `e` with top-side parity `x`.
    fn push_exp(&mut self, par: &Par, e: usize, x: u32) {
        for (f, tb) in self.f.iter_mut().zip(&par.tables) {
            *f = mulmod(*f, tb.one_minus[e], tb.p);
        }
        self.sum += e;
        self.parity ^= x & 1;
    }
}

/// The option groups of one budget+M1+M2 pair: at most one group per
/// canonical index (plus the torsion slot), at most two options each
/// — fixed-size, no heap. Group count is bounded by `quarter <= 16`.
struct Groups {
    opts: [[Opt; 2]; MAXK + 1],
    len: [u8; MAXK + 1],
    n: usize,
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
    let mut base = OPT_ONE;
    for e in 1..half {
        if pmask >> e & 1 == 1 {
            base.push_exp(par, e, 0);
            base.push_exp(par, e + half, 0);
        }
    }
    if combo.sp == 1 {
        base.push_exp(par, half, 0);
    }
    let mut groups = Groups {
        opts: [[OPT_ONE; 2]; MAXK + 1],
        len: [0; MAXK + 1],
        n: 0,
    };
    for c in 1..quarter {
        let inc = hmask >> c & 1 == 1;
        let incc = hmask >> (half - c) & 1 == 1;
        if !inc && !incc {
            continue;
        }
        let mv = m[c - 1];
        let mut len = 0usize;
        let xcs: &[i32] = if inc { &[0, 1] } else { &[-1] };
        let xccs: &[i32] = if incc { &[0, 1] } else { &[-1] };
        for &xc in xcs {
            for &xcc in xccs {
                if xc.max(0) - xcc.max(0) != mv {
                    continue;
                }
                let mut o = OPT_ONE;
                if inc {
                    o.push_exp(par, c + half * xc as usize, xc as u32);
                }
                if incc {
                    o.push_exp(par, (half - c) + half * xcc as usize, xcc as u32);
                }
                groups.opts[groups.n][len] = o;
                len += 1;
            }
        }
        if len == 0 {
            return 0;
        }
        groups.len[groups.n] = len as u8;
        groups.n += 1;
    }
    if hmask >> quarter & 1 == 1 {
        for x in 0..2u32 {
            let mut o = OPT_ONE;
            o.push_exp(par, quarter + half * x as usize, x);
            groups.opts[groups.n][x as usize] = o;
        }
        groups.len[groups.n] = 2;
        groups.n += 1;
    }
    let mut cx = RealCtx {
        par,
        pi: combo.pi,
        lq: (l / 4) as u64,
        groups: &groups,
        count: 0,
        mismatch: 0,
    };
    dfs(&mut cx, 0, base.f, base.sum, base.parity);
    *mismatch += cx.mismatch;
    cx.count
}

/// Shared state of one realization enumeration.
struct RealCtx<'a> {
    par: &'a Par,
    pi: u32,
    lq: u64,
    groups: &'a Groups,
    count: u64,
    mismatch: u64,
}

fn dfs(cx: &mut RealCtx, depth: usize, prod: [u64; 3], sum: usize, parity: u32) {
    if depth == cx.groups.n {
        if parity != cx.pi {
            return;
        }
        let s = sum % cx.par.l;
        let mut eq = 0;
        for (&f, tb) in prod.iter().zip(&cx.par.tables) {
            let rhs = mulmod(cx.lq % tb.p, (1 + tb.pow_w[s]) % tb.p, tb.p);
            if f == rhs {
                eq += 1;
            }
        }
        if eq == 3 {
            cx.count += 1;
        } else if eq > 0 {
            cx.mismatch += 1;
        }
        return;
    }
    for oi in 0..cx.groups.len[depth] as usize {
        let o = cx.groups.opts[depth][oi];
        let mut np = prod;
        for ((f, &of), tb) in np.iter_mut().zip(&o.f).zip(&cx.par.tables) {
            *f = mulmod(*f, of, tb.p);
        }
        dfs(cx, depth + 1, np, sum + o.sum, parity ^ o.parity);
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
    let max_vs_a: i64 = a_slots.iter().map(|&e| 2 * par.vgcd[e] as i64).sum();
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
                    for a in atab.get(&key) {
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
        return Err(Error::Verification(format!(
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
