//! The integer-exact `Z[zeta_s]` value census (the floor preamble).

use crate::error::{Error, Result};

/// Exact `Z[zeta_s]` value census of one elementary-symmetric coordinate
/// over every `r`-subset of the `s`-th roots of unity (`s` a power of
/// two): the number of distinct exact values `N` and the multiplicity
/// profile of the value map — its maximum `M_inf` (the intrinsic,
/// prime-independent floor of the reduction histogram `A_j(p)`) and the
/// top multiplicities.
///
/// Arithmetic is in `Z[x]/(x^{s/2} + 1)`; multiplication by a root is a
/// signed coefficient rotation, and subsets are enumerated by DFS with
/// exact apply/undo updates, so the whole census is integer-exact. For
/// `r = s/2` the family equals its own complement family, so these are
/// the floors of the vanishing-syndrome census cell `(s, s/2 - 1)`.
///
/// Values are folded to 128-bit fingerprints before counting (collision
/// probability over `C(32,16)` values is `~2^-70`); memory is one `u128`
/// per subset — about 9.6 GB at `(32, 16)`.
///
/// Returns `(distinct, max_mult, top5_mults)`.
pub fn exact_value_census(s: usize, r: usize, coord: usize) -> Result<(u64, u64, Vec<u64>)> {
    if !s.is_power_of_two() || !(4..=64).contains(&s) {
        return Err(Error::OutOfRange(
            "census requires s a power of two in [4, 64]".into(),
        ));
    }
    if r == 0 || r > s || coord == 0 || coord > r {
        return Err(Error::OutOfRange(
            "census requires 1 <= coord <= r <= s".into(),
        ));
    }
    let half = s / 2;

    // e[c] holds the degree-c elementary symmetric value of the chosen
    // roots, as coefficients over the power basis of Z[x]/(x^{half}+1).
    let mut e = vec![vec![0i64; half]; coord + 1];
    e[0][0] = 1;

    fn fingerprint(v: &[i64]) -> u128 {
        let mut a = 0x9E37_79B9_7F4A_7C15u64;
        let mut b = 0xC2B2_AE3D_27D4_EB4Fu64;
        for &x in v {
            let x = x as u64;
            a = (a ^ x).wrapping_mul(0xFF51_AFD7_ED55_8CCD).rotate_left(31);
            b = b.wrapping_add(x).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            b ^= b >> 29;
        }
        ((a as u128) << 64) | b as u128
    }

    // e[c] += zeta^i * e[c-1] (signed rotation); exact inverse below.
    fn apply(e: &mut [Vec<i64>], hi: usize, i: usize, half: usize, add: bool) {
        let range: Vec<usize> = if add {
            (1..=hi).rev().collect()
        } else {
            (1..=hi).collect()
        };
        for c in range {
            for idx in 0..half {
                let v = e[c - 1][idx];
                if v == 0 {
                    continue;
                }
                let (m, sign) = crate::ring::fold(half, idx + i);
                let w = sign * v;
                if add {
                    e[c][m] += w;
                } else {
                    e[c][m] -= w;
                }
            }
        }
    }

    let total = crate::field::binom(s as u64, r as u64) as usize;
    let mut fps: Vec<u128> = Vec::new();
    fps.try_reserve_exact(total)
        .map_err(|_| Error::Unsupported("census allocation (memory)".into()))?;

    #[allow(clippy::too_many_arguments)] // recursive hot-loop helper; a
                                         // params struct would force per-node indirection for no clarity gain
    fn dfs(
        e: &mut [Vec<i64>],
        fps: &mut Vec<u128>,
        next: usize,
        chosen: usize,
        s: usize,
        r: usize,
        coord: usize,
        half: usize,
        fp: fn(&[i64]) -> u128,
    ) {
        if chosen == r {
            fps.push(fp(&e[coord]));
            return;
        }
        // prune: not enough elements left to finish the subset
        for i in next..=(s - (r - chosen)) {
            let hi = coord.min(chosen + 1);
            apply(e, hi, i, half, true);
            dfs(e, fps, i + 1, chosen + 1, s, r, coord, half, fp);
            apply(e, hi, i, half, false);
        }
    }
    if total < 1 << 20 || r < 2 {
        dfs(&mut e, &mut fps, 0, 0, s, r, coord, half, fingerprint);
        fps.sort_unstable();
    } else {
        // parallel: split by the first TWO chosen elements (~s^2/2
        // work-stealing tasks — first-element splits are badly
        // imbalanced: i0 = 0 alone holds half the subsets), each task
        // with an exactly-reserved output vector (growth-by-doubling
        // at GB scale was the measured cost of the first attempt).
        use rayon::prelude::*;
        let mut prefixes: Vec<(usize, usize)> = Vec::new();
        for i0 in 0..=(s - r) {
            for i1 in (i0 + 1)..=(s - r + 1) {
                prefixes.push((i0, i1));
            }
        }
        let parts: Vec<Vec<u128>> = prefixes
            .par_iter()
            .map(|&(i0, i1)| {
                let mut e0 = vec![vec![0i64; half]; coord + 1];
                e0[0][0] = 1;
                apply(&mut e0, coord.min(1), i0, half, true);
                apply(&mut e0, coord.min(2), i1, half, true);
                let cap = crate::field::binom((s - 1 - i1) as u64, (r - 2) as u64) as usize;
                let mut out: Vec<u128> = Vec::with_capacity(cap);
                dfs(&mut e0, &mut out, i1 + 1, 2, s, r, coord, half, fingerprint);
                out
            })
            .collect();
        for p_ in parts {
            fps.extend_from_slice(&p_);
        }
        fps.par_sort_unstable();
    }
    let mut mults: Vec<u64> = Vec::new();
    let mut run = 1u64;
    for w in fps.windows(2) {
        if w[0] == w[1] {
            run += 1;
        } else {
            mults.push(run);
            run = 1;
        }
    }
    mults.push(run);
    mults.sort_unstable_by(|a, b| b.cmp(a));
    let distinct = mults.len() as u64;
    let max_mult = mults[0];
    mults.truncate(5);
    Ok((distinct, max_mult, mults))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact-census pins from the independently-verified Python
    /// preamble at (16, 8): (coord, distinct N_j, intrinsic floor).
    #[test]
    fn exact_census_s16_pins() {
        for (coord, n, m) in [(1, 3281, 70), (4, 1629, 266), (8, 16, 810)] {
            let (d, mx, tops) = exact_value_census(16, 8, coord).unwrap();
            assert_eq!((d, mx), (n, m), "coord {coord}");
            assert_eq!(tops[0], m);
        }
    }
}
