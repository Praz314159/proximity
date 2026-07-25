//! Kernel censuses: counting nonzero vectors `v` with bounded coefficients and
//! weight such that `sum_i v_i w^i = 0 (mod p)` on the half-basis
//! `i in [0, s/2)`.
//!
//! These vectors are the *arithmetic accidents* of the landscape: each one
//! merges structural classes, in dilation orbits of size `s`, and the
//! (exactly validated) decomposition law says bucket inflation is precisely a
//! weighted count of them. Their norms obey `N(v) <= (sum v_i^2)^{s/4}`
//! (Parseval + AM–GM), which confines weight-`w` orbits to primes
//! `p <= ~w^{s/4}` — the anticorrelation law.
//!
//! Two engines:
//! - [`direct`]: weight-capped depth-first enumeration, rayon-parallel; cost
//!   `~ sum_w C(s/2, w) (2c)^w`, any `s`.
//! - [`mitm`]: full census by weight via meet-in-the-middle halves; cost and
//!   memory `(2c + 1)^{s/4}`, so `s <= 32` at `c = 2`.

use crate::domain::MultiplicativeSubgroup;
use crate::error::{Error, Result};
use rayon::prelude::*;

/// `counts[w]` = number of nonzero kernel vectors of weight exactly `w`
/// (`w <= wmax`), coefficients in `[-cmax, cmax] \ {0}` on the support.
pub fn direct(sg: &MultiplicativeSubgroup, cmax: i64, wmax: usize) -> Result<Vec<u64>> {
    if cmax < 1 {
        return Err(Error::OutOfRange("cmax must be >= 1".into()));
    }
    let p = sg.p();
    let half = sg.order() / 2;
    if wmax > half {
        return Err(Error::OutOfRange("wmax exceeds s/2".into()));
    }
    let pows = sg.pow_table(half);
    let coefs: Vec<i64> = (-cmax..=cmax).filter(|&c| c != 0).collect();
    let residues: Vec<Vec<u64>> = (0..half)
        .map(|i| {
            coefs
                .iter()
                .map(|&c| {
                    let cc = if c >= 0 {
                        c as u64 % p
                    } else {
                        p - ((-c) as u64 % p)
                    };
                    (cc as u128 * pows[i] as u128 % p as u128) as u64
                })
                .collect()
        })
        .collect();

    #[allow(clippy::too_many_arguments)]
    fn recurse(
        pos: usize,
        used: usize,
        acc: u64,
        p: u64,
        half: usize,
        wmax: usize,
        residues: &[Vec<u64>],
        counts: &mut [u64],
    ) {
        if used > 0 && acc == 0 {
            counts[used] += 1;
        }
        if used == wmax || pos == half {
            return;
        }
        for i in pos..half {
            for &rv in &residues[i] {
                recurse(
                    i + 1,
                    used + 1,
                    (acc + rv) % p,
                    p,
                    half,
                    wmax,
                    residues,
                    counts,
                );
            }
        }
    }

    let firsts: Vec<(usize, u64)> = (0..half)
        .flat_map(|i| {
            residues[i]
                .iter()
                .map(move |&rv| (i, rv))
                .collect::<Vec<_>>()
        })
        .collect();
    Ok(firsts
        .par_iter()
        .map(|&(i, rv)| {
            let mut c = vec![0u64; wmax + 1];
            recurse(i + 1, 1, rv % p, p, half, wmax, &residues, &mut c);
            c
        })
        .reduce(
            || vec![0u64; wmax + 1],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                a
            },
        ))
}

/// Half-side kernel table: enumerate every coefficient vector with entries in
/// `[-cmax, cmax]` over pow-table slots `[from, to)`, mapping each residue to
/// the weights of the vectors achieving it. Shared by the MitM census and the
/// bucket-decomposition engine — the decomposition law requires the two to
/// agree, so they must enumerate identically.
pub(crate) fn kernel_side(
    pows: &[u64],
    p: u64,
    cmax: i64,
    from: usize,
    to: usize,
) -> std::collections::HashMap<u64, Vec<u8>> {
    let mut map: std::collections::HashMap<u64, Vec<u8>> = std::collections::HashMap::new();
    let n = to - from;
    let base = (2 * cmax + 1) as u64;
    for code in 0..base.pow(n as u32) {
        let mut c = code;
        let mut acc: u64 = 0;
        let mut w: u8 = 0;
        for i in 0..n {
            let digit = (c % base) as i64 - cmax;
            c /= base;
            if digit != 0 {
                w += 1;
                let cc = if digit >= 0 {
                    digit as u64 % p
                } else {
                    p - ((-digit) as u64 % p)
                };
                acc = (acc + (cc as u128 * pows[from + i] as u128 % p as u128) as u64) % p;
            }
        }
        map.entry(acc).or_default().push(w);
    }
    map
}

/// Full census by weight via meet-in-the-middle over coordinate halves.
/// Requires `s <= 32` at `cmax = 2` (`(2c+1)^{s/4}` table entries).
pub fn mitm(sg: &MultiplicativeSubgroup, cmax: i64) -> Result<Vec<u64>> {
    let s = sg.order();
    if s > 32 || s % 4 != 0 {
        return Err(Error::Unsupported(
            "MitM census requires s <= 32, 4 | s".into(),
        ));
    }
    if !(1..=4).contains(&cmax) {
        return Err(Error::OutOfRange("cmax in [1, 4]".into()));
    }
    let p = sg.p();
    let half = s / 2;
    let pows = sg.pow_table(half);
    let a = kernel_side(&pows, p, cmax, 0, half / 2);
    let b = kernel_side(&pows, p, cmax, half / 2, half);
    let mut counts = vec![0u64; half + 1];
    for (val, wsb) in &b {
        let need = (p - val % p) % p;
        if let Some(wsa) = a.get(&need) {
            for &wb in wsb {
                for &wa in wsa {
                    counts[(wa + wb) as usize] += 1;
                }
            }
        }
    }
    counts[0] = counts[0].saturating_sub(1); // remove the zero vector
    Ok(counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::Cyclo;

    /// Glue (design/negacyclic_ring.md): the census counts exactly the
    /// bounded-height [`Cyclo`] elements in the kernel of
    /// [`Cyclo::eval_at`] at the subgroup generator — both engines, one
    /// ring definition.
    #[test]
    fn census_is_the_cyclo_eval_kernel() {
        let sg = MultiplicativeSubgroup::new(17, 8).unwrap();
        let g = sg.elements()[1];
        let mut expected = vec![0u64; 5];
        for pat in 0..5u64.pow(4) {
            let mut v = vec![0i64; 4];
            let mut t = pat;
            for slot in v.iter_mut() {
                *slot = (t % 5) as i64 - 2;
                t /= 5;
            }
            let w = v.iter().filter(|&&c| c != 0).count();
            if w == 0 {
                continue;
            }
            if Cyclo::from_coeffs(v).unwrap().eval_at(g, 17) == 0 {
                expected[w] += 1;
            }
        }
        assert_eq!(mitm(&sg, 2).unwrap(), expected);
        assert_eq!(direct(&sg, 2, 4).unwrap(), expected);
    }
}
