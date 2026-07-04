//! Kernel censuses: counting nonzero vectors v with coefficients in
//! [-cmax, cmax] and bounded weight such that sum_i v_i w^i = 0 (mod p),
//! i on the half-basis [0, s/2). These drive the (exp20a-validated)
//! decomposition law: bucket inflation is a weighted count of exactly these
//! vectors, in dilation orbits of size s.
//!
//! Two engines:
//!  - `census_direct`: weight-capped depth-first enumeration, rayon over the
//!    first (position, coefficient) choice. Cost ~ sum_w C(s/2, w) (2c)^w.
//!  - `census_mitm`: full census by weight via meet-in-the-middle over the two
//!    coordinate halves; requires (2c+1)^{s/4} tables (s <= 32 at c = 2).

use crate::field::{powmod, subgroup};
use rayon::prelude::*;
use std::collections::HashMap;

fn pow_table(p: u64, s: usize) -> Vec<u64> {
    let els = subgroup(p, s);
    let w = els[1];
    let mut t = Vec::with_capacity(s / 2);
    let mut x = 1u64;
    for _ in 0..s / 2 {
        t.push(x);
        x = (x as u128 * w as u128 % p as u128) as u64;
    }
    debug_assert_eq!(powmod(w, (s / 2) as u64, p), p - 1);
    t
}

/// counts[w] = # nonzero kernel vectors of weight exactly w (<= wmax),
/// coefficients in [-cmax, cmax] \ {0} on the support.
pub fn census_direct(p: u64, s: usize, cmax: u64, wmax: usize) -> Vec<u64> {
    let pows = pow_table(p, s);
    let half = s / 2;
    // residues[i][k] = (c * w^i) mod p for c = -cmax..-1, 1..cmax (2*cmax entries)
    let coefs: Vec<i64> = (-(cmax as i64)..=cmax as i64).filter(|&c| c != 0).collect();
    let residues: Vec<Vec<u64>> = (0..half)
        .map(|i| {
            coefs
                .iter()
                .map(|&c| {
                    let cc = if c >= 0 { c as u64 % p } else { p - ((-c) as u64 % p) };
                    (cc as u128 * pows[i] as u128 % p as u128) as u64
                })
                .collect()
        })
        .collect();

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
        // enough positions left?
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

    // parallel over the first (position, coefficient) pair
    let firsts: Vec<(usize, u64)> = (0..half)
        .flat_map(|i| residues[i].iter().map(move |&rv| (i, rv)).collect::<Vec<_>>())
        .collect();
    let counts = firsts
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
        );
    counts
}

/// Full census by weight via MitM over coordinate halves; s/2 <= 16 at cmax=2.
pub fn census_mitm(p: u64, s: usize, cmax: i64) -> Vec<u64> {
    let pows = pow_table(p, s);
    let half = s / 2;
    let (lo, hi) = (0, half / 2); // coords [0, hi) and [hi, half)
    let side = |from: usize, to: usize| -> HashMap<u64, Vec<u8>> {
        // value -> list of weights of vectors on these coords
        let mut map: HashMap<u64, Vec<u8>> = HashMap::new();
        let n = to - from;
        let base = (2 * cmax + 1) as u64;
        let total = base.pow(n as u32);
        for code in 0..total {
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
    };
    let a = side(lo, hi);
    let b = side(hi, half);
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
    counts
}
