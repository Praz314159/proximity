//! Exact bucket-distribution DPs.
//!
//! q = 1: N[lambda] = #{ r-subsets S of mu_s : e_1(S) = lambda } for every
//! lambda in F_p, via the in-place subset-sum recurrence
//!     T[j][lambda] += T[j-1][lambda - g]   (g over subgroup, j descending),
//! i.e. a rotated vector add per (g, j). Counts are exact u64 (valid for
//! s <= 64: C(64,32) < 2^64; guarded by assertion).
//!
//! q = 2: joint distribution over (e_1, e_2); adding element g maps
//! (e1, e2) -> (e1 + g, e2 + g*e1), a row permutation + per-row rotation.

use crate::field::subgroup;
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 1 << 15;

fn add_rotated(dst: &mut [u64], src: &[u64], shift: usize) {
    // dst[lambda] += src[(lambda - shift) mod p]
    let n = dst.len();
    let s = shift % n;
    let (d1, d2) = dst.split_at_mut(s);
    let add = |d: &mut [u64], sr: &[u64]| {
        if d.len() >= PAR_THRESHOLD {
            d.par_chunks_mut(1 << 14)
                .zip(sr.par_chunks(1 << 14))
                .for_each(|(dc, sc)| dc.iter_mut().zip(sc).for_each(|(x, y)| *x += *y));
        } else {
            d.iter_mut().zip(sr).for_each(|(x, y)| *x += *y);
        }
    };
    add(d1, &src[n - s..]);
    add(d2, &src[..n - s]);
}

/// Full q=1 bucket distribution; returns Vec of length p.
pub fn bucket_dist_q1(p: u64, s: usize, r: usize) -> Vec<u64> {
    assert!(s <= 64, "u64 counts require s <= 64 (CRT variant TODO)");
    assert!(r <= s);
    let els = subgroup(p, s);
    let pp = p as usize;
    let mut t: Vec<Vec<u64>> = (0..=r).map(|_| vec![0u64; pp]).collect();
    t[0][0] = 1;
    for (cnt, &g) in els.iter().enumerate() {
        let top = r.min(cnt + 1);
        for j in (1..=top).rev() {
            let (lo, hi) = t.split_at_mut(j);
            add_rotated(&mut hi[0], &lo[j - 1], g as usize);
        }
    }
    t.pop().unwrap()
}

/// Full q=2 joint distribution; returns row-major Vec of length p*p
/// (index = e1 * p + e2). Intended for p <= ~700.
pub fn bucket_dist_q2(p: u64, s: usize, r: usize) -> Vec<u64> {
    assert!(s <= 64 && r <= s);
    let els = subgroup(p, s);
    let pp = p as usize;
    let mut t: Vec<Vec<u64>> = (0..=r).map(|_| vec![0u64; pp * pp]).collect();
    t[0][0] = 1;
    for (cnt, &g) in els.iter().enumerate() {
        let top = r.min(cnt + 1);
        for j in (1..=top).rev() {
            let (lo, hi) = t.split_at_mut(j);
            let src = &lo[j - 1];
            let dst = &mut hi[0];
            let gg = g as usize;
            // dst[(e1+g) % p][(e2 + g*e1) % p] += src[e1][e2]
            for e1 in 0..pp {
                let e1p = (e1 + gg) % pp;
                let shift = (gg * e1) % pp;
                let drow = &mut dst[e1p * pp..(e1p + 1) * pp];
                let srow = &src[e1 * pp..(e1 + 1) * pp];
                let (d1, d2) = drow.split_at_mut(shift);
                d1.iter_mut()
                    .zip(&srow[pp - shift..])
                    .for_each(|(x, y)| *x += *y);
                d2.iter_mut()
                    .zip(&srow[..pp - shift])
                    .for_each(|(x, y)| *x += *y);
            }
        }
    }
    t.pop().unwrap()
}

pub fn max_and_argmax(dist: &[u64]) -> (u64, usize) {
    let mut best = 0u64;
    let mut arg = 0usize;
    for (i, &v) in dist.iter().enumerate() {
        if v > best {
            best = v;
            arg = i;
        }
    }
    (best, arg)
}
