//! Full bucket distributions by in-place subset-sum DP.
//!
//! `q = 1`: `T[j][lambda] += T[j-1][lambda - g]` over subgroup elements `g`
//! with `j` descending — a rotated vector add per `(g, j)`; memory
//! `(r + 1) * p * 8` bytes, exact `u64` counts for `s <= 64`
//! (`C(64, 32) < 2^64`).
//!
//! `q = 2`: joint distribution over `(e_1, e_2)`; adding `g` maps
//! `(e1, e2) -> (e1 + g, e2 + g * e1)` — a row permutation plus per-row
//! rotation. Intended for `p <= ~700`.

use crate::buckets::BucketDistribution;
use crate::domain::Subgroup;
use crate::error::{Error, Result};
use rayon::prelude::*;

const PAR_THRESHOLD: usize = 1 << 15;

fn add_rotated(dst: &mut [u64], src: &[u64], shift: usize) {
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

fn check(sg: &Subgroup, r: usize) -> Result<()> {
    if sg.order() > 64 {
        return Err(Error::Unsupported(
            "u64-exact DP requires s <= 64 (CRT variant is a tracked issue)".into(),
        ));
    }
    if r == 0 || r > sg.order() {
        return Err(Error::OutOfRange(format!("r = {r} outside [1, s]")));
    }
    Ok(())
}

/// Exact `q = 1` distribution: `N(lambda) = #{ |S| = r : e_1(S) = lambda }`
/// for every `lambda in F_p`.
pub fn distribution_q1(sg: &Subgroup, r: usize) -> Result<BucketDistribution> {
    check(sg, r)?;
    let pp = sg.p() as usize;
    let mut t: Vec<Vec<u64>> = (0..=r).map(|_| vec![0u64; pp]).collect();
    t[0][0] = 1;
    for (cnt, &g) in sg.elements().iter().enumerate() {
        let top = r.min(cnt + 1);
        for j in (1..=top).rev() {
            let (lo, hi) = t.split_at_mut(j);
            add_rotated(&mut hi[0], &lo[j - 1], g as usize);
        }
    }
    Ok(BucketDistribution {
        values: t.pop().unwrap(),
    })
}

/// Exact `q = 2` joint distribution, row-major `p x p`
/// (`index = e1 * p + e2`).
pub fn distribution_q2(sg: &Subgroup, r: usize) -> Result<Vec<u64>> {
    check(sg, r)?;
    let pp = sg.p() as usize;
    if pp > 1 << 11 {
        return Err(Error::Unsupported(
            "q=2 grid DP intended for p <= ~2048".into(),
        ));
    }
    let mut t: Vec<Vec<u64>> = (0..=r).map(|_| vec![0u64; pp * pp]).collect();
    t[0][0] = 1;
    for (cnt, &g) in sg.elements().iter().enumerate() {
        let top = r.min(cnt + 1);
        for j in (1..=top).rev() {
            let (lo, hi) = t.split_at_mut(j);
            let src = &lo[j - 1];
            let dst = &mut hi[0];
            let gg = g as usize;
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
    Ok(t.pop().unwrap())
}
