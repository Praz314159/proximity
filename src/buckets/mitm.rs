//! Meet-in-the-middle bucket machinery — cost `2^{s/2}`-type, **independent of
//! `p`**: interrogate primes of any magnitude at chosen `lambda`.
//!
//! - [`HalfTables`]: exact single buckets at arbitrary `q <= 8` (split the
//!   subgroup into halves; the top-`q` signed coefficients of `V_A * V_B` are
//!   triangular in those of `V_A`, `V_B`).
//! - [`decompose_bucket_q1`]: the anatomy law — enumerate the structural
//!   classes composing a `q = 1` bucket; the class sizes must sum to the
//!   bucket exactly.
//!
//! Sign convention (institutionalized after a real bug): the coefficient of
//! `Y^{r-i}` in `prod (Y - a)` is `c_i = (-1)^i e_i`. All convolutions are on
//! the signed `c_i`; unsigned `e_i` appear only at the API boundary.

use crate::code::class_size;
use crate::domain::Subgroup;
use crate::error::{Error, Result};
use std::collections::HashMap;

const QCAP: usize = 8;

type Key = (u8, [u64; QCAP]);

/// Precomputed half-subset tables for exact bucket queries at fixed
/// `(subgroup, r, q)`.
pub struct HalfTables {
    p: u64,
    s: usize,
    r: usize,
    q: usize,
    a: HashMap<Key, u64>,
    b: Vec<(u8, [u64; QCAP], u64)>,
}

fn top_signed_coeffs(els: &[u64], q: usize, p: u64) -> [u64; QCAP] {
    let mut c = [0u64; QCAP + 1];
    c[0] = 1;
    for (cnt, &a) in els.iter().enumerate() {
        let na = (p - a % p) % p;
        let top = q.min(cnt + 1);
        for i in (1..=top).rev() {
            c[i] = (c[i] + (na as u128 * c[i - 1] as u128 % p as u128) as u64) % p;
        }
    }
    let mut out = [0u64; QCAP];
    out[..q].copy_from_slice(&c[1..=q]);
    out
}

fn signed_to_e(c: &[u64; QCAP], q: usize, p: u64) -> Vec<u64> {
    (0..q)
        .map(|i| if i % 2 == 0 { (p - c[i]) % p } else { c[i] })
        .collect()
}

fn e_to_signed(e: &[u64], p: u64) -> [u64; QCAP] {
    let mut c = [0u64; QCAP];
    for (i, &ei) in e.iter().enumerate() {
        c[i] = if (i + 1) % 2 == 1 {
            (p - ei % p) % p
        } else {
            ei % p
        };
    }
    c
}

impl HalfTables {
    /// Build the tables. Requires an even subgroup order `s <= 32` and
    /// `1 <= q <= 8`, `r <= s`.
    pub fn build(sg: &Subgroup, r: usize, q: usize) -> Result<Self> {
        let s = sg.order();
        if s > 32 || s % 2 != 0 {
            return Err(Error::Unsupported(
                "MitM tables require even s <= 32 (the 2^{s/2} enumeration)".into(),
            ));
        }
        if q == 0 || q > QCAP || r > s {
            return Err(Error::OutOfRange("need 1 <= q <= 8 and r <= s".into()));
        }
        let p = sg.p();
        let half = s / 2;
        let els = sg.elements();
        let mut tables: Vec<HashMap<Key, u64>> = vec![HashMap::new(), HashMap::new()];
        for (t, chunk) in [&els[..half], &els[half..]].iter().enumerate() {
            let mut subset = Vec::with_capacity(half);
            for mask in 0u32..(1u32 << half) {
                subset.clear();
                for (i, &el) in chunk.iter().enumerate() {
                    if mask >> i & 1 == 1 {
                        subset.push(el);
                    }
                }
                let key = (subset.len() as u8, top_signed_coeffs(&subset, q, p));
                *tables[t].entry(key).or_insert(0) += 1;
            }
        }
        let b = tables
            .pop()
            .unwrap()
            .into_iter()
            .map(|((j, c), n)| (j, c, n))
            .collect();
        let a = tables.pop().unwrap();
        Ok(HalfTables { p, s, r, q, a, b })
    }

    /// Exact bucket size at target `e`-values `lam` (length `q`).
    pub fn bucket(&self, lam: &[u64]) -> Result<u64> {
        if lam.len() != self.q {
            return Err(Error::OutOfRange("lambda length != q".into()));
        }
        let (p, q) = (self.p, self.q);
        let cl = e_to_signed(lam, p);
        let mut total = 0u64;
        for &(j, cb, cnt) in &self.b {
            let j = j as usize;
            if j > self.r || self.r - j > self.s / 2 {
                continue;
            }
            let mut ca = [0u64; QCAP + 1];
            ca[0] = 1;
            for i in 1..=q {
                let mut acc: u64 = 0;
                for u in 0..i {
                    let cbv = cb[i - u - 1];
                    acc = (acc + (ca[u] as u128 * cbv as u128 % p as u128) as u64) % p;
                }
                ca[i] = (cl[i - 1] + p - acc) % p;
            }
            let mut ka = [0u64; QCAP];
            ka[..q].copy_from_slice(&ca[1..=q]);
            if let Some(&na) = self.a.get(&((self.r - j) as u8, ka)) {
                total += na * cnt;
            }
        }
        Ok(total)
    }

    /// The `e`-values corresponding to signed coefficients (exposed for
    /// round-tripping in tests).
    pub fn signed_roundtrip(lam: &[u64], p: u64) -> Vec<u64> {
        let q = lam.len();
        signed_to_e(&e_to_signed(lam, p), q, p)
    }
}

/// Exact structural decomposition of a `q = 1` bucket (the anatomy law):
/// enumerate every `eps in {-1, 0, 1}^{s/2}` with `sum eps_i w^i = lambda`
/// by MitM over coordinate halves; returns `(sum of class sizes,
/// per-weight class counts)`. The sum equals the DP bucket `N(lambda)`.
pub fn decompose_bucket_q1(sg: &Subgroup, r: usize, lam: u64) -> Result<(u64, Vec<u64>)> {
    let s = sg.order();
    if s > 32 || !sg.is_two_smooth() {
        return Err(Error::Unsupported(
            "decomposition requires power-of-two s <= 32".into(),
        ));
    }
    let p = sg.p();
    let half = s / 2;
    let pows = sg.pow_table(half);
    let hh = half / 2;
    let side = |from: usize, to: usize| -> HashMap<u64, Vec<u8>> {
        let mut map: HashMap<u64, Vec<u8>> = HashMap::new();
        let n = to - from;
        for code in 0..3u64.pow(n as u32) {
            let mut c = code;
            let mut acc = 0u64;
            let mut wt = 0u8;
            for i in 0..n {
                let d = (c % 3) as i64 - 1;
                c /= 3;
                if d != 0 {
                    wt += 1;
                    let cc = if d > 0 { 1u64 } else { p - 1 };
                    acc = (acc + (cc as u128 * pows[from + i] as u128 % p as u128) as u64) % p;
                }
            }
            map.entry(acc).or_default().push(wt);
        }
        map
    };
    let a = side(0, hh);
    let b = side(hh, half);
    let mut per_weight = vec![0u64; half + 1];
    let mut total = 0u64;
    for (val, wsb) in &b {
        let need = (lam % p + p - val % p) % p;
        if let Some(wsa) = a.get(&need) {
            for &wb in wsb {
                for &wa in wsa {
                    let wt = (wa + wb) as usize;
                    let cs = class_size(s, r, wt);
                    if cs > 0 {
                        per_weight[wt] += 1;
                        total += cs;
                    }
                }
            }
        }
    }
    Ok((total, per_weight))
}
