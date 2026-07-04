//! Meet-in-the-middle machinery at s <= 32:
//!  - exact single buckets at arbitrary q (split halves; the top-q signed
//!    coefficients of V_A * V_B are triangular in those of V_A, V_B);
//!  - the rung lambda of the Theorem-A construction (exp20c conventions);
//!  - exact eps-decomposition of a q=1 bucket (the anatomy law).
//!
//! Sign convention (the exp20c bug, institutionalized): the coefficient of
//! Y^{r-i} in prod (Y - a) is c_i = (-1)^i e_i. All products/convolutions are
//! done on the signed c_i; unsigned e_i only at the API boundary.

use crate::field::{binom, subgroup};
use std::collections::HashMap;

const QCAP: usize = 8;

type Key = (u8, [u64; QCAP]);

pub struct HalfTables {
    pub p: u64,
    pub s: usize,
    pub r: usize,
    pub q: usize,
    a: HashMap<Key, u64>,
    b: Vec<(u8, [u64; QCAP], u64)>, // (size, signed coeffs c_1..c_q, count)
}

fn top_signed_coeffs(els: &[u64], q: usize, p: u64) -> [u64; QCAP] {
    // c_i of prod (Y - a): recurrence c_i += (-a) * c_{i-1}
    let mut c = [0u64; QCAP + 1];
    c[0] = 1;
    let mut cnt = 0usize;
    for &a in els {
        cnt += 1;
        let na = (p - a % p) % p;
        let top = q.min(cnt);
        for i in (1..=top).rev() {
            c[i] = (c[i] + (na as u128 * c[i - 1] as u128 % p as u128) as u64) % p;
        }
    }
    let mut out = [0u64; QCAP];
    out[..q].copy_from_slice(&c[1..=q]);
    out
}

pub fn signed_to_e(c: &[u64; QCAP], q: usize, p: u64) -> Vec<u64> {
    (0..q)
        .map(|i| if i % 2 == 0 { (p - c[i]) % p } else { c[i] })
        .collect() // e_i = (-1)^i c_i: i odd in 1-based => e_1 = -c_1 etc.
}

pub fn e_to_signed(e: &[u64], p: u64) -> [u64; QCAP] {
    let mut c = [0u64; QCAP];
    for (i, &ei) in e.iter().enumerate() {
        // 1-based index i+1: c = (-1)^{i+1} e
        c[i] = if (i + 1) % 2 == 1 { (p - ei % p) % p } else { ei % p };
    }
    c
}

impl HalfTables {
    pub fn build(p: u64, s: usize, r: usize, q: usize) -> Self {
        assert!(s <= 32 && s % 2 == 0 && q >= 1 && q <= QCAP && r <= s);
        let els = subgroup(p, s);
        let half = s / 2;
        let mut tables: Vec<HashMap<Key, u64>> = vec![HashMap::new(), HashMap::new()];
        for (t, chunk) in [&els[..half], &els[half..]].iter().enumerate() {
            for mask in 0u32..(1u32 << half) {
                let subset: Vec<u64> = (0..half)
                    .filter(|i| mask >> i & 1 == 1)
                    .map(|i| chunk[i])
                    .collect();
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
        HalfTables { p, s, r, q, a, b }
    }

    /// Exact bucket size at target e-values lam (length q).
    pub fn bucket_e(&self, lam: &[u64]) -> u64 {
        assert_eq!(lam.len(), self.q);
        let p = self.p;
        let q = self.q;
        let cl = e_to_signed(lam, p);
        let mut total = 0u64;
        for &(j, cb, cnt) in &self.b {
            let j = j as usize;
            if j > self.r || self.r - j > self.s / 2 {
                continue;
            }
            // solve triangular: cl_i = sum_{u+v=i} ca_u cb_v with ca_0 = cb_0 = 1
            let mut ca = [0u64; QCAP + 1];
            ca[0] = 1;
            let mut ok = true;
            for i in 1..=q {
                let mut acc: u64 = 0;
                for u in 0..i {
                    let cbv = if i - u == 0 { 1 } else { cb[i - u - 1] };
                    acc = (acc + (ca[u] as u128 * cbv as u128 % p as u128) as u64) % p;
                }
                let cli = cl[i - 1];
                ca[i] = (cli + p - acc) % p;
                let _ = &mut ok;
            }
            let mut ka = [0u64; QCAP];
            ka[..q].copy_from_slice(&ca[1..=q]);
            if let Some(&na) = self.a.get(&((self.r - j) as u8, ka)) {
                total += na * cnt;
            }
        }
        total
    }
}

/// The rung lambda (Theorem A construction), exp20c conventions:
/// t minimal with 2^t - 1 >= q; b, r0 = divmod(r, 2^t); cosets
/// C_i = { G[(i + j * ncos) mod s] }; S = C_0[..r0] ++ C_1 ++ .. ++ C_b.
/// Returns the e-values (length q) of any member (all members share them).
pub fn rung_lambda_e(p: u64, s: usize, r: usize, q: usize) -> Vec<u64> {
    let els = subgroup(p, s);
    let mut t = 0usize;
    while (1usize << t) - 1 < q {
        t += 1;
    }
    let block = 1usize << t;
    let (b, r0) = (r / block, r % block);
    let ncos = s / block;
    assert!(b + 1 <= ncos || (r0 == 0 && b <= ncos));
    let coset = |i: usize| -> Vec<u64> {
        (0..block).map(|j| els[(i + j * ncos) % s]).collect()
    };
    let mut sset: Vec<u64> = coset(0)[..r0].to_vec();
    for i in 1..=b {
        sset.extend(coset(i));
    }
    assert_eq!(sset.len(), r);
    let c = top_signed_coeffs(&sset, q, p);
    signed_to_e(&c, q, p)
}

/// Exact eps-decomposition of the q=1 bucket at value lam (s <= 32):
/// enumerate all eps in {-1,0,1}^{s/2} with sum eps_i w^i = lam (mod p) by MitM
/// over coordinate halves; return (sum of class sizes, per-weight class counts).
/// The sum must equal the DP bucket N(lam) — the anatomy law.
pub fn decompose_bucket_q1(p: u64, s: usize, r: usize, lam: u64) -> (u64, Vec<u64>) {
    assert!(s <= 32);
    let els = subgroup(p, s);
    let w1 = els[1];
    let half = s / 2;
    let mut pows = Vec::with_capacity(half);
    let mut x = 1u64;
    for _ in 0..half {
        pows.push(x);
        x = (x as u128 * w1 as u128 % p as u128) as u64;
    }
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
    let class_size = |wt: usize| -> u64 {
        if wt > r || (r - wt) % 2 == 1 {
            return 0;
        }
        let z = (half - wt) as u64;
        binom(z, ((r - wt) / 2) as u64)
    };
    let mut per_weight = vec![0u64; half + 1];
    let mut total = 0u64;
    for (val, wsb) in &b {
        let need = (lam % p + p - val % p) % p;
        if let Some(wsa) = a.get(&need) {
            for &wb in wsb {
                for &wa in wsa {
                    let wt = (wa + wb) as usize;
                    let cs = class_size(wt);
                    if cs > 0 {
                        per_weight[wt] += 1;
                        total += cs;
                    }
                }
            }
        }
    }
    (total, per_weight)
}
