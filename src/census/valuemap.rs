//! `census::valuemap` — the meet-in-the-middle value-map census kernel.
//!
//! The workhorse behind the shell and rung censuses (stages 49/50/54,
//! the four rung computations, the collision instruments): subset
//! products `prod (point - dom[e])` over all subsets of a half
//! exponent list, grouped by `(subset size, exponent sum mod level)`,
//! joined across two halves under a size-and-sum constraint, and
//! reduced to fiber statistics — total, distinct values, maximum
//! fiber, and the exact second moment (the collision count, the
//! quantity of the program's open core).
//!
//! Scope: in-memory MITM, half lists up to 20 exponents (the
//! `s <= 32` regime; the level-64 halves at `2^31` entries are the
//! GPU pod's territory, not this kernel's). The Python mirror is
//! `experiments/analysis/toolkit.py::half_tables`; the golden pins
//! here were generated from it at the standard prime and are
//! cross-checked exactly in `Z[zeta_s]` via [`Cyclo`] glue tests.

use crate::census::join::SortedMultiMap;
use crate::error::{Error, Result};
use crate::field::mulmod;
use rayon::prelude::*;

/// Grouped half-products over all subsets of one exponent list:
/// `groups[size][sigma] = (products mod p, subset masks)`.
pub struct HalfTable {
    level: usize,
    hexp: Vec<usize>,
    groups: Vec<Vec<(Vec<u64>, Vec<u32>)>>,
}

/// Build the half table of `prod_{e in mask} (point - dom[e]) mod p`
/// over all `2^n` subsets of `hexp` (indices into `dom`, the
/// consecutive-powers list of the order-`level` subgroup).
pub fn half_table(
    dom: &[u64],
    hexp: &[usize],
    level: usize,
    point: u64,
    p: u64,
) -> Result<HalfTable> {
    let n = hexp.len();
    if n == 0 || n > 20 {
        return Err(Error::OutOfRange(
            "half_table requires 1..=20 exponents (in-memory MITM scope)".into(),
        ));
    }
    if dom.len() != level {
        return Err(Error::OutOfRange(
            "dom must list the level consecutive subgroup powers".into(),
        ));
    }
    let full = 1usize << n;
    let vals: Vec<u64> = hexp
        .iter()
        .map(|&e| (point + p - dom[e % level] % p) % p)
        .collect();
    let mut q = vec![0u64; full];
    let mut esum = vec![0u32; full];
    q[0] = 1 % p;
    for m in 1..full {
        let b = m & m.wrapping_neg();
        let i = b.trailing_zeros() as usize;
        let r = m ^ b;
        q[m] = mulmod(q[r], vals[i], p);
        esum[m] = (esum[r] + hexp[i] as u32) % level as u32;
    }
    let mut groups = vec![vec![(Vec::new(), Vec::new()); level]; n + 1];
    for (m, (&qm, &sm)) in q.iter().zip(esum.iter()).enumerate() {
        let j = (m as u32).count_ones() as usize;
        let (ps, ms) = &mut groups[j][sm as usize];
        ps.push(qm);
        ms.push(m as u32);
    }
    Ok(HalfTable {
        level,
        hexp: hexp.to_vec(),
        groups,
    })
}

/// Fiber statistics of a joined census: the census in five numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CensusSummary {
    /// Number of subsets in the ensemble (values counted with
    /// multiplicity).
    pub total: u64,
    /// Number of distinct values.
    pub distinct: u64,
    /// The census maximum: the largest fiber.
    pub max_fiber: u64,
    /// A value attaining the maximum fiber.
    pub argmax: u64,
    /// `sum fiber^2` — the exact collision count (ordered pairs of
    /// subsets sharing a value), the open core's quantity.
    pub second_moment: u128,
}

/// One `(size, sigma)` group of a half table: products and masks.
type Group = (Vec<u64>, Vec<u32>);

fn matched_pairs<'t>(
    a: &'t HalfTable,
    b: &'t HalfTable,
    size: usize,
    class: usize,
) -> Result<Vec<(&'t Group, &'t Group)>> {
    if a.level != b.level {
        return Err(Error::OutOfRange("mismatched levels".into()));
    }
    let level = a.level;
    let mut out = Vec::new();
    for (j, row) in a.groups.iter().enumerate() {
        if j > size || size - j > b.hexp.len() {
            continue;
        }
        for (sigma, ga) in row.iter().enumerate() {
            if ga.0.is_empty() {
                continue;
            }
            let gb = &b.groups[size - j][(level + class - sigma % level) % level];
            if gb.0.is_empty() {
                continue;
            }
            out.push((ga, gb));
        }
    }
    Ok(out)
}

/// The full value-resolved census: `(distinct values sorted,
/// multiplicities)` — the data every distribution figure and the
/// character-spectrum pipeline consume. Rayon-parallel; memory is
/// one `u64` per ensemble member plus the output.
pub fn join_distribution(
    a: &HalfTable,
    b: &HalfTable,
    size: usize,
    class: usize,
    p: u64,
) -> Result<(Vec<u64>, Vec<u64>)> {
    let pairs = matched_pairs(a, b, size, class)?;
    let mut vals: Vec<u64> = pairs
        .par_iter()
        .flat_map_iter(|(ga, gb)| {
            ga.0.iter()
                .flat_map(move |&x| gb.0.iter().map(move |&y| mulmod(x, y, p)))
                .collect::<Vec<u64>>()
        })
        .collect();
    vals.par_sort_unstable();
    let mut values = Vec::new();
    let mut counts = Vec::new();
    let mut i = 0usize;
    while i < vals.len() {
        let v = vals[i];
        let mut j = i + 1;
        while j < vals.len() && vals[j] == v {
            j += 1;
        }
        values.push(v);
        counts.push((j - i) as u64);
        i = j;
    }
    Ok((values, counts))
}

/// The fiber-size histogram: `out[k]` = number of values with fiber
/// size exactly `k` (index 0 unused). The lightest full-shape output
/// for sweeps and CCDF figures.
pub fn join_histogram(
    a: &HalfTable,
    b: &HalfTable,
    size: usize,
    class: usize,
    p: u64,
) -> Result<Vec<u64>> {
    let (_, counts) = join_distribution(a, b, size, class, p)?;
    let max = counts.iter().copied().max().unwrap_or(0) as usize;
    let mut hist = vec![0u64; max + 1];
    for &c in &counts {
        hist[c as usize] += 1;
    }
    Ok(hist)
}

/// Join two half tables under `(size, sum class)` and reduce to fiber
/// statistics.
pub fn join_census(
    a: &HalfTable,
    b: &HalfTable,
    size: usize,
    class: usize,
    p: u64,
) -> Result<CensusSummary> {
    let (values, counts) = join_distribution(a, b, size, class, p)?;
    let mut total = 0u64;
    let (mut max_fiber, mut argmax) = (0u64, 0u64);
    let mut second_moment = 0u128;
    for (&v, &f) in values.iter().zip(&counts) {
        total += f;
        second_moment += (f as u128) * (f as u128);
        if f > max_fiber {
            max_fiber = f;
            argmax = v;
        }
    }
    Ok(CensusSummary {
        total,
        distinct: values.len() as u64,
        max_fiber,
        argmax,
        second_moment,
    })
}

/// The fiber size of one target value (no large allocation).
pub fn fiber_count(
    a: &HalfTable,
    b: &HalfTable,
    size: usize,
    class: usize,
    p: u64,
    value: u64,
) -> Result<u64> {
    let pairs = matched_pairs(a, b, size, class)?;
    Ok(pairs
        .par_iter()
        .map(|(ga, gb)| {
            let tab = SortedMultiMap::new(gb.0.iter().map(|&y| (y, ())).collect());
            ga.0.iter()
                .map(|&x| {
                    // x * y = value  =>  y = value * x^{-1}
                    let y = mulmod(value, crate::field::powmod(x, p - 2, p), p);
                    tab.get(&y).len() as u64
                })
                .sum::<u64>()
        })
        .sum())
}

/// The members of one fiber, as sorted exponent lists (capped).
pub fn fiber_members(
    a: &HalfTable,
    b: &HalfTable,
    size: usize,
    class: usize,
    p: u64,
    value: u64,
    cap: usize,
) -> Result<Vec<Vec<usize>>> {
    let pairs = matched_pairs(a, b, size, class)?;
    let mut out = Vec::new();
    'outer: for (ga, gb) in pairs {
        let tab = SortedMultiMap::new(gb.0.iter().zip(&gb.1).map(|(&y, &mb)| (y, mb)).collect());
        for (&x, &ma) in ga.0.iter().zip(&ga.1) {
            // x * y = value  =>  y = value * x^{-1}
            let y = mulmod(value, crate::field::powmod(x, p - 2, p), p);
            for &mb in tab.get(&y) {
                let mut exps: Vec<usize> = Vec::with_capacity(size);
                for (t, &e) in a.hexp.iter().enumerate() {
                    if ma >> t & 1 == 1 {
                        exps.push(e);
                    }
                }
                for (t, &e) in b.hexp.iter().enumerate() {
                    if mb >> t & 1 == 1 {
                        exps.push(e);
                    }
                }
                exps.sort_unstable();
                out.push(exps);
                if out.len() >= cap {
                    break 'outer;
                }
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::MultiplicativeSubgroup;
    use crate::ring::Cyclo;

    const P: u64 = 2_130_706_433;

    fn dom32() -> Vec<u64> {
        MultiplicativeSubgroup::new(P, 32)
            .unwrap()
            .elements()
            .to_vec()
    }

    #[test]
    fn shell_census_golden() {
        // stage 49/50/54 pins, regenerated from the python toolkit at
        // the standard prime (2026-07-27)
        let dom = dom32();
        let h1: Vec<usize> = (1..16).collect();
        let h2: Vec<usize> = (17..32).collect();
        let a = half_table(&dom, &h1, 32, 1, P).unwrap();
        let b = half_table(&dom, &h2, 32, 1, P).unwrap();
        let c = join_census(&a, &b, 16, 0, P).unwrap();
        assert_eq!(c.total, 4_544_445);
        assert_eq!(c.distinct, 275_247);
        assert_eq!(c.max_fiber, 1_250);
        assert_eq!(c.argmax, 4);
        assert_eq!(c.second_moment, 448_183_873);
        assert_eq!(fiber_count(&a, &b, 16, 0, P, 4).unwrap(), 1_250);

        // distribution and histogram agree with the summary exactly
        let (values, counts) = join_distribution(&a, &b, 16, 0, P).unwrap();
        assert_eq!(values.len() as u64, c.distinct);
        assert_eq!(counts.iter().sum::<u64>(), c.total);
        let hist = join_histogram(&a, &b, 16, 0, P).unwrap();
        assert_eq!(hist.len(), 1_251);
        assert_eq!(hist[1_250], 1);
        assert_eq!(hist.iter().sum::<u64>(), c.distinct);
        assert_eq!(
            hist.iter()
                .enumerate()
                .map(|(k, &n)| (k as u128) * (k as u128) * (n as u128))
                .sum::<u128>(),
            c.second_moment
        );

        // glue: a fiber member verifies exactly in Z[zeta_32]
        let members = fiber_members(&a, &b, 16, 0, P, 4, 3).unwrap();
        assert_eq!(members.len(), 3);
        for m in &members {
            assert_eq!(m.len(), 16);
            assert_eq!(m.iter().sum::<usize>() % 32, 0);
            assert!(Cyclo::prod_one_minus(32, m).unwrap().eq_int(4));
        }
    }

    #[test]
    fn rung_census_golden() {
        // the level-32 rung-ensemble class-1 census (stage 63 pins)
        let dom = dom32();
        let h1: Vec<usize> = (1..16).collect();
        let h2: Vec<usize> = (16..32).collect();
        let a = half_table(&dom, &h1, 32, 1, P).unwrap();
        let b = half_table(&dom, &h2, 32, 1, P).unwrap();
        let c = join_census(&a, &b, 16, 1, P).unwrap();
        assert_eq!(c.total, 9_391_680);
        assert_eq!(c.distinct, 550_667);
        assert_eq!(c.max_fiber, 952);
        assert_eq!(c.second_moment, 1_117_081_898);
        // the unit-equation target 8(1 + g): the stage-63 merger
        // measurement's fiber
        let target = (8 * (1 + dom[1])) % P;
        assert_eq!(fiber_count(&a, &b, 16, 1, P, target).unwrap(), 892);
    }
}
