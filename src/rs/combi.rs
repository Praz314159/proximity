//! Combinatorial enumeration and the deterministic PRNG that samples
//! it — one home for the three subset walks the crate uses
//! (sequential, unranked for parallel iteration, uniform random) and
//! the one SplitMix64 behind every reproducible sample.

/// One SplitMix64 step: the crate's deterministic PRNG (no `rand`
/// dependency; reproducible in the seed).
pub(crate) fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// SplitMix64 as a stream.
pub(crate) struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        SplitMix64 { state: seed }
    }
    pub(crate) fn next_u64(&mut self) -> u64 {
        splitmix(&mut self.state)
    }
    /// A uniform `k`-subset of `0..n` (partial Fisher–Yates), sorted.
    pub(crate) fn combination(&mut self, n: usize, k: usize) -> Vec<usize> {
        let mut pool: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = i + (self.next_u64() as usize) % (n - i);
            pool.swap(i, j);
        }
        let mut idx = pool[..k].to_vec();
        idx.sort_unstable();
        idx
    }
    /// A word of `len` uniform values below `p`.
    #[cfg(test)]
    pub(crate) fn word(&mut self, p: u64, len: usize) -> Vec<u64> {
        (0..len).map(|_| self.next_u64() % p).collect()
    }
}

/// Call `f` on each `k`-subset of `0..n`, as a sorted index slice.
pub(crate) fn for_each_combination(n: usize, k: usize, mut f: impl FnMut(&[usize])) {
    if k > n {
        return;
    }
    let mut c: Vec<usize> = (0..k).collect();
    loop {
        f(&c);
        let mut i = k as isize - 1;
        while i >= 0 && c[i as usize] == i as usize + n - k {
            i -= 1;
        }
        if i < 0 {
            break;
        }
        let i = i as usize;
        c[i] += 1;
        for j in i + 1..k {
            c[j] = c[j - 1] + 1;
        }
    }
}

/// The `idx`-th `k`-subset of `0..n` in colexicographic order — the
/// random-access form of [`for_each_combination`], for parallel
/// iteration over `0..C(n, k)`.
pub(crate) fn unrank_combination(mut idx: u64, n: u64, k: u64) -> Vec<usize> {
    let mut out = Vec::with_capacity(k as usize);
    let mut remaining = k;
    let mut top = n;
    while remaining > 0 {
        top -= 1;
        let c = crate::field::checked_binom(top, remaining).unwrap_or(u64::MAX);
        if idx >= c {
            out.push(top as usize);
            idx -= c;
            remaining -= 1;
        }
    }
    out.reverse();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unrank_matches_enumeration() {
        let (n, k) = (7u64, 3u64);
        let mut all = Vec::new();
        for_each_combination(n as usize, k as usize, |c| all.push(c.to_vec()));
        let mut unranked: Vec<Vec<usize>> = (0..crate::field::binom(n, k))
            .map(|i| unrank_combination(i, n, k))
            .collect();
        unranked.sort();
        all.sort();
        assert_eq!(unranked, all);
    }
}
