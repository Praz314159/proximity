//! Ingestion of externally computed norm tables (e.g. the GPU campaign's
//! JSON output) into bad sets — the s = 64 pipeline.
//!
//! The GPU sweep emits `{"<norm>": {"<w>": count, ...}, ...}` per shard
//! (shards partition supports, so the same norm may appear in several
//! shards; counts add). Files are gigabytes, so parsing is streaming: a
//! hand-rolled scanner over the byte buffer feeds (norm, counts) entries
//! straight into parallel factoring without materializing the table.
//!
//! Normalization follows `norms::bad_set` exactly (valuation / (s/2);
//! per-weight censuses are Galois-invariant), except that primes where the
//! equal split is unsafe (`p^2 | N(v)`, or a valuation not divisible by
//! s/2) are *flagged* rather than census-corrected — at s = 64 the direct
//! census fallback is not yet feasible, and the flags mark exactly the
//! rows a downstream analysis must treat as approximate.

use crate::error::{Error, Result};
use crate::field::factor;
use rayon::prelude::*;
use std::collections::HashMap;

/// Accumulated bad-set row (pre-normalization counts are valuation-weighted).
#[derive(Debug, Clone)]
pub struct IngestEntry {
    /// The prime.
    pub p: u64,
    /// Galois-normalized per-weight kernel-vector counts (index = weight).
    pub counts: Vec<u64>,
    /// True when the equal-split normalization is unsafe for this prime
    /// (some norm divisible by p^2, or a non-divisible valuation sum).
    pub unsafe_split: bool,
}

/// Global invariants returned alongside the bad set, for validation.
#[derive(Debug, Clone)]
pub struct IngestStats {
    /// Sum over norms of per-weight vector counts; must equal
    /// C(s/2, w) * 2^w when shards are complete (cmax = 1).
    pub mass_by_weight: Vec<u64>,
    /// Largest norm observed at each weight (anticorrelation profile).
    pub n_max_by_weight: Vec<u64>,
    /// Number of distinct (norm, shard-occurrence) entries parsed.
    pub entries_parsed: u64,
}

struct Scanner<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Scanner<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.buf.len()
            && matches!(self.buf[self.pos], b' ' | b'\t' | b'\n' | b'\r')
        {
            self.pos += 1;
        }
    }
    fn expect(&mut self, c: u8) -> Result<()> {
        self.skip_ws();
        if self.pos < self.buf.len() && self.buf[self.pos] == c {
            self.pos += 1;
            Ok(())
        } else {
            Err(Error::Unsupported(format!(
                "JSON parse: expected '{}' at byte {}",
                c as char, self.pos
            )))
        }
    }
    fn peek(&mut self) -> u8 {
        self.skip_ws();
        if self.pos < self.buf.len() {
            self.buf[self.pos]
        } else {
            0
        }
    }
    /// Parse a quoted decimal string as u64.
    fn quoted_u64(&mut self) -> Result<u64> {
        self.expect(b'"')?;
        let mut v: u64 = 0;
        while self.pos < self.buf.len() && self.buf[self.pos].is_ascii_digit() {
            v = v
                .checked_mul(10)
                .and_then(|x| x.checked_add((self.buf[self.pos] - b'0') as u64))
                .ok_or_else(|| Error::Unsupported("norm exceeds u64".into()))?;
            self.pos += 1;
        }
        self.expect(b'"')?;
        Ok(v)
    }
    /// Parse a bare decimal number as u64 (integer counts).
    fn bare_u64(&mut self) -> Result<u64> {
        self.skip_ws();
        let start = self.pos;
        let mut v: u64 = 0;
        while self.pos < self.buf.len() && self.buf[self.pos].is_ascii_digit() {
            v = v * 10 + (self.buf[self.pos] - b'0') as u64;
            self.pos += 1;
        }
        if self.pos == start {
            return Err(Error::Unsupported(format!(
                "JSON parse: expected number at byte {}",
                self.pos
            )));
        }
        // GPU counts arrive as floats ("123.0") from the f64 accumulator.
        if self.peek() == b'.' {
            self.pos += 1;
            while self.pos < self.buf.len() && self.buf[self.pos].is_ascii_digit() {
                self.pos += 1;
            }
        }
        Ok(v)
    }
}

/// Parse one shard buffer, invoking `sink` per (norm, counts_by_weight).
fn parse_shard(buf: &[u8], wmax: usize, mut sink: impl FnMut(u64, &[u64])) -> Result<u64> {
    let mut sc = Scanner { buf, pos: 0 };
    let mut n_entries = 0u64;
    sc.expect(b'{')?;
    if sc.peek() == b'}' {
        return Ok(0);
    }
    let mut counts = vec![0u64; wmax + 1];
    loop {
        let norm = sc.quoted_u64()?;
        sc.expect(b':')?;
        sc.expect(b'{')?;
        counts.iter_mut().for_each(|c| *c = 0);
        if sc.peek() != b'}' {
            loop {
                let w = sc.quoted_u64()? as usize;
                sc.expect(b':')?;
                let c = sc.bare_u64()?;
                if w > wmax {
                    return Err(Error::OutOfRange(format!("weight {w} > wmax {wmax}")));
                }
                counts[w] += c;
                if sc.peek() == b',' {
                    sc.pos += 1;
                } else {
                    break;
                }
            }
        }
        sc.expect(b'}')?;
        sink(norm, &counts);
        n_entries += 1;
        if sc.peek() == b',' {
            sc.pos += 1;
        } else {
            break;
        }
    }
    sc.expect(b'}')?;
    Ok(n_entries)
}

/// Ingest GPU-campaign shard files into a bad set.
///
/// Streams each file, factors every norm in parallel, keeps primes
/// `p = 1 mod s`, `p > s`, and Galois-normalizes valuation-weighted counts.
pub fn badset_from_gpu_json(
    paths: &[String],
    s: usize,
    wmax: usize,
) -> Result<(Vec<IngestEntry>, IngestStats)> {
    if !s.is_power_of_two() || s < 4 {
        return Err(Error::Unsupported("power-of-two s >= 4 required".into()));
    }
    let half = (s / 2) as u64;
    let mut acc: HashMap<u64, (Vec<u64>, bool)> = HashMap::new();
    let mut stats = IngestStats {
        mass_by_weight: vec![0; wmax + 1],
        n_max_by_weight: vec![0; wmax + 1],
        entries_parsed: 0,
    };
    for path in paths {
        let buf =
            std::fs::read(path).map_err(|e| Error::Unsupported(format!("read {path}: {e}")))?;
        // collect entries in batches, factor in parallel
        let mut batch: Vec<(u64, Vec<u64>)> = Vec::with_capacity(1 << 20);
        let flush = |batch: &mut Vec<(u64, Vec<u64>)>,
                         acc: &mut HashMap<u64, (Vec<u64>, bool)>| {
            let partial: Vec<HashMap<u64, (Vec<u64>, bool)>> = batch
                .par_chunks(1.max(batch.len() / 128))
                .map(|chunk| {
                    let mut local: HashMap<u64, (Vec<u64>, bool)> = HashMap::new();
                    for (n, counts) in chunk {
                        if *n <= 1 {
                            continue;
                        }
                        let fs = factor(*n);
                        let mut i = 0;
                        while i < fs.len() {
                            let p = fs[i];
                            let mut e = 0u64;
                            while i < fs.len() && fs[i] == p {
                                e += 1;
                                i += 1;
                            }
                            if p > s as u64 && (p - 1) % s as u64 == 0 {
                                let entry = local
                                    .entry(p)
                                    .or_insert_with(|| (vec![0; counts.len()], false));
                                for (w, &c) in counts.iter().enumerate() {
                                    entry.0[w] += e * c;
                                }
                                if e >= 2 {
                                    entry.1 = true;
                                }
                            }
                        }
                    }
                    local
                })
                .collect();
            for m in partial {
                for (p, (cs, flag)) in m {
                    let entry = acc.entry(p).or_insert_with(|| (vec![0; cs.len()], false));
                    for (w, c) in cs.iter().enumerate() {
                        entry.0[w] += c;
                    }
                    entry.1 |= flag;
                }
            }
            batch.clear();
        };
        stats.entries_parsed += parse_shard(&buf, wmax, |n, counts| {
            for (w, &c) in counts.iter().enumerate() {
                stats.mass_by_weight[w] += c;
                if c > 0 && n > stats.n_max_by_weight[w] {
                    stats.n_max_by_weight[w] = n;
                }
            }
            batch.push((n, counts.to_vec()));
        })?;
        flush(&mut batch, &mut acc);
    }
    let mut out: Vec<IngestEntry> = acc
        .into_iter()
        .map(|(p, (val_counts, mut flag))| {
            let counts: Vec<u64> = val_counts
                .iter()
                .map(|&v| {
                    if v % half != 0 {
                        flag = true;
                    }
                    v / half
                })
                .collect();
            IngestEntry {
                p,
                counts,
                unsafe_split: flag,
            }
        })
        .collect();
    out.sort_by_key(|e| e.p);
    Ok((out, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::norms::{bad_set, norm_table};

    /// The ingest path must reproduce `bad_set` exactly on the golden
    /// s = 16 landscape when fed the CPU table serialized as GPU JSON
    /// (excluding census-fallback rows, which ingest only flags).
    #[test]
    fn ingest_matches_bad_set_s16() {
        let t = norm_table(16, 8, 1).unwrap();
        let mut js = String::from("{");
        for (i, (n, counts)) in t.entries.iter().enumerate() {
            if i > 0 {
                js.push(',');
            }
            js.push_str(&format!("\"{n}\": {{"));
            let mut first = true;
            for (w, &c) in counts.iter().enumerate() {
                if c > 0 {
                    if !first {
                        js.push(',');
                    }
                    js.push_str(&format!("\"{w}\": {c}.0"));
                    first = false;
                }
            }
            js.push('}');
        }
        js.push('}');
        let tmp = std::env::temp_dir().join("vanish_ingest_test.json");
        std::fs::write(&tmp, js).unwrap();
        let (rows, stats) = badset_from_gpu_json(&[tmp.to_str().unwrap().into()], 16, 8).unwrap();
        // mass invariant: sum_w counts = C(8,w) * 2^w
        for w in 1..=8usize {
            let expect = binom(8, w) * (1u64 << w);
            assert_eq!(stats.mass_by_weight[w], expect, "mass at w={w}");
        }
        let reference = bad_set(16, 8, 1).unwrap();
        assert_eq!(rows.len(), reference.len(), "same prime set");
        for (a, b) in rows.iter().zip(reference.iter()) {
            assert_eq!(a.p, b.p);
            if !b.census_fallback {
                assert_eq!(a.counts, b.counts, "counts at p={}", a.p);
                assert!(!a.unsafe_split);
            } else {
                assert!(a.unsafe_split, "p={} must be flagged", a.p);
            }
        }
    }

    fn binom(n: u64, k: usize) -> u64 {
        let mut r = 1u64;
        for i in 0..k as u64 {
            r = r * (n - i) / (i + 1);
        }
        r
    }
}
