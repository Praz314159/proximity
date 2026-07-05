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

/// Inline per-weight counts: avoids a heap allocation per parsed entry
/// (the JSON/bin ingest touches billions of entries; per-entry Vec
/// allocation dominated the w=12 ingest wall time).
pub(crate) const MAXW: usize = 16;
type Counts = [u64; MAXW];

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

/// Read one binary weight dump pair (`<prefix>.w<w>.norms.bin` u64-le +
/// `.counts.bin` u64-le), invoking `sink` per (norm, counts_by_weight).
fn parse_bin_weight(
    prefix: &str,
    w: usize,
    wmax: usize,
    mut sink: impl FnMut(u64, &[u64]),
) -> Result<u64> {
    let nb = std::fs::read(format!("{prefix}.w{w}.norms.bin"))
        .map_err(|e| Error::Unsupported(format!("read {prefix}.w{w}.norms.bin: {e}")))?;
    let cb = std::fs::read(format!("{prefix}.w{w}.counts.bin"))
        .map_err(|e| Error::Unsupported(format!("read {prefix}.w{w}.counts.bin: {e}")))?;
    if nb.len() != cb.len() || nb.len() % 8 != 0 {
        return Err(Error::Unsupported("bin length mismatch".into()));
    }
    let mut counts = vec![0u64; wmax + 1];
    let n = nb.len() / 8;
    for i in 0..n {
        let norm = u64::from_le_bytes(nb[8 * i..8 * i + 8].try_into().unwrap());
        let c = u64::from_le_bytes(cb[8 * i..8 * i + 8].try_into().unwrap());
        counts[w] = c;
        sink(norm, &counts);
    }
    Ok(n as u64)
}

fn flush_batch(
    batch: &mut Vec<(u64, Counts)>,
    acc: &mut HashMap<u64, (Counts, bool)>,
    s: usize,
    _wmax: usize,
) {
    let partial: Vec<HashMap<u64, (Counts, bool)>> = batch
        .par_chunks(1.max(batch.len() / 128))
        .map(|chunk| {
            let mut local: HashMap<u64, (Counts, bool)> = HashMap::new();
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
                        let entry = local.entry(p).or_insert(([0; MAXW], false));
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
            let entry = acc.entry(p).or_insert(([0; MAXW], false));
            for (w, c) in cs.iter().enumerate() {
                entry.0[w] += c;
            }
            entry.1 |= flag;
        }
    }
    batch.clear();
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
    if wmax >= MAXW {
        return Err(Error::OutOfRange(
            "wmax >= 16 unsupported by inline counts".into(),
        ));
    }
    if !s.is_power_of_two() || s < 4 {
        return Err(Error::Unsupported("power-of-two s >= 4 required".into()));
    }
    let half = (s / 2) as u64;
    let mut acc: HashMap<u64, (Counts, bool)> = HashMap::new();
    let mut stats = IngestStats {
        mass_by_weight: vec![0; wmax + 1],
        n_max_by_weight: vec![0; wmax + 1],
        entries_parsed: 0,
    };
    for path in paths {
        if !path.ends_with(".json") {
            // binary prefix: ingest every existing per-weight dump
            let mut batch: Vec<(u64, Counts)> = Vec::with_capacity(1 << 20);
            for w in 1..=wmax {
                if !std::path::Path::new(&format!("{path}.w{w}.norms.bin")).exists() {
                    continue;
                }
                stats.entries_parsed += parse_bin_weight(path, w, wmax, |n, counts| {
                    for (wi, &c) in counts.iter().enumerate() {
                        if c > 0 {
                            stats.mass_by_weight[wi] += c;
                            if n > stats.n_max_by_weight[wi] {
                                stats.n_max_by_weight[wi] = n;
                            }
                        }
                    }
                    let mut c = [0u64; MAXW];
                    c[..counts.len()].copy_from_slice(counts);
                    batch.push((n, c));
                })?;
            }
            flush_batch(&mut batch, &mut acc, s, wmax);
            continue;
        }
        let buf =
            std::fs::read(path).map_err(|e| Error::Unsupported(format!("read {path}: {e}")))?;
        // collect entries in batches, factor in parallel
        let mut batch: Vec<(u64, Counts)> = Vec::with_capacity(1 << 20);

        stats.entries_parsed += parse_shard(&buf, wmax, |n, counts| {
            for (w, &c) in counts.iter().enumerate() {
                stats.mass_by_weight[w] += c;
                if c > 0 && n > stats.n_max_by_weight[w] {
                    stats.n_max_by_weight[w] = n;
                }
            }
            let mut c = [0u64; MAXW];
            c[..counts.len()].copy_from_slice(counts);
            batch.push((n, c));
        })?;
        flush_batch(&mut batch, &mut acc, s, wmax);
    }
    let mut out: Vec<IngestEntry> = acc
        .into_iter()
        .map(|(p, (val_counts, mut flag))| {
            let counts: Vec<u64> = val_counts[..=wmax]
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

    /// The binary-dump ingest path must produce the identical bad set.
    #[test]
    fn ingest_bin_matches_bad_set_s16() {
        let t = norm_table(16, 8, 1).unwrap();
        let dir = std::env::temp_dir();
        let prefix = dir.join("vanish_ingest_bin_test");
        let prefix = prefix.to_str().unwrap();
        // group entries per weight, dump as (norms, counts) u64-le pairs
        for w in 1..=8usize {
            let mut nb: Vec<u8> = Vec::new();
            let mut cb: Vec<u8> = Vec::new();
            for (&n, counts) in &t.entries {
                if counts[w] > 0 {
                    nb.extend_from_slice(&(u64::try_from(n).unwrap()).to_le_bytes());
                    cb.extend_from_slice(&counts[w].to_le_bytes());
                }
            }
            std::fs::write(format!("{prefix}.w{w}.norms.bin"), nb).unwrap();
            std::fs::write(format!("{prefix}.w{w}.counts.bin"), cb).unwrap();
        }
        let (rows, stats) = badset_from_gpu_json(&[prefix.to_string()], 16, 8).unwrap();
        for w in 1..=8usize {
            assert_eq!(stats.mass_by_weight[w], binom(8, w) * (1u64 << w));
        }
        let reference = bad_set(16, 8, 1).unwrap();
        assert_eq!(rows.len(), reference.len());
        for (a, b) in rows.iter().zip(reference.iter()) {
            assert_eq!(a.p, b.p);
            if !b.census_fallback {
                assert_eq!(a.counts, b.counts, "counts at p={}", a.p);
            } else {
                assert!(a.unsafe_split);
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
