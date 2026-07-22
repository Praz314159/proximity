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

use super::{for_each_bad_prime, BadSetEntry, Provenance};
use crate::error::{Error, Result};
use rayon::prelude::*;
use std::collections::HashMap;

/// Inline per-weight counts: avoids a heap allocation per parsed entry
/// (the JSON/bin ingest touches billions of entries; per-entry Vec
/// allocation dominated the w=12 ingest wall time).
pub(crate) const MAXW: usize = 16;
type Counts = [u64; MAXW];

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
            Err(Error::MalformedInput(format!(
                "JSON: expected '{}' at byte {}",
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
                .ok_or_else(|| Error::MalformedInput("norm exceeds u64".into()))?;
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
            return Err(Error::MalformedInput(format!(
                "JSON: expected number at byte {}",
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
    let read = |path: String| std::fs::read(&path).map_err(|source| Error::Io { path, source });
    let nb = read(format!("{prefix}.w{w}.norms.bin"))?;
    let cb = read(format!("{prefix}.w{w}.counts.bin"))?;
    if nb.len() != cb.len() || nb.len() % 8 != 0 {
        return Err(Error::MalformedInput(format!(
            "{prefix}.w{w}: norms/counts length mismatch or not 8-byte aligned"
        )));
    }
    let mut counts = vec![0u64; wmax + 1];
    let n = nb.len() / 8;
    for i in 0..n {
        let norm = u64::from_le_bytes(
            nb[8 * i..8 * i + 8]
                .try_into()
                .expect("slice is exactly 8 bytes"),
        );
        let c = u64::from_le_bytes(
            cb[8 * i..8 * i + 8]
                .try_into()
                .expect("slice is exactly 8 bytes"),
        );
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
                for_each_bad_prime(*n, s as u64, |p, e| {
                    let entry = local.entry(p).or_insert(([0; MAXW], false));
                    for (w, &c) in counts.iter().enumerate() {
                        entry.0[w] += e * c;
                    }
                    if e >= 2 {
                        entry.1 = true;
                    }
                });
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

/// Checkpoint state: the accumulator plus everything needed to resume.
struct Checkpoint {
    acc: HashMap<u64, (Counts, bool)>,
    stats: IngestStats,
    done_paths: Vec<String>,
}

fn ckpt_names(prefix: &str) -> (String, String) {
    (format!("{prefix}.ckpt.bin"), format!("{prefix}.ckpt.meta"))
}

/// Atomically persist the accumulator after a completed shard. A failure
/// here is an early disk-space canary: it fires at shard granularity
/// instead of after the final hour of factoring.
fn save_checkpoint(
    prefix: &str,
    acc: &HashMap<u64, (Counts, bool)>,
    stats: &IngestStats,
    done_paths: &[String],
    wmax: usize,
) -> Result<()> {
    let (bin, meta) = ckpt_names(prefix);
    let werr = |path: &str| {
        let path = path.to_string();
        move |source: std::io::Error| Error::Io { path, source }
    };
    let row = 8 + (wmax + 1) * 8 + 1;
    let mut buf = Vec::with_capacity(acc.len() * row);
    for (&p, (counts, flag)) in acc {
        buf.extend_from_slice(&p.to_le_bytes());
        for &c in &counts[..=wmax] {
            buf.extend_from_slice(&c.to_le_bytes());
        }
        buf.push(u8::from(*flag));
    }
    let tmp = format!("{bin}.tmp");
    std::fs::write(&tmp, &buf).map_err(werr(&tmp))?;
    std::fs::rename(&tmp, &bin).map_err(werr(&bin))?;
    let mut m = format!(
        "wmax {}\nentries {}\nmass {}\nnmax {}\n",
        wmax,
        stats.entries_parsed,
        stats
            .mass_by_weight
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(","),
        stats
            .n_max_by_weight
            .iter()
            .map(u64::to_string)
            .collect::<Vec<_>>()
            .join(","),
    );
    for p in done_paths {
        m.push_str("done ");
        m.push_str(p);
        m.push('\n');
    }
    let tmpm = format!("{meta}.tmp");
    std::fs::write(&tmpm, m).map_err(werr(&tmpm))?;
    std::fs::rename(&tmpm, &meta).map_err(werr(&meta))?;
    Ok(())
}

fn load_checkpoint(prefix: &str, wmax: usize) -> Option<Checkpoint> {
    let (bin, meta) = ckpt_names(prefix);
    let m = std::fs::read_to_string(&meta).ok()?;
    let buf = std::fs::read(&bin).ok()?;
    let mut stats = IngestStats {
        mass_by_weight: vec![0; wmax + 1],
        n_max_by_weight: vec![0; wmax + 1],
        entries_parsed: 0,
    };
    let mut done_paths = Vec::new();
    let mut ck_wmax = usize::MAX;
    for line in m.lines() {
        let (key, val) = line.split_once(' ')?;
        match key {
            "wmax" => ck_wmax = val.parse().ok()?,
            "entries" => stats.entries_parsed = val.parse().ok()?,
            "mass" => {
                stats.mass_by_weight = val
                    .split(',')
                    .map(str::parse)
                    .collect::<std::result::Result<_, _>>()
                    .ok()?
            }
            "nmax" => {
                stats.n_max_by_weight = val
                    .split(',')
                    .map(str::parse)
                    .collect::<std::result::Result<_, _>>()
                    .ok()?
            }
            "done" => done_paths.push(val.to_string()),
            _ => return None,
        }
    }
    if ck_wmax != wmax || stats.mass_by_weight.len() != wmax + 1 {
        return None;
    }
    let row = 8 + (wmax + 1) * 8 + 1;
    if buf.len() % row != 0 {
        return None;
    }
    let mut acc = HashMap::with_capacity(buf.len() / row);
    for chunk in buf.chunks_exact(row) {
        let p = u64::from_le_bytes(chunk[..8].try_into().expect("record is at least 8 bytes"));
        let mut counts = [0u64; MAXW];
        for (w, c) in counts.iter_mut().enumerate().take(wmax + 1) {
            let off = 8 + w * 8;
            *c = u64::from_le_bytes(
                chunk[off..off + 8]
                    .try_into()
                    .expect("slice is exactly 8 bytes"),
            );
        }
        let flag = *chunk.last().expect("chunks_exact yields non-empty records") != 0;
        acc.insert(p, (counts, flag));
    }
    Some(Checkpoint {
        acc,
        stats,
        done_paths,
    })
}

/// Delete a run's checkpoint files. Call after the caller has durably
/// written its outputs; until then the checkpoint is the crash-recovery
/// state for the whole factoring run.
pub fn clear_checkpoint(prefix: &str) {
    let (bin, meta) = ckpt_names(prefix);
    let _ = std::fs::remove_file(bin);
    let _ = std::fs::remove_file(meta);
}

/// Ingest GPU-campaign shard files into a bad set.
///
/// Streams each file, factors every norm in parallel, keeps primes
/// `p = 1 mod s`, `p > s`, and Galois-normalizes valuation-weighted counts.
///
/// With `ckpt_prefix` set, the accumulator is persisted after every
/// completed shard and a matching checkpoint on disk resumes the run,
/// re-factoring at most one shard. The checkpoint survives this function —
/// call [`clear_checkpoint`] once downstream outputs are safely written.
pub fn badset_from_gpu_json(
    paths: &[String],
    s: usize,
    wmax: usize,
    ckpt_prefix: Option<&str>,
) -> Result<(Vec<BadSetEntry>, IngestStats)> {
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
    let mut done_paths: Vec<String> = Vec::new();
    if let Some(prefix) = ckpt_prefix {
        if let Some(ck) = load_checkpoint(prefix, wmax) {
            if ck.done_paths.iter().all(|p| paths.contains(p)) {
                eprintln!(
                    "[ingest] resuming from checkpoint: {} primes, {} shard(s) done",
                    ck.acc.len(),
                    ck.done_paths.len()
                );
                acc = ck.acc;
                stats = ck.stats;
                done_paths = ck.done_paths;
            } else {
                eprintln!("[ingest] checkpoint does not match requested paths; starting fresh");
            }
        }
    }
    // Incremental flushing caps the batch at ~1M entries (a whole-shard batch
    // costs tens of GB at w = 12) and provides progress heartbeats on stderr.
    const FLUSH_AT: usize = 1 << 20;
    let mut done: u64 = 0;
    for path in paths {
        if done_paths.contains(path) {
            continue;
        }
        if !path.ends_with(".json") {
            // binary prefix: ingest every existing per-weight dump
            let mut batch: Vec<(u64, Counts)> = Vec::with_capacity(FLUSH_AT);
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
                    if batch.len() >= FLUSH_AT {
                        done += batch.len() as u64;
                        flush_batch(&mut batch, &mut acc, s, wmax);
                        if done % (64 << 20) < FLUSH_AT as u64 {
                            eprintln!("[ingest] {done} entries factored, {} bad primes", acc.len());
                        }
                    }
                })?;
            }
            done += batch.len() as u64;
            flush_batch(&mut batch, &mut acc, s, wmax);
            eprintln!(
                "[ingest] {path}: done ({done} entries, {} primes)",
                acc.len()
            );
            if let Some(prefix) = ckpt_prefix {
                done_paths.push(path.clone());
                save_checkpoint(prefix, &acc, &stats, &done_paths, wmax)?;
                eprintln!(
                    "[ingest] checkpoint saved ({} shards done)",
                    done_paths.len()
                );
            }
            continue;
        }
        let buf = std::fs::read(path).map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        let mut batch: Vec<(u64, Counts)> = Vec::with_capacity(FLUSH_AT);
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
            if batch.len() >= FLUSH_AT {
                done += batch.len() as u64;
                flush_batch(&mut batch, &mut acc, s, wmax);
                if done % (64 << 20) < FLUSH_AT as u64 {
                    eprintln!("[ingest] {done} entries factored, {} bad primes", acc.len());
                }
            }
        })?;
        done += batch.len() as u64;
        flush_batch(&mut batch, &mut acc, s, wmax);
        eprintln!(
            "[ingest] {path}: done ({done} entries, {} primes)",
            acc.len()
        );
        if let Some(prefix) = ckpt_prefix {
            done_paths.push(path.clone());
            save_checkpoint(prefix, &acc, &stats, &done_paths, wmax)?;
            eprintln!(
                "[ingest] checkpoint saved ({} shards done)",
                done_paths.len()
            );
        }
    }
    let mut out: Vec<BadSetEntry> = acc
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
            BadSetEntry {
                p,
                counts,
                // no census fallback exists at ingest scale: unsafe splits
                // are flagged, never corrected
                provenance: if flag {
                    Provenance::UnsafeSplit
                } else {
                    Provenance::ValuationSplit
                },
            }
        })
        .collect();
    out.sort_by_key(|e| e.p);
    Ok((out, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smooth::norms::{bad_set, norm_table};

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
        let (rows, stats) =
            badset_from_gpu_json(&[tmp.to_str().unwrap().into()], 16, 8, None).unwrap();
        // mass invariant: sum_w counts = C(8,w) * 2^w
        for w in 1..=8usize {
            let expect = binom(8, w) * (1u64 << w);
            assert_eq!(stats.mass_by_weight[w], expect, "mass at w={w}");
        }
        let reference = bad_set(16, 8, 1).unwrap();
        assert_eq!(rows.len(), reference.len(), "same prime set");
        for (a, b) in rows.iter().zip(reference.iter()) {
            assert_eq!(a.p, b.p);
            if b.provenance != Provenance::CensusCorrected {
                assert_eq!(a.counts, b.counts, "counts at p={}", a.p);
                assert_eq!(a.provenance, Provenance::ValuationSplit);
            } else {
                assert_eq!(
                    a.provenance,
                    Provenance::UnsafeSplit,
                    "p={} must be flagged",
                    a.p
                );
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
        let (rows, stats) = badset_from_gpu_json(&[prefix.to_string()], 16, 8, None).unwrap();
        for w in 1..=8usize {
            assert_eq!(stats.mass_by_weight[w], binom(8, w) * (1u64 << w));
        }
        let reference = bad_set(16, 8, 1).unwrap();
        assert_eq!(rows.len(), reference.len());
        for (a, b) in rows.iter().zip(reference.iter()) {
            assert_eq!(a.p, b.p);
            if b.provenance != Provenance::CensusCorrected {
                assert_eq!(a.counts, b.counts, "counts at p={}", a.p);
            } else {
                assert_eq!(a.provenance, Provenance::UnsafeSplit);
            }
        }
    }

    /// A checkpointed run interrupted between shards must resume to results
    /// identical to an uninterrupted run.
    #[test]
    fn checkpoint_resume_roundtrip() {
        let t = norm_table(16, 8, 1).unwrap();
        let entries: Vec<_> = t.entries.iter().collect();
        let dir = std::env::temp_dir();
        // split the table into two JSON "shards"
        let mut paths = Vec::new();
        for (i, half) in entries.chunks(entries.len().div_ceil(2)).enumerate() {
            let mut js = String::from("{");
            for (j, (n, counts)) in half.iter().enumerate() {
                if j > 0 {
                    js.push(',');
                }
                js.push_str(&format!("\"{n}\": {{"));
                let mut first = true;
                for (w, &c) in counts.iter().enumerate() {
                    if c > 0 {
                        if !first {
                            js.push(',');
                        }
                        js.push_str(&format!("\"{w}\": {c}"));
                        first = false;
                    }
                }
                js.push('}');
            }
            js.push('}');
            let p = dir.join(format!("vanish_ckpt_shard{i}.json"));
            std::fs::write(&p, js).unwrap();
            paths.push(p.to_str().unwrap().to_string());
        }
        let prefix = dir.join("vanish_ckpt_test");
        let prefix = prefix.to_str().unwrap();
        clear_checkpoint(prefix);

        // reference: uninterrupted run, no checkpointing
        let (ref_rows, ref_stats) = badset_from_gpu_json(&paths, 16, 8, None).unwrap();
        // interrupted run: shard 0 only, checkpoint persists...
        let _ = badset_from_gpu_json(&paths[..1], 16, 8, Some(prefix)).unwrap();
        assert!(
            load_checkpoint(prefix, 8).is_some(),
            "checkpoint must exist"
        );
        // ...then the full path list resumes from it
        let (rows, stats) = badset_from_gpu_json(&paths, 16, 8, Some(prefix)).unwrap();
        assert_eq!(rows, ref_rows, "resumed rows differ from uninterrupted run");
        assert_eq!(stats.mass_by_weight, ref_stats.mass_by_weight);
        assert_eq!(stats.n_max_by_weight, ref_stats.n_max_by_weight);
        assert_eq!(stats.entries_parsed, ref_stats.entries_parsed);
        clear_checkpoint(prefix);
        assert!(load_checkpoint(prefix, 8).is_none(), "checkpoint cleared");
    }

    /// Missing input files surface as [`crate::Error::Io`], not as an
    /// engine-limit error.
    #[test]
    fn missing_file_is_io_error() {
        let r = badset_from_gpu_json(&["/nonexistent/vanish_test.json".into()], 16, 8, None);
        assert!(matches!(r, Err(crate::Error::Io { .. })));
    }

    fn binom(n: u64, k: usize) -> u64 {
        let mut r = 1u64;
        for i in 0..k as u64 {
            r = r * (n - i) / (i + 1);
        }
        r
    }
}
