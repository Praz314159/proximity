//! Ingestion of externally computed norm tables (the GPU campaign's
//! output) into bad sets and accident events — the s = 64 pipeline.
//!
//! Two dump formats: JSON shards
//! (`{"<norm>": {"<w>": count, ...}, ...}`; shards partition supports,
//! so the same norm may appear in several shards and counts add) and
//! per-weight binary dumps (norms + counts, optionally + exemplar
//! vectors). Files are gigabytes, so parsing is streaming: entries feed
//! straight into parallel factoring without materializing the table,
//! checkpointed at shard granularity.
//!
//! Normalization follows `norms::bad_set` exactly (valuation / (s/2);
//! per-weight censuses are Galois-invariant), except that primes where the
//! equal split is unsafe (`p^2 | N(v)`, or a valuation not divisible by
//! s/2) are *flagged* rather than census-corrected — at s = 64 the direct
//! census fallback is not yet feasible, and the flags mark exactly the
//! rows a downstream analysis must treat as approximate.
//!
//! Events change that calculus where they exist: an
//! [`AccidentEvent`] row holds the witness vector itself, so its
//! valuation needs no split assumption at all — the same factoring pass
//! that builds the bad set retains, for filtered primes, the identity
//! of each accident ([`badset_and_events_from_gpu_bin`]).

use super::events::{
    event_row, orbit, AccidentEvent, CoeffVec, EventFilter, EventProvenance, EventSource,
};
use super::{for_each_bad_prime, BadSetEntry, NormEngine, Provenance};
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

/// Read one binary weight dump (`<prefix>.w<w>.norms.bin` u64-le +
/// `.counts.bin` u64-le, plus `.exemplars.bin` when `half` is set:
/// `s/2` i8 half-basis coefficients per row, row-aligned with the
/// norms), invoking `sink` per (norm, counts_by_weight, exemplar).
fn parse_bin_weight(
    prefix: &str,
    w: usize,
    wmax: usize,
    half: Option<usize>,
    mut sink: impl FnMut(u64, &[u64], Option<&CoeffVec>),
) -> Result<u64> {
    let read = |path: String| std::fs::read(&path).map_err(|source| Error::Io { path, source });
    let nb = read(format!("{prefix}.w{w}.norms.bin"))?;
    let cb = read(format!("{prefix}.w{w}.counts.bin"))?;
    if nb.len() != cb.len() || nb.len() % 8 != 0 {
        return Err(Error::MalformedInput(format!(
            "{prefix}.w{w}: norms/counts length mismatch or not 8-byte aligned"
        )));
    }
    let n = nb.len() / 8;
    let eb = match half {
        Some(h) => {
            let eb = read(format!("{prefix}.w{w}.exemplars.bin"))?;
            if eb.len() != n * h {
                return Err(Error::MalformedInput(format!(
                    "{prefix}.w{w}: exemplars not row-aligned with norms \
                     ({} bytes for {n} rows of {h})",
                    eb.len()
                )));
            }
            Some((eb, h))
        }
        None => None,
    };
    let mut counts = vec![0u64; wmax + 1];
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
        let ex = eb.as_ref().map(|(eb, h)| {
            let mut v = [0i8; 32];
            for (slot, &b) in v.iter_mut().zip(&eb[h * i..h * (i + 1)]) {
                *slot = b as i8;
            }
            v
        });
        sink(norm, &counts, ex.as_ref());
    }
    Ok(n as u64)
}

/// Everything the events-retaining mode threads through a run: the
/// shared exact-norm kernel (exemplar validation), the retention
/// filter, and the accumulator keyed `(norm, weight) -> rep -> size`.
struct EventsMode {
    engine: NormEngine,
    filter: EventFilter,
    cmax: i64,
    acc: HashMap<(u64, usize), HashMap<CoeffVec, u64>>,
}

/// Validate one retained exemplar against its recorded norm and reduce
/// it to canonical orbit form. The norm recompute is the association
/// gate: a dump whose vector/norm pairing was scrambled (the device
/// dedup hazard) fails here, loudly, instead of poisoning the table.
fn canonical_exemplar(
    s: usize,
    engine: &NormEngine,
    cmax: i64,
    norm: u64,
    ex: &CoeffVec,
) -> Result<(usize, CoeffVec, u64)> {
    let half = s / 2;
    let mut sup: Vec<u8> = Vec::new();
    let mut cvec = [0i64; 32];
    for (i, &c) in ex.iter().enumerate().take(half) {
        if c != 0 {
            if (c as i64).abs() > cmax {
                return Err(Error::MalformedInput(format!(
                    "exemplar coefficient {c} exceeds cmax {cmax}"
                )));
            }
            cvec[sup.len()] = c as i64;
            sup.push(i as u8);
        }
    }
    if sup.is_empty() {
        return Err(Error::MalformedInput("empty exemplar".into()));
    }
    let recomputed = engine.norm(&engine.folds(&sup), &cvec[..sup.len()]);
    if recomputed != norm as u128 {
        return Err(Error::MalformedInput(format!(
            "exemplar recomputes to norm {recomputed}, dump says {norm}: \
             the vector/norm association is broken"
        )));
    }
    let members = orbit(s, ex);
    Ok((sup.len(), members[0], members.len() as u64))
}

fn flush_batch(
    batch: &mut Vec<(u64, Counts)>,
    exemplars: &mut Vec<Option<CoeffVec>>,
    acc: &mut HashMap<u64, (Counts, bool)>,
    mut events: Option<&mut EventsMode>,
    s: usize,
) -> Result<()> {
    let chunk_size = 1.max(batch.len() / 128);
    type Partial = (
        HashMap<u64, (Counts, bool)>,
        Vec<((u64, usize), CoeffVec, u64)>,
    );
    let partial: Vec<Partial> = batch
        .par_chunks(chunk_size)
        .enumerate()
        .map(|(ci, chunk)| -> Result<Partial> {
            let mut local: HashMap<u64, (Counts, bool)> = HashMap::new();
            let mut found: Vec<((u64, usize), CoeffVec, u64)> = Vec::new();
            for (j, (n, counts)) in chunk.iter().enumerate() {
                let mut admitted = false;
                for_each_bad_prime(*n, s as u64, |p, e| {
                    let entry = local.entry(p).or_insert(([0; MAXW], false));
                    for (w, &c) in counts.iter().enumerate() {
                        entry.0[w] += e * c;
                    }
                    if e >= 2 {
                        entry.1 = true;
                    }
                    if let Some(ev) = events.as_deref() {
                        admitted |= ev.filter.admits(p);
                    }
                });
                if admitted {
                    let ev = events
                        .as_deref()
                        .expect("admitted is only set in events mode");
                    let ex = exemplars[ci * chunk_size + j]
                        .as_ref()
                        .expect("events mode retains an exemplar per entry");
                    let (w, rep, size) = canonical_exemplar(s, &ev.engine, ev.cmax, *n, ex)?;
                    found.push(((*n, w), rep, size));
                }
            }
            Ok((local, found))
        })
        .collect::<Result<_>>()?;
    for (m, found) in partial {
        for (p, (cs, flag)) in m {
            let entry = acc.entry(p).or_insert(([0; MAXW], false));
            for (w, c) in cs.iter().enumerate() {
                entry.0[w] += c;
            }
            entry.1 |= flag;
        }
        if let Some(ev) = events.as_deref_mut() {
            for (key, rep, size) in found {
                let prior = ev.acc.entry(key).or_default().insert(rep, size);
                assert!(
                    prior.map_or(true, |s0| s0 == size),
                    "orbit size disagrees across shards at norm {}, weight {}",
                    key.0,
                    key.1
                );
            }
        }
    }
    batch.clear();
    exemplars.clear();
    Ok(())
}

/// Checkpoint state: the accumulators plus everything needed to resume.
struct Checkpoint {
    acc: HashMap<u64, (Counts, bool)>,
    events_acc: HashMap<(u64, usize), HashMap<CoeffVec, u64>>,
    stats: IngestStats,
    done_paths: Vec<String>,
}

fn ckpt_names(prefix: &str) -> (String, String, String) {
    (
        format!("{prefix}.ckpt.bin"),
        format!("{prefix}.ckpt.meta"),
        format!("{prefix}.ckpt.events.bin"),
    )
}

/// Atomically persist the accumulators after a completed shard. A failure
/// here is an early disk-space canary: it fires at shard granularity
/// instead of after the final hour of factoring. The meta file is the
/// commit point and records the row count of each binary, so a crash
/// between the renames leaves a rejected (never a replayed-on-top)
/// checkpoint.
fn save_checkpoint(
    prefix: &str,
    acc: &HashMap<u64, (Counts, bool)>,
    stats: &IngestStats,
    done_paths: &[String],
    wmax: usize,
    events: Option<&EventsMode>,
) -> Result<()> {
    let (bin, meta, ebin) = ckpt_names(prefix);
    let werr = |path: &str| {
        let path = path.to_string();
        move |source: std::io::Error| Error::Io { path, source }
    };
    let atomic = |path: &str, buf: &[u8]| -> Result<()> {
        let tmp = format!("{path}.tmp");
        std::fs::write(&tmp, buf).map_err(werr(&tmp))?;
        std::fs::rename(&tmp, path).map_err(werr(path))
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
    atomic(&bin, &buf)?;
    let mut erows = 0usize;
    if let Some(ev) = events {
        let mut ebuf = Vec::new();
        for (&(n, w), reps) in &ev.acc {
            for (rep, &size) in reps {
                ebuf.extend_from_slice(&n.to_le_bytes());
                ebuf.push(u8::try_from(w).expect("weight fits u8 by MAXW"));
                ebuf.extend_from_slice(
                    &u16::try_from(size)
                        .expect("orbit size is at most s^2/2 <= 2048")
                        .to_le_bytes(),
                );
                ebuf.extend_from_slice(&rep.map(|c| c as u8));
                erows += 1;
            }
        }
        atomic(&ebin, &ebuf)?;
    }
    let mut m = format!(
        "wmax {}\nentries {}\nmass {}\nnmax {}\nrows {}\n",
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
        acc.len(),
    );
    if let Some(ev) = events {
        m.push_str(&format!("events {}\nerows {erows}\n", ev.filter.spec()));
    }
    for p in done_paths {
        m.push_str("done ");
        m.push_str(p);
        m.push('\n');
    }
    atomic(&meta, m.as_bytes())
}

/// Load a checkpoint, or `None` when absent or incompatible. The row
/// counts recorded in the meta (the commit point) must match the binary
/// payloads exactly, and an events checkpoint must have been retaining
/// exactly what this run retains (`events_spec`) — anything else is
/// rejected so a resume can never silently drop or replay work.
fn load_checkpoint(prefix: &str, wmax: usize, events_spec: Option<&str>) -> Option<Checkpoint> {
    let (bin, meta, ebin) = ckpt_names(prefix);
    let m = std::fs::read_to_string(&meta).ok()?;
    let buf = std::fs::read(&bin).ok()?;
    let mut stats = IngestStats {
        mass_by_weight: vec![0; wmax + 1],
        n_max_by_weight: vec![0; wmax + 1],
        entries_parsed: 0,
    };
    let mut done_paths = Vec::new();
    let mut ck_wmax = usize::MAX;
    let mut rows = usize::MAX;
    let mut erows = usize::MAX;
    let mut ck_spec: Option<String> = None;
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
            "rows" => rows = val.parse().ok()?,
            "events" => ck_spec = Some(val.to_string()),
            "erows" => erows = val.parse().ok()?,
            "done" => done_paths.push(val.to_string()),
            _ => return None,
        }
    }
    if ck_wmax != wmax || stats.mass_by_weight.len() != wmax + 1 {
        return None;
    }
    if ck_spec.as_deref() != events_spec {
        return None;
    }
    let row = 8 + (wmax + 1) * 8 + 1;
    if buf.len() % row != 0 || buf.len() / row != rows {
        return None;
    }
    let mut acc = HashMap::with_capacity(rows);
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
    let mut events_acc: HashMap<(u64, usize), HashMap<CoeffVec, u64>> = HashMap::new();
    if events_spec.is_some() {
        const EROW: usize = 8 + 1 + 2 + 32;
        let ebuf = std::fs::read(&ebin).ok()?;
        if ebuf.len() % EROW != 0 || ebuf.len() / EROW != erows {
            return None;
        }
        for chunk in ebuf.chunks_exact(EROW) {
            let n = u64::from_le_bytes(chunk[..8].try_into().expect("record starts with 8 bytes"));
            let w = chunk[8] as usize;
            let size =
                u16::from_le_bytes(chunk[9..11].try_into().expect("slice is exactly 2 bytes"))
                    as u64;
            let mut rep = [0i8; 32];
            for (slot, &b) in rep.iter_mut().zip(&chunk[11..]) {
                *slot = b as i8;
            }
            events_acc.entry((n, w)).or_default().insert(rep, size);
        }
    }
    Some(Checkpoint {
        acc,
        events_acc,
        stats,
        done_paths,
    })
}

/// Delete a run's checkpoint files. Call after the caller has durably
/// written its outputs; until then the checkpoint is the crash-recovery
/// state for the whole factoring run.
pub fn clear_checkpoint(prefix: &str) {
    let (bin, meta, ebin) = ckpt_names(prefix);
    let _ = std::fs::remove_file(bin);
    let _ = std::fs::remove_file(meta);
    let _ = std::fs::remove_file(ebin);
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
    ingest_core(paths, s, wmax, ckpt_prefix, None).map(|(rows, _, stats)| (rows, stats))
}

/// The events-retaining ingest: [`badset_from_gpu_json`] plus one
/// [`AccidentEvent`] row per retained (prime, orbit) incidence, built
/// from the dumps' exemplar files (`<prefix>.w<w>.exemplars.bin`,
/// `s/2` i8 coefficients per row, row-aligned with the norms).
///
/// Binary dumps only — the JSON shard format carries no vectors. Every
/// exemplar whose norm admits a filtered prime is validated by
/// recomputing its norm through the shared `NormEngine` (`cmax` sizes
/// the schedule and bounds the coefficients), then reduced to canonical
/// orbit form; rows carry [`EventProvenance::ExemplarOnly`] because
/// sibling orbits at the same (norm, weight) may not have been retained
/// by the producer. Checkpoints record the retention filter and resume
/// only under the identical one.
pub fn badset_and_events_from_gpu_bin(
    paths: &[String],
    s: usize,
    wmax: usize,
    cmax: i64,
    ckpt_prefix: Option<&str>,
    filter: EventFilter,
) -> Result<(Vec<BadSetEntry>, Vec<AccidentEvent>, IngestStats)> {
    ingest_core(paths, s, wmax, ckpt_prefix, Some((cmax, filter)))
}

fn ingest_core(
    paths: &[String],
    s: usize,
    wmax: usize,
    ckpt_prefix: Option<&str>,
    events: Option<(i64, EventFilter)>,
) -> Result<(Vec<BadSetEntry>, Vec<AccidentEvent>, IngestStats)> {
    if wmax >= MAXW {
        return Err(Error::OutOfRange(
            "wmax >= 16 unsupported by inline counts".into(),
        ));
    }
    if !s.is_power_of_two() || s < 4 {
        return Err(Error::Unsupported("power-of-two s >= 4 required".into()));
    }
    let half = (s / 2) as u64;
    let mut events = events
        .map(|(cmax, filter)| -> Result<EventsMode> {
            Ok(EventsMode {
                engine: NormEngine::new(s, wmax, cmax)?,
                filter,
                cmax,
                acc: HashMap::new(),
            })
        })
        .transpose()?;
    let spec = events.as_ref().map(|ev| ev.filter.spec());
    let mut acc: HashMap<u64, (Counts, bool)> = HashMap::new();
    let mut stats = IngestStats {
        mass_by_weight: vec![0; wmax + 1],
        n_max_by_weight: vec![0; wmax + 1],
        entries_parsed: 0,
    };
    let mut done_paths: Vec<String> = Vec::new();
    if let Some(prefix) = ckpt_prefix {
        if let Some(ck) = load_checkpoint(prefix, wmax, spec.as_deref()) {
            if ck.done_paths.iter().all(|p| paths.contains(p)) {
                eprintln!(
                    "[ingest] resuming from checkpoint: {} primes, {} shard(s) done",
                    ck.acc.len(),
                    ck.done_paths.len()
                );
                acc = ck.acc;
                stats = ck.stats;
                done_paths = ck.done_paths;
                if let Some(ev) = events.as_mut() {
                    ev.acc = ck.events_acc;
                }
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
        let mut batch: Vec<(u64, Counts)> = Vec::with_capacity(FLUSH_AT);
        let mut exemplars: Vec<Option<CoeffVec>> = Vec::new();
        if !path.ends_with(".json") {
            // binary prefix: ingest every existing per-weight dump
            let ex_half = events.as_ref().map(|_| s / 2);
            for w in 1..=wmax {
                if !std::path::Path::new(&format!("{path}.w{w}.norms.bin")).exists() {
                    continue;
                }
                let mut flush_err: Result<()> = Ok(());
                stats.entries_parsed +=
                    parse_bin_weight(path, w, wmax, ex_half, |n, counts, ex| {
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
                        if ex_half.is_some() {
                            exemplars.push(ex.copied());
                        }
                        if batch.len() >= FLUSH_AT && flush_err.is_ok() {
                            done += batch.len() as u64;
                            flush_err = flush_batch(
                                &mut batch,
                                &mut exemplars,
                                &mut acc,
                                events.as_mut(),
                                s,
                            );
                            if done % (64 << 20) < FLUSH_AT as u64 {
                                eprintln!(
                                    "[ingest] {done} entries factored, {} bad primes",
                                    acc.len()
                                );
                            }
                        }
                    })?;
                flush_err?;
            }
        } else {
            if events.is_some() {
                return Err(Error::Unsupported(
                    "events retention requires binary dumps with exemplars (JSON \
                     shards carry no vectors)"
                        .into(),
                ));
            }
            let buf = std::fs::read(path).map_err(|source| Error::Io {
                path: path.clone(),
                source,
            })?;
            let mut flush_err: Result<()> = Ok(());
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
                if batch.len() >= FLUSH_AT && flush_err.is_ok() {
                    done += batch.len() as u64;
                    flush_err = flush_batch(&mut batch, &mut exemplars, &mut acc, None, s);
                    if done % (64 << 20) < FLUSH_AT as u64 {
                        eprintln!("[ingest] {done} entries factored, {} bad primes", acc.len());
                    }
                }
            })?;
            flush_err?;
        }
        done += batch.len() as u64;
        flush_batch(&mut batch, &mut exemplars, &mut acc, events.as_mut(), s)?;
        eprintln!(
            "[ingest] {path}: done ({done} entries, {} primes)",
            acc.len()
        );
        if let Some(prefix) = ckpt_prefix {
            done_paths.push(path.clone());
            save_checkpoint(prefix, &acc, &stats, &done_paths, wmax, events.as_ref())?;
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
    let mut rows: Vec<AccidentEvent> = Vec::new();
    if let Some(ev) = events {
        for ((n, _w), reps) in ev.acc {
            let mut incidences = Vec::new();
            for_each_bad_prime(n, s as u64, |p, e| {
                if ev.filter.admits(p) {
                    incidences.push((p, e));
                }
            });
            for (rep, size) in reps {
                for &pe in &incidences {
                    rows.push(event_row(
                        s,
                        n as u128,
                        pe,
                        (&rep, size),
                        EventProvenance::ExemplarOnly,
                        EventSource::GpuIngest,
                    ));
                }
            }
        }
        rows.sort_by(|a, b| {
            (a.p, a.weight, a.norm, &a.orbit_rep).cmp(&(b.p, b.weight, b.norm, &b.orbit_rep))
        });
    }
    Ok((out, rows, stats))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ring::primes::norms::events::accident_events;
    use crate::ring::Cyclo;
    use crate::smooth::norms::{bad_set, norm_table};

    /// Per-weight dump rows: (norm, count, exemplar).
    type ShardRows = HashMap<usize, Vec<(u64, u64, [i8; 4])>>;

    /// The full s = 8, cmax = 2 enumeration grouped per weight as
    /// (norm, count, first-encountered exemplar) — the raw material for
    /// fabricating exemplar-bearing binary dumps.
    fn brute_s8() -> ShardRows {
        let mut grouped: ShardRows = HashMap::new();
        for w in 1..=4usize {
            let mut by_norm: HashMap<u64, (u64, [i8; 4])> = HashMap::new();
            for pat in 0..5u64.pow(4) {
                let mut v = [0i8; 4];
                let mut t = pat;
                for slot in v.iter_mut() {
                    *slot = (t % 5) as i8 - 2;
                    t /= 5;
                }
                if v.iter().filter(|&&c| c != 0).count() != w {
                    continue;
                }
                let n = Cyclo::from_coeffs(v.iter().map(|&c| c as i64).collect())
                    .unwrap()
                    .norm_i128()
                    .unwrap() as u64;
                by_norm.entry(n).or_insert((0, v)).0 += 1;
            }
            grouped.insert(
                w,
                by_norm.into_iter().map(|(n, (c, v))| (n, c, v)).collect(),
            );
        }
        grouped
    }

    /// Write one shard's per-weight dumps (norms + counts + exemplars).
    fn write_shard(prefix: &str, shard: &ShardRows) {
        for (&w, rows) in shard {
            let (mut nb, mut cb, mut eb) = (Vec::new(), Vec::new(), Vec::new());
            for (n, c, ex) in rows {
                nb.extend_from_slice(&n.to_le_bytes());
                cb.extend_from_slice(&c.to_le_bytes());
                eb.extend_from_slice(&ex.map(|x| x as u8));
            }
            std::fs::write(format!("{prefix}.w{w}.norms.bin"), nb).unwrap();
            std::fs::write(format!("{prefix}.w{w}.counts.bin"), cb).unwrap();
            std::fs::write(format!("{prefix}.w{w}.exemplars.bin"), eb).unwrap();
        }
    }

    /// The events-retaining ingest against the CPU inversion, s = 8,
    /// cmax = 2: every ingest event must match its CPU counterpart on
    /// all orbit invariants — the fabricated exemplars are deliberately
    /// non-canonical, so this pins the shared canonicalization across
    /// both pipelines.
    #[test]
    fn ingest_events_match_the_cpu_path() {
        let dir = std::env::temp_dir();
        let prefix = dir.join("vanish_events_ingest_test");
        let prefix = prefix.to_str().unwrap();
        write_shard(prefix, &brute_s8());
        let (rows, events, _) = badset_and_events_from_gpu_bin(
            &[prefix.to_string()],
            8,
            4,
            2,
            None,
            EventFilter::at_least(1),
        )
        .unwrap();
        assert!(!rows.is_empty() && !events.is_empty());
        let reference = accident_events(&norm_table(8, 4, 2).unwrap()).unwrap();
        for e in &events {
            let m = reference
                .iter()
                .find(|r| {
                    (r.p, r.norm, r.weight, &r.orbit_rep) == (e.p, e.norm, e.weight, &e.orbit_rep)
                })
                .unwrap_or_else(|| panic!("ingest event at p={} unmatched", e.p));
            assert_eq!(
                (e.valuation, e.cofactor, e.orbit_size, e.height, e.max_coeff),
                (m.valuation, m.cofactor, m.orbit_size, m.height, m.max_coeff)
            );
            assert_eq!(e.provenance, EventProvenance::ExemplarOnly);
            assert_eq!(e.source, EventSource::GpuIngest);
        }
        // the poster event of this cell: N(2 + zeta) = 17, prime norm
        assert!(
            events
                .iter()
                .any(|e| e.p == 17 && e.norm == 17 && e.cofactor == 1),
            "the 2 + zeta accident is missing"
        );
    }

    /// A dump whose vector/norm association is scrambled (the device
    /// dedup hazard the exemplar format exists to guard against) must be
    /// rejected loudly, not ingested.
    #[test]
    fn scrambled_exemplar_is_rejected() {
        let mut table = brute_s8();
        // find a weight holding two accident norms and swap their exemplars
        let reference = accident_events(&norm_table(8, 4, 2).unwrap()).unwrap();
        let mut swapped = false;
        'outer: for rows in table.values_mut() {
            let bad: Vec<usize> = (0..rows.len())
                .filter(|&i| reference.iter().any(|e| e.norm == rows[i].0 as u128))
                .collect();
            for pair in bad.windows(2) {
                if rows[pair[0]].0 != rows[pair[1]].0 {
                    let ex = rows[pair[0]].2;
                    rows[pair[0]].2 = rows[pair[1]].2;
                    rows[pair[1]].2 = ex;
                    swapped = true;
                    break 'outer;
                }
            }
        }
        assert!(swapped, "test needs two accident norms at one weight");
        let dir = std::env::temp_dir();
        let prefix = dir.join("vanish_events_scrambled_test");
        let prefix = prefix.to_str().unwrap();
        write_shard(prefix, &table);
        let r = badset_and_events_from_gpu_bin(
            &[prefix.to_string()],
            8,
            4,
            2,
            None,
            EventFilter::at_least(1),
        );
        assert!(
            matches!(r, Err(Error::MalformedInput(ref m)) if m.contains("association")),
            "scrambled dump must fail the association gate: {r:?}"
        );
    }

    /// An events-mode run interrupted between shards must resume to
    /// results identical to an uninterrupted run, and a checkpoint
    /// written under one retention filter must never be resumed by a
    /// run with a different one (or by a badset-only run).
    #[test]
    fn events_checkpoint_resume_roundtrip() {
        let table = brute_s8();
        // split into two shards by norm parity
        let (mut a, mut b) = (HashMap::new(), HashMap::new());
        for (w, rows) in &table {
            let (ra, rb): (Vec<_>, Vec<_>) = rows.iter().partition(|(n, _, _)| n % 2 == 0);
            a.insert(*w, ra);
            b.insert(*w, rb);
        }
        let dir = std::env::temp_dir();
        let pa = dir.join("vanish_evck_shard_a");
        let pb = dir.join("vanish_evck_shard_b");
        write_shard(pa.to_str().unwrap(), &a);
        write_shard(pb.to_str().unwrap(), &b);
        let paths = vec![
            pa.to_str().unwrap().to_string(),
            pb.to_str().unwrap().to_string(),
        ];
        let ck = dir.join("vanish_evck_test");
        let ck = ck.to_str().unwrap();
        clear_checkpoint(ck);
        let filter = EventFilter::at_least(1);
        let (ref_rows, ref_events, ref_stats) =
            badset_and_events_from_gpu_bin(&paths, 8, 4, 2, None, filter.clone()).unwrap();
        // interrupted: shard A only, checkpoint persists...
        let _ =
            badset_and_events_from_gpu_bin(&paths[..1], 8, 4, 2, Some(ck), filter.clone()).unwrap();
        assert!(load_checkpoint(ck, 4, Some(&filter.spec())).is_some());
        // ...a different filter or a badset-only run must NOT resume it...
        assert!(load_checkpoint(ck, 4, Some(&EventFilter::at_least(1000).spec())).is_none());
        assert!(load_checkpoint(ck, 4, None).is_none());
        // ...and the matching run resumes to the uninterrupted answer
        let (rows, events, stats) =
            badset_and_events_from_gpu_bin(&paths, 8, 4, 2, Some(ck), filter).unwrap();
        assert_eq!(rows, ref_rows);
        assert_eq!(events, ref_events);
        assert_eq!(stats.mass_by_weight, ref_stats.mass_by_weight);
        assert_eq!(stats.entries_parsed, ref_stats.entries_parsed);
        clear_checkpoint(ck);
    }

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
            let expect = crate::field::binom(8, w as u64) * (1u64 << w);
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
            assert_eq!(
                stats.mass_by_weight[w],
                crate::field::binom(8, w as u64) * (1u64 << w)
            );
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
            load_checkpoint(prefix, 8, None).is_some(),
            "checkpoint must exist"
        );
        // ...then the full path list resumes from it
        let (rows, stats) = badset_from_gpu_json(&paths, 16, 8, Some(prefix)).unwrap();
        assert_eq!(rows, ref_rows, "resumed rows differ from uninterrupted run");
        assert_eq!(stats.mass_by_weight, ref_stats.mass_by_weight);
        assert_eq!(stats.n_max_by_weight, ref_stats.n_max_by_weight);
        assert_eq!(stats.entries_parsed, ref_stats.entries_parsed);
        clear_checkpoint(prefix);
        assert!(
            load_checkpoint(prefix, 8, None).is_none(),
            "checkpoint cleared"
        );
    }

    /// Missing input files surface as [`crate::Error::Io`], not as an
    /// engine-limit error.
    #[test]
    fn missing_file_is_io_error() {
        let r = badset_from_gpu_json(&["/nonexistent/vanish_test.json".into()], 16, 8, None);
        assert!(matches!(r, Err(crate::Error::Io { .. })));
    }
}
