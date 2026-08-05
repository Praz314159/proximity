//! Accident events — the atomic rows of the accident inventory.
//!
//! An *accident* at a prime `p = 1 (mod s)` is an extra zero of the
//! shadow: a small vector `v` in `Z[zeta_s]` with `p | N(v)`, so `v`
//! lands in the kernel of some embedding `Z[zeta_s] -> F_p`. The bad set
//! ([`super::bad_set`]) records how *many* such vectors each prime
//! admits; an [`AccidentEvent`] records *which*: one row per
//! (prime, symmetry orbit) incidence, carrying the witness vector
//! itself. Downstream consumers interrogate the vector, not the count —
//! class merges, support structure, tower descent, and bad-word recipes
//! all need the relation in hand.
//!
//! The symmetry group is stated once, here, and owned by this module:
//! monomial rotation `v -> zeta * v` (order `s`; contains negation as
//! `zeta^{s/2} = -1`) and the Galois group `v -> sigma_a(v)` for odd
//! `a` (order `s/2`), together of order `s^2/2`. Both act by signed
//! permutation of half-basis coefficients, so they preserve weight,
//! height, and the coefficient bound; both preserve the norm exactly
//! (Galois permutes the embedding factors, and `N(zeta) = zeta^{(s/2)^2}
//! = 1` for power-of-two `s >= 4`). Orbits are therefore closed inside
//! every enumeration stratum, which is what makes the occupancy
//! certificate below possible: per (norm, weight), the orbit sizes must
//! sum to the norm table's independently aggregated count, or
//! [`accident_events`] refuses to return.
//!
//! The canonical representative of an orbit is its lexicographically
//! least coefficient tuple (signed-integer order on the half-basis
//! sequence). Canonical means canonical: two pipelines that meet the
//! same orbit report the same representative, so event rows can be
//! joined across sources by `(norm, weight, orbit_rep)`.

use super::{combinations, decode_pattern, for_each_bad_prime, NormEngine, NormTable};
use crate::error::{Error, Result};
use crate::ring::Cyclo;
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};

/// Half-basis coefficient vector in fixed-width form (slots beyond
/// `s/2` stay zero) — the orbit machinery's working representation.
pub(crate) type CoeffVec = [i8; 32];

/// How complete an event row's orbit coverage is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventProvenance {
    /// Every orbit at the row's (norm, weight) is present in the event
    /// set, and their sizes sum to the norm table's count (checked at
    /// build — the occupancy certificate).
    ExhaustiveOrbits,
    /// The row came from a retained exemplar: the representative and
    /// its invariants are exact for *this* orbit, but sibling orbits
    /// sharing the (norm, weight) may be absent from the event set.
    ExemplarOnly,
}

/// Which pipeline produced an event row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventSource {
    /// The CPU norm-table inversion ([`accident_events`]).
    CpuNormTable,
    /// The GPU-dump ingest ([`super::ingest`]).
    GpuIngest,
}

/// Which accident primes an ingest-scale run retains events for.
///
/// The CPU path ([`accident_events`]) is always complete — at `s <= 32`
/// the whole inventory is small. At ingest scale (`s = 64`, billions of
/// norms) a large fraction of norms carry *some* bad prime, so
/// unfiltered retention would produce an event table nobody consumes;
/// the actual demand is named deployment primes (bad-word recipes) and
/// the die-out frontier (large primes). A prime is admitted when it is
/// listed explicitly or reaches `min_p`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventFilter {
    /// Primes retained regardless of size (deployment primes).
    pub primes: Vec<u64>,
    /// Retain every prime at or above this bound (die-out hunting).
    pub min_p: u64,
}

impl EventFilter {
    /// Retain only the listed primes.
    #[must_use]
    pub fn named(primes: Vec<u64>) -> Self {
        EventFilter {
            primes,
            min_p: u64::MAX,
        }
    }
    /// Retain every prime at or above `min_p`.
    #[must_use]
    pub fn at_least(min_p: u64) -> Self {
        EventFilter {
            primes: Vec::new(),
            min_p,
        }
    }
    /// Whether events at `p` are retained.
    #[must_use]
    pub fn admits(&self, p: u64) -> bool {
        p >= self.min_p || self.primes.contains(&p)
    }
    /// Canonical one-line form, used to guard checkpoint compatibility:
    /// a resumed run must be retaining exactly what the checkpoint did.
    #[must_use]
    pub fn spec(&self) -> String {
        let ps: Vec<String> = self.primes.iter().map(u64::to_string).collect();
        format!("min_p={};primes={}", self.min_p, ps.join(","))
    }
}

/// One accident event: a prime meeting one symmetry orbit of kernel
/// vectors. The canonical row of the accidents table.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccidentEvent {
    /// The level `s` (domain size; the vector lives in `Z[zeta_s]`).
    pub level: usize,
    /// The accident prime, `p = 1 (mod s)`, `p > s`.
    pub p: u64,
    /// Number of nonzero half-basis coefficients (an orbit invariant).
    pub weight: usize,
    /// The exact norm `N(v)` shared by the whole orbit.
    pub norm: u128,
    /// `e = v_p(norm)` — exact integer arithmetic, no split assumption.
    pub valuation: u64,
    /// `norm / p^e` — the identity-cards headline ("norms are barely
    /// composite": accident norms are near-prime lattice points).
    pub cofactor: u128,
    /// The orbit's canonical representative: half-basis coefficients,
    /// length `s/2`, the lexicographically least member.
    pub orbit_rep: Vec<i64>,
    /// Number of distinct vectors in the orbit (divides `s^2/2`;
    /// smaller than the group order exactly when the representative
    /// has extra symmetry).
    pub orbit_size: u64,
    /// `sum v_i^2` — the Parseval mass, capping the norm at
    /// `height^{s/4}`.
    pub height: u64,
    /// `max |v_i|` — the unit-confined vs beyond-unit distinction.
    pub max_coeff: i64,
    /// Orbit-coverage completeness of this row.
    pub provenance: EventProvenance,
    /// Producing pipeline.
    pub source: EventSource,
}

/// Pack a ring element into the fixed-width working form (set key /
/// serialization; the arithmetic itself lives on [`Cyclo`]).
fn pack(c: &Cyclo) -> CoeffVec {
    let mut out = [0i8; 32];
    for (slot, &v) in out.iter_mut().zip(c.coeffs()) {
        *slot = i8::try_from(v).expect("orbit coefficients stay within the enumeration bound");
    }
    out
}

/// Lift the packed form back into the ring.
fn unpack(half: usize, v: &CoeffVec) -> Cyclo {
    Cyclo::from_coeffs(v[..half].iter().map(|&c| c as i64).collect())
        .expect("half is a power of two >= 2 whenever s >= 4")
}

/// The full symmetry orbit of `v`, sorted, deduplicated — so `[0]` is
/// the canonical representative and `len()` is the orbit size. The
/// generators are the ring's own: [`Cyclo::galois`] and
/// [`Cyclo::dilate`] (rotation is `dilate(1)`; negation is `dilate(s/2)`).
pub(crate) fn orbit(s: usize, v: &CoeffVec) -> Vec<CoeffVec> {
    let half = s / 2;
    let start = unpack(half, v);
    let mut members = Vec::with_capacity(s * half);
    for a in (1..s).step_by(2) {
        let mut w = start.galois(a).expect("a is odd by construction");
        for _ in 0..s {
            members.push(pack(&w));
            w = w.dilate(1);
        }
    }
    members.sort_unstable();
    members.dedup();
    members
}

/// Build one event row from an orbit's canonical form `(rep, orbit_size)`
/// and one `(p, valuation)` incidence of its norm. The orbit invariants
/// come from the ring element itself — note the vocabulary seam: the
/// row's `height` is the ring's [`Cyclo::sq_sum`] (Parseval mass), the
/// row's `max_coeff` is the ring's [`Cyclo::height`].
pub(crate) fn event_row(
    s: usize,
    norm: u128,
    (p, e): (u64, u64),
    (rep, orbit_size): (&CoeffVec, u64),
    provenance: EventProvenance,
    source: EventSource,
) -> AccidentEvent {
    let rc = unpack(s / 2, rep);
    let pe = (0..e).fold(1u128, |acc, _| acc * p as u128);
    AccidentEvent {
        level: s,
        p,
        weight: rc.weight(),
        norm,
        valuation: e,
        cofactor: norm / pe,
        orbit_size,
        height: u64::try_from(rc.sq_sum()).expect("Parseval mass of a bounded vector fits u64"),
        max_coeff: rc.height(),
        orbit_rep: rc.coeffs().to_vec(),
        provenance,
        source,
    }
}

/// The complete accident inventory of a norm table, one row per
/// (prime, orbit) incidence: factor the table's norms, re-enumerate the
/// vectors behind the accident-bearing ones through the shared
/// `NormEngine`, and decompose them into symmetry orbits.
///
/// Exhaustive and certified: per (norm, weight), the emitted orbit
/// sizes must sum to the table's count — a mismatch (which would mean
/// the two enumeration passes or the orbit closure disagree) is an
/// error, never a silent discrepancy. Rows are sorted by
/// `(p, weight, norm, orbit_rep)`.
pub fn accident_events(table: &NormTable) -> Result<Vec<AccidentEvent>> {
    let s = table.s;
    let half = s / 2;
    // factor once per distinct norm; keep the accident-bearing ones
    let mut accident_norms: HashMap<u128, Vec<(u64, u64)>> = HashMap::new();
    for &n in table.entries.keys() {
        if n <= 1 {
            continue;
        }
        let n64 = u64::try_from(n).map_err(|_| {
            Error::Unsupported("factoring norms above 2^64 not yet supported".into())
        })?;
        let mut ps = Vec::new();
        for_each_bad_prime(n64, s as u64, |p, e| ps.push((p, e)));
        if !ps.is_empty() {
            accident_norms.insert(n, ps);
        }
    }
    let engine = NormEngine::new(s, table.wmax, table.cmax)?;
    let coefs: Vec<i64> = (-table.cmax..=table.cmax).filter(|&c| c != 0).collect();
    let ncoef = coefs.len();
    let mut events: Vec<AccidentEvent> = Vec::new();
    for w in 1..=table.wmax {
        let supports = combinations(half, w);
        let npat: u64 = (ncoef as u64).pow(w as u32);
        // parallel sweep: collect the vectors landing on accident norms,
        // grouped per norm inside each chunk (one move, no flat
        // intermediate — the hit set reaches hundreds of MB at s = 32)
        let hits: Vec<HashMap<u128, Vec<CoeffVec>>> = supports
            .par_chunks(1.max(supports.len() / 64))
            .map(|chunk| {
                let mut local: HashMap<u128, Vec<CoeffVec>> = HashMap::new();
                for sup in chunk {
                    let folds = engine.folds(sup);
                    for pat in 0..npat {
                        let mut cvec = [0i64; 32];
                        decode_pattern(pat, &coefs, w, &mut cvec);
                        let n = engine.norm(&folds, &cvec);
                        if accident_norms.contains_key(&n) {
                            let mut v = [0i8; 32];
                            for (i, &si) in sup.iter().enumerate() {
                                v[si as usize] = cvec[i] as i8;
                            }
                            local.entry(n).or_default().push(v);
                        }
                    }
                }
                local
            })
            .collect();
        // orbit decomposition per norm, with the occupancy certificate
        let mut by_norm: HashMap<u128, Vec<CoeffVec>> = HashMap::new();
        for local in hits {
            for (n, mut vs) in local {
                match by_norm.entry(n) {
                    std::collections::hash_map::Entry::Occupied(mut e) => {
                        e.get_mut().append(&mut vs)
                    }
                    std::collections::hash_map::Entry::Vacant(e) => {
                        e.insert(vs);
                    }
                }
            }
        }
        for (n, vs) in by_norm {
            let mut seen: HashSet<CoeffVec> = HashSet::with_capacity(vs.len());
            let mut covered = 0u64;
            for v in &vs {
                if seen.contains(v) {
                    continue;
                }
                let members = orbit(s, v);
                covered += members.len() as u64;
                for m in &members {
                    seen.insert(*m);
                }
                let rep = members[0];
                for &pe in &accident_norms[&n] {
                    events.push(event_row(
                        s,
                        n,
                        pe,
                        (&rep, members.len() as u64),
                        EventProvenance::ExhaustiveOrbits,
                        EventSource::CpuNormTable,
                    ));
                }
            }
            let expect = table.entries[&n][w];
            assert!(
                covered == expect && vs.len() as u64 == expect,
                "orbit occupancy certificate at norm {n}, weight {w}: table has \
                 {expect}, enumeration hit {}, orbits cover {covered} — the two \
                 passes or the orbit closure disagree",
                vs.len()
            );
        }
    }
    events.sort_by(|a, b| {
        (a.p, a.weight, a.norm, &a.orbit_rep).cmp(&(b.p, b.weight, b.norm, &b.orbit_rep))
    });
    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::census::kernel as census;
    use crate::domain::MultiplicativeSubgroup;
    use crate::ring::primes::norms::{bad_set, norm_table, Provenance};

    /// The orbit machinery against the exact ring: every member of an
    /// orbit has the same `Cyclo::norm_i128`, the representative is the
    /// least member, canonicalization is idempotent, and the orbit size
    /// divides the group order `s^2/2`. Full `s = 8`, `cmax = 2` sweep.
    #[test]
    fn orbits_are_norm_classes_of_the_ring() {
        let s = 8usize;
        for pat in 0..5u64.pow(4) {
            let mut v = [0i8; 32];
            let mut t = pat;
            for slot in v.iter_mut().take(4) {
                *slot = (t % 5) as i8 - 2;
                t /= 5;
            }
            if v[..4].iter().all(|&c| c == 0) {
                continue;
            }
            let members = orbit(s, &v);
            assert_eq!((s * s / 2) % members.len(), 0, "size divides group order");
            let rep = members[0];
            assert!(members.iter().all(|m| *m >= rep), "rep is least");
            assert_eq!(orbit(s, &rep)[0], rep, "canonicalization idempotent");
            let n0 = Cyclo::from_coeffs(v[..4].iter().map(|&c| c as i64).collect())
                .unwrap()
                .norm_i128()
                .unwrap();
            for m in &members {
                let nm = Cyclo::from_coeffs(m[..4].iter().map(|&c| c as i64).collect())
                    .unwrap()
                    .norm_i128()
                    .unwrap();
                assert_eq!(nm, n0, "orbit members share the exact ring norm");
            }
        }
    }

    /// The s = 16 accident inventory, measured 2026-08-05 and pinned:
    /// 11 bad primes, die-out at p = 881 whose witness norm is 881
    /// itself — prime norm, cofactor 1, one full orbit of 128 weight-7
    /// unit vectors. The near-prime-norm law at its sharpest.
    #[test]
    fn s16_inventory_has_prime_norm_dieout() {
        let table = norm_table(16, 8, 1).unwrap();
        let events = accident_events(&table).unwrap();
        let mut primes: Vec<u64> = events.iter().map(|e| e.p).collect();
        primes.sort_unstable();
        primes.dedup();
        assert_eq!(primes.len(), 11);
        assert_eq!(*primes.last().unwrap(), 881);
        let dieout: Vec<_> = events.iter().filter(|e| e.p == 881).collect();
        assert_eq!(dieout.len(), 1, "one orbit carries the die-out prime");
        let e = dieout[0];
        assert_eq!((e.norm, e.valuation, e.cofactor), (881, 1, 1));
        assert_eq!((e.weight, e.orbit_size), (7, 128));
        assert_eq!((e.height, e.max_coeff), (7, 1));
        assert_eq!(e.provenance, EventProvenance::ExhaustiveOrbits);
        // the representative really is a witness: its ring norm is 881
        let n = Cyclo::from_coeffs(e.orbit_rep.clone())
            .unwrap()
            .norm_i128()
            .unwrap();
        assert_eq!(n, 881);
    }

    /// Events re-derive the bad set: valuation-weighted orbit sizes,
    /// Galois-normalized by s/2, must reproduce every split-safe row of
    /// [`bad_set`] — two independent inversions of the same table.
    #[test]
    fn events_rederive_the_bad_set() {
        let (s, wmax, cmax) = (16usize, 8usize, 1i64);
        let events = accident_events(&norm_table(s, wmax, cmax).unwrap()).unwrap();
        let mut derived: HashMap<u64, Vec<u64>> = HashMap::new();
        for e in &events {
            derived.entry(e.p).or_insert_with(|| vec![0; wmax + 1])[e.weight] +=
                e.valuation * e.orbit_size;
        }
        let reference = bad_set(s, wmax, cmax).unwrap();
        assert_eq!(derived.len(), reference.len(), "same prime set");
        for row in &reference {
            let counts: Vec<u64> = derived[&row.p]
                .iter()
                .map(|&v| v / (s as u64 / 2))
                .collect();
            if row.provenance == Provenance::ValuationSplit {
                assert_eq!(counts, row.counts, "counts at p={}", row.p);
            }
        }
    }

    /// The poster-prime identity card, re-derived from the events path:
    /// at s = 32 the zero-class-dirty prime 77,569 must surface among
    /// the weight-capped events, and the event-derived per-weight counts
    /// must match an independent meet-in-the-middle kernel census in
    /// F_p — enumeration + factoring + orbits against modular counting.
    #[test]
    fn events_rederive_the_poster_prime_census() {
        let (s, wmax) = (32usize, 6usize);
        let events = accident_events(&norm_table(s, wmax, 1).unwrap()).unwrap();
        let poster: Vec<_> = events.iter().filter(|e| e.p == 77_569).collect();
        assert!(!poster.is_empty(), "poster prime missing from events");
        assert!(
            poster.iter().all(|e| e.valuation == 1),
            "split-safe, so the census comparison is exact"
        );
        let mut derived = vec![0u64; wmax + 1];
        for e in &poster {
            derived[e.weight] += e.valuation * e.orbit_size;
        }
        let sg = MultiplicativeSubgroup::new(77_569, s).unwrap();
        let mitm = census::mitm(&sg, 1).unwrap();
        for w in 1..=wmax {
            assert_eq!(derived[w] / (s as u64 / 2), mitm[w], "census at weight {w}");
        }
    }
}
