//! The envelope's gates: exact-rational oracles for the step (the
//! candidate-min right-hand side mirrored independently), provider
//! pins against measured strata, tower-level regressions.
//!
//! The functions marked `#[ignore]` are EXPLORATORY SCANS, not
//! tests: they assemble full towers at the reduced box and print
//! tables that the SPLICE ledger quotes. They cannot fail, so they
//! are excluded from the default run; reproduce one with
//! `cargo test --features certified <name> -- --ignored --nocapture`.

use super::charges::derived_johnson;
use super::*;
use rug::float::Round;
use rug::{Integer, Rational};
use std::collections::BTreeSet;

/// The reduced box: the level `2^12` at rate one half, the cell
/// every gauge below is read at (the full box `2^21` is the CLI's).
const BOX_TOTAL: u64 = 1 << 12;

/// The reduced box's `(total, k)`.
fn box_cell() -> (u64, u64) {
    (BOX_TOTAL, BOX_TOTAL / 2 - 1)
}

/// The certified extension field the ceiling rows are read in:
/// KoalaBear to the sixth.
fn box_ext() -> Integer {
    use rug::ops::Pow;
    Integer::from(crate::field::named::KOALABEAR).pow(6)
}

/// The certified disagreement radius `z*` of the tower `(total, k)`
/// assembled from `base` under `data`, at budget `2^-128` in the
/// box extension; `0` when the tower does not assemble or no
/// positive ceiling exists. The one number every gauge reads.
fn zstar(total: u64, k: u64, base: u64, data: &dyn Interface) -> u64 {
    let Ok(prof) = assemble(total, k, base, data, DEFAULT_RESOLUTION) else {
        return 0;
    };
    crate::soundness::ceiling::list_ceiling_row(1, total, total - k - 1, &box_ext(), -128.0, |z| {
        prof.lg_at_disagreement(k, z)
    })
    .map_or(0, |r| r.z_star)
}

fn binom(a: u64, b: u64) -> Integer {
    if b > a {
        return Integer::from(0);
    }
    let mut v = Integer::from(1);
    for i in 0..b {
        v *= a - i;
        v /= i + 1;
    }
    v
}

/// The parameterized step oracle: level 8 to 16 at dimension 7,
/// every threshold, against exact rational arithmetic — the
/// provider's `D_c` mirrored by `dc_mirror`, everything else (deep
/// candidates, mid, graded min, analytic clamps) shared. Endpoint
/// mirror rather than containment: the extended-split candidate
/// carries a deliberately loose mid bracket.
fn run_step_oracle(data: &dyn Interface, dc_mirror: impl Fn(u64) -> Integer) {
    let (n, k) = (8u64, 7u64);
    let s = 2 * n;
    let (kev, kod) = channel_dims(k); // (4, 3)
    let r = k + 1;
    // the coverage threshold, RE-DERIVED rather than read from the
    // charges — the oracle mirrors the theorem, not the code
    let lstar = (n + k - 1).div_ceil(3); // 5
    let base = interpolation_base(n, &BTreeSet::from([kev, kod])).expect("base");
    let prof = step(&base, &BTreeSet::from([k]), data, u64::MAX).expect("step");
    for t in r..=s {
        let lmin = t.saturating_sub(n);
        let mut small = Rational::new();
        for l in lmin..kod {
            let cut = Rational::from((dc_mirror(l), binom(t - 2 * l, r - 2 * l)));
            small += graded_min(cut, s, k, n, l, t);
        }
        let chan = binom(n, kev).max(binom(n, kod));
        let (mut want_lo, mut want_hi) = rhs_mirror(&small, &chan, s, n, kod, lstar, t);
        for c in analytic_clamps(s, k, t) {
            if c < want_lo {
                want_lo = c.clone();
            }
            if c < want_hi {
                want_hi = c;
            }
        }
        let got = prof.eval(k, t).expect("in domain");
        let lo = got.lo.to_f64_round(Round::Down);
        let hi = got.hi.to_f64_round(Round::Up);
        let (wl, wh) = (want_lo.to_f64().log2(), want_hi.to_f64().log2());
        assert!(
            (lo - wl).abs() < 1e-9 && (hi - wh).abs() < 1e-9,
            "t = {t}: [{lo}, {hi}] vs mirror [{wl}, {wh}]"
        );
    }
}

/// The analytic clamps mirrored in exact rationals (unfloored,
/// matching the per-level brackets), the Johnson clause through the
/// production kernel [`super::base::johnson_agreement`] so the u128
/// contract is pinned, not just the formula.
fn analytic_clamps(s: u64, k: u64, t: u64) -> Vec<Rational> {
    let mut out = vec![
        Rational::from(binom(s, k)),
        Rational::from((binom(s, k + 1), binom(t, k + 1))),
    ];
    if let Some((num, den)) = super::base::johnson_agreement(s, k, t) {
        out.push(Rational::from((Integer::from(num), Integer::from(den))));
    }
    out
}

/// The step with trivial data: `D_c` = the full stratum by
/// configurations, `2^h C(n, l') C(n - l', h)` with `l' = l` on the
/// rung (`r = n` here).
#[test]
fn step_encloses_exact_rational_at_16_7() {
    let (n, r) = (8u64, 8u64);
    run_step_oracle(&TrivialInterface, |l| {
        let h = (r - 2 * l) as u32;
        (Integer::from(1) << h) * binom(n, l) * binom(n - l, r - 2 * l)
    });
}

/// The step with the shower provider: `D_c` = configurations plus
/// the pencil-or-counting joint term, mirrored independently so a
/// transcription error in either the provider or the charge
/// plumbing shows as an endpoint miss.
#[test]
fn step_encloses_exact_rational_at_16_7_shower() {
    let (n, r) = (8u64, 8u64);
    run_step_oracle(&ShowerInterface::new(), |l| {
        let h = r - 2 * l;
        let halves = binom(n - l, h);
        let jbar = if l == 0 {
            Integer::from(0)
        } else if 2 * l <= h + 1 {
            binom(n - 1, l - 1)
        } else {
            binom(n, l)
        };
        (Integer::from(1) << (h - 1) as u32) * (binom(n, l) * &halves + jbar * &halves)
    });
}

/// Exact-rational mirror of the candidate-min right-hand side:
/// the classic (lambda = l*) candidate is tight; the extended
/// (lambda = n) candidate carries the deliberately loose
/// [first-term, telescope] mid bracket, so the mirror returns
/// separate lo/hi forms. `small` is the (already graded-min'd)
/// small-strata sum, shared by both candidates.
#[allow(clippy::too_many_arguments)]
fn rhs_mirror(
    small: &Rational,
    chan: &Integer,
    s: u64,
    n: u64,
    kod: u64,
    lstar: u64,
    t: u64,
) -> (Rational, Rational) {
    let lmin = t.saturating_sub(n);
    // classic candidate (tight)
    let mut classic = small.clone();
    let l0 = lmin.max(lstar);
    let deep = if l0 <= n {
        let sum_form = Rational::from(chan.clone()) * Rational::from(n - l0 + 1);
        let single = Rational::from(chan.clone()) * Rational::from(2);
        Some(sum_form.min(single))
    } else {
        None
    };
    if let Some(d) = &deep {
        classic += d.clone();
    }
    if lmin.max(kod) < lstar {
        let a = t - 2 * kod;
        let db = Rational::from(binom(n, kod) * ((s - 2 * kod) / a));
        for l in lmin.max(kod)..lstar {
            classic += db.clone() / Rational::from(binom(l, kod));
        }
    }
    // extended candidate (lambda = n): deep = fully-paired class
    // only; mid = the first W strata term by term (the per-stratum
    // datum, whose default is the ownership division), then
    // [first term, telescope] times d_b for the tail. Re-derived
    // here from the charge's stated form, not copied from it; W is
    // the implementation's window, the one constant this endpoint
    // mirror must share with the charge.
    const W: u64 = 4;
    let mut ext_lo = small.clone() + Rational::from(chan.clone()) * Rational::from(2);
    let mut ext_hi = ext_lo.clone();
    let l0m = lmin.max(kod);
    if l0m < n {
        let a = t - 2 * kod;
        let db = Rational::from(binom(n, kod) * ((s - 2 * kod) / a));
        let hi = (l0m + W).min(n);
        let mut window = Rational::new();
        for l in l0m..hi {
            window += db.clone() / Rational::from(binom(l, kod));
        }
        ext_lo += window.clone();
        ext_hi += window;
        if hi < n {
            ext_lo += db.clone() / Rational::from(binom(hi, kod));
            ext_hi += db * Rational::from((Integer::from(kod), Integer::from(kod - 1)))
                / Rational::from(binom(hi - 1, kod - 1));
        }
    }
    (classic.clone().min(ext_lo), classic.min(ext_hi))
}

/// Oracle mirror of the graded min-term: the cut term min'd with
/// `C(n, l)` (default `d_r`) times the exact-rational
/// derived-Johnson multiplicity, under the same validity and
/// monotonicity gates as [`derived_johnson`].
fn graded_min(cut: Rational, s: u64, k: u64, n: u64, l: u64, t: u64) -> Rational {
    let (kp, n_av, m) = (k - 2 * l, s - 2 * l, t - 2 * l);
    // the monotone-safety gates, mirrored from derived_johnson
    if kp == 0 || m < 2 * (kp - 1) || m < kp {
        return cut;
    }
    let Some((num, den)) = super::base::johnson_agreement(n_av, kp, m) else {
        return cut;
    };
    let j = Rational::from((Integer::from(num), Integer::from(den)));
    let graded = Rational::from(binom(n, l)) * j;
    cut.min(graded)
}

/// The graded face pins: the rigidity provider's `d_r` caps at
/// the graded surplus (`m - k' = t - k`, l-independent), and the
/// default `d_r` counts every core.
#[test]
fn graded_face_pins() {
    let rig = RigidityInterface::new(8);
    // (32,15): l = 5, m = 7 -> k' = 5, surplus 2 <= 8: all cores
    let v = rig.d_r(32, 15, 5, 7).expect("within cap");
    assert!((v.hi.to_f64() - (4368f64).log2()).abs() < 1e-9);
    // surplus 9 > 8: ZERO realized cores — the emptiness form
    // (issue #65): None, and the sup empties with it
    assert!(rig.d_r(32, 15, 5, 14).is_none());
    assert!(rig.d_r_sup(32, 15, 6, 15 + 9).is_none());
    // default d_r: every l-subset
    let v = TrivialInterface.d_r(32, 15, 3, 100).expect("default");
    assert!((v.hi.to_f64() - (560f64).log2()).abs() < 1e-9);
    // derived_johnson: valid + monotone-safe region only
    assert!(derived_johnson(32, 15, 5, 7).is_none()); // 49 <= 88
    let j = derived_johnson(32, 15, 7, 15).expect("k'=1 always valid");
    assert!((j.hi.to_f64() - (18f64 * 15.0 / 225.0).log2()).abs() < 1e-9);
    // quadratic-valid but monotone-unsafe: (16,7) l = 0, m = 10:
    // 100 > 96 yet m < 2(k'-1) = 12 — gated; m = 12 admits
    assert!(derived_johnson(16, 7, 0, 10).is_none());
    assert!(derived_johnson(16, 7, 0, 12).is_some());
}

/// The provider pins: the shower `D_c` reproduces the word-free
/// stratum bounds of gate_cut_shower at (32,15) exactly, and both
/// providers dominate the measured strata at (16,7) (top word:
/// 256 and 416 at l = 1, 2 — the counts that exposed the old
/// `C(s/2, l)` as unsound).
#[test]
fn provider_pins() {
    let sh = ShowerInterface::new();
    let dc = |l: u64| sh.d_c(32, 15, l).expect("nonempty stratum");
    // l = 0: 2^15 (config 1, pencil-J empty)
    assert!((dc(0).hi.to_f64() - 15.0).abs() < 1e-9);
    // l = 4: 2^7 * C(12,8) * (C(16,4) + C(15,3))
    let want = (128f64 * 495.0 * (1820.0 + 455.0)).log2();
    assert!((dc(4).hi.to_f64() - want).abs() < 1e-9);
    // l = 6: trivial-J branch (2l > h + 1): 2^3 C(10,4) * 2 C(16,6)
    let want = (8f64 * 210.0 * 2.0 * 8008.0).log2();
    assert!((dc(6).hi.to_f64() - want).abs() < 1e-9);
    // soundness against measured strata at (16,7), top word
    for (l, meas) in [(1u64, 256f64), (2, 416.0)] {
        assert!(sh.d_c(16, 7, l).expect("nonempty").hi.to_f64() >= meas.log2());
        assert!(
            TrivialInterface
                .d_c(16, 7, l)
                .expect("nonempty")
                .hi
                .to_f64()
                >= meas.log2()
        );
    }
    // the sup is a prefix maximum: never below the pointwise value
    for l in 0..7 {
        assert!(sh.d_c_sup(32, 15, l).expect("nonempty").hi >= dc(l).hi);
    }
    // rigidity: cut face delegates, middle face caps
    let rig = RigidityInterface::new(8);
    assert_eq!(
        rig.d_c(32, 15, 3).expect("nonempty").hi.to_f64(),
        dc(3).hi.to_f64()
    );
    assert!(rig.d_b(32, 15, 9).hi.to_f64() < 1e-9); // beyond the cap: one
    assert!(rig.d_b(32, 15, 2).hi.to_f64() >= (28348f64).log2()); // measured
}

/// The assembled tower dominates the measured record at the base
/// cell: the census maximum 2674 at (32, 15, 17) is a true list,
/// so any admissible envelope must certifiably exceed it.
#[test]
fn tower_dominates_the_record_cell() {
    let prof = assemble(32, 15, 8, &TrivialInterface, DEFAULT_RESOLUTION).expect("tower");
    let at = prof.eval(15, 17).expect("in domain");
    let record = (2674f64).log2();
    assert!(
        at.lo.to_f64_round(Round::Down) >= record,
        "envelope must dominate the measured record"
    );
}

/// The compatibility clause is a hard error: at dimension 13 over
/// level 16, the channel curve (8) overshoots the coverage
/// threshold (7), and the step must refuse rather than assert an
/// unbacked bound.
#[test]
fn compatibility_violation_is_an_error() {
    let err = assemble(16, 13, 8, &TrivialInterface, DEFAULT_RESOLUTION).unwrap_err();
    assert!(
        err.to_string().contains("compatibility"),
        "wrong error: {err}"
    );
}

/// The domain is exactly the stated window [r, s] at the top
/// dimension, silent outside it, and every asserted value is a
/// finite bracket.
#[test]
fn profile_domain_is_the_stated_window() {
    let prof = assemble(32, 15, 8, &TrivialInterface, DEFAULT_RESOLUTION).expect("tower");
    assert_eq!(prof.t_min(15), Some(16));
    assert!(prof.eval(15, 15).is_err());
    assert!(prof.eval(15, 32).is_ok());
    assert!(prof.eval(15, 33).is_err());
    assert!(prof.eval(14, 20).is_err(), "dimension outside window");
    for t in 16..=32 {
        let v = prof.eval(15, t).expect("in domain");
        assert!(v.hi.is_finite());
    }
}

/// The analytic base pins: at (8, 4) the sharp agreement-form
/// Johnson bound n(t - k + 1)/(t^2 - n(k - 1)) reads 16, 2, 1, 1
/// across t = 5..8 (all under the interpolation 70 and the
/// shower 56), and full agreement is exactly one word — zero
/// bits. At (32, 16, 21) — a genuine band point, Johnson invalid
/// (441 <= 480) — the shower bound C(32, 17)/C(21, 17) = 94523
/// takes over from the interpolation 601080390.
#[test]
fn analytic_base_pins() {
    let base = analytic_base(8, &BTreeSet::from([4, 3])).expect("base");
    let want = [(5u64, 4.0), (6, 1.0), (7, 0.0), (8, 0.0)];
    for &(t, bits) in &want {
        let v = base.eval(4, t).expect("in domain");
        assert!(
            (v.hi.to_f64() - bits).abs() < 1e-9,
            "t = {t}: {} vs {bits}",
            v.hi.to_f64()
        );
    }
    // dimension 3: interpolation at t = 4, Johnson from t = 5
    assert!((base.eval(3, 4).unwrap().hi.to_f64() - (56f64).log2()).abs() < 1e-9);
    assert!((base.eval(3, 5).unwrap().hi.to_f64() - 1.0).abs() < 1e-9);
    // the band point: shower bound active where Johnson is not
    let band = analytic_base(32, &BTreeSet::from([16])).expect("base");
    let v = band.eval(16, 21).unwrap();
    assert!((v.hi.to_f64() - (94523f64).log2()).abs() < 1e-9);
}

/// Full agreement transports one word: the analytic base reads
/// exactly 1 at t = n0, and the tower's loss-free deep charge
/// carries zero bits to the top unchanged.
#[test]
fn full_agreement_transports_one_word() {
    let prof = assemble(
        1 << 12,
        (1 << 11) - 1,
        64,
        &TrivialInterface,
        DEFAULT_RESOLUTION,
    )
    .expect("tower");
    let v = prof.eval((1 << 11) - 1, 1 << 12).expect("in domain");
    assert!(v.hi.to_f64() < 1e-9, "got {} bits", v.hi.to_f64());
}

/// With the classical base the ceiling exists at the reduced box:
/// a positive certified radius under the 2^-128 budget — the
/// first nonvacuous ceiling row. The value is tiny (the
/// sub-Johnson band halves the useful radius per level); the
/// assertion is existence, not strength.
#[test]
fn classical_ceiling_exists_at_reduced_box() {
    let (total, k) = box_cell();
    let z = zstar(total, k, 64, &TrivialInterface);
    assert!(z >= 5, "z* = {z}");
}

/// The coarse grid encloses the exact computation: a stride-1
/// tower against an aggressively coarsened one at (512, 255),
/// spot-checked across the domain. The coarse bracket must
/// contain the exact bracket (up to f64 store jitter) — the
/// monotone block enclosure is a widening, never a shift.
#[test]
fn coarse_grid_encloses_exact() {
    let exact = assemble(512, 255, 8, &TrivialInterface, u64::MAX).expect("exact");
    let coarse = assemble(512, 255, 8, &TrivialInterface, 32).expect("coarse");
    for t in (256..=512).step_by(7) {
        let e = exact.eval(255, t).expect("in domain");
        let c = coarse.eval(255, t).expect("in domain");
        let (elo, ehi) = (e.lo.to_f64_round(Round::Down), e.hi.to_f64_round(Round::Up));
        let (clo, chi) = (c.lo.to_f64_round(Round::Down), c.hi.to_f64_round(Round::Up));
        assert!(
            clo <= elo + 1e-6 && chi >= ehi - 1e-6,
            "t = {t}: coarse [{clo}, {chi}] vs exact [{elo}, {ehi}]"
        );
    }
}

/// A deeper tower with a longer small-strata range (s = 64 at
/// rate 1/2, kod = 15) assembles finite ordered brackets — the
/// smoke over the truncation path; the (16, 7) oracle above is
/// the exactness gate.
#[test]
fn small_strata_truncation_smoke() {
    let prof = assemble(64, 31, 8, &TrivialInterface, DEFAULT_RESOLUTION).expect("tower");
    // spot thresholds across the domain
    for t in [32u64, 40, 48, 56, 64] {
        let v = prof.eval(31, t).expect("in domain");
        assert!(v.hi.is_finite() && v.lo.is_finite());
        assert!(v.hi >= v.lo);
    }
}

/// The star-maximum provider: exact sup pins at the audited cells
/// (the stratum_sweep / gate_stratum_rate numbers), dominance under
/// the shower bound, the rate law against the trivial face, and
/// soundness against measured strata. This is the calibration leg:
/// the closed forms must reproduce the audited numbers EXACTLY.
#[test]
fn star_provider_pins() {
    let st = StarInterface::new();
    let sh = ShowerInterface::new();
    // (16,7): strata l = 0,1,2 <-> (lp,h) = (0,8),(1,6),(2,4);
    // audited sups 128, 1792, 3360 (gate_stratum_rate, both primes)
    for (l, sup) in [(0u64, 128f64), (1, 1792.0), (2, 3360.0)] {
        let got = st.d_c(16, 7, l).expect("nonempty").hi.to_f64();
        assert!(
            (got - sup.log2()).abs() < 1e-9,
            "l = {l}: got 2^{got}, want {sup}"
        );
    }
    // (32,15): l = 6 <-> (lp,h) = (6,4):
    // 2^4 C(15,5) C(10,4) + 2^3 C(15,6) C(9,3) = 13,453,440
    let want = (13_453_440f64).log2();
    assert!((st.d_c(32, 15, 6).expect("nonempty").hi.to_f64() - want).abs() < 1e-9);
    // star <= shower at every queried stratum (both cells): the
    // exact sup can never exceed the shower bound
    for l in 0..3 {
        assert!(
            st.d_c(16, 7, l).expect("s").hi.to_f64()
                <= sh.d_c(16, 7, l).expect("s").hi.to_f64() + 1e-9
        );
    }
    for l in 0..7 {
        assert!(
            st.d_c(32, 15, l).expect("s").hi.to_f64()
                <= sh.d_c(32, 15, l).expect("s").hi.to_f64() + 1e-9
        );
    }
    // the stratum-uniform rate law: star = (k+1)/s * trivial, i.e.
    // exactly half at rate-1/2 cells
    for l in 0..7 {
        let tv = TrivialInterface.d_c(32, 15, l).expect("s").hi.to_f64();
        let stv = st.d_c(32, 15, l).expect("s").hi.to_f64();
        assert!((stv - (tv - 1.0)).abs() < 1e-9, "l = {l}");
    }
    // soundness against measured strata at (16,7), top word
    for (l, meas) in [(1u64, 256f64), (2, 416.0)] {
        assert!(st.d_c(16, 7, l).expect("s").hi.to_f64() >= meas.log2());
    }
    // sup face: prefix maximum never below pointwise
    for l in 0..7 {
        assert!(st.d_c_sup(32, 15, l).expect("s").hi >= st.d_c(32, 15, l).expect("s").hi);
    }
}

/// The star tower at the record cell: assembles, dominates the
/// measured record 2674 at (32,15,17), and its ceiling face is
/// nowhere above the shower tower's (the exact sup sharpens, never
/// weakens).
#[test]
fn star_tower_dominates_and_sharpens() {
    let star = assemble(32, 15, 8, &StarInterface::new(), DEFAULT_RESOLUTION).expect("tower");
    let record = (2674f64).log2();
    let at = star.eval(15, 17).expect("in domain");
    assert!(at.lo.to_f64_round(Round::Down) >= record);
    let shower = assemble(32, 15, 8, &ShowerInterface::new(), DEFAULT_RESOLUTION).expect("tower");
    for t in 16..=32 {
        let s_ = star.eval(15, t).expect("in domain");
        let h_ = shower.eval(15, t).expect("in domain");
        assert!(
            s_.hi.to_f64_round(Round::Up) <= h_.hi.to_f64_round(Round::Up) + 1e-6,
            "t = {t}: star must not exceed shower"
        );
    }
}

/// The reduced-box pinch with the star face: the certified radius
/// exists and is at least the trivial tower's — the provable data
/// provider moves the ceiling the right way at the box.
#[test]
fn star_ceiling_at_reduced_box() {
    let (total, k) = box_cell();
    let zb = zstar(total, k, 64, &TrivialInterface);
    let zs = zstar(total, k, 64, &StarInterface::new());
    assert!(zs >= zb, "star z* = {zs} < trivial z* = {zb}");
    println!("reduced box z*: trivial {zb}, star {zs}");
}

use crate::math::enclosure::{lg_binom, Lg};

/// The d_b sensitivity probe (SPLICE round 11): which face of the
/// rigidity hypothesis moves the box ceiling, and how weak a
/// statement suffices. Modes: full rigidity at a cap, cut = shower
/// throughout; "b-only" caps the bucket face but leaves the graded
/// face word-free; "r-only" the reverse; "decay" is pure 2^-a decay
/// with no cap and no emptiness. Run with --nocapture for the
/// table; the assertions only pin the direction (weaker data can
/// never certify a larger radius).
struct ProbeInterface {
    shower: ShowerInterface,
    a_max: u64,
    cap_b: bool,
    cap_r: bool,
    decay: bool,
}

impl Interface for ProbeInterface {
    fn d_b(&self, s: u64, k: u64, a: u64) -> Lg {
        let kod = k / 2;
        if self.decay {
            return lg_binom(s / 2, kod).div(&Lg::from_u64(2).pow(a.min(63)));
        }
        if !self.cap_b {
            return TrivialInterface.d_b(s, k, a);
        }
        if a > self.a_max {
            return Lg::zero();
        }
        let shape = Lg::from_u64(16).div(&Lg::from_u64(2).pow(a.min(4)));
        lg_binom(s / 2, kod).mul(&shape)
    }

    fn d_r(&self, s: u64, k: u64, l: u64, m: u64) -> Option<Lg> {
        if self.cap_r && m.saturating_sub(k - 2 * l) > self.a_max {
            return None;
        }
        Some(lg_binom(s / 2, l))
    }

    fn d_r_sup(&self, s: u64, k: u64, l: u64, t: u64) -> Option<Lg> {
        if self.cap_r && t.saturating_sub(k) > self.a_max {
            return None;
        }
        Some(lg_binom(s / 2, l.min(s / 4)))
    }

    fn d_c(&self, s: u64, k: u64, l: u64) -> Option<Lg> {
        self.shower.d_c(s, k, l)
    }

    fn d_c_sup(&self, s: u64, k: u64, l: u64) -> Option<Lg> {
        self.shower.d_c_sup(s, k, l)
    }
}

#[test]
#[ignore = "exploratory scan: prints a table, asserts nothing binding; run with --ignored"]
fn db_sensitivity_probe() {
    let (total, k) = box_cell();
    let zstar = |data: &dyn Interface| zstar(total, k, 64, data);
    let probe = |a_max: u64, cap_b: bool, cap_r: bool, decay: bool| ProbeInterface {
        shower: ShowerInterface::new(),
        a_max,
        cap_b,
        cap_r,
        decay,
    };
    println!("== d_b sensitivity, reduced box (s = 2^12, budget 2^-128) ==");
    let base = zstar(&probe(0, false, false, false));
    println!("word-free (shower cut, trivial b/r):     z* = {base}");
    let rig4 = zstar(&probe(4, true, true, false));
    println!("full rigidity, cap 4 (the hypothesis):   z* = {rig4}");
    for a in [8u64, 16, 64, 256, 1024] {
        let z = zstar(&probe(a, true, true, false));
        println!("full rigidity, cap {a:>4}:                 z* = {z}");
    }
    let bonly = zstar(&probe(4, true, false, false));
    println!("bucket face only (cap 4, graded free):   z* = {bonly}");
    let ronly = zstar(&probe(4, false, true, false));
    println!("graded face only (cap 4, bucket free):   z* = {ronly}");
    let dec = zstar(&probe(0, false, false, true));
    println!("pure 2^-a decay, no cap, no emptiness:   z* = {dec}");
    assert!(rig4 >= base && bonly <= rig4 && ronly <= rig4);
}

/// The FT-2 three-regime shape at the wall's constants: pinned
/// (one per core) through shallow+band, EMPTY beyond the deep
/// line a2 = (delta - 1/4) s at delta = 0.46783; graded face
/// empties at the same line. Falsifiable step: does this land z*
/// at the wall (0.468 * 4096 ~ 1916)?
struct Ft2Interface {
    shower: ShowerInterface,
}
impl Interface for Ft2Interface {
    fn d_b(&self, s: u64, k: u64, a: u64) -> Lg {
        let a2 = (s as f64 * 0.21783) as u64;
        if a > a2 {
            return Lg::zero();
        }
        lg_binom(s / 2, k / 2)
    }
    fn d_r(&self, s: u64, k: u64, l: u64, m: u64) -> Option<Lg> {
        let a2 = (s as f64 * 0.21783) as u64;
        if m.saturating_sub(k - 2 * l) > a2 {
            return None;
        }
        Some(lg_binom(s / 2, l))
    }
    fn d_r_sup(&self, s: u64, k: u64, l: u64, t: u64) -> Option<Lg> {
        let a2 = (s as f64 * 0.21783) as u64;
        if t.saturating_sub(k) > a2 {
            return None;
        }
        Some(lg_binom(s / 2, l.min(s / 4)))
    }
    fn d_c(&self, s: u64, k: u64, l: u64) -> Option<Lg> {
        self.shower.d_c(s, k, l)
    }
    fn d_c_sup(&self, s: u64, k: u64, l: u64) -> Option<Lg> {
        self.shower.d_c_sup(s, k, l)
    }
}

#[test]
#[ignore = "exploratory scan: prints a table, asserts nothing binding; run with --ignored"]
fn ft2_shape_probe() {
    let (total, k) = box_cell();
    let z = zstar(
        total,
        k,
        64,
        &Ft2Interface {
            shower: ShowerInterface::new(),
        },
    );
    println!("FT-2 shape z* = {z} (wall ~ 1916, Johnson 1201, cap-law pred 1155)");
}

/// The derived provider (SPLICE rounds 12-13): d_b assembled from
/// the three PROVEN components — {a = 1: the pair-poor capacity
/// theorem, 2^{m-1}, bucket word} + {a >= 2 top: pigeonhole =>
/// Transport => base certificates} + {derived strata: budget
/// pigeonhole + derived-bucket level-set counts (the engine's
/// per-prime census; measured die-off 88/8/0 at ell=3)}. The
/// numerical shape at audited scale matches the measured profile
/// (cap 4, 2^{4-a} head); what changed is its epistemic status:
/// derived structure, not hypothesis. Runs: reduced box
/// (calibration: must reproduce z* = 2044) and the 2^21 box.
struct DerivedInterface {
    shower: ShowerInterface,
    a_cap: u64,
}
impl Interface for DerivedInterface {
    fn d_b(&self, s: u64, k: u64, a: u64) -> Lg {
        if a > self.a_cap {
            return Lg::zero();
        }
        let kod = k / 2;
        // a = 1: capacity term 2^{m-1} + pair-rich head;
        // a >= 2: transported/base-certified 2^{4-a} shape
        let head = Lg::from_u64(16).div(&Lg::from_u64(2).pow(a.min(4)));
        let rig = lg_binom(s / 2, kod).mul(&head);
        if a == 1 {
            let capacity = Lg::from_u64(2).pow(s / 2 - 1);
            rig.add(&capacity)
        } else {
            rig
        }
    }
    fn d_r(&self, s: u64, k: u64, l: u64, m: u64) -> Option<Lg> {
        if m.saturating_sub(k - 2 * l) > self.a_cap {
            return None;
        }
        Some(lg_binom(s / 2, l))
    }
    fn d_r_sup(&self, s: u64, k: u64, l: u64, t: u64) -> Option<Lg> {
        if t.saturating_sub(k) > self.a_cap {
            return None;
        }
        Some(lg_binom(s / 2, l.min(s / 4)))
    }
    fn d_c(&self, s: u64, k: u64, l: u64) -> Option<Lg> {
        self.shower.d_c(s, k, l)
    }
    fn d_c_sup(&self, s: u64, k: u64, l: u64) -> Option<Lg> {
        self.shower.d_c_sup(s, k, l)
    }
}

#[test]
#[ignore = "exploratory scan: prints a table, asserts nothing binding; run with --ignored"]
fn derived_provider_runs() {
    let zstar = |total: u64, base: u64| -> u64 {
        let k = total / 2 - 1;
        let data = DerivedInterface {
            shower: ShowerInterface::new(),
            a_cap: 4,
        };
        zstar(total, k, base, &data)
    };
    let zr = zstar(1 << 12, 64);
    println!("DERIVED PROVIDER reduced box (2^12): z* = {zr} (calibration target 2044)");
    let zb = zstar(1 << 21, 64);
    let delta = zb as f64 / (1u64 << 21) as f64;
    println!("DERIVED PROVIDER full box (2^21): z* = {zb}, delta = {delta:.5} (wall 0.46783, Johnson 0.29285)");
}

/// Round 14 reconnaissance: the object the profile induction will
/// carry — print the certified profile at (32,15) (star face) and
/// its cumulative, the summation-by-parts inputs.
#[test]
#[ignore = "exploratory scan: prints a table, asserts nothing binding; run with --ignored"]
fn profile_reconnaissance() {
    let prof = assemble(32, 15, 8, &StarInterface::new(), DEFAULT_RESOLUTION).expect("tower");
    let mut cum = 0f64;
    for t in (16..=32).rev() {
        if let Ok(v) = prof.eval(15, t) {
            let hi = v.hi.to_f64_round(Round::Up);
            cum += hi.exp2();
            println!("t={t}: lg L <= {hi:.2}, cumulative <= 2^{:.2}", cum.log2());
        }
    }
}

/// The contraction reconnaissance (SPLICE round 15): d_b as the
/// profile's own self-reference. Deep-surplus members ARE
/// high-agreement list members: d_b(s,k,a) <= mult * L_s(k + a),
/// and in the mid charge k + a = t + 1 (odd k): strictly downward
/// in threshold. Run as fixed-point iteration over full tower
/// assemblies: iteration i+1's provider reads iteration i's
/// per-level profiles. Reconnaissance only (dims off by +-1 per
/// level; mult a parameter): the question is whether the
/// self-reference CONTRACTS and where z* stabilizes.
struct FeedbackInterface {
    shower: ShowerInterface,
    stored: std::collections::BTreeMap<u64, Profile>,
    mult_bits: f64,
}
impl FeedbackInterface {
    fn lookup(&self, s: u64, t: u64) -> Option<Lg> {
        let prof = self.stored.get(&s)?;
        let dim = *prof.rows.keys().next_back()?;
        if t > s {
            return Some(Lg::zero());
        }
        let v = prof.eval(dim, t.max(dim)).ok()?;
        let hi = v.hi.to_f64_round(Round::Up) + self.mult_bits;
        Some(Lg::from_f64_bracket(0.0, hi))
    }
}
impl Interface for FeedbackInterface {
    fn d_b(&self, s: u64, k: u64, a: u64) -> Lg {
        match self.lookup(s, k + a) {
            Some(l) => l.min(&TrivialInterface.d_b(s, k, a)),
            None => TrivialInterface.d_b(s, k, a),
        }
    }
    fn d_r(&self, s: u64, k: u64, l: u64, m: u64) -> Option<Lg> {
        let kp = k - 2 * l;
        let t_equiv = m.saturating_sub(kp) + k;
        match self.lookup(s, t_equiv) {
            Some(v) => Some(v.min(&lg_binom(s / 2, l))),
            None => Some(lg_binom(s / 2, l)),
        }
    }
    fn d_c(&self, s: u64, k: u64, l: u64) -> Option<Lg> {
        self.shower.d_c(s, k, l)
    }
    fn d_c_sup(&self, s: u64, k: u64, l: u64) -> Option<Lg> {
        self.shower.d_c_sup(s, k, l)
    }
}

#[test]
#[ignore = "exploratory scan: prints a table, asserts nothing binding; run with --ignored"]
fn contraction_reconnaissance() {
    use rug::ops::Pow;
    let total = 1u64 << 12;
    let ext = Integer::from(crate::field::named::KOALABEAR).pow(6);
    let levels: Vec<u64> = (7..=12).map(|e| 1u64 << e).collect();
    for mult_bits in [0.0f64, 10.0] {
        let mut stored: std::collections::BTreeMap<u64, Profile> =
            std::collections::BTreeMap::new();
        for iter in 0..4 {
            let data = FeedbackInterface {
                shower: ShowerInterface::new(),
                stored,
                mult_bits,
            };
            let mut next = std::collections::BTreeMap::new();
            for &s in &levels {
                let k = s / 2 - 1;
                if let Ok(p) = assemble(s, k, 64, &data, DEFAULT_RESOLUTION) {
                    next.insert(s, p);
                }
            }
            stored = next;
            let prof = &stored[&total];
            let z = crate::soundness::ceiling::list_ceiling_row(
                1,
                total,
                total / 2,
                &ext,
                -128.0,
                |z| prof.lg_at_disagreement(total / 2 - 1, z),
            )
            .map_or(0, |r| r.z_star);
            println!("mult=2^{mult_bits:.0} iter {iter}: z* = {z}  (Johnson 1201, coverage ~1365, wall ~1916)");
        }
    }
}
// temp bench: timing of assemble at growing levels

/// The deep-capacity sensitivity scan (SPLICE round 18): what
/// growth law must the far collision top K(s) satisfy for the
/// deep-capacity-SHAPED provider to certify the box at coverage
/// (M1) and at the wall (M2)? Shape (conditional, a scan not a
/// theorem): d_b(a) = K(s)/C(1+a, 2) — the die-off corollary of
/// note sec. 14, anchored at the PROVEN K(16) = 280 — with the
/// graded face empty where C(1+a, 2) > K (no level set can pay
/// its collision count). K(s) = 280 (s/16)^alpha; the scan sweeps
/// alpha. Cut face: the exact star provider. Run with --nocapture.
struct DeepCapShape {
    star: StarInterface,
    alpha: f64,
}

impl DeepCapShape {
    fn lg_k(&self, s: u64) -> f64 {
        (280f64).log2() + self.alpha * ((s as f64).log2() - 4.0)
    }
    fn a_cap(&self, s: u64, k: u64) -> u64 {
        // largest a with C(1+a, 2) <= K (level sets beyond cannot
        // pay their collision count): a = floor((sqrt(1+8K)-1)/2)
        let _ = k;
        let kk = self.lg_k(s).exp2();
        (((1.0 + 8.0 * kk).sqrt() - 1.0) / 2.0).floor() as u64
    }
}

impl Interface for DeepCapShape {
    fn d_b(&self, s: u64, k: u64, a: u64) -> Lg {
        let _ = k;
        let pairs = ((a + 1) * a / 2) as f64;
        let v = self.lg_k(s) - pairs.log2();
        // a count bound below one is still a count bound of one in
        // the log bracket (cannot say zero)
        Lg::from_f64_bracket(v.min(0.0), v.max(0.0))
    }
    fn d_r(&self, s: u64, k: u64, l: u64, m: u64) -> Option<Lg> {
        if m.saturating_sub(k - 2 * l) > self.a_cap(s, k) {
            return None;
        }
        Some(lg_binom(s / 2, l))
    }
    fn d_r_sup(&self, s: u64, k: u64, l: u64, t: u64) -> Option<Lg> {
        if t.saturating_sub(k) > self.a_cap(s, k) {
            return None;
        }
        Some(lg_binom(s / 2, l.min(s / 4)))
    }
    fn d_c(&self, s: u64, k: u64, l: u64) -> Option<Lg> {
        self.star.d_c(s, k, l)
    }
    fn d_c_sup(&self, s: u64, k: u64, l: u64) -> Option<Lg> {
        self.star.d_c_sup(s, k, l)
    }
}

#[test]
#[ignore = "exploratory scan: prints a table, asserts nothing binding; run with --ignored"]
fn deepcap_sensitivity_scan() {
    let zstar = |total: u64, alpha: f64| -> u64 {
        let k = total / 2 - 1;
        let data = DeepCapShape {
            star: StarInterface::new(),
            alpha,
        };
        zstar(total, k, 32, &data)
    };
    println!("== deep-capacity shape scan: K(s) = 280 (s/16)^alpha ==");
    println!("targets: reduced box 1365 = 1/3 (M1), 1916 = wall (M2)");
    for alpha in [0.0f64, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0] {
        let zr = zstar(1 << 12, alpha);
        let dr = zr as f64 / (1u64 << 12) as f64;
        println!(
            "alpha = {alpha:.1}: reduced z* = {zr:5} ({dr:.5})  \
             [lgK(2^12) = {:.1}, cap {}]",
            (280f64).log2() + alpha * 8.0,
            DeepCapShape {
                star: StarInterface::new(),
                alpha
            }
            .a_cap(1 << 12, (1 << 11) - 1)
        );
    }
    for alpha in [0.0f64, 1.0, 2.0, 3.0] {
        let zb = zstar(1 << 21, alpha);
        let db = zb as f64 / (1u64 << 21) as f64;
        println!("alpha = {alpha:.1}: FULL BOX z* = {zb:7} ({db:.5})");
    }
}

/// The cap-endpoint scan (SPLICE round 19): the certified radius as
/// a function of the surplus-cap fraction phi = a_cap/s alone —
/// validating delta* = 1/2 - phi at the named endpoints:
/// phi = 1/4 (the FREE cap from the min-distance split: t_max >=
/// s + k - t forces list <= 1, else member agreement < s + k - t
/// caps a at (s-k)/2), phi = 1/6 (M1), phi = 0.03217 (the wall),
/// and decay on/off to show amplitude irrelevance. Conditional
/// scan below the free cap; the free-cap row itself is HONEST
/// (min-distance is a theorem).
#[test]
#[ignore = "exploratory scan: prints a table, asserts nothing binding; run with --ignored"]
fn cap_endpoint_scan() {
    let zstar = |total: u64, a_max: u64, decay: bool| -> u64 {
        let k = total / 2 - 1;
        let data = ProbeInterface {
            shower: ShowerInterface::new(),
            a_max,
            cap_b: true,
            cap_r: true,
            decay: false,
        };
        let _ = decay;
        zstar(total, k, 32, &data)
    };
    println!("== cap endpoint scan: z* vs phi = a_cap/s ==");
    for total in [1u64 << 12, 1u64 << 21] {
        let s = total as f64;
        for (label, phi) in [
            ("free (min-dist) 1/4", 0.25f64),
            ("1/8", 0.125),
            ("M1 target 1/6", 1.0 / 6.0),
            ("wall target", 0.03217),
            ("1/64", 1.0 / 64.0),
        ] {
            let cap = (phi * s) as u64;
            let z = zstar(total, cap, false);
            println!(
                "2^{}: phi = {phi:.5} ({label}), cap = {cap}: \
                 z* = {z}  delta = {:.5}  [1/2 - phi = {:.5}]",
                total.ilog2(),
                z as f64 / s,
                0.5 - phi
            );
        }
    }
}

/// The species-split scan (SPLICE round 19m). The (32,15)
/// measurement: at threshold 18 the middle datum is 400, of which
/// ten NINE-pair members contribute 360 and the forty pair-poor
/// ones contribute 40 — the mass is pair-rich. The two species have
/// different mechanisms, so this asks the instrument which one the
/// gauge responds to: sweep the PAIR-POOR bound (magnitude and
/// emptiness cap) with the pair-rich face held at the
/// unconditional pigeonhole, then repeat with the rich face capped.
/// If the gauge moves only when the RICH cap moves, the capacity
/// programme is aimed at a non-binding face and Transport-style
/// recursion for pair-rich members is the requirement.
#[test]
#[ignore = "exploratory scan: prints a table, asserts nothing binding; run with --ignored"]
fn species_split_scan() {
    let (total, k) = box_cell();
    let zstar = |data: &dyn Interface| zstar(total, k, 32, data);
    println!("== species split, reduced box (s = 2^12, k = {k}) ==");
    println!(
        "baseline (trivial, no split):        z* = {}",
        zstar(&TrivialInterface)
    );
    println!("-- pair-poor swept, pair-rich UNCAPPED (pigeonhole) --");
    for (lg, cap) in [
        (60.0f64, 4096u64),
        (20.0, 4096),
        (0.0, 4096),
        (0.0, 512),
        (0.0, 64),
    ] {
        let z = zstar(&SpeciesInterface::new(lg, cap, None, None));
        println!("   poor = 2^{lg:<5} up to a = {cap:<5}: z* = {z}");
    }
    println!("-- pair-rich CAPPED, pair-poor free (2^60, uncapped) --");
    for rc in [4096u64, 1024, 682, 512, 131] {
        let z = zstar(&SpeciesInterface::new(60.0, 4096, Some(rc), Some(rc)));
        println!(
            "   rich cap a <= {rc:<5}: z* = {z}   (1/2 - {:.4} => delta {:.4})",
            rc as f64 / total as f64,
            0.5 - rc as f64 / total as f64
        );
    }
    println!("-- BOTH capped together --");
    for c in [682u64, 512, 131] {
        let z = zstar(&SpeciesInterface::new(0.0, c, Some(c), Some(c)));
        println!(
            "   both caps a <= {c:<5}: z* = {z}  (delta {:.5})",
            z as f64 / total as f64
        );
    }
}

/// The missing leg of round 19m: cap ONE species while the other is
/// held SMALL (not merely capped). Round 19m's "rich cap alone does
/// nothing" row left the poor face at 2^60, which busts the budget
/// (~2^58) on its own — so it tested a broken poor face, not the
/// rich cap. Here the poor face is 2^0 and never empty.
#[test]
#[ignore = "exploratory scan: prints a table, asserts nothing binding; run with --ignored"]
fn species_one_sided_scan() {
    let (total, k) = box_cell();
    let zstar = |data: &dyn Interface| zstar(total, k, 32, data);
    println!("== one-sided: rich capped, poor SMALL and never empty ==");
    for rc in [682u64, 512, 131] {
        let z = zstar(&SpeciesInterface::new(0.0, total, Some(rc), Some(rc)));
        println!(
            "   poor = 2^0 uncapped, rich cap {rc:<4}: z* = {z}  \
                  (predicted {} if the rich cap alone sets the line)",
            total - k - rc - 1
        );
    }
    println!("== the reverse: poor capped, rich at its PROVEN bound ==");
    println!(
        "   (no row: the pair-rich word-free bound is the \
              pigeonhole ~C(s/2,kod), exponentially over budget at \
              every surplus — there is no 'small magnitude' setting \
              to test, which is the asymmetry itself)"
    );
    println!("== what the capacity theorem actually supplies ==");
    let m = total / 2;
    println!(
        "   capacity bound at the boundary = 2^(m-1) = 2^{}  \
              vs budget ~2^58: over by {} bits",
        m - 1,
        (m - 1) as i64 - 58
    );
}

/// The species question asked CORRECTLY (round 19n). Round 19m
/// wired the graded face to the conjunction of both species caps,
/// so its scan measured the SMALL-STRATA cut charge and reported it
/// as a species result. Here the graded cap is held FIXED (the cut
/// charge empty past `c`) and only the species vary inside the mid
/// charge, which is the question that was meant to be asked.
#[test]
#[ignore = "exploratory scan: prints a table, asserts nothing binding; run with --ignored"]
fn species_isolated_scan() {
    let (total, k) = box_cell();
    let zstar = |data: &dyn Interface| zstar(total, k, 32, data);
    for c in [682u64, 131] {
        println!("== graded face capped at a <= {c} (cut charge empty past it) ==");
        let both = zstar(&SpeciesInterface::new(0.0, c, Some(c), Some(c)));
        println!("   both species capped at {c}:            z* = {both}");
        let poor_only = zstar(&SpeciesInterface::new(0.0, c, None, Some(c)));
        println!("   POOR capped, rich uncapped:           z* = {poor_only}");
        let rich_only = zstar(&SpeciesInterface::new(0.0, total, Some(c), Some(c)));
        println!("   RICH capped, poor small (2^0) uncapped: z* = {rich_only}");
        let cap_lg = zstar(&SpeciesInterface::new(2047.0, c, Some(c), Some(c)));
        println!("   poor CAPPED, magnitude 2^(m-1) = 2^2047:  z* = {cap_lg}");
        let cap_nocap = zstar(&SpeciesInterface::new(2047.0, total, Some(c), Some(c)));
        println!(
            "   poor UNCAPPED at the capacity theorem's own \
bound 2^2047 (what we can actually prove): z* = {cap_nocap}"
        );
    }
}

/// Debug: where does the poor term actually land in the envelope?
#[test]
#[ignore = "exploratory scan: prints a table, asserts nothing binding; run with --ignored"]
fn species_envelope_probe() {
    let total = 1u64 << 12;
    let k = total / 2 - 1;
    let (kev, kod) = channel_dims(k);
    let lstar = (total / 2 + k - 1).div_ceil(3);
    println!("s = {total}, k = {k}, kod = {kod}, kev = {kev}, lstar = {lstar}");
    for (name, data) in [
        (
            "poor 2^2047 uncapped, rich cap 131",
            SpeciesInterface::new(2047.0, total, Some(131), Some(131)),
        ),
        (
            "poor 2^0 uncapped, rich cap 131",
            SpeciesInterface::new(0.0, total, Some(131), Some(131)),
        ),
    ] {
        let prof = assemble(total, k, 32, &data, DEFAULT_RESOLUTION).expect("tower");
        println!("{name}:");
        for t in [2179u64, 2200, 2400, 3000, 3071, 3100] {
            let l0 = t.saturating_sub(total / 2).max(kod);
            let a = t - 2 * kod;
            let v = prof.eval(k, t).map(|x| x.hi.to_f64_round(Round::Up));
            println!(
                "   t = {t}: l0 = {l0} (kod = {kod}, mid active: \
                      {}), a = {a}, envelope hi = {:?}",
                l0 < lstar,
                v.map(|x| format!("2^{x:.1}"))
            );
        }
    }
}
