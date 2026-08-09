//! The envelope's gates: exact-rational oracles for the step (the
//! candidate-min right-hand side mirrored independently), provider
//! pins against measured strata, tower-level regressions.

use super::charges::derived_johnson;
use super::*;
use rug::float::Round;
use rug::{Integer, Rational};
use std::collections::BTreeSet;

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
    if l0 <= n {
        let sum_form = Rational::from(chan.clone()) * Rational::from(n - l0 + 1);
        let single = Rational::from(chan.clone()) * Rational::from(2);
        classic += sum_form.min(single);
    }
    if lmin.max(kod) < lstar {
        let a = t - 2 * kod;
        let db = Rational::from(binom(n, kod) * ((s - 2 * kod) / a));
        for l in lmin.max(kod)..lstar {
            classic += db.clone() / Rational::from(binom(l, kod));
        }
    }
    // extended candidate (lambda = n): deep = fully-paired class
    // only; mid = [first term, telescope] times d_b
    let mut ext_lo = small.clone() + Rational::from(chan.clone()) * Rational::from(2);
    let mut ext_hi = ext_lo.clone();
    let l0m = lmin.max(kod);
    if l0m < n {
        let a = t - 2 * kod;
        let db = Rational::from(binom(n, kod) * ((s - 2 * kod) / a));
        ext_lo += db.clone() / Rational::from(binom(l0m, kod));
        ext_hi += db * Rational::from((Integer::from(kod), Integer::from(kod - 1)))
            / Rational::from(binom(l0m - 1, kod - 1));
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
    // surplus 9 > 8: at most one realized core
    let v = rig.d_r(32, 15, 5, 14).expect("capped");
    assert!(v.hi.to_f64() < 1e-9);
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
    use rug::ops::Pow;
    let total = 1u64 << 12;
    let k = total / 2 - 1;
    let ext = Integer::from(crate::field::named::KOALABEAR).pow(6);
    let prof = assemble(total, k, 64, &TrivialInterface, DEFAULT_RESOLUTION).expect("tower");
    let row =
        crate::soundness::ceiling::list_ceiling_row(1, total, total - k - 1, &ext, -128.0, |z| {
            prof.lg_at_disagreement(k, z)
        })
        .expect("a positive ceiling");
    assert!(row.z_star >= 5, "z* = {}", row.z_star);
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
