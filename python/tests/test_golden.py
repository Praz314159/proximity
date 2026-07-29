"""Python-binding golden parity tests (mirror of tests/golden.rs pins).

Run with pytest after installing the wheel:
    maturin build --release --features python && pip install target/wheels/*.whl
    pytest python/tests/
"""
import numpy as np
import vanish


def test_bucket_dist_q1_golden():
    d = np.asarray(vanish.bucket_dist_q1(89633, 32, 16))
    assert d.max() == 29382
    assert d.sum() == 601080390
    d = np.asarray(vanish.bucket_dist_q1(65537, 32, 16))
    assert d.max() == 12870


def test_mitm_rung_and_decompose():
    assert vanish.bucket_e(3457, 32, 16, vanish.rung_lambda_e(3457, 32, 16, 2)) == 422
    d = np.asarray(vanish.bucket_dist_q1(77569, 32, 16))
    total, per_w = vanish.decompose_bucket_q1(77569, 32, 16, int(d.argmax()))
    assert total == d.max() == 30598
    assert (per_w[0], per_w[6], per_w[8], per_w[10], per_w[12]) == (1, 32, 96, 128, 64)


def test_census_and_utils():
    assert vanish.census_mitm(65537, 32, 2)[2:5] == [32, 32, 448]
    assert vanish.m_struct(32, 16, 3) == 70
    assert vanish.is_prime(2130706433)
    fs = vanish.factor(1331716)
    assert np.prod(np.array(fs, dtype=object)) == 1331716


def test_errors_raise_valueerror():
    import pytest

    with pytest.raises(ValueError):
        vanish.bucket_dist_q1(3458, 32, 16)  # composite p
    with pytest.raises(ValueError):
        vanish.subgroup(97, 31)  # s does not divide p - 1


if __name__ == "__main__":
    test_bucket_dist_q1_golden()
    test_mitm_rung_and_decompose()
    test_census_and_utils()
    print("python golden parity OK")


def test_sweep_stats_and_certify_bindings():
    # dist_stats_q1 must agree with the full distribution
    d = np.asarray(vanish.bucket_dist_q1(3457, 32, 16))
    mx, arg, occ, tot, m2 = vanish.dist_stats_q1(3457, 32, 16)
    assert (mx, arg) == (int(d.max()), int(d.argmax()))
    assert occ == int((d > 0).sum()) and tot == int(d.sum())
    assert m2 == int((d.astype(object) ** 2).sum())
    # parallel sweep returns the same rows
    rows = vanish.sweep_stats_q1(32, 16, [97, 3457, 89633])
    by_p = {r[0]: r for r in rows}
    assert by_p[89633][1] == 29382 and by_p[97][1] == 6196723
    # certify tiers: inflated / tier-2 / tier-1
    assert vanish.certify_q1(89633, 32, 16) == (3, 12870, 29382)
    assert vanish.certify_q1(65537, 32, 16)[0] == 2
    assert vanish.certify_q1(562949953421729, 32, 16) == (1, 12870, 12870)


def test_ergonomics_pass_bindings():
    # primes_1_mod matches a known population count
    ps = vanish.primes_1_mod(32, 33, 300000)
    assert len(ps) == 1622 and ps[0] == 97
    # class_size pins
    assert vanish.class_size(32, 16, 0) == 12870
    assert vanish.class_size(32, 16, 6) == 252
    assert vanish.class_size(32, 16, 7) == 0
    # decompose_many agrees with singles
    rows = {p: (t, pw) for p, t, pw in vanish.decompose_many(32, 16, [89633, 77569])}
    assert rows[89633][0] == 29382 and rows[77569][0] == 30598
    # attack bindings reproduce the golden thresholds
    lam = 6 * np.log2(2130706433) - 128
    assert abs(vanish.attack_antipodal(1 << 21, 1 << 20, lam)[0] - 0.492188) < 1e-5
    best = vanish.attack_best(1 << 21, 1 << 20, lam)
    assert abs(best[0] - 0.4843755) < 1e-6 and best[2] == 15
    # toy soundness pins
    w, s_, c = vanish.toy_soundness(35521, 16, 8)
    assert (w, c) == (3281, 3281) and abs(s_ - 3281 / 35521) < 1e-12


def test_round3_parallel_bindings():
    rows = {p: b for p, b in vanish.rung_buckets_many(32, 16, [1, 2, 8], [3457, 89633])}
    assert rows[3457] == [220134, 422, 2]
    assert rows[89633][0] == 29382
    certs = {p: (t, m, z) for p, t, m, z in vanish.certify_many(32, 16, [89633, 65537])}
    assert certs[89633] == (3, 12870, 29382) and certs[65537][0] == 2


def test_cyclo_content_map():
    # the hand census at s = 8 (counting chapter, sec:cc-census):
    # three shell sets, values 2, 4 + 2*sqrt2, 4 - 2*sqrt2
    assert vanish.Cyclo.prod_one_minus(8, [1, 3, 5, 7]).eq_int(2)
    assert vanish.Cyclo.prod_one_minus(8, [2, 3, 5, 6]).coeffs() == [4, 2, 0, -2]
    assert vanish.Cyclo.prod_one_minus(8, [1, 2, 6, 7]).coeffs() == [4, -2, 0, 2]
    # fold identity at s = 32
    for e in (1, 3, 8, 11):
        lhs = vanish.Cyclo.one_minus(32, e).mul(vanish.Cyclo.one_minus(32, e + 16))
        assert lhs == vanish.Cyclo.one_minus(32, 2 * e)
    # integer discrimination
    assert vanish.Cyclo.one_minus(8, 4).as_int() == 2
    assert vanish.Cyclo.one_minus(8, 2).as_int() is None


def test_cyclo_e_vector():
    # e_j of all nonzero exponents = (-1)^j
    es = vanish.Cyclo.e_vector(16, list(range(1, 16)), 5)
    for j, ej in enumerate(es):
        assert ej.eq_int(1 if j % 2 == 0 else -1)
    # alternating sum of the e-vector = the content product (Vieta)
    exps = [1, 3, 5, 7]
    es = vanish.Cyclo.e_vector(16, exps, 4)
    alt = vanish.Cyclo([0] * 8)
    for j, ej in enumerate(es):
        alt = alt.add(ej) if j % 2 == 0 else alt.sub(ej)
    assert alt == vanish.Cyclo.prod_one_minus(16, exps)


def test_fold_units():
    # closed form vs quotient: u_e * (1 - z^e) = 1 + z^e
    for e in (1, 5, 8, 12):
        u = vanish.fold_unit(32, e)
        lhs = u.mul(vanish.Cyclo.one_minus(32, e))
        rhs = vanish.Cyclo.monomial(32, 0).add(vanish.Cyclo.monomial(32, e))
        assert lhs == rhs
    # the two exact identities
    assert vanish.fold_unit(32, 3).mul(vanish.fold_unit(32, 13)).eq_int(-1)
    assert vanish.fold_unit(32, 8) == vanish.Cyclo.monomial(32, 8)
    # the rank certificate at the working levels
    for s in (16, 32, 64):
        lo, hi, independent = vanish.foldunit_rank_certificate(s)
        assert independent and not (lo <= 0.0 <= hi)


def test_valuemap_census():
    # the shell census pins (stages 49/50/54), standard prime
    P = 2130706433
    h1, h2 = list(range(1, 16)), list(range(17, 32))
    total, distinct, mx, arg, sm = vanish.valuemap_census(P, 32, h1, h2, 16, 0, 1)
    assert (total, distinct, mx, arg) == (4544445, 275247, 1250, 4)
    assert sm == 448183873
    assert vanish.valuemap_fiber(P, 32, h1, h2, 16, 0, 1, 4) == 1250
    members = vanish.valuemap_fiber_members(P, 32, h1, h2, 16, 0, 1, 4, 2)
    for m in members:
        assert vanish.Cyclo.prod_one_minus(32, m).eq_int(4)


def test_valuemap_distribution_and_sweep():
    P = 2130706433
    h1, h2 = list(range(1, 16)), list(range(17, 32))
    hist = np.asarray(vanish.valuemap_histogram(P, 32, h1, h2, 16, 0, 1))
    assert hist[1250] == 1 and hist.sum() == 275247
    values, counts = vanish.valuemap_distribution(P, 32, h1, h2, 16, 0, 1)
    values, counts = np.asarray(values), np.asarray(counts)
    assert counts.sum() == 4544445 and counts.max() == 1250
    assert values[counts.argmax()] == 4
    # histogram consistent with distribution
    assert np.array_equal(np.bincount(counts.astype(int)), hist)
    # sweep: max fiber and argmax are p-independent (floors)
    rows = vanish.valuemap_sweep(32, h1, h2, 16, 0, 1, [2130706433, 2113929217])
    for p, total, distinct, mx, arg, sm in rows:
        assert (total, mx, arg) == (4544445, 1250, 4)


def test_skeleton_census():
    # G1 skeleton kernel: S3/S4-certified golden pins (2026-07-28).
    assert vanish.skeleton_totals(32) == (3492117, 356588, 178304)
    assert vanish.skeleton_totals(64) == (
        106495542464222, 3049510275016, 1524755137544)
    m1, m2, solvable, sols = vanish.skeleton_census(32)
    assert (m1, m2, solvable) == (31788, 20288, 15564)
    # the stage-67-certified counting instrument: |solutions(32)| exactly
    assert sols == 26084


def test_alpha_certificate():
    # certified atom addresses: D = 8 at the skeleton levels, sixteenths
    # at 128 (first certified table beyond the measured regime)
    denom, alpha, tors, resid, gap = vanish.foldunit_alpha_certificate(64)
    assert denom == 8 and len(alpha) == 63
    assert all(x == 0 for x in alpha[0]) and tors[0] == 0
    assert resid < 1e-4 * gap
    import math
    for ji, row in enumerate(alpha):
        ord_ = 64 // math.gcd(ji + 1, 64)
        step = 8 // min(max(ord_ // 8, 1), 8)
        assert all(x % step == 0 for x in row), f"atom {ji+1}"
    denom128, alpha128, _, resid128, gap128 = vanish.foldunit_alpha_certificate(128)
    assert denom128 == 16 and len(alpha128) == 127 and resid128 < 1e-4 * gap128
