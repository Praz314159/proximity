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
