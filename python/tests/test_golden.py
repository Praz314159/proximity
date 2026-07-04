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
