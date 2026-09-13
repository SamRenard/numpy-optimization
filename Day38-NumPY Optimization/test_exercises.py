"""
Day 38 — Automated correctness checks for numpy_exercises_01_25.py
NIZAM AI · 150-Day AI Engineering Protocol
"""

import numpy as np
from numpy_exercises_01_25 import (
    ex02, ex03, ex05, ex06, ex07, ex08, ex09, ex10, ex11, ex12, ex13,
    ex14, ex15, ex16, ex17, ex18, ex19, ex20, ex21, ex22, ex23, ex24, ex25,
)


def test_ex02():
    a = ex02()
    assert a.shape == (10,) and np.all(a == 0)
    print("[PASS] 02: null vector of size 10")


def test_ex03():
    assert ex03(np.arange(10)) == 80  # 10 elements * 8 bytes (int64)
    print("[PASS] 03: memory size calculation")


def test_ex05():
    a = ex05()
    assert a[4] == 1 and a.sum() == 1
    print("[PASS] 05: null vector with 1 at index 4")


def test_ex06():
    a = ex06()
    assert a[0] == 10 and a[-1] == 49 and len(a) == 40
    print("[PASS] 06: range 10..49")


def test_ex07():
    a = ex07(np.arange(5))
    assert list(a) == [4, 3, 2, 1, 0]
    print("[PASS] 07: reverse vector")


def test_ex08():
    a = ex08()
    assert a.shape == (3, 3) and a[0, 0] == 0 and a[2, 2] == 8
    print("[PASS] 08: 3x3 matrix 0..8")


def test_ex09():
    idx = ex09()[0]
    assert list(idx) == [0, 1, 4]
    print("[PASS] 09: nonzero indices")


def test_ex10():
    a = ex10()
    assert np.array_equal(a, np.eye(3))
    print("[PASS] 10: identity matrix")


def test_ex11():
    a = ex11()
    assert a.shape == (3, 3, 3) and a.min() >= 0 and a.max() <= 1
    print("[PASS] 11: 3x3x3 random array")


def test_ex12():
    mn, mx = ex12()
    assert 0 <= mn <= mx <= 1
    print("[PASS] 12: min/max of random array")


def test_ex13():
    m = ex13()
    assert 0 <= m <= 1
    print("[PASS] 13: mean of random vector")


def test_ex14():
    a = ex14(5)
    assert np.all(a[0, :] == 1) and np.all(a[-1, :] == 1)
    assert np.all(a[1:-1, 1:-1] == 0)
    print("[PASS] 14: bordered array")


def test_ex15():
    a = ex15(np.ones((2, 2)))
    assert a.shape == (4, 4)
    assert np.all(a[0, :] == 0) and np.all(a[:, 0] == 0)
    print("[PASS] 15: zero-padded border")


def test_ex16():
    results = ex16()
    assert np.isnan(results["0 * np.nan"])
    assert results["np.nan == np.nan"] is False
    assert results["np.inf > np.nan"] is False
    assert results["np.nan in set([np.nan])"] is True
    assert results["0.3 == 3 * 0.1"] is False
    print("[PASS] 16: NaN/inf edge cases")


def test_ex17():
    a = ex17()
    assert a[1, 0] == 1 and a[2, 1] == 2 and a[3, 2] == 3 and a[4, 3] == 4
    print("[PASS] 17: sub-diagonal matrix")


def test_ex18():
    a = ex18()
    assert a.shape == (8, 8)
    assert a[0, 0] == 0 and a[0, 1] == 1 and a[1, 0] == 1
    print("[PASS] 18: checkerboard via slicing")


def test_ex19():
    idx = ex19()
    assert idx == (1, 5, 3)
    flat_index = np.ravel_multi_index(idx, (6, 7, 8))
    assert flat_index == 99
    print("[PASS] 19: unravel_index for 100th element")


def test_ex20():
    a = ex20()
    assert a.shape == (8, 8)
    assert np.array_equal(a, ex18())  # both checkerboards should match
    print("[PASS] 20: checkerboard via tile")


def test_ex21():
    a = ex21()
    assert abs(a.mean()) < 1e-9
    assert abs(a.std() - 1.0) < 1e-9
    print("[PASS] 21: normalization (mean=0, std=1)")


def test_ex22():
    dt = ex22()
    assert dt.names == ("r", "g", "b", "a")
    print("[PASS] 22: RGBA custom dtype")


def test_ex23():
    result = ex23()
    assert result.shape == (5, 2)
    assert np.all(result == 3)  # each entry = sum of 3 ones
    print("[PASS] 23: matrix multiplication shape/values")


def test_ex24():
    a = ex24()
    assert list(a) == [0, 1, 2, 3, -4, -5, -6, -7, 8, 9, 10]
    print("[PASS] 24: conditional negation")


def test_ex25():
    assert ex25() == 9  # sum(range(5)) = 10, plus start=-1 -> 9
    print("[PASS] 25: sum(range(5), -1) gotcha")


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_ex")]
    tests.sort(key=lambda f: f.__name__)
    for t in tests:
        t()
    print(f"\n{len(tests)}/{len(tests)} checks passed.")
