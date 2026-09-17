"""
test_exercises.py
Day 41 -- correctness tests for loop_vs_vectorized.py and a sanity check
on vectorization_anatomy.py's core claims.
"""

import numpy as np
import loop_vs_vectorized as lv
import vectorization_anatomy as va


def test_sum_squares_match():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert abs(lv.sum_squares_loop(a.tolist()) - lv.sum_squares_vectorized(a)) < 1e-9


def test_normalize_match():
    a = np.array([2.0, 4.0, 6.0, 8.0])
    loop_result = lv.normalize_loop(a.tolist())
    vec_result = lv.normalize_vectorized(a)
    np.testing.assert_allclose(loop_result, vec_result)


def test_clip_and_scale_match():
    a = np.array([-1.0, 0.5, 0.9, 2.0])
    loop_result = lv.clip_and_scale_loop(a.tolist(), 0.0, 1.0, 10.0)
    vec_result = lv.clip_and_scale_vectorized(a, 0.0, 1.0, 10.0)
    np.testing.assert_allclose(loop_result, vec_result)


def test_pairwise_products_match():
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0, 5.0])
    loop_result = lv.pairwise_products_loop(a.tolist(), b.tolist())
    vec_result = lv.pairwise_products_vectorized(a, b)
    np.testing.assert_allclose(np.array(loop_result), vec_result)


def test_running_max_match():
    a = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    loop_result = lv.running_max_loop(a.tolist())
    vec_result = lv.running_max_vectorized(a)
    np.testing.assert_allclose(loop_result, vec_result)


def test_safe_sqrt_match():
    a = np.array([-4.0, 0.0, 4.0, 9.0])
    loop_result = lv.safe_sqrt_loop(a.tolist())
    vec_result = lv.safe_sqrt_vectorized(a)
    np.testing.assert_allclose(loop_result, vec_result)


def test_trig_transform_match():
    a = np.array([-1.0, 0.0, 1.0, 2.5])
    loop_result = lv.trig_transform_loop(a.tolist())
    vec_result = lv.trig_transform_vectorized(a)
    np.testing.assert_allclose(loop_result, vec_result, atol=1e-9)


def test_benchmark_returns_positive_speedup():
    row = lv.benchmark(
        "unit-test sum of squares",
        lv.sum_squares_loop, lv.sum_squares_vectorized,
        ([1.0, 2.0, 3.0],), (np.array([1.0, 2.0, 3.0]),),
        repeats=1,
    )
    assert float(row["speedup_x"]) > 0


def test_run_all_benchmarks_logs_file(tmp_path_csv="test_benchmark_log.csv"):
    rows = lv.run_all_benchmarks(n=5000, log_path=tmp_path_csv)
    assert len(rows) == 7
    for row in rows:
        assert float(row["speedup_x"]) > 1.0  # vectorized must be faster
    import os
    assert os.path.exists(tmp_path_csv)
    os.remove(tmp_path_csv)


# ---------------------------------------------------------------- vectorization_anatomy.py sanity
def test_vectorized_multiply_matches_loop_small():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    loop_result = [a[i] * b[i] + 1.0 for i in range(3)]
    vec_result = a * b + 1.0
    np.testing.assert_allclose(loop_result, vec_result)


def test_row_and_column_sums_agree():
    m = np.arange(16, dtype=float).reshape(4, 4)
    row_sum = sum(row.sum() for row in m)
    col_sum = sum(col.sum() for col in m.T)
    assert abs(row_sum - col_sum) < 1e-9


if __name__ == "__main__":
    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_")]
    passed, failed = 0, 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL  {t.__name__}  ->  {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed, {passed + failed} total")
