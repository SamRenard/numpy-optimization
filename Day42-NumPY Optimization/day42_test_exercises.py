"""
test_exercises.py
Day 42 -- test suite for numpy_exercises_76_100.py, monte_carlo_pi.py,
and random_and_memory_layout.py
"""

import numpy as np
import numpy_exercises_76_100 as ex
import monte_carlo_pi as mc


def test_ex76_sliding_window_shape_and_values():
    result = ex.ex76(np.array([1, 2, 3, 4, 5]), window=3)
    assert result.shape == (3, 3)
    np.testing.assert_array_equal(result[0], [1, 2, 3])
    np.testing.assert_array_equal(result[-1], [3, 4, 5])


def test_ex77_negates_in_place():
    b = np.array([True, False])
    f = np.array([1.0, -2.0])
    b_out, f_out = ex.ex77(b, f)
    np.testing.assert_array_equal(b_out, [False, True])
    np.testing.assert_array_equal(f_out, [-1.0, 2.0])
    assert b_out is b and f_out is f


def test_ex78_distance_to_line():
    p0 = np.array([[0.0, 0.0]])
    p1 = np.array([[1.0, 0.0]])
    p = np.array([0.5, 2.0])
    result = ex.ex78(p0, p1, p)
    assert abs(result[0] - 2.0) < 1e-9


def test_ex79_shape_matches_points_by_lines():
    p0 = np.array([[0.0, 0.0], [0.0, 0.0]])
    p1 = np.array([[1.0, 0.0], [0.0, 1.0]])
    points = np.array([[0.5, 1.0], [2.0, 2.0], [1.0, 1.0]])
    result = ex.ex79(p0, p1, points)
    assert result.shape == (3, 2)


def test_ex80_pads_out_of_bounds():
    a = np.arange(16).reshape(4, 4)
    result = ex.ex80(a, (0, 0), (3, 3), fill=-1)
    assert result.shape == (3, 3)
    assert result[0, 0] == -1
    assert result[1, 1] == a[0, 0]


def test_ex81_matches_expected_pattern():
    result = ex.ex81(np.arange(1, 6), window=3)
    np.testing.assert_array_equal(result[1], [2, 3, 4])


def test_ex82_identity_has_full_rank():
    assert ex.ex82(np.eye(4)) == 4


def test_ex83_most_frequent_value():
    assert ex.ex83(np.array([5, 5, 5, 2, 2, 9])) == 5


def test_ex84_block_count_and_shape():
    result = ex.ex84(np.arange(25).reshape(5, 5))
    assert result.shape == (9, 3, 3)


def test_ex85_symmetric_writes_mirror():
    sym = ex.ex85(3)
    sym[0, 2] = 7
    assert np.asarray(sym)[2, 0] == 7


def test_ex86_shape_and_values():
    matrices = np.stack([np.eye(2), np.eye(2) * 2])
    vectors = np.stack([np.array([[1.0], [1.0]]), np.array([[1.0], [1.0]])])
    result = ex.ex86(matrices, vectors)
    assert result.shape == (2, 1)
    np.testing.assert_allclose(result.ravel(), [3.0, 3.0])


def test_ex87_block_sums():
    a = np.ones((8, 8))
    result = ex.ex87(a, block=4)
    assert result.shape == (2, 2)
    assert result[0, 0] == 16.0


def test_ex88_game_of_life_blinker_oscillates():
    blinker = np.zeros((5, 5), dtype=int)
    blinker[2, 1:4] = 1
    step1 = ex.ex88(blinker, steps=1)
    step2 = ex.ex88(blinker, steps=2)
    np.testing.assert_array_equal(step2, blinker)   # period-2 oscillator
    assert step1[1, 2] == 1 and step1[3, 2] == 1


def test_ex89_n_largest_values():
    result = ex.ex89(np.array([3, 1, 9, 4, 7]), 3)
    assert set(result.tolist()) == {9, 7, 4}


def test_ex90_cartesian_product_size():
    result = ex.ex90(np.array([0, 1]), np.array([10, 20]), np.array([100]))
    assert result.shape == (4, 3)


def test_ex91_record_array_fields():
    result = ex.ex91(np.array([[1, 2.5], [3, 4.5]]), names="a,b")
    assert result.a.tolist() == [1.0, 3.0]
    assert result.b.tolist() == [2.5, 4.5]


def test_ex92_three_methods_agree():
    z = np.array([1, 2, 3, 4])
    m1, m2, m3 = ex.ex92(z)
    np.testing.assert_array_equal(m1, m2)
    np.testing.assert_array_equal(m2, m3)


def test_ex93_requires_match_with_every_b_row():
    a = np.array([[1, 2, 3], [4, 5, 6], [1, 4, 9]])
    b = np.array([[1, 2], [4, 5]])
    result = ex.ex93(a, b)
    np.testing.assert_array_equal(result, [[1, 4, 9]])


def test_ex94_drops_all_equal_rows():
    a = np.array([[1, 1, 1], [1, 2, 3], [5, 5, 5]])
    result = ex.ex94(a)
    np.testing.assert_array_equal(result, [[1, 2, 3]])


def test_ex95_binary_representation():
    result = ex.ex95(np.array([2, 5]), width=4)
    np.testing.assert_array_equal(result, [[0, 0, 1, 0], [0, 1, 0, 1]])


def test_ex96_unique_rows():
    result = ex.ex96(np.array([[1, 2], [1, 2], [3, 4]]))
    np.testing.assert_array_equal(result, [[1, 2], [3, 4]])


def test_ex97_einsum_matches_numpy_builtins():
    a = np.array([1.0, 2.0])
    b = np.array([3.0, 4.0])
    inner, outer, total_sum, mul = ex.ex97(a, b)
    assert abs(inner - np.dot(a, b)) < 1e-9
    np.testing.assert_allclose(outer, np.outer(a, b))
    assert abs(total_sum - a.sum()) < 1e-9
    np.testing.assert_allclose(mul, a * b)


def test_ex98_resampled_path_endpoints_match():
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([0.0, 1.0, 0.0])
    x_new, y_new = ex.ex98(x, y, n_samples=5)
    assert len(x_new) == 5
    assert abs(x_new[0] - x[0]) < 1e-9
    assert abs(x_new[-1] - x[-1]) < 1e-9


def test_ex99_selects_valid_multinomial_rows():
    x = np.array([[1, 2, 2], [0, 5, 0], [1, 1, 1]])
    result = ex.ex99(x, 5)
    assert len(result) == 2
    assert [1, 1, 1] not in result.tolist()


def test_ex100_ci_contains_true_mean_direction():
    x = np.array([10.0, 11.0, 9.0, 10.5, 9.5, 10.2])
    lo, hi = ex.ex100(x, n_resamples=500, seed=1)
    assert lo < x.mean() < hi


# ---------------------------------------------------------------- monte_carlo_pi.py
def test_monte_carlo_pi_reasonably_close():
    pi_hat, fraction = mc.estimate_pi(200_000, seed=1)
    assert abs(pi_hat - np.pi) < 0.05
    assert 0.0 <= fraction <= 1.0


def test_monte_carlo_reproducible_with_seed():
    pi1, _ = mc.estimate_pi(5000, seed=99)
    pi2, _ = mc.estimate_pi(5000, seed=99)
    assert pi1 == pi2


def test_monte_carlo_error_shrinks_with_more_points():
    results = mc.estimate_pi_convergence([100, 1_000_000], seed=0)
    small_n_error = results[0][2]
    large_n_error = results[1][2]
    assert large_n_error < small_n_error


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
