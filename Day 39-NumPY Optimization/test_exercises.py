"""
test_exercises.py
Day 39 -- test suite for numpy_exercises_26_50.py and broadcasting_anatomy.py

Run with:
    pytest test_exercises.py -v
or plainly:
    python test_exercises.py
"""

import numpy as np
import numpy_exercises_26_50 as ex
import broadcasting_anatomy as ba


# ---------------------------------------------------------------- 26-30
def test_ex26_builtin_vs_numpy_sum_differ():
    builtin_result, numpy_result = ex.ex26()
    assert builtin_result == 9
    assert numpy_result == 10


def test_ex27_chained_comparison_is_illegal():
    results = ex.ex27()
    assert results["Z**Z"] is True
    assert results["Z<Z>Z"] is False


def test_ex28_zero_div_zero_is_nan_not_error():
    true_div, floor_div, nan_cast = ex.ex28()
    assert np.isnan(true_div)
    assert floor_div == 0


def test_ex29_rounds_away_from_zero():
    result = ex.ex29(np.array([-2.3, 1.1, -0.5, 3.9]))
    np.testing.assert_array_equal(result, [-3.0, 2.0, -1.0, 4.0])


def test_ex30_intersection():
    result = ex.ex30(np.array([1, 2, 3, 4]), np.array([3, 4, 5, 6]))
    np.testing.assert_array_equal(result, [3, 4])


# ---------------------------------------------------------------- 31-35
def test_ex31_errstate_does_not_raise():
    assert ex.ex31() is True


def test_ex32_sqrt_vs_emath_sqrt():
    regular, complex_aware, are_equal = ex.ex32()
    assert np.isnan(regular)
    assert complex_aware == 1j
    assert are_equal == False  # noqa: E712  (numpy bool, keep explicit)


def test_ex33_yesterday_today_tomorrow_are_consecutive():
    yesterday, today, tomorrow = ex.ex33()
    assert today - yesterday == np.timedelta64(1, "D")
    assert tomorrow - today == np.timedelta64(1, "D")


def test_ex34_july_2016_has_31_days():
    days = ex.ex34()
    assert len(days) == 31
    assert str(days[0]) == "2016-07-01"
    assert str(days[-1]) == "2016-07-31"


def test_ex35_in_place_formula():
    A = np.array([1.0, 2.0])
    B = np.array([3.0, 4.0])
    result = ex.ex35(A, B)
    # ((A+B) * (-A/2)) computed with the ORIGINAL A, B values
    expected = ((1.0 + 3.0) * (-1.0 / 2), (2.0 + 4.0) * (-2.0 / 2))
    np.testing.assert_allclose(result, expected)


# ---------------------------------------------------------------- 36-40
def test_ex36_four_methods_agree():
    a = np.array([3.7, 2.1, 9.99])
    m1, m2, m3, m4 = ex.ex36(a)
    np.testing.assert_array_equal(m1, [3.0, 2.0, 9.0])
    np.testing.assert_array_equal(m2, m1)
    np.testing.assert_array_equal(m3, m1)
    np.testing.assert_array_equal(m4, [3, 2, 9])


def test_ex37_row_pattern():
    result = ex.ex37()
    assert result.shape == (5, 5)
    for row in result:
        np.testing.assert_array_equal(row, [0, 1, 2, 3, 4])


def test_ex38_from_generator():
    result = ex.ex38()
    np.testing.assert_array_equal(result, np.arange(10, dtype=float))


def test_ex39_bounds_are_excluded():
    result = ex.ex39()
    assert len(result) == 10
    assert result.min() > 0
    assert result.max() < 1


def test_ex40_sorted_ascending():
    result = ex.ex40(20)
    assert len(result) == 20
    assert np.all(np.diff(result) >= 0)


# ---------------------------------------------------------------- 41-45
def test_ex41_matches_np_sum():
    a = np.arange(10)
    assert ex.ex41(a) == np.sum(a)


def test_ex42_equality_checks():
    same, close = ex.ex42(np.array([1, 2]), np.array([1, 2]))
    assert same is True or bool(same) is True
    assert bool(close) is True


def test_ex43_array_is_readonly():
    result = ex.ex43(np.arange(3))
    assert result.flags.writeable is False
    try:
        result[0] = 99
        raised = False
    except ValueError:
        raised = True
    assert raised is True


def test_ex44_polar_conversion():
    cart = np.array([[3.0, 4.0]])   # classic 3-4-5 triangle
    polar = ex.ex44(cart)
    r, theta = polar[0]
    assert abs(r - 5.0) < 1e-9
    assert abs(theta - np.arctan2(4.0, 3.0)) < 1e-9


def test_ex45_max_replaced_with_zero():
    result = ex.ex45(np.array([1, 9, 3, 2]))
    np.testing.assert_array_equal(result, [1, 0, 3, 2])


# ---------------------------------------------------------------- 46-50
def test_ex46_structured_grid_shape():
    grid = ex.ex46(4)
    assert grid.shape == (4, 4)
    assert grid.dtype.names == ("x", "y")
    assert grid["x"][0, 0] == 0.0
    assert grid["x"][0, -1] == 1.0


def test_ex47_cauchy_matrix_values():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0])
    C = ex.ex47(x, y)
    assert C.shape == (3, 2)
    assert abs(C[0, 0] - 1.0 / (1.0 - 4.0)) < 1e-9


def test_ex48_int8_bounds():
    report = ex.ex48()
    assert report["int8"] == (-128, 127)


def test_ex49_prints_full_array_no_ellipsis():
    big = np.arange(2000)
    text = ex.ex49(big)
    assert "..." not in text


def test_ex50_closest_value():
    a = np.array([1, 5, 9, 14, 20])
    assert ex.ex50(a, 12) == 14
    assert ex.ex50(a, 6) == 5


# ---------------------------------------------------------------- broadcasting_anatomy.py
def test_broadcast_rule1_shape():
    a = np.ones((3, 4))
    b = np.ones(4)
    assert (a + b).shape == (3, 4)


def test_broadcast_rule2_stretch():
    col = np.array([[1], [2], [3]])
    row = np.array([10, 20, 30, 40])
    assert (col + row).shape == (3, 4)


def test_broadcast_rule3_raises():
    x = np.ones((3, 4))
    y = np.ones((3, 5))
    try:
        x + y
        raised = False
    except ValueError:
        raised = True
    assert raised is True


def test_broadcast_to_is_a_view_not_a_copy():
    v = np.array([1, 2, 3])
    view = np.broadcast_to(v, (4, 3))
    assert view.base is v
    assert view.strides[0] == 0
    assert view.flags.writeable is False


def test_broadcasting_matches_loop_result():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([10.0, 20.0])
    vectorized = a[:, None] - b[None, :]
    manual = np.empty((3, 2))
    for i in range(3):
        for j in range(2):
            manual[i, j] = a[i] - b[j]
    np.testing.assert_allclose(vectorized, manual)


# ---------------------------------------------------------------- runner
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
