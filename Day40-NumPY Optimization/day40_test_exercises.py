"""
test_exercises.py
Day 40 -- test suite for numpy_exercises_51_75.py and fancy_indexing_anatomy.py
"""

import numpy as np
import numpy_exercises_51_75 as ex
import fancy_indexing_anatomy as fi


def test_ex51_field_names():
    a = ex.ex51(2)
    assert a.dtype.names == ("position", "color")
    assert a["position"].dtype.names == ("x", "y")


def test_ex52_distance_matrix():
    coords = np.array([[0.0, 0.0], [3.0, 4.0]])
    d = ex.ex52(coords)
    assert d.shape == (2, 2)
    assert abs(d[0, 1] - 5.0) < 1e-9
    assert d[0, 0] == 0.0


def test_ex53_inplace_cast_same_buffer():
    a = np.array([1.9, -2.1, 3.5], dtype=np.float32)
    result = ex.ex53(a)
    assert result.dtype == np.int32
    np.testing.assert_array_equal(result, [1, -2, 3])


def test_ex54_missing_values_become_nan():
    result = ex.ex54()
    assert result.shape == (3, 5)
    assert np.isnan(result[1, 1])


def test_ex55_matches_length():
    a = np.arange(6).reshape(2, 3)
    result = ex.ex55(a)
    assert len(result) == 6
    assert result[0] == ((0, 0), 0)


def test_ex56_peak_at_center():
    g = ex.ex56((11, 11))
    assert g.shape == (11, 11)
    assert g[5, 5] == g.max()


def test_ex57_exact_count_placed():
    arr = ex.ex57((6, 6), 5)
    assert (arr == 1).sum() == 5


def test_ex58_rows_sum_to_zero():
    a = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    result = ex.ex58(a)
    np.testing.assert_allclose(result.sum(axis=1), [0.0, 0.0], atol=1e-9)


def test_ex59_sorted_by_column():
    a = np.array([[3, 1], [1, 2], [2, 0]])
    result = ex.ex59(a, 0)
    np.testing.assert_array_equal(result[:, 0], [1, 2, 3])


def test_ex60_detects_null_column():
    has_null = ex.ex60(np.array([[1, 0, 2], [3, 0, 4]]))
    no_null = ex.ex60(np.array([[1, 5, 2], [3, 6, 4]]))
    assert has_null is True
    assert no_null is False


def test_ex61_closest_value():
    a = np.array([1, 5, 9, 14, 20])
    assert ex.ex61(a, 12) == 14


def test_ex62_matches_broadcast_sum():
    a = np.array([[1, 2, 3]])
    b = np.array([[10], [20]])
    result = ex.ex62(a, b)
    np.testing.assert_array_equal(result, a + b)


def test_ex63_name_attribute_survives():
    arr, name = ex.ex63()
    assert name == "day40-array"
    sliced = arr[1:3]
    assert sliced.name == "day40-array"


def test_ex64_handles_duplicate_indices():
    result = ex.ex64(5, np.array([0, 0, 1]))
    # index 0 hit twice -> should be 1 (base) + 2 = 3
    assert result[0] == 3
    assert result[1] == 2
    assert result[2] == 1


def test_ex65_accumulates_correctly():
    result = ex.ex65(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0, 1, 0, 2]))
    np.testing.assert_array_equal(result, [4.0, 2.0, 4.0])


def test_ex66_counts_unique_colors():
    img = np.zeros((2, 2, 3), dtype=np.uint8)
    img[0, 1] = [255, 255, 255]
    assert ex.ex66(img) == 2


def test_ex67_sums_last_two_axes():
    a = np.ones((2, 3, 4, 5))
    result = ex.ex67(a)
    assert result.shape == (2, 3)
    assert result[0, 0] == 20.0


def test_ex68_grouped_means():
    result = ex.ex68(np.array([1.0, 3.0, 2.0, 4.0]), np.array([0, 0, 1, 1]))
    np.testing.assert_allclose(result, [2.0, 3.0])


def test_ex69_diagonal_of_dot_product():
    a = np.eye(3) * 2
    b = np.eye(3) * 3
    result = ex.ex69(a, b)
    np.testing.assert_array_equal(result, [6.0, 6.0, 6.0])


def test_ex70_interleaved_zeros():
    result = ex.ex70(np.array([1, 2, 3]), n_zeros=2)
    np.testing.assert_array_equal(result, [1, 0, 0, 2, 0, 0, 3])


def test_ex71_broadcast_multiply():
    a = np.ones((2, 2, 3))
    b = np.array([[2.0, 3.0], [4.0, 5.0]])
    result = ex.ex71(a, b)
    assert result.shape == (2, 2, 3)
    assert result[0, 0, 0] == 2.0


def test_ex72_rows_swapped():
    a = np.arange(9).reshape(3, 3)
    result = ex.ex72(a, 0, 2)
    np.testing.assert_array_equal(result[0], a[2])
    np.testing.assert_array_equal(result[2], a[0])
    # original untouched
    np.testing.assert_array_equal(a[0], [0, 1, 2])


def test_ex73_unique_edges_of_two_shared_triangles():
    triangles = np.array([[0, 1, 2], [1, 2, 3]])
    edges = ex.ex73(triangles)
    assert len(edges) == 5  # 6 raw edges, 1 shared -> 5 unique


def test_ex74_reconstructs_bincount():
    counts = np.array([2, 1, 0, 3])
    a = ex.ex74(counts)
    np.testing.assert_array_equal(np.bincount(a, minlength=4), counts)


def test_ex75_sliding_window_average():
    result = ex.ex75(np.arange(10, dtype=float), 3)
    np.testing.assert_allclose(result[0], (0 + 1 + 2) / 3)


# ---------------------------------------------------------------- fancy_indexing_anatomy.py
def test_basic_slice_is_view():
    a = np.arange(10)
    sl = a[2:5]
    assert sl.base is a
    sl[0] = 999
    assert a[2] == 999


def test_fancy_index_is_copy():
    a = np.arange(10)
    fancy = a[[2, 3, 4]]
    assert fancy.base is None
    fancy[0] = 999
    assert a[2] == 2


def test_boolean_mask_is_copy():
    a = np.arange(10)
    mask = a % 2 == 0
    evens = a[mask]
    evens[:] = -1
    assert a[0] == 0  # untouched


def test_boolean_mask_inplace_edit_works():
    a = np.arange(10)
    a[a % 2 == 0] = -1
    assert a[0] == -1
    assert a[1] == 1


def test_mixed_indexing_is_copy():
    a = np.arange(12).reshape(3, 4)
    mixed = a[1:, [0, 2]]
    assert mixed.base is not a
    mixed[0, 0] = -1
    assert a[1, 0] == 4  # original untouched


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
