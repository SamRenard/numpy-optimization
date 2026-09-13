"""
Day 38 — ndarray Anatomy: shape, dtype, stride, contiguous memory
NIZAM AI · 150-Day AI Engineering Protocol

Goal: understand WHY NumPy is fast by looking under the hood of the
ndarray — how shape, dtype, and strides work together to let a 1D
block of memory behave like an N-dimensional array, and how
view/copy/contiguity affect performance.
"""

import numpy as np
import time


def explore_shape_dtype():
    print("=" * 70)
    print("shape & dtype")
    print("=" * 70)
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)
    print(f"array:\n{a}")
    print(f"shape: {a.shape}   (2 rows, 3 cols)")
    print(f"dtype: {a.dtype}   (each element = {a.itemsize} bytes)")
    print(f"total memory: {a.nbytes} bytes = shape product ({a.size}) * itemsize ({a.itemsize})")


def explore_strides():
    print("\n" + "=" * 70)
    print("strides — how NumPy 'walks' through a flat memory block")
    print("=" * 70)
    a = np.arange(12, dtype=np.int64).reshape(3, 4)
    print(f"array (3x4):\n{a}")
    print(f"strides: {a.strides}  (bytes to step: dim0={a.strides[0]}, dim1={a.strides[1]})")
    print(f"-> to move one row down, jump {a.strides[0]} bytes "
          f"(= {a.strides[0] // a.itemsize} elements = row length)")
    print(f"-> to move one column right, jump {a.strides[1]} bytes "
          f"(= {a.strides[1] // a.itemsize} element)")

    # Transposing doesn't move any data -- it just swaps the strides!
    at = a.T
    print(f"\ntransposed (4x3):\n{at}")
    print(f"transposed strides: {at.strides}  <- SWAPPED, no data was copied")
    print(f"same underlying memory? {np.shares_memory(a, at)}")


def explore_view_vs_copy():
    print("\n" + "=" * 70)
    print("view vs copy")
    print("=" * 70)
    a = np.arange(10)
    view = a[2:8:2]       # slicing -> view (shares memory)
    copy = a[[2, 4, 6]]   # fancy indexing -> copy (new memory)

    print(f"original: {a}")
    print(f"slice view a[2:8:2]: {view}   shares memory: {np.shares_memory(a, view)}")
    print(f"fancy index a[[2,4,6]]: {copy}   shares memory: {np.shares_memory(a, copy)}")

    print("\nModifying the view changes the original:")
    view[0] = 999
    print(f"  view[0]=999 -> original is now: {a}")

    a2 = np.arange(10)  # reset
    copy2 = a2[[2, 4, 6]]
    copy2[0] = 999
    print(f"Modifying the copy does NOT change the original: {a2}")


def explore_contiguity_performance():
    print("\n" + "=" * 70)
    print("contiguous memory & performance")
    print("=" * 70)
    n = 2000
    a = np.random.rand(n, n)

    a_row_major = np.ascontiguousarray(a)          # C-order: rows are contiguous
    a_col_major = np.asfortranarray(a)              # F-order: columns are contiguous

    print(f"a_row_major.flags['C_CONTIGUOUS']: {a_row_major.flags['C_CONTIGUOUS']}")
    print(f"a_col_major.flags['F_CONTIGUOUS']: {a_col_major.flags['F_CONTIGUOUS']}")

    # Summing along rows (axis=1) is fast for C-order because each row
    # is a contiguous block -- great for CPU cache locality.
    start = time.perf_counter()
    for _ in range(50):
        _ = a_row_major.sum(axis=1)
    t_row = time.perf_counter() - start

    start = time.perf_counter()
    for _ in range(50):
        _ = a_col_major.sum(axis=1)
    t_col = time.perf_counter() - start

    print(f"\nsum(axis=1) on C-contiguous array:  {t_row*1000:.2f} ms (50 runs)")
    print(f"sum(axis=1) on F-contiguous array:  {t_col*1000:.2f} ms (50 runs)")
    print("-> accessing memory in its natural (contiguous) order is faster: "
          "fewer cache misses.")


if __name__ == "__main__":
    explore_shape_dtype()
    explore_strides()
    explore_view_vs_copy()
    explore_contiguity_performance()
