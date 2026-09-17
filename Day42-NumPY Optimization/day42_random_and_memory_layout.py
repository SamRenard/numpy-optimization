"""
random_and_memory_layout.py
Day 42 -- Random Module, Statistical Functions, Memory Layout (C vs F order)
==============================================================================

Three topics for today's last theory block of the NumPy optimization
week:

  1. The modern `np.random.Generator` API (replacing the legacy
     `np.random.seed()` / global-state functions) and its main
     distributions.
  2. The core statistical functions: mean, median, std, var, percentile,
     correlation.
  3. Memory layout -- C order (row-major) vs Fortran order (column
     -major) -- and how it changes which access pattern is fast.

Run directly:
    python random_and_memory_layout.py
"""

import time
import numpy as np


def _rule_line():
    print("=" * 70)


def demo_modern_random_api():
    """
    The recommended way to generate random numbers: create a Generator
    with an explicit seed for reproducibility, then sample from it.
    This replaces the older global np.random.seed()/np.random.rand().
    """
    _rule_line()
    print("1) MODERN RANDOM API -- np.random.default_rng()")
    _rule_line()
    rng = np.random.default_rng(seed=42)
    print("uniform [0,1)   :", rng.random(4))
    print("integers [0,10) :", rng.integers(0, 10, 4))
    print("normal(0,1)     :", rng.normal(0, 1, 4))
    print("choice no repl. :", rng.choice(10, 4, replace=False))
    print("shuffled 0..5   :", rng.permutation(6))
    print()
    print("Two Generators built with the SAME seed produce the SAME")
    print("sequence -- this is what makes experiments reproducible:")
    rng_a = np.random.default_rng(seed=7)
    rng_b = np.random.default_rng(seed=7)
    print("rng_a.random(3) :", rng_a.random(3))
    print("rng_b.random(3) :", rng_b.random(3), "\n")


def demo_statistical_functions():
    """Core descriptive statistics numpy provides out of the box."""
    _rule_line()
    print("2) STATISTICAL FUNCTIONS")
    _rule_line()
    rng = np.random.default_rng(0)
    data = rng.normal(loc=50, scale=10, size=10_000)
    print(f"mean       : {data.mean():.3f}")
    print(f"median     : {np.median(data):.3f}")
    print(f"std        : {data.std():.3f}")
    print(f"var        : {data.var():.3f}")
    print(f"25/50/75 pct: {np.percentile(data, [25, 50, 75])}")

    x = rng.normal(size=1000)
    y = 2 * x + rng.normal(scale=0.5, size=1000)   # y correlated with x
    corr_matrix = np.corrcoef(x, y)
    print(f"correlation(x, y) : {corr_matrix[0, 1]:.3f}\n")


def demo_c_vs_f_order():
    """
    C order stores rows contiguously (last axis changes fastest).
    Fortran order stores columns contiguously (first axis changes
    fastest). Walking an array in an order that MATCHES its layout is
    fast; walking against it forces the CPU to jump around memory.
    """
    _rule_line()
    print("3) MEMORY LAYOUT -- C order vs Fortran order")
    _rule_line()
    n = 4000
    c_array = np.random.rand(n, n)                 # default: C order
    f_array = np.asfortranarray(c_array)            # same data, F order

    print(f"c_array.flags['C_CONTIGUOUS'] = {c_array.flags['C_CONTIGUOUS']}")
    print(f"f_array.flags['F_CONTIGUOUS'] = {f_array.flags['F_CONTIGUOUS']}")
    print(f"c_array.strides = {c_array.strides}   (row step is small)")
    print(f"f_array.strides = {f_array.strides}   (column step is small)")

    # Summing row by row matches C order -> fast on c_array
    start = time.perf_counter()
    for row in c_array:
        row.sum()
    c_row_time = time.perf_counter() - start

    # Same row-by-row loop on the F-ordered array fights its layout
    start = time.perf_counter()
    for row in f_array:
        row.sum()
    f_row_time = time.perf_counter() - start

    print(f"\nrow-wise sum over C-ordered array : {c_row_time:.4f} s")
    print(f"row-wise sum over F-ordered array : {f_row_time:.4f} s")
    print(f"F-array is {f_row_time / c_row_time:.2f}x slower for the SAME")
    print("loop, purely because its memory layout doesn't match the")
    print("access pattern.\n")


def demo_reshape_is_usually_a_view():
    """
    reshape() tries to return a view when the new shape is compatible
    with the existing memory layout, and silently falls back to a copy
    otherwise (this is also a layout / memory question).
    """
    _rule_line()
    print("Bonus: reshape() and memory layout")
    _rule_line()
    a = np.arange(12)
    b = a.reshape(3, 4)
    print(f"a.reshape(3, 4).base is a: {b.base is a}   (no copy needed)")

    c = a[::2]                 # a non-contiguous view (every other element)
    try:
        d = c.reshape(2, 3)
        print(f"c.reshape(2, 3).base is c: {d.base is c}")
    except ValueError as e:
        print(f"reshape failed on non-contiguous data: {e}")
    print()


if __name__ == "__main__":
    demo_modern_random_api()
    demo_statistical_functions()
    demo_c_vs_f_order()
    demo_reshape_is_usually_a_view()
