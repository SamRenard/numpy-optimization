"""
vectorization_anatomy.py
Day 41 -- Why Vectorization Is Fast: SIMD, C backend, Cache Locality
======================================================================

A pure-Python `for` loop over array elements is slow for three
compounding reasons that this script demonstrates one at a time:

  1. C BACKEND      -- a Python loop pays bytecode-interpretation and
                        type-checking overhead on every single
                        iteration; a vectorized numpy call runs one
                        pre-compiled C loop with none of that overhead.
  2. SIMD            -- modern CPUs can apply one instruction to
                        several numbers at once (Single Instruction,
                        Multiple Data). NumPy's C loops are written so
                        the compiler can emit these instructions;
                        Python bytecode cannot.
  3. CACHE LOCALITY  -- numpy arrays are contiguous blocks of memory,
                        so walking through them sequentially reuses
                        CPU cache lines efficiently. Python lists of
                        boxed objects, or strided/non-contiguous
                        access patterns, defeat this.

Run directly:
    python vectorization_anatomy.py
"""

import time
import numpy as np


def _rule_line():
    print("=" * 70)


def demo_c_backend_overhead():
    """
    Same math (squaring every element), one version pays Python's
    per-iteration interpreter overhead, the other runs a single
    compiled C loop inside numpy.
    """
    _rule_line()
    print("1) C BACKEND -- interpreter overhead per element vs one C call")
    _rule_line()
    n = 2_000_000
    data = list(range(n))
    arr = np.arange(n)

    start = time.perf_counter()
    squared_loop = [x * x for x in data]
    loop_time = time.perf_counter() - start

    start = time.perf_counter()
    squared_vec = arr * arr
    vec_time = time.perf_counter() - start

    print(f"Python list comprehension : {loop_time:.4f} s")
    print(f"numpy vectorized (arr*arr): {vec_time:.6f} s")
    print(f"speedup                   : {loop_time / vec_time:,.0f}x\n")


def demo_simd_on_contiguous_floats():
    """
    SIMD instructions operate on chunks of contiguous float64 values
    at once. This shows a large elementwise multiply-add, which is
    exactly the pattern SIMD accelerates, against the scalar loop
    equivalent.
    """
    _rule_line()
    print("2) SIMD -- one instruction, many floats, per CPU cycle")
    _rule_line()
    n = 3_000_000
    a = np.random.rand(n)
    b = np.random.rand(n)

    start = time.perf_counter()
    result_loop = [a[i] * b[i] + 1.0 for i in range(n)]
    loop_time = time.perf_counter() - start

    start = time.perf_counter()
    result_vec = a * b + 1.0
    vec_time = time.perf_counter() - start

    print(f"scalar loop (a[i]*b[i]+1) : {loop_time:.4f} s")
    print(f"vectorized  (a*b+1)       : {vec_time:.6f} s")
    print(f"speedup                   : {loop_time / vec_time:,.0f}x")
    print("(numpy's C loop lets the CPU pack several multiplies into")
    print(" one SIMD instruction; the Python loop cannot)\n")


def demo_cache_locality():
    """
    Summing a large matrix row-by-row (C order, matches memory layout)
    versus column-by-column (F order, jumps around memory) shows how
    access pattern -- not amount of work -- drives performance.
    """
    _rule_line()
    print("3) CACHE LOCALITY -- same data, different access pattern")
    _rule_line()
    n = 4000
    m = np.random.rand(n, n)   # C-contiguous by default

    start = time.perf_counter()
    row_major_sum = 0.0
    for row in m:               # walks memory sequentially
        row_major_sum += row.sum()
    row_time = time.perf_counter() - start

    start = time.perf_counter()
    col_major_sum = 0.0
    for col in m.T:              # walks memory with a large stride
        col_major_sum += col.sum()
    col_time = time.perf_counter() - start

    print(f"row-wise iteration (cache-friendly)    : {row_time:.4f} s")
    print(f"column-wise iteration (cache-unfriendly): {col_time:.4f} s")
    print(f"slowdown from bad access pattern         : {col_time / row_time:.2f}x")
    print(f"results agree: {abs(row_major_sum - col_major_sum) < 1e-6}\n")


def demo_stacking_all_three():
    """
    Put it all together: a realistic 'distance from origin' calculation
    over 5 million 2D points, loop vs fully vectorized.
    """
    _rule_line()
    print("All three effects combined on a real computation")
    _rule_line()
    n = 5_000_000
    points = np.random.rand(n, 2)

    start = time.perf_counter()
    distances_loop = [
        (points[i, 0] ** 2 + points[i, 1] ** 2) ** 0.5 for i in range(n)
    ]
    loop_time = time.perf_counter() - start

    start = time.perf_counter()
    distances_vec = np.sqrt((points ** 2).sum(axis=1))
    vec_time = time.perf_counter() - start

    print(f"python loop : {loop_time:.4f} s")
    print(f"vectorized  : {vec_time:.6f} s")
    print(f"speedup     : {loop_time / vec_time:,.0f}x\n")


if __name__ == "__main__":
    demo_c_backend_overhead()
    demo_simd_on_contiguous_floats()
    demo_cache_locality()
    demo_stacking_all_three()
