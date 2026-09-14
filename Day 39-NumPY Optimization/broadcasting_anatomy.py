"""
broadcasting_anatomy.py
Day 39 -- NumPy Broadcasting Rules, Live Demo
================================================

NumPy's broadcasting lets you combine arrays of different shapes without
writing explicit loops or copying data. This script walks through the
three rules that decide whether two shapes are "broadcastable", then
shows what actually happens in memory when broadcasting occurs (a view
with zero-stride, not a real copy) and finally benchmarks a broadcasted
computation against the equivalent hand-written Python loop.

Run this file directly to see the output of every example:
    python broadcasting_anatomy.py
"""

import time
import numpy as np


def _rule_line():
    print("=" * 70)


def demo_rule_1():
    """
    RULE 1 -- Align shapes from the RIGHT.

    When two arrays have a different number of dimensions, the shape of
    the smaller one is padded with 1s on its LEFT side until both shapes
    have the same length. Only after this padding can the shapes be
    compared dimension by dimension.
    """
    _rule_line()
    print("RULE 1 - align shapes from the right, pad missing dims with 1")
    _rule_line()
    a = np.ones((3, 4))          # shape (3, 4)
    b = np.ones(4)                # shape    (4,)  -> padded to (1, 4)
    print(f"a.shape = {a.shape}")
    print(f"b.shape = {b.shape}   (treated as (1, 4) for this op)")
    c = a + b
    print(f"(a + b).shape = {c.shape}\n")


def demo_rule_2():
    """
    RULE 2 -- Stretch size-1 dimensions.

    Once both shapes have the same length, any axis where one array has
    size 1 is conceptually "stretched" to match the other array's size
    on that axis. Nothing is physically duplicated in memory; NumPy just
    reuses the same value along that axis.
    """
    _rule_line()
    print("RULE 2 - size-1 dimensions are stretched to match")
    _rule_line()
    col = np.array([[1], [2], [3]])     # shape (3, 1)
    row = np.array([10, 20, 30, 40])    # shape (4,) -> (1, 4)
    result = col + row
    print(f"col.shape = {col.shape}")
    print(f"row.shape = {row.shape}")
    print(f"result.shape = {result.shape}   (3,1) + (1,4) -> (3,4)")
    print(result, "\n")


def demo_rule_3():
    """
    RULE 3 -- Mismatched, non-1 dimensions are illegal.

    After rules 1 and 2 have been applied, if any axis still disagrees
    in size AND neither side is 1 on that axis, the shapes cannot be
    broadcast together and NumPy raises a ValueError.
    """
    _rule_line()
    print("RULE 3 - mismatched non-1 dimensions raise ValueError")
    _rule_line()
    x = np.ones((3, 4))
    y = np.ones((3, 5))
    try:
        x + y
    except ValueError as err:
        print(f"x.shape={x.shape}, y.shape={y.shape} -> ValueError: {err}\n")


def demo_view_vs_copy():
    """
    Broadcasting is implemented with stride tricks, not real copies.
    np.broadcast_to() makes this explicit: it returns a read-only VIEW
    whose stride is 0 along the broadcast axis, so every "row" points
    at the same underlying memory.
    """
    _rule_line()
    print("Broadcasting is a zero-stride view, not a copy")
    _rule_line()
    v = np.array([1, 2, 3])
    view = np.broadcast_to(v, (4, 3))
    print(f"original strides : {v.strides}")
    print(f"broadcast strides: {view.strides}   <- 0 on the new axis")
    print(f"view.base is v?  : {view.base is v}")
    print(f"view is writeable: {view.flags.writeable}\n")


def demo_performance():
    """
    Compare a hand-written double Python loop against the broadcasted
    vectorized equivalent for an outer-difference matrix a[i] - b[j].
    """
    _rule_line()
    print("Loop vs broadcasting: timing an outer subtraction")
    _rule_line()
    n = 600
    a = np.random.rand(n)
    b = np.random.rand(n)

    start = time.perf_counter()
    loop_result = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            loop_result[i, j] = a[i] - b[j]
    loop_time = time.perf_counter() - start

    start = time.perf_counter()
    vec_result = a[:, None] - b[None, :]
    vec_time = time.perf_counter() - start

    print(f"pure python loop : {loop_time:.4f} s")
    print(f"broadcasting     : {vec_time:.6f} s")
    print(f"speedup          : {loop_time / vec_time:,.0f}x")
    print(f"results match    : {np.allclose(loop_result, vec_result)}\n")


if __name__ == "__main__":
    demo_rule_1()
    demo_rule_2()
    demo_rule_3()
    demo_view_vs_copy()
    demo_performance()
