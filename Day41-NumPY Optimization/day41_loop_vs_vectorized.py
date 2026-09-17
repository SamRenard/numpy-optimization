"""
loop_vs_vectorized.py
Day 41 -- Practical Coding: Vectorize Loop-Based Code, Measure the Speedup
=============================================================================

Five small "before/after" pairs: a loop-based function exactly as a
numpy beginner would write it, and its vectorized rewrite. Each pair
is benchmarked with `timeit` (the script's own %timeit-equivalent) and
logged to `benchmark_log.csv` alongside the measured speedup, so today's
target ("measure and record the ~100x difference") is captured on disk
and not just printed to the terminal.

Run directly:
    python loop_vs_vectorized.py
"""

import csv
import math
import timeit
import numpy as np


# ---------------------------------------------------------------- 1
def sum_squares_loop(a):
    total = 0.0
    for x in a:
        total += x * x
    return total


def sum_squares_vectorized(a):
    return float((a * a).sum())


# ---------------------------------------------------------------- 2
def normalize_loop(a):
    lo, hi = min(a), max(a)
    out = [0.0] * len(a)
    for i in range(len(a)):
        out[i] = (a[i] - lo) / (hi - lo)
    return out


def normalize_vectorized(a):
    lo, hi = a.min(), a.max()
    return (a - lo) / (hi - lo)


# ---------------------------------------------------------------- 3
def clip_and_scale_loop(a, lo, hi, factor):
    out = [0.0] * len(a)
    for i in range(len(a)):
        v = a[i]
        if v < lo:
            v = lo
        elif v > hi:
            v = hi
        out[i] = v * factor
    return out


def clip_and_scale_vectorized(a, lo, hi, factor):
    return np.clip(a, lo, hi) * factor


# ---------------------------------------------------------------- 4
def pairwise_products_loop(a, b):
    n, m = len(a), len(b)
    out = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            out[i][j] = a[i] * b[j]
    return out


def pairwise_products_vectorized(a, b):
    return a[:, None] * b[None, :]


# ---------------------------------------------------------------- 5
def running_max_loop(a):
    out = [0.0] * len(a)
    current_max = float("-inf")
    for i, v in enumerate(a):
        if v > current_max:
            current_max = v
        out[i] = current_max
    return out


def running_max_vectorized(a):
    return np.maximum.accumulate(a)


# ---------------------------------------------------------------- 6
def safe_sqrt_loop(a):
    out = [0.0] * len(a)
    for i in range(len(a)):
        v = a[i]
        out[i] = math.sqrt(v) if v >= 0 else 0.0
    return out


def safe_sqrt_vectorized(a):
    return np.sqrt(np.clip(a, 0, None))


# ---------------------------------------------------------------- 7
def trig_transform_loop(a):
    out = [0.0] * len(a)
    for i in range(len(a)):
        x = a[i]
        out[i] = math.sin(x) * math.cos(x) + math.sqrt(abs(x))
    return out


def trig_transform_vectorized(a):
    return np.sin(a) * np.cos(a) + np.sqrt(np.abs(a))


# ---------------------------------------------------------------- benchmark harness
def benchmark(label, loop_fn, vec_fn, list_args, array_args, repeats=3):
    """Time both versions with timeit and return a result row."""
    loop_time = timeit.timeit(lambda: loop_fn(*list_args), number=repeats) / repeats
    vec_time = timeit.timeit(lambda: vec_fn(*array_args), number=repeats) / repeats
    speedup = loop_time / vec_time
    print(f"{label:28s}  loop: {loop_time:10.6f}s   "
          f"vectorized: {vec_time:10.6f}s   speedup: {speedup:9,.0f}x")
    return {
        "task": label,
        "loop_seconds": f"{loop_time:.6f}",
        "vectorized_seconds": f"{vec_time:.6f}",
        "speedup_x": f"{speedup:.1f}",
    }


def run_all_benchmarks(n=2_000_000, log_path="benchmark_log.csv"):
    rng = np.random.default_rng(0)
    a_arr = rng.random(n)
    b_arr = rng.random(2500)
    a_list = a_arr.tolist()
    b_list = b_arr.tolist()

    small_a_arr = a_arr[:2500]
    small_a_list = small_a_arr.tolist()

    rows = []
    rows.append(benchmark(
        "sum of squares",
        sum_squares_loop, sum_squares_vectorized,
        (a_list,), (a_arr,),
    ))
    rows.append(benchmark(
        "min-max normalize",
        normalize_loop, normalize_vectorized,
        (a_list,), (a_arr,),
    ))
    rows.append(benchmark(
        "clip and scale",
        clip_and_scale_loop, clip_and_scale_vectorized,
        (a_list, 0.2, 0.8, 10.0), (a_arr, 0.2, 0.8, 10.0),
    ))
    rows.append(benchmark(
        "pairwise products (2000x2000)",
        pairwise_products_loop, pairwise_products_vectorized,
        (small_a_list, b_list), (small_a_arr, b_arr),
        repeats=1,
    ))
    rows.append(benchmark(
        "running max",
        running_max_loop, running_max_vectorized,
        (a_list,), (a_arr,),
    ))
    rows.append(benchmark(
        "safe sqrt (math.sqrt vs np.sqrt)",
        safe_sqrt_loop, safe_sqrt_vectorized,
        (a_list,), (a_arr,),
    ))
    rows.append(benchmark(
        "trig transform (sin*cos+sqrt)",
        trig_transform_loop, trig_transform_vectorized,
        (a_list,), (a_arr,),
    ))

    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["task", "loop_seconds", "vectorized_seconds", "speedup_x"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nLogged {len(rows)} benchmark rows to {log_path}")
    return rows


if __name__ == "__main__":
    run_all_benchmarks()
