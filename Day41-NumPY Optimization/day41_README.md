# Day 41 — NumPy Optimization (Vectorization)

**Program:** NIZAM AI — 150 Day AI Engineering Protocol
**Month 2:** Math & Deep Learning · Block 4/5 · 4 hours total

## What this covers

Today's theory is *why* vectorization is fast: the C backend removing
per-element interpreter overhead, SIMD letting the CPU process several
values per instruction, and cache locality from numpy's contiguous
memory layout. The practical block takes loop-based code, rewrites it
in vectorized numpy, and measures the real speedup with `timeit`
(the script's own `%timeit` equivalent), logging results to
`benchmark_log.csv`.

## Files

| File | Description | Status |
|---|---|---|
| `vectorization_anatomy.py` | Demos of C-backend overhead, SIMD, and cache locality, each with a timed before/after | ✅ runs clean |
| `loop_vs_vectorized.py` | 7 loop-vs-vectorized function pairs, benchmarked and logged to `benchmark_log.csv` | ✅ runs clean |
| `test_exercises.py` | 11 tests: correctness of every vectorized rewrite + benchmark harness sanity checks | ✅ 11/11 PASS |
| `README.md` | This file | ✅ |
| `benchmark_log.csv` | Generated when `loop_vs_vectorized.py` runs — one row per task with measured timings and speedup | generated on run |

## Measured results (this machine)

| Task | Loop (s) | Vectorized (s) | Speedup |
|---|---|---|---|
| sum of squares | 0.0399 | 0.0045 | ~9x |
| min-max normalize | 0.1877 | 0.0046 | ~41x |
| clip and scale | 0.1542 | 0.0028 | ~54x |
| pairwise products (2000×2000) | 0.4051 | 0.0173 | ~23x |
| running max | 0.0727 | 0.0096 | ~8x |
| safe sqrt (`math.sqrt` vs `np.sqrt`) | 0.1784 | 0.0042 | ~42x |
| trig transform (`sin·cos+sqrt`) | 0.2778 | 0.0398 | ~7x |

Speedup is not a fixed constant — it depends on how much per-element
Python overhead the loop version pays (attribute lookups, function
calls, branching) versus how much real math work numpy can push into
its C loop. Operations with more Python-level overhead per element
(normalize, clip, safe-sqrt) get closer to 40–55x here; on larger
arrays, older CPUs, or GPU-backed workloads it's common to see this
climb past 100x, which is why the plan's "measure the ~100x gap" goal
is treated as an order-of-magnitude target, not a fixed number.

## How to run

```bash
python vectorization_anatomy.py      # prints the SIMD/cache/backend demos
python loop_vs_vectorized.py         # runs the 7 benchmarks, writes benchmark_log.csv
python test_exercises.py             # runs the 11-test suite (or use pytest)
```

## Sources used today

- NumPy official documentation on performance
- "From Python to NumPy" (N. Rougier) — vectorization chapter
- Python `timeit` standard library docs
