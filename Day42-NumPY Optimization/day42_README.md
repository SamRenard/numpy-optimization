# Day 42 — NumPy Optimization (Random, Statistics, Memory Layout)

**Program:** NIZAM AI — 150 Day AI Engineering Protocol
**Month 2:** Math & Deep Learning · Block 5/5 (final day of the block) · 4 hours total

## What this covers

The closing theory block of the NumPy optimization week: the modern
`np.random.Generator` API, the core statistical functions, and C-order
vs Fortran-order memory layout and why it changes performance. The
practical block finishes the "100 NumPy Exercises" set (76–100) and
estimates pi with the Monte Carlo method.

## Files

| File | Description | Status |
|---|---|---|
| `random_and_memory_layout.py` | `default_rng()` API, mean/median/std/percentile/correlation, and a timed C-order vs F-order comparison | ✅ runs clean |
| `numpy_exercises_76_100.py` | Exercises 76–100 — the final batch, completing the full 100 | ✅ 25/25 implemented |
| `monte_carlo_pi.py` | Vectorized Monte Carlo estimate of π, plus a convergence table as N grows | ✅ runs clean |
| `test_exercises.py` | 28 tests covering every exercise + the Monte Carlo estimator | ✅ 28/28 PASS |
| `README.md` | This file | ✅ |

## Monte Carlo π — the idea

Throw N random points into the square `[-1,1] x [-1,1]`. The circle
inscribed in it has area π, the square has area 4, so the fraction of
points landing inside the circle approximates π/4. Multiply by 4 to
estimate π. Sample result from this run:

| N | π estimate | absolute error |
|---|---|---|
| 10 | 2.400000 | 0.741593 |
| 1,000 | 3.112000 | 0.029593 |
| 100,000 | 3.143840 | 0.002247 |
| 10,000,000 | 3.141219 | 0.000374 |

## 100 NumPy Exercises — complete

This closes out the exercise set spread across four days:

| Day | Range |
|---|---|
| 38 | 1–25 |
| 39 | 26–50 |
| 40 | 51–75 |
| 42 | 76–100 |

## How to run

```bash
python random_and_memory_layout.py   # prints random/statistics/memory-layout demos
python numpy_exercises_76_100.py     # prints a short demo of all 25 solutions
python monte_carlo_pi.py             # prints the pi estimate + convergence table
python test_exercises.py             # runs the 28-test suite (or use pytest)
```

## Sources used today

- NumPy official documentation: `numpy.random.Generator`, statistics functions
- "100 NumPy Exercises" (rougier/numpy-100)
- "From Python to NumPy" (N. Rougier) — memory layout chapter
