# Day 39 — NumPy Optimization (Broadcasting)

**Program:** NIZAM AI — 150 Day AI Engineering Protocol
**Month 2:** Math & Deep Learning · Block 2/5 · 4 hours total

## What this covers

Today's focus is NumPy broadcasting: the three rules that let arrays of
different shapes interact without explicit loops, plus 25 more exercises
from the classic "100 NumPy Exercises" set, all solved without a single
Python-level loop over array elements.

## Files

| File | Description | Status |
|---|---|---|
| `broadcasting_anatomy.py` | Live demo of the 3 broadcasting rules, a view-vs-copy check with `np.broadcast_to`, and a loop-vs-vectorized speed benchmark | ✅ runs clean |
| `numpy_exercises_26_50.py` | Exercises 26–50, all vectorized, no loops | ✅ 25/25 implemented |
| `test_exercises.py` | 30 tests covering every exercise + every broadcasting rule | ✅ 30/30 PASS |
| `README.md` | This file | ✅ |

## The 3 broadcasting rules (quick reference)

1. **Align from the right.** If two arrays have a different number of
   dimensions, the smaller shape is padded with 1s on its left until
   both shapes have equal length.
2. **Stretch size-1 dimensions.** Once shapes have equal length, any
   axis where one array has size 1 is treated as if it were repeated
   to match the other array's size — with no data actually copied.
3. **Mismatched dimensions are illegal.** If two axes disagree in size
   and neither is 1, the shapes cannot be broadcast and NumPy raises
   `ValueError`.

## How to run

```bash
python broadcasting_anatomy.py       # prints the rule-by-rule demo + benchmark
python numpy_exercises_26_50.py      # prints a short demo of all 25 solutions
python test_exercises.py             # runs the 30-test suite (or use pytest)
```

## Remaining work

Exercises 51–100 of the "100 NumPy Exercises" set are still left for a
future day (today's task was limited to 26–50, continuing from
yesterday's 1–25).

## Sources used today

- NumPy official broadcasting documentation
- "100 NumPy Exercises" (rougier/numpy-100)
- "From Python to NumPy" (N. Rougier)
