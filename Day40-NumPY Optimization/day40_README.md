# Day 40 — NumPy Optimization (Fancy Indexing & Boolean Masking)

**Program:** NIZAM AI — 150 Day AI Engineering Protocol
**Month 2:** Math & Deep Learning · Block 3/5 · 4 hours total

## What this covers

Today's theory is the view-vs-copy split between basic and advanced
indexing: fancy (integer array) indexing and boolean masking, and the
silent bug that happens when you forget which one you're using. The
practical block continues the "100 NumPy Exercises" set with items
51–75.

## Files

| File | Description | Status |
|---|---|---|
| `fancy_indexing_anatomy.py` | Basic vs fancy indexing, boolean masking, the in-place-edit-via-mask pattern, and the classic "silent copy" bug | ✅ runs clean |
| `numpy_exercises_51_75.py` | Exercises 51–75, vectorized | ✅ 25/25 implemented |
| `test_exercises.py` | 30 tests covering every exercise + every indexing rule | ✅ 30/30 PASS |
| `README.md` | This file | ✅ |

## Key rule for today

- **Basic indexing** (single ints, slices, `...`) → **view**, shares memory.
- **Fancy indexing** (integer array) and **boolean masking** → **copy**, new memory.
- To edit the original array conditionally, index the array itself on
  the left of `=` (`a[a > 5] = 0`), don't edit an extracted copy.

## How to run

```bash
python fancy_indexing_anatomy.py     # prints the view/copy demos
python numpy_exercises_51_75.py      # prints a short demo of all 25 solutions
python test_exercises.py             # runs the 30-test suite (or use pytest)
```

## Remaining work

Exercises 76–100 are scheduled for a later day, continuing from 1–25
(Day 38) and 26–50 (Day 39).

## Sources used today

- NumPy official documentation on indexing
- "100 NumPy Exercises" (rougier/numpy-100)
- "From Python to NumPy" (N. Rougier)
