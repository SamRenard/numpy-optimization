# Day 38 — NumPy Optimallaşdırması: ndarray Anatomiyası + Exercises #1-25

Part of the **NIZAM AI · 150-Day AI Engineering Protocol**
Month 2 · Mathematics & Deep Learning · Block Day 1/5

## 📌 Concept

NumPy's speed comes from how `ndarray` is structured under the hood.
`shape` gives the dimensions, `dtype` fixes each element's byte size,
and **strides** tell NumPy how many bytes to jump in memory to move
along each dimension — this is what lets one flat block of memory
*look* multi-dimensional. Slicing usually creates a **view** (shares
memory, just changes strides — free, instant), while fancy indexing or
reshape sometimes forces a **copy**. **Contiguous memory** (elements
stored back-to-back) is what makes CPU cache access — and therefore
NumPy — fast.

## 📂 Files

| File | Description |
|---|---|
| `ndarray_anatomy.py` | Hands-on exploration: shape/dtype/itemsize, strides (incl. transpose = stride swap, zero copy), view vs. copy, contiguity performance timing |
| `numpy_exercises_01_25.py` | Solutions to exercises #1–25 from the classic "100 numpy exercises" set |
| `test_exercises.py` | Automated assertion-based tests for exercises #2–25 |

## ▶️ Usage

```bash
pip install numpy
python3 ndarray_anatomy.py          # explore shape/stride/view/contiguity
python3 numpy_exercises_01_25.py    # run exercises 1-25 with printed output
python3 test_exercises.py           # automated correctness checks
```

## ✅ Test Results

```
23/23 checks passed
```
(Exercises #1 and #4 return the NumPy version string and a docstring —
not deterministic values to assert against, so they're demonstrated
but not unit-tested.)

## 🔑 Key takeaways

- `transpose()` never copies data — it just swaps strides.
- Slicing (`a[::2]`) → view; fancy indexing (`a[[1,3,5]]`) → copy.
- `sum(range(5), -1)` is a classic gotcha: Python's builtin `sum` (not
  `np.sum`) takes a `start` argument, so this equals `10 + (-1) = 9`.

## 📚 Sources

- NumPy official quickstart
- 100 NumPy Exercises (rougier/numpy-100)
- From Python to NumPy (Nicolas P. Rougier)

## 🔁 Refactor Log

- Initial pass: solved exercises with vectorized NumPy operations, no explicit loops
- Added performance timing comparison for C- vs F-contiguous arrays
- Added automated test suite covering all deterministic exercises

---
*NIZAM AI Day 38/150*
