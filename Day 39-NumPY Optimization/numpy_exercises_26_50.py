"""
numpy_exercises_26_50.py
Day 39 -- 100 NumPy Exercises, items 26-50
================================================

Continuation of Day 38's set (1-25). Every solution below is fully
vectorized: no Python-level `for`/`while` loop touches array elements.
Each function is self-contained and returns a value so it can be
imported and asserted against by test_exercises.py.

Run this file directly to print a short demo of every exercise.
"""

import numpy as np


# 26 -----------------------------------------------------------------
def ex26():
    """
    Builtin sum(iterable, start) and np.sum(array, axis=-1) look similar
    but are NOT the same function. Builtin sum(range(5), -1) adds -1 as
    the starting value: 0+1+2+3+4-1 = 9. Once numpy's names are
    imported with `from numpy import *`, calling sum(range(5), -1)
    resolves to np.sum, whose second positional argument is `axis`, so
    it just sums range(5) along the last axis: 0+1+2+3+4 = 10.
    """
    builtin_result = sum(range(5), -1)          # uses Python's builtin sum
    numpy_result = np.sum(range(5), axis=-1)    # uses np.sum explicitly
    return builtin_result, numpy_result


# 27 -----------------------------------------------------------------
def ex27():
    """
    Given an integer vector Z, check which expressions are valid numpy
    operations and which raise errors. Returns a dict mapping each
    expression's label to True (legal) or False (raises an exception).
    """
    Z = np.arange(5)
    results = {}

    def legal(label, fn):
        try:
            fn()
            results[label] = True
        except Exception:
            results[label] = False

    legal("Z**Z", lambda: Z ** Z)
    legal("2 << Z >> 2", lambda: 2 << Z >> 2)
    legal("Z <- Z", lambda: Z < -Z)          # parses as Z < (-Z)
    legal("1j*Z", lambda: 1j * Z)
    legal("Z/1/1", lambda: Z / 1 / 1)
    legal("Z<Z>Z", lambda: Z < Z > Z)        # chained comparison -> error
    return results


# 28 -----------------------------------------------------------------
def ex28():
    """
    Evaluate a few edge-case numeric expressions with 0-arrays and NaN,
    silencing the runtime warnings numpy would normally print so the
    demo output stays clean.
    """
    with np.errstate(all="ignore"):
        true_div = np.array(0) / np.array(0)
        floor_div = np.array(0) // np.array(0)
        nan_cast = np.array([np.nan]).astype(int).astype(float)
    return true_div, floor_div, nan_cast


# 29 -----------------------------------------------------------------
def ex29(a):
    """Round a float array away from zero (not to nearest-even)."""
    return np.copysign(np.ceil(np.abs(a)), a)


# 30 -----------------------------------------------------------------
def ex30(a, b):
    """Return the values common to both 1-D arrays a and b."""
    return np.intersect1d(a, b)


# 31 -----------------------------------------------------------------
def ex31():
    """
    Temporarily silence all numpy floating point warnings using a
    context manager, then confirm normal warning behaviour resumes
    afterwards.
    """
    with np.errstate(all="ignore"):
        _ = np.array(0) / np.array(0)   # would normally warn, now silent
    return True


# 32 -----------------------------------------------------------------
def ex32():
    """
    Compare the regular sqrt (returns NaN for negative input) against
    numpy's complex-aware emath.sqrt (returns a complex number).
    """
    with np.errstate(invalid="ignore"):
        regular = np.sqrt(-1)
    complex_aware = np.emath.sqrt(-1)
    return regular, complex_aware, regular == complex_aware


# 33 -----------------------------------------------------------------
def ex33():
    """Return yesterday, today and tomorrow as datetime64[D] values."""
    today = np.datetime64("today", "D")
    yesterday = today - np.timedelta64(1, "D")
    tomorrow = today + np.timedelta64(1, "D")
    return yesterday, today, tomorrow


# 34 -----------------------------------------------------------------
def ex34():
    """Return every date belonging to July 2016 as a datetime64 array."""
    return np.arange("2016-07", "2016-08", dtype="datetime64[D]")


# 35 -----------------------------------------------------------------
def ex35(A, B):
    """
    Compute ((A + B) * (-A / 2)) in place, i.e. without allocating any
    extra temporary arrays beyond what is strictly required.
    """
    np.add(A, B, out=B)          # B = A + B
    np.divide(A, 2, out=A)       # A = A / 2
    np.negative(A, out=A)        # A = -A
    np.multiply(A, B, out=A)     # A = A * B
    return A


# 36 -----------------------------------------------------------------
def ex36(a):
    """Extract the integer part of a positive float array 4 ways."""
    m1 = a - a % 1
    m2 = np.floor(a)
    m3 = np.trunc(a)
    m4 = a.astype(int)
    return m1, m2, m3, m4


# 37 -----------------------------------------------------------------
def ex37():
    """Build a 5x5 matrix where every row holds the values 0..4."""
    return np.zeros((5, 5)) + np.arange(5)


# 38 -----------------------------------------------------------------
def ex38():
    """Build a numpy array directly from a generator function."""
    def generate():
        for x in range(10):
            yield x
    return np.fromiter(generate(), dtype=float, count=10)


# 39 -----------------------------------------------------------------
def ex39():
    """A size-10 vector of values strictly between 0 and 1 (both excl.)."""
    return np.linspace(0, 1, 12)[1:-1]


# 40 -----------------------------------------------------------------
def ex40(size=10):
    """A random vector of the given size, sorted ascending."""
    v = np.random.random(size)
    return np.sort(v)


# 41 -----------------------------------------------------------------
def ex41(a):
    """Sum a small array faster than np.sum using np.add.reduce."""
    return np.add.reduce(a)


# 42 -----------------------------------------------------------------
def ex42(a, b):
    """Check whether two arrays are element-wise equal."""
    same_shape_and_values = np.array_equal(a, b)
    allow_tolerance = np.allclose(a, b)
    return same_shape_and_values, allow_tolerance


# 43 -----------------------------------------------------------------
def ex43(a):
    """Make an array read-only (immutable)."""
    a = a.copy()
    a.flags.writeable = False
    return a


# 44 -----------------------------------------------------------------
def ex44(cartesian):
    """
    Convert an (n, 2) array of cartesian (x, y) coordinates into polar
    (r, theta) coordinates, fully vectorized.
    """
    x, y = cartesian[:, 0], cartesian[:, 1]
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)
    return np.column_stack((r, theta))


# 45 -----------------------------------------------------------------
def ex45(a):
    """Return a copy of the vector with its maximum value replaced by 0."""
    a = a.copy()
    a[a.argmax()] = 0
    return a


# 46 -----------------------------------------------------------------
def ex46(n=5):
    """
    A structured array with named fields x and y covering an n x n grid
    over the [0, 1] x [0, 1] area.
    """
    grid = np.zeros((n, n), dtype=[("x", float), ("y", float)])
    grid["x"], grid["y"] = np.meshgrid(
        np.linspace(0, 1, n), np.linspace(0, 1, n)
    )
    return grid


# 47 -----------------------------------------------------------------
def ex47(x, y):
    """Build the Cauchy matrix C where C[i, j] = 1 / (x[i] - y[j])."""
    return 1.0 / (x[:, None] - y[None, :])


# 48 -----------------------------------------------------------------
def ex48():
    """Print the min/max representable value for every numpy scalar type."""
    report = {}
    for dtype in [np.int8, np.int32, np.int64]:
        info = np.iinfo(dtype)
        report[dtype.__name__] = (info.min, info.max)
    for dtype in [np.float32, np.float64]:
        info = np.finfo(dtype)
        report[dtype.__name__] = (info.min, info.max)
    return report


# 49 -----------------------------------------------------------------
def ex49(a):
    """
    Print an entire array without numpy's default truncation
    ("..." for large arrays). Returns the printed string.
    """
    old_opts = np.get_printoptions()
    np.set_printoptions(threshold=np.inf)
    text = np.array2string(a)
    np.set_printoptions(**old_opts)
    return text


# 50 -----------------------------------------------------------------
def ex50(a, target):
    """Return the value in vector a that is closest to the scalar target."""
    return a[np.abs(a - target).argmin()]


if __name__ == "__main__":
    print("26:", ex26())
    print("27:", ex27())
    print("28:", ex28())
    print("29:", ex29(np.array([-2.3, 1.1, -0.5, 3.9])))
    print("30:", ex30(np.array([1, 2, 3, 4]), np.array([3, 4, 5, 6])))
    ex31()
    print("31: warnings context worked")
    print("32:", ex32())
    print("33:", ex33())
    print("34 (first 3 days):", ex34()[:3])
    print("35:", ex35(np.array([1.0, 2.0]), np.array([3.0, 4.0])))
    print("36:", ex36(np.array([3.7, 2.1, 9.99])))
    print("37:\n", ex37())
    print("38:", ex38())
    print("39:", ex39())
    print("40:", ex40())
    print("41:", ex41(np.arange(10)))
    print("42:", ex42(np.array([1, 2]), np.array([1, 2])))
    print("43 writeable flag:", ex43(np.arange(3)).flags.writeable)
    print("44:", ex44(np.array([[1.0, 1.0], [0.0, 2.0]])))
    print("45:", ex45(np.array([1, 9, 3, 2])))
    print("46:\n", ex46(3))
    print("47:\n", ex47(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])))
    print("48:", ex48())
    print("49:", ex49(np.arange(5)))
    print("50:", ex50(np.array([1, 5, 9, 14, 20]), 12))
