"""
numpy_exercises_51_75.py
Day 40 -- 100 NumPy Exercises, items 51-75
================================================

Continuation of Day 39's set (26-50). Solutions favor vectorized numpy
operations; a handful of exercises (structured arrays, subclasses) are
inherently about numpy's type system rather than about avoiding loops,
and are implemented the idiomatic numpy way.

Run this file directly to print a short demo of every exercise.
"""

import numpy as np
from io import StringIO


# 51 -----------------------------------------------------------------
def ex51(n=3):
    """Structured array: a position (x, y) and a color (r, g, b) per row."""
    dtype = [("position", [("x", float), ("y", float)]),
             ("color", [("r", float), ("g", float), ("b", float)])]
    return np.zeros(n, dtype=dtype)


# 52 -----------------------------------------------------------------
def ex52(coords):
    """
    coords: (n, 2) array of cartesian points.
    Return the (n, n) matrix of pairwise euclidean distances.
    """
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


# 53 -----------------------------------------------------------------
def ex53(a):
    """
    a: float32 array. Cast to int32 IN PLACE, reusing the same buffer.
    float32 and int32 share the same item size (4 bytes), so a same
    -sized view lets the cast happen without allocating new memory.
    """
    view = a.view(np.int32)
    view[:] = a
    return view


# 54 -----------------------------------------------------------------
def ex54():
    """Read a CSV-like block of text that has missing values."""
    raw = "1, 2, 3, 4, 5\n6, , , 7, 8\n, , 9,10,11"
    return np.genfromtxt(StringIO(raw), delimiter=",")


# 55 -----------------------------------------------------------------
def ex55(a):
    """The array equivalent of Python's enumerate()."""
    return list(np.ndenumerate(a))


# 56 -----------------------------------------------------------------
def ex56(shape=(10, 10), sigma=1.0):
    """A generic 2D Gaussian-shaped array."""
    y, x = np.mgrid[-1:1:shape[0] * 1j, -1:1:shape[1] * 1j]
    d = np.sqrt(x * x + y * y)
    return np.exp(-(d ** 2) / (2.0 * sigma ** 2))


# 57 -----------------------------------------------------------------
def ex57(shape=(5, 5), p=3):
    """Randomly place p ones inside a 2D array of zeros."""
    arr = np.zeros(shape)
    idx = np.random.choice(arr.size, p, replace=False)
    np.put(arr, idx, 1)
    return arr


# 58 -----------------------------------------------------------------
def ex58(a):
    """Subtract the mean of each row from that row."""
    return a - a.mean(axis=1, keepdims=True)


# 59 -----------------------------------------------------------------
def ex59(a, column):
    """Sort the rows of a 2D array by a given column."""
    return a[a[:, column].argsort()]


# 60 -----------------------------------------------------------------
def ex60(a):
    """True if the 2D array has at least one all-zero column."""
    return bool((~a.any(axis=0)).any())


# 61 -----------------------------------------------------------------
def ex61(a, value):
    """The value inside array a that is closest to the given scalar."""
    return a.flat[np.abs(a - value).argmin()]


# 62 -----------------------------------------------------------------
def ex62(a, b):
    """
    a shape (1, n), b shape (n, 1): compute their broadcast sum using
    an explicit nditer, as the exercise requires (rather than `a + b`).
    """
    it = np.nditer([a, b, None])
    for x, y, z in it:
        z[...] = x + y
    return it.operands[2]


# 63 -----------------------------------------------------------------
class NamedArray(np.ndarray):
    """An ndarray subclass that carries a `name` attribute."""

    def __new__(cls, array, name="no name"):
        obj = np.asarray(array).view(cls)
        obj.name = name
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.name = getattr(obj, "name", "no name")


def ex63():
    arr = NamedArray(np.arange(5), name="day40-array")
    return arr, arr.name


# 64 -----------------------------------------------------------------
def ex64(size=10, index=None, repeats=20):
    """
    Add 1 to Z at each position listed in `index`, correctly handling
    repeated indices (a plain Z[index] += 1 would silently drop
    duplicates).
    """
    z = np.ones(size)
    if index is None:
        index = np.random.randint(0, size, repeats)
    np.add.at(z, index, 1)
    return z


# 65 -----------------------------------------------------------------
def ex65(values, index):
    """
    Accumulate `values` into an output array `F`, where each value's
    destination slot is given by the matching entry in `index`.
    """
    f = np.zeros(index.max() + 1)
    np.add.at(f, index, values)
    return f


# 66 -----------------------------------------------------------------
def ex66(image):
    """image: (w, h, 3) ubyte array. Count the number of unique colors."""
    pixels = image.reshape(-1, image.shape[-1])
    return len(np.unique(pixels, axis=0))


# 67 -----------------------------------------------------------------
def ex67(a):
    """Sum a 4D array over its last two axes in a single call."""
    return a.sum(axis=(-2, -1))


# 68 -----------------------------------------------------------------
def ex68(values, subset_index):
    """
    Compute the mean of `values` grouped by `subset_index` (same
    length vector of group labels), without a Python loop.
    """
    sums = np.bincount(subset_index, weights=values)
    counts = np.bincount(subset_index)
    return sums / counts


# 69 -----------------------------------------------------------------
def ex69(a, b):
    """Diagonal of A @ B, without forming the full matrix product."""
    return np.einsum("ij,ji->i", a, b)


# 70 -----------------------------------------------------------------
def ex70(z, n_zeros=3):
    """Interleave n_zeros zeros between every consecutive pair of z."""
    size = len(z) + (len(z) - 1) * n_zeros
    out = np.zeros(size)
    out[::n_zeros + 1] = z
    return out


# 71 -----------------------------------------------------------------
def ex71(a, b):
    """a: (5,5,3), b: (5,5). Multiply every channel of a by b."""
    return a * b[:, :, None]


# 72 -----------------------------------------------------------------
def ex72(a, row1, row2):
    """Return a copy of the array with two rows swapped."""
    out = a.copy()
    out[[row1, row2]] = out[[row2, row1]]
    return out


# 73 -----------------------------------------------------------------
def ex73(triangles):
    """
    triangles: (n, 3) array of vertex indices, one row per triangle.
    Return the set of unique undirected edges (line segments) that
    make up all the triangles.
    """
    edges = np.roll(triangles.repeat(2, axis=1), -1, axis=1)
    edges = edges.reshape(len(edges) * 3, 2)
    edges = np.sort(edges, axis=1)
    structured = edges.view(dtype=[("p0", edges.dtype), ("p1", edges.dtype)])
    return np.unique(structured)


# 74 -----------------------------------------------------------------
def ex74(counts):
    """Given a bincount array C, rebuild an array A with bincount(A)==C."""
    return np.repeat(np.arange(len(counts)), counts)


# 75 -----------------------------------------------------------------
def ex75(a, window):
    """Sliding-window average of `a` with the given window size."""
    cumsum = np.cumsum(np.insert(a, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window


if __name__ == "__main__":
    print("51:\n", ex51())
    print("52:\n", ex52(np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])))
    print("53:", ex53(np.array([1.9, 2.1, -3.7], dtype=np.float32)))
    print("54:\n", ex54())
    print("55:", ex55(np.array([[1, 2], [3, 4]]))[:2], "...")
    print("56:\n", ex56((5, 5)))
    print("57:\n", ex57((4, 4), 3))
    print("58:\n", ex58(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 9.0]])))
    print("59:\n", ex59(np.array([[3, 1], [1, 2], [2, 0]]), 0))
    print("60:", ex60(np.array([[1, 0, 2], [3, 0, 4]])))
    print("61:", ex61(np.array([1, 5, 9, 14, 20]), 12))
    print("62:", ex62(np.array([[1, 2, 3]]), np.array([[10], [20]])))
    print("63:", ex63())
    print("64:", ex64(5, np.array([0, 0, 1, 2, 2, 2])))
    print("65:", ex65(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0, 1, 0, 2])))
    print("66: (2x2 image, 2 unique colors expected) ->",
          ex66(np.array([[[0, 0, 0], [255, 255, 255]],
                          [[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)))
    print("67 shape:", ex67(np.ones((2, 3, 4, 5))).shape)
    print("68:", ex68(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0, 0, 1, 1])))
    print("69:", ex69(np.eye(3), np.eye(3) * 2))
    print("70:", ex70(np.array([1, 2, 3, 4, 5])))
    print("71 shape:", ex71(np.ones((5, 5, 3)), np.ones((5, 5)) * 2).shape)
    print("72:\n", ex72(np.arange(9).reshape(3, 3), 0, 2))
    print("73:", ex73(np.array([[0, 1, 2], [1, 2, 3]])))
    print("74:", ex74(np.array([2, 1, 0, 3])))
    print("75:", ex75(np.arange(10, dtype=float), 3))
