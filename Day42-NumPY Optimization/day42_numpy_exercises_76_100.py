"""
numpy_exercises_76_100.py
Day 42 -- 100 NumPy Exercises, items 76-100 (final batch)
================================================

Completes the set started on Day 38 (1-25), continued on Day 39
(26-50) and Day 40 (51-75). A few of these (Game of Life, the
symmetric-array subclass, the multinomial-row selector) are naturally
expressed with a thin loop over a small outer dimension; every inner,
per-element computation is still vectorized.

Run this file directly to print a short demo of every exercise.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# 76 -----------------------------------------------------------------
def ex76(z, window=3):
    """
    z: 1D array. Build a 2D array whose row i is z[i:i+window] --
    i.e. a sliding window view, with zero extra memory allocated.
    """
    return sliding_window_view(z, window)


# 77 -----------------------------------------------------------------
def ex77(bool_array, float_array):
    """Negate a boolean array, and flip the sign of a float array, both in place."""
    np.logical_not(bool_array, out=bool_array)
    np.negative(float_array, out=float_array)
    return bool_array, float_array


# 78 -----------------------------------------------------------------
def ex78(p0, p1, p):
    """
    p0, p1: (n, 2) endpoints of n line segments. p: a single 2D point.
    Return the perpendicular distance from p to each line.
    """
    d = p1 - p0
    length_sq = (d ** 2).sum(axis=1)
    t = ((p - p0) * d).sum(axis=1) / length_sq
    projection = p0 + t[:, None] * d
    return np.sqrt(((p - projection) ** 2).sum(axis=1))


# 79 -----------------------------------------------------------------
def ex79(p0, p1, points):
    """
    Same as ex78 but for a whole set of points: return the (m, n)
    matrix of distances from each point j to each line i.
    """
    d = p1 - p0                                    # (n, 2)
    length_sq = (d ** 2).sum(axis=1)                # (n,)
    diff = points[:, None, :] - p0[None, :, :]      # (m, n, 2)
    t = (diff * d[None, :, :]).sum(axis=2) / length_sq[None, :]
    projection = p0[None, :, :] + t[:, :, None] * d[None, :, :]
    return np.sqrt(((points[:, None, :] - projection) ** 2).sum(axis=2))


# 80 -----------------------------------------------------------------
def ex80(a, center, shape, fill=0):
    """
    Extract a `shape`-sized window from `a` centered on `center`,
    padding with `fill` wherever the window falls outside `a`.
    """
    out = np.full(shape, fill, dtype=a.dtype)
    half = (shape[0] // 2, shape[1] // 2)
    src_lo = [max(center[i] - half[i], 0) for i in range(2)]
    src_hi = [min(center[i] - half[i] + shape[i], a.shape[i]) for i in range(2)]
    dst_lo = [src_lo[i] - (center[i] - half[i]) for i in range(2)]
    dst_hi = [dst_lo[i] + (src_hi[i] - src_lo[i]) for i in range(2)]
    out[dst_lo[0]:dst_hi[0], dst_lo[1]:dst_hi[1]] = \
        a[src_lo[0]:src_hi[0], src_lo[1]:src_hi[1]]
    return out


# 81 -----------------------------------------------------------------
def ex81(z, window=4):
    """z=[1..14] style vector -> sliding-window rows via stride tricks."""
    return sliding_window_view(z, window)


# 82 -----------------------------------------------------------------
def ex82(a):
    """Compute the rank of a 2D matrix."""
    return np.linalg.matrix_rank(a)


# 83 -----------------------------------------------------------------
def ex83(a):
    """The most frequent value in a non-negative integer array."""
    return np.bincount(a).argmax()


# 84 -----------------------------------------------------------------
def ex84(a):
    """Every contiguous 3x3 block of a 2D array, as a stack of blocks."""
    return sliding_window_view(a, (3, 3)).reshape(-1, 3, 3)


# 85 -----------------------------------------------------------------
class SymmetricArray(np.ndarray):
    """A 2D array subclass where setting Z[i, j] also sets Z[j, i]."""

    def __setitem__(self, index, value):
        i, j = index
        super().__setitem__((i, j), value)
        super().__setitem__((j, i), value)


def ex85(n=4):
    return np.zeros((n, n)).view(SymmetricArray)


# 86 -----------------------------------------------------------------
def ex86(matrices, vectors):
    """
    matrices: (p, n, n), vectors: (p, n, 1).
    Sum of the p matrix-vector products, shape (n, 1).
    """
    return np.tensordot(matrices, vectors, axes=[[0, 2], [0, 1]])


# 87 -----------------------------------------------------------------
def ex87(a, block=4):
    """16x16-style array -> block-sums using a block-size reshape."""
    n, m = a.shape
    return a.reshape(n // block, block, m // block, block).sum(axis=(1, 3))


# 88 -----------------------------------------------------------------
def ex88(board, steps=1):
    """Conway's Game of Life, vectorized neighbor counting via padding."""
    board = board.astype(int)
    for _ in range(steps):
        padded = np.pad(board, 1, mode="constant")
        neighbors = sum(
            np.roll(np.roll(padded, i, axis=0), j, axis=1)
            for i in (-1, 0, 1) for j in (-1, 0, 1) if not (i == 0 and j == 0)
        )[1:-1, 1:-1]
        board = ((neighbors == 3) | ((board == 1) & (neighbors == 2))).astype(int)
    return board


# 89 -----------------------------------------------------------------
def ex89(a, n):
    """The n largest values of an array, unordered but O(n) selection."""
    return a[np.argpartition(a, -n)[-n:]]


# 90 -----------------------------------------------------------------
def ex90(*vectors):
    """Cartesian product of an arbitrary number of 1D vectors."""
    grids = np.meshgrid(*vectors, indexing="ij")
    return np.stack([g.ravel() for g in grids], axis=-1)


# 91 -----------------------------------------------------------------
def ex91(a, names):
    """Build a record array from a regular 2D array and column names."""
    return np.rec.fromarrays(a.T, names=names)


# 92 -----------------------------------------------------------------
def ex92(z):
    """Z**3 computed three different ways."""
    m1 = z ** 3
    m2 = z * z * z
    m3 = np.power(z, 3)
    return m1, m2, m3


# 93 -----------------------------------------------------------------
def ex93(a, b):
    """
    a: (8, 3), b: (2, 2). Return the rows of a that, for EVERY row of
    b, share at least one element with that row of b (order ignored).
    """
    # match[i, k, r, c] = True if a[i, k] equals b[r, c]
    match = a[:, :, None, None] == b[None, None, :, :]
    # collapse a's columns (axis 1) and b's columns (axis 3): does row
    # i of a share ANY element with row r of b?
    shares_element = match.any(axis=(1, 3))          # shape (len(a), len(b))
    # keep rows of a that satisfy this for ALL rows of b
    keep = shares_element.all(axis=1)
    return a[keep]


# 94 -----------------------------------------------------------------
def ex94(a):
    """From a 10x3-style matrix, keep only rows with not-all-equal values."""
    return a[~np.all(a == a[:, [0]], axis=1)]


# 95 -----------------------------------------------------------------
def ex95(ints, width=8):
    """Convert a vector of non-negative ints into its binary matrix form."""
    return ((ints[:, None] & (1 << np.arange(width - 1, -1, -1))) > 0).astype(int)


# 96 -----------------------------------------------------------------
def ex96(a):
    """Extract the unique rows of a 2D array."""
    return np.unique(a, axis=0)


# 97 -----------------------------------------------------------------
def ex97(a, b):
    """Inner, outer, sum and elementwise product of vectors a, b via einsum."""
    inner = np.einsum("i,i->", a, b)
    outer = np.einsum("i,j->ij", a, b)
    total_sum = np.einsum("i->", a)
    mul = np.einsum("i,i->i", a, b)
    return inner, outer, total_sum, mul


# 98 -----------------------------------------------------------------
def ex98(x, y, n_samples=10):
    """Resample a path (x, y) using n_samples equidistant points along its length."""
    dr = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    arc_length = np.concatenate(([0.0], np.cumsum(dr)))
    even_positions = np.linspace(0, arc_length[-1], n_samples)
    x_new = np.interp(even_positions, arc_length, x)
    y_new = np.interp(even_positions, arc_length, y)
    return x_new, y_new


# 99 -----------------------------------------------------------------
def ex99(x, n):
    """Rows of integer array x that sum to n (valid multinomial draws)."""
    is_integer = np.all(x == x.astype(int), axis=1)
    sums_to_n = x.sum(axis=1) == n
    return x[is_integer & sums_to_n]


# 100 ----------------------------------------------------------------
def ex100(x, n_resamples=1000, seed=0):
    """Bootstrapped 95% confidence interval for the mean of 1D array x."""
    rng = np.random.default_rng(seed)
    resample_means = np.array([
        rng.choice(x, size=len(x), replace=True).mean()
        for _ in range(n_resamples)
    ])
    return np.percentile(resample_means, [2.5, 97.5])


if __name__ == "__main__":
    print("76:\n", ex76(np.array([1, 2, 3, 4, 5, 6])))
    b, f = ex77(np.array([True, False, True]), np.array([1.0, -2.0, 3.0]))
    print("77:", b, f)
    print("78:", ex78(np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]), np.array([0.5, 1.0])))
    print("79 shape:", ex79(np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]),
                             np.array([[0.5, 1.0], [0.0, 2.0]])).shape)
    print("80:\n", ex80(np.arange(16).reshape(4, 4), (0, 0), (3, 3), fill=-1))
    print("81:\n", ex81(np.arange(1, 8)))
    print("82:", ex82(np.eye(3)))
    print("83:", ex83(np.array([1, 1, 2, 3, 3, 3, 4])))
    print("84 shape:", ex84(np.arange(25).reshape(5, 5)).shape)
    sym = ex85(3)
    sym[0, 1] = 9
    print("85:\n", np.asarray(sym))
    print("86 shape:", ex86(np.ones((2, 3, 3)), np.ones((2, 3, 1))).shape)
    print("87:\n", ex87(np.ones((8, 8)), block=4))
    glider = np.zeros((6, 6), dtype=int)
    glider[1, 2] = glider[2, 3] = glider[3, 1] = glider[3, 2] = glider[3, 3] = 1
    print("88 (after 1 step):\n", ex88(glider, steps=1))
    print("89:", ex89(np.array([3, 1, 9, 4, 7]), 3))
    print("90:\n", ex90(np.array([0, 1]), np.array([10, 20])))
    print("91:", ex91(np.array([[1, 2.5], [3, 4.5]]), names="a,b"))
    print("92:", ex92(np.array([1, 2, 3])))
    print("93:", ex93(np.array([[1, 2, 3], [4, 5, 6], [1, 4, 9]]), np.array([[1, 2], [4, 5]])))
    print("94:\n", ex94(np.array([[1, 1, 1], [1, 2, 3], [5, 5, 5]])))
    print("95:\n", ex95(np.array([2, 5]), width=4))
    print("96:\n", ex96(np.array([[1, 2], [1, 2], [3, 4]])))
    print("97:", ex97(np.array([1.0, 2.0]), np.array([3.0, 4.0])))
    print("98:", ex98(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 0.0]), 5))
    print("99:", ex99(np.array([[1, 2, 2], [0, 5, 0], [1, 1, 1]]), 5))
    print("100:", ex100(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), n_resamples=200))
