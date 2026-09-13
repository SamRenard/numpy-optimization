"""
Day 38 — 100 NumPy Exercises: #1-25
NIZAM AI · 150-Day AI Engineering Protocol

Solutions to the first 25 exercises from the well-known "100 numpy
exercises" set (rougier/numpy-100). Each function is self-contained,
solved without loops where NumPy offers a vectorized alternative, and
demonstrated with a printed example when run directly.
"""

import numpy as np


# 1. Import numpy as np and print the version
def ex01():
    return np.__version__


# 2. Create a null vector of size 10
def ex02():
    return np.zeros(10)


# 3. How to find the memory size of any array
def ex03(arr: np.ndarray) -> int:
    return arr.size * arr.itemsize


# 4. Get the documentation of numpy add function (returns a string, not run here)
def ex04() -> str:
    return np.add.__doc__


# 5. Create a null vector of size 10 but the fifth value is 1
def ex05():
    a = np.zeros(10)
    a[4] = 1
    return a


# 6. Create a vector with values ranging from 10 to 49
def ex06():
    return np.arange(10, 50)


# 7. Reverse a vector (first element becomes last)
def ex07(arr: np.ndarray):
    return arr[::-1]


# 8. Create a 3x3 matrix with values ranging from 0 to 8
def ex08():
    return np.arange(9).reshape(3, 3)


# 9. Find indices of non-zero elements from [1,2,0,0,4,0]
def ex09():
    return np.nonzero([1, 2, 0, 0, 4, 0])


# 10. Create a 3x3 identity matrix
def ex10():
    return np.eye(3)


# 11. Create a 3x3x3 array with random values
def ex11():
    return np.random.random((3, 3, 3))


# 12. Create a 10x10 array with random values and find the min/max
def ex12():
    a = np.random.random((10, 10))
    return a.min(), a.max()


# 13. Create a random vector of size 30 and find the mean value
def ex13():
    a = np.random.random(30)
    return a.mean()


# 14. Create a 2D array with 1 on the border and 0 inside
def ex14(size: int = 5):
    a = np.ones((size, size))
    a[1:-1, 1:-1] = 0
    return a


# 15. Add a border of zeros around an existing array
def ex15(arr: np.ndarray):
    return np.pad(arr, pad_width=1, mode="constant", constant_values=0)


# 16. What are the results of these NaN/inf expressions? (returns explanations)
def ex16():
    return {
        "0 * np.nan": 0 * np.nan,                       # nan
        "np.nan == np.nan": np.nan == np.nan,            # False
        "np.inf > np.nan": np.inf > np.nan,               # False
        "np.nan - np.nan": np.nan - np.nan,               # nan
        "np.nan in set([np.nan])": np.nan in set([np.nan]),  # True (identity, not equality)
        "0.3 == 3 * 0.1": 0.3 == 3 * 0.1,                 # False (float precision)
    }


# 17. Create a 5x5 matrix with values 1,2,3,4 just below the diagonal
def ex17():
    return np.diag(1 + np.arange(4), k=-1)


# 18. Create an 8x8 checkerboard pattern using slicing
def ex18():
    a = np.zeros((8, 8), dtype=int)
    a[1::2, ::2] = 1
    a[::2, 1::2] = 1
    return a


# 19. Given a (6,7,8) shape array, find the index (x,y,z) of the 100th element
def ex19():
    return np.unravel_index(99, (6, 7, 8))


# 20. Create a checkerboard 8x8 using tile
def ex20():
    return np.tile(np.array([[0, 1], [1, 0]]), (4, 4))


# 21. Normalize a 5x5 random matrix (subtract mean, divide by std)
def ex21():
    a = np.random.random((5, 5))
    return (a - a.mean()) / a.std()


# 22. Create a custom dtype describing a color as four unsigned bytes (RGBA)
def ex22():
    return np.dtype(
        [("r", np.ubyte), ("g", np.ubyte), ("b", np.ubyte), ("a", np.ubyte)]
    )


# 23. Multiply a 5x3 matrix by a 3x2 matrix (real matrix product)
def ex23():
    a = np.ones((5, 3))
    b = np.ones((3, 2))
    return a @ b  # shape (5, 2)


# 24. Given a 1D array, negate all elements between 3 and 8, in place
def ex24():
    a = np.arange(11)
    a[(a > 3) & (a < 8)] *= -1
    return a


# 25. Sum of a range (without loops)
def ex25():
    return sum(range(5), -1)  # python builtin sum, -1 start -> result 9


if __name__ == "__main__":
    print(f"1.  numpy version: {ex01()}")
    print(f"2.  null vector: {ex02()}")
    print(f"3.  memory size of arange(10): {ex03(np.arange(10))} bytes")
    print(f"5.  vector with 1 at index 4: {ex05()}")
    print(f"6.  10..49: {ex06()}")
    print(f"7.  reversed: {ex07(np.arange(10))}")
    print(f"8.  3x3 range:\n{ex08()}")
    print(f"9.  nonzero indices: {ex09()}")
    print(f"10. identity:\n{ex10()}")
    print(f"11. random 3x3x3 shape: {ex11().shape}")
    a12 = ex12()
    print(f"12. min/max of random 10x10: {a12[0]:.4f}, {a12[1]:.4f}")
    print(f"13. mean of random 30: {ex13():.4f}")
    print(f"14. bordered array:\n{ex14()}")
    print(f"15. padded array:\n{ex15(np.ones((2, 2)))}")
    print(f"16. NaN/inf gotchas: {ex16()}")
    print(f"17. sub-diagonal matrix:\n{ex17()}")
    print(f"18. checkerboard (slicing):\n{ex18()}")
    print(f"19. index of 100th element in (6,7,8): {ex19()}")
    print(f"20. checkerboard (tile):\n{ex20()}")
    print(f"21. normalized matrix mean~0, std~1: mean={ex21().mean():.6f}, std={ex21().std():.4f}")
    print(f"22. RGBA dtype: {ex22()}")
    print(f"23. matrix product shape: {ex23().shape}")
    print(f"24. negate 3<x<8: {ex24()}")
    print(f"25. sum(range(5), -1) = {ex25()}  (python's sum, not np.sum -- classic gotcha!)")
