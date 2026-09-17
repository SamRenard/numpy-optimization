"""
fancy_indexing_anatomy.py
Day 40 -- Fancy Indexing, Boolean Masking, View vs Copy Danger
================================================================

NumPy has two indexing regimes that behave completely differently:

  * BASIC indexing (single integers, slices, `...`, `None`) always
    returns a VIEW -- a new array header pointing at the SAME memory.
  * ADVANCED / "fancy" indexing (an integer array, or a boolean mask)
    always returns a COPY -- brand new memory.

Mixing this up is one of the most common silent bugs in numpy code:
you think you're editing the original array, but you're editing a
throwaway copy (or vice versa). This script makes the distinction
concrete.

Run directly:
    python fancy_indexing_anatomy.py
"""

import numpy as np


def _rule_line():
    print("=" * 70)


def demo_basic_indexing_is_a_view():
    """Slicing shares memory with the original array."""
    _rule_line()
    print("BASIC indexing (slicing) -> a VIEW")
    _rule_line()
    a = np.arange(10)
    sl = a[2:5]
    print(f"a   = {a}")
    print(f"sl = a[2:5] = {sl}")
    print(f"sl.base is a: {sl.base is a}")
    sl[0] = 999
    print(f"after sl[0] = 999, a = {a}   <- original changed!\n")


def demo_fancy_indexing_is_a_copy():
    """Integer-array indexing allocates new memory."""
    _rule_line()
    print("FANCY indexing (integer array) -> a COPY")
    _rule_line()
    a = np.arange(10)
    idx = np.array([2, 3, 4])
    fancy = a[idx]
    print(f"a     = {a}")
    print(f"fancy = a[[2,3,4]] = {fancy}")
    print(f"fancy.base is a: {fancy.base is a}")
    fancy[0] = 999
    print(f"after fancy[0] = 999, a = {a}   <- original UNCHANGED\n")


def demo_boolean_masking_is_a_copy():
    """Boolean masks also always return a copy, never a view."""
    _rule_line()
    print("BOOLEAN masking -> a COPY")
    _rule_line()
    a = np.arange(10)
    mask = a % 2 == 0
    evens = a[mask]
    print(f"a     = {a}")
    print(f"mask  = {mask}")
    print(f"evens = a[mask] = {evens}")
    print(f"evens.base is a: {evens.base is a}")
    evens[:] = -1
    print(f"after evens[:] = -1, a = {a}   <- original UNCHANGED\n")


def demo_boolean_masking_in_place_edit():
    """
    To actually MODIFY the original array based on a condition, index
    the ORIGINAL array with the mask on the left-hand side of `=`
    instead of assigning the extracted copy.
    """
    _rule_line()
    print("Editing the ORIGINAL through a boolean mask (correct pattern)")
    _rule_line()
    a = np.arange(10)
    print(f"before: {a}")
    a[a % 2 == 0] = -1          # this DOES mutate a
    print(f"after a[a % 2 == 0] = -1 : {a}\n")


def demo_mixed_indexing_returns_copy():
    """
    Even ONE fancy/boolean component inside an otherwise "basic" index
    expression forces the whole result to be a copy.
    """
    _rule_line()
    print("Mixed indexing (slice + fancy) still returns a COPY")
    _rule_line()
    a = np.arange(12).reshape(3, 4)
    mixed = a[1:, [0, 2]]        # slice + fancy column selection
    print(f"a =\n{a}")
    print(f"a[1:, [0, 2]] =\n{mixed}")
    print(f"mixed.base is a: {mixed.base is a}\n")


def demo_the_danger_in_practice():
    """
    A realistic bug: someone assumes `subset = a[a > 5]` lets them
    edit `a` in place, silently loses the edit, then wonders why
    nothing changed.
    """
    _rule_line()
    print("The danger, in practice")
    _rule_line()
    a = np.array([1, 8, 3, 9, 2, 7])
    subset = a[a > 5]             # copy! easy to forget
    subset *= 0                   # only zeroes the COPY
    print(f"a (still has the big values!) = {a}")
    print("Fix: a[a > 5] = 0   -- index the array itself, not the copy")
    a[a > 5] = 0
    print(f"a (fixed)                     = {a}\n")


if __name__ == "__main__":
    demo_basic_indexing_is_a_view()
    demo_fancy_indexing_is_a_copy()
    demo_boolean_masking_is_a_copy()
    demo_boolean_masking_in_place_edit()
    demo_mixed_indexing_returns_copy()
    demo_the_danger_in_practice()
