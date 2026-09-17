"""
monte_carlo_pi.py
Day 42 -- Practical Coding: Estimate pi with the Monte Carlo Method
=======================================================================

Idea: throw N random points uniformly into the [-1, 1] x [-1, 1]
square. The unit circle inscribed in that square has area pi, the
square has area 4, so:

    P(point lands inside the circle) = pi / 4

Counting the fraction of random points that land inside the circle
(x^2 + y^2 <= 1) and multiplying by 4 gives an estimate of pi that
gets more accurate as N grows -- a direct, hands-on use of today's
random-module and vectorization material.

Run directly:
    python monte_carlo_pi.py
"""

import numpy as np


def estimate_pi(n_points, seed=None):
    """
    Vectorized Monte Carlo estimate of pi using n_points random samples.
    Returns (pi_estimate, fraction_inside).
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, n_points)
    y = rng.uniform(-1.0, 1.0, n_points)
    inside = (x ** 2 + y ** 2) <= 1.0
    fraction_inside = inside.mean()
    pi_estimate = 4.0 * fraction_inside
    return pi_estimate, fraction_inside


def estimate_pi_convergence(sample_sizes, seed=0):
    """
    Run the estimator at several sample sizes and report how the
    estimate's error shrinks as N grows.
    """
    rng = np.random.default_rng(seed)
    results = []
    for n in sample_sizes:
        x = rng.uniform(-1.0, 1.0, n)
        y = rng.uniform(-1.0, 1.0, n)
        inside = (x ** 2 + y ** 2) <= 1.0
        pi_hat = 4.0 * inside.mean()
        error = abs(pi_hat - np.pi)
        results.append((n, pi_hat, error))
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("Monte Carlo estimate of pi")
    print("=" * 70)

    pi_hat, fraction = estimate_pi(1_000_000, seed=42)
    print(f"N = 1,000,000 points")
    print(f"fraction inside the circle : {fraction:.5f}")
    print(f"pi estimate                : {pi_hat:.5f}")
    print(f"true pi                    : {np.pi:.5f}")
    print(f"absolute error             : {abs(pi_hat - np.pi):.5f}\n")

    print("Convergence as N grows:")
    print(f"{'N':>10} | {'pi estimate':>12} | {'abs error':>10}")
    print("-" * 38)
    for n, pi_hat, error in estimate_pi_convergence(
        [10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
    ):
        print(f"{n:>10,} | {pi_hat:>12.6f} | {error:>10.6f}")
