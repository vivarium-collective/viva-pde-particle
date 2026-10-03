"""Exact well-mixed reference for the coupled calcium-spark benchmark (Schaff et al. 2016, Test 2).

In the fast-diffusion limit the spatial model reduces to a piecewise-deterministic Markov
process (PDMP). Calcium is well mixed over the volume V, and the open-channel count n jumps:

    dU/dt = J·n/V − V_p (U − U₀)                      (between jumps)
    closed → open at k_on·U/U₀ per closed channel,   open → closed at k_off per channel.

Between jumps U relaxes exponentially to U* = U₀ + J·n/(V_p·V), so the integrated total
hazard over a waiting time τ is analytic:

    Λ(τ) = a·τ + b·(1 − e^(−V_p τ)),
    a = n_c·k_on·U*/U₀ + n_o·k_off,  b = n_c·k_on·(U(t₀) − U*)/(U₀·V_p).

The next event time solves Λ(τ) = E with E ~ Exp(1). The event type is drawn from the
hazards at that time. This is exact, with no time discretization: the event-driven
reference the paper obtained with the Gibson–Bruck method.
"""
from __future__ import annotations

import numpy as np

from viva_pde_particle.benchmarks.calcium_sparks import SparkParams


def _solve_tau(a: float, b: float, vp: float, target: float) -> float:
    """Smallest τ > 0 with a·τ + b·(1 − e^(−vp τ)) = target (Λ is increasing)."""
    lam = lambda t: a * t + b * (1.0 - np.exp(-vp * t))  # noqa: E731
    hi = max(target / max(a, 1e-300), 1e-6)
    while lam(hi) < target:
        hi *= 2.0
        if hi > 1e12:
            return np.inf
    lo = 0.0
    for _ in range(200):  # bisection, then a Newton polish
        mid = 0.5 * (lo + hi)
        if lam(mid) < target:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-13 * max(1.0, hi):
            break
    t = 0.5 * (lo + hi)
    d = a + b * vp * np.exp(-vp * t)
    if d > 0:
        t = t - (lam(t) - target) / d
    return float(t)


def simulate_well_mixed(p: SparkParams, n_channels: int, volume: float, times: np.ndarray,
                        n_realizations: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Sample (U, n_open) at ``times`` for each realization; arrays shaped (n_realizations, len(times))."""
    rng = np.random.default_rng(seed)
    times = np.asarray(times, dtype=float)
    U_out = np.empty((n_realizations, len(times)))
    n_out = np.empty((n_realizations, len(times)), dtype=int)
    vp, src = p.Vp, p.J / volume
    for r in range(n_realizations):
        t, U, n_open = 0.0, p.U0, 0
        k = 0
        while k < len(times):
            n_closed = n_channels - n_open
            u_star = p.U0 + src * n_open / vp
            on = p.k_on / p.U0 if p.coupled else p.k_on
            a = n_closed * on * (u_star if p.coupled else 1.0) + n_open * p.k_off
            b = n_closed * on * (U - u_star) / vp if p.coupled else 0.0
            tau = _solve_tau(a, b, vp, rng.exponential())
            t_next = t + tau
            # record every requested time before the next event
            while k < len(times) and times[k] < t_next:
                dt = times[k] - t
                U_out[r, k] = u_star + (U - u_star) * np.exp(-vp * dt)
                n_out[r, k] = n_open
                k += 1
            if k >= len(times):
                break
            U = u_star + (U - u_star) * np.exp(-vp * tau)
            h_open = n_closed * on * (U if p.coupled else 1.0)
            h_close = n_open * p.k_off
            if rng.random() * (h_open + h_close) < h_open:
                n_open += 1
            else:
                n_open -= 1
            t = t_next
    return U_out, n_out
