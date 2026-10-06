"""Single-channel Fokker–Planck references for Schaff et al. 2016, Tests 3–5 (Study C3).

One channel at the origin switches between closed (ξ = 0) and open (ξ = 1). Dimensionless calcium ρ
(ρ = (U − U₀)/U₀, τ = t·V_p) obeys

    ∂τ ρ = d·∂²ₓρ + a·δ(x)·ξ(τ) − ρ,

with opening rate αβ(ρ(0) + 1) and closing rate α. The references below solve for the joint
density of (ρ, ξ) directly, without Monte Carlo.

- **Fast diffusion (Test 3, Eq 4):** ρ is well mixed, dρ/dτ = aξ − ρ, and p₀(ρ, τ), p₁(ρ, τ) obey
  ∂τ p₀ = −∂ρ[−ρ p₀] + R and ∂τ p₁ = −∂ρ[(a − ρ) p₁] − R, with R = α(p₁ − β(ρ + 1) p₀),
  p₀(ρ, 0) = δ(ρ), p₁(ρ, 0) = 0.
  - :func:`fast_steady_state` is the exact stationary density: p₀ ∝ (a − ρ)^α ρ^(αβ−1) e^(αβρ),
    p₁ = ρ p₀/(a − ρ).
  - :func:`solve_fast` integrates Eq 4 in time (first-order upwind finite volumes, backward Euler).
- **Finite diffusion (Tests 4–5, Eq 5):** the functional Fokker–Planck equation, discretized as in the
  paper's S2 Text (Eq T2.1).
  - ``imax`` cells of width Δx, the channel in cell 0: dρᵢ/dτ = d/Δx²·Σⱼ(ρⱼ − ρᵢ) + (a/Δx)·δᵢ₀·ξ − ρᵢ.
  - The "mass" mᵢ = ρᵢΔx is quantized in units ΔρΔx. Pᵏ({nᵢ}, τ) is the probability of
    {ρᵢ = nᵢΔρ} with the channel in state k; it obeys a master equation. Quanta enter cell 0 at
    a/(ΔρΔx) while the channel is open, decay at rate 1 each, and hop to each neighbouring cell at
    d/Δx² each.
  - :func:`solve_master` integrates it (explicit Euler, matrix-free). Its error is O(Δρ), and
    :func:`extrapolate` takes several Δρ to Δρ → 0, as the paper did for Test 5.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ChannelTest:
    """Parameters of one test (dimensionless), with the hybrid solver's settings from the paper."""

    alpha: float
    beta: float
    a: float
    d: float
    imax: int = 0          # 0: the fast-diffusion (well-mixed) test
    dx: float = 2.0
    tau: float = 1.0
    dtau: float = 1e-4
    trials: int = 12500


TEST3 = ChannelTest(alpha=1.0, beta=1.0, a=24.0, d=1e4, tau=30.0, dtau=0.002, trials=10000)
TEST4 = ChannelTest(alpha=10.0, beta=1.0, a=20.0 / 3.0, d=1.0, imax=2)
TEST5 = ChannelTest(alpha=20.0, beta=0.5, a=20.0, d=50.0, imax=3, dtau=2e-5)


# ---------------------------------------------------------------- fast diffusion (Eq 4)

def fast_steady_state(rho, a: float, alpha: float, beta: float) -> np.ndarray:
    """Exact stationary density p(ρ) = p₀ + p₁ of Eq 4 on (0, a), normalized.

    Zero probability flux at steady state gives p₁ = ρ p₀/(a − ρ), and then
    d(ρ p₀)/dρ = −α ρ p₀ (1/(a − ρ) − β(ρ + 1)/ρ), so p₀ ∝ (a − ρ)^α ρ^(αβ−1) e^(αβρ) and
    p = p₀ + p₁ ∝ a (a − ρ)^(α−1) ρ^(αβ−1) e^(αβρ).
    """
    rho = np.asarray(rho, dtype=float)
    grid = np.linspace(0.0, a, 200001)[1:-1]
    log = lambda r: (alpha - 1) * np.log(a - r) + (alpha * beta - 1) * np.log(r) + alpha * beta * r  # noqa: E731
    shift = log(grid).max()
    norm = np.trapezoid(np.exp(log(grid) - shift), grid)
    out = np.zeros_like(rho)
    inside = (rho > 0) & (rho < a)
    out[inside] = np.exp(log(rho[inside]) - shift) / norm
    return out


def solve_fast(a: float, alpha: float, beta: float, tau: float, n: int = 2400, dtau: float = 0.005):
    """Eq 4 to time ``tau``: cell centres ρ (n cells on [0, a]) and p(ρ, τ) = p₀ + p₁.

    First-order upwind fluxes (velocities −ρ and a − ρ vanish at ρ = 0 and ρ = a: zero flux), the
    switching term per cell, backward Euler. The initial δ(ρ) sits in the first cell.
    """
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu

    h = a / n
    centres = (np.arange(n) + 0.5) * h
    faces = np.arange(1, n) * h  # interior faces

    def advection(v_face):  # flux through each interior face, upwind; d/dt p = -(F_right - F_left)/h
        rows, cols, vals = [], [], []
        for f, v in enumerate(v_face):  # face between cell f and f+1
            src = f if v > 0 else f + 1
            for cell, sign in ((f, -1.0), (f + 1, 1.0)):
                rows.append(cell)
                cols.append(src)
                vals.append(sign * v / h)
        return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))

    A0 = advection(-faces)
    A1 = advection(a - faces)
    on = sp.diags(alpha * beta * (centres + 1.0))
    off = alpha * sp.identity(n)
    M = sp.bmat([[A0 - on, off], [on, A1 - off]], format="csc")
    lu = splu((sp.identity(2 * n, format="csc") - dtau * M).tocsc())
    p = np.zeros(2 * n)
    p[0] = 1.0 / h
    for _ in range(int(round(tau / dtau))):
        p = lu.solve(p)
    return centres, p[:n] + p[n:]


# ---------------------------------------------------------------- finite diffusion (Eq 5 / T2.1)

def coupling_matrix(imax: int, d: float, dx: float) -> np.ndarray:
    """dρ/dτ = (L − I) ρ + s ξ on ``imax`` equal cells: L is the nearest-neighbour diffusion."""
    L = np.zeros((imax, imax))
    for i in range(imax):
        for j in (i - 1, i + 1):
            if 0 <= j < imax:
                L[i, j] += d / dx**2
                L[i, i] -= d / dx**2
    return L


def open_channel_reach(test: ChannelTest) -> np.ndarray:
    """ρᵢ reached at τ when the channel is open the whole time: the range ρ_max(i) of each cell."""
    from scipy.linalg import expm

    A = coupling_matrix(test.imax, test.d, test.dx) - np.eye(test.imax)
    s = np.zeros(test.imax)
    s[0] = test.a / test.dx
    # ρ(τ) = ∫₀^τ e^{A(τ−u)} s du = A⁻¹(e^{Aτ} − I) s
    return np.linalg.solve(A, (expm(A * test.tau) - np.eye(test.imax)) @ s)


def solve_master(test: ChannelTest, nmax0: int, margin: float = 1.04, tau: float | None = None,
                 dtype=np.float64) -> tuple[float, list[np.ndarray]]:
    """Eq T2.1 to time τ with Δρ = ρ_max(0)/nmax0. Returns Δρ and each cell's marginal probabilities.

    ``marginals[i][n]`` is P(ρᵢ = nΔρ); the ranges are ρ_max(i) = margin × :func:`open_channel_reach`.
    States that would exceed a range are blocked (zero flux at ρ_max), as in the paper.
    """
    tau = test.tau if tau is None else tau
    rmax = margin * open_channel_reach(test)
    drho = rmax[0] / nmax0
    nmax = [int(np.ceil(r / drho)) + 1 for r in rmax]
    imax = test.imax
    shape = tuple(nmax)
    n = [np.arange(m, dtype=dtype).reshape([-1 if k == i else 1 for k in range(imax)]) for i, m in enumerate(nmax)]
    h = test.d / test.dx**2
    lam = test.a / (drho * test.dx)
    on = test.alpha * test.beta * (n[0] * drho + 1.0)
    neighbours = [(i, j) for i in range(imax) for j in (i - 1, i + 1) if 0 <= j < imax]
    not_full = [(n[i] < nmax[i] - 1).astype(dtype) for i in range(imax)]
    # total out-rate per state, for the explicit step (dt ≤ 1/max rate keeps it positive and stable)
    out0 = sum(n[i] for i in range(imax)) + on + sum(h * n[i] * not_full[j] for i, j in neighbours)
    out1 = sum(n[i] for i in range(imax)) + test.alpha + lam * not_full[0] + sum(h * n[i] * not_full[j] for i, j in neighbours)
    rate_max = float(max(np.max(np.broadcast_to(out0, shape)), np.max(np.broadcast_to(out1, shape))))
    steps = int(np.ceil(tau * rate_max / 0.95))
    dt = tau / steps
    P = np.zeros((2, *shape), dtype=dtype)
    P[(0,) + (0,) * imax] = 1.0

    def sl(i, lo, hi):  # slice along cell i's axis of a (2, *shape) array
        idx = [slice(None)] * (imax + 1)
        idx[i + 1] = slice(lo, hi)
        return tuple(idx)

    for _ in range(steps):
        dP = np.zeros_like(P)
        flip = test.alpha * P[1] - on * P[0]
        dP[0] += flip
        dP[1] -= flip
        # source: P1(n0 − 1) → P1(n0) at λ, blocked at the top of cell 0's range
        src = lam * P[1] * not_full[0]
        dP[1] -= src
        dP[1][sl(0, 1, None)[1:]] += src[sl(0, None, -1)[1:]]
        for i in range(imax):
            # decay: a quantum leaves cell i at rate nᵢ
            dec = n[i] * P
            dP -= dec
            dP[sl(i, None, -1)] += dec[sl(i, 1, None)]
        for i, j in neighbours:
            # hop i → j at h·nᵢ, blocked when cell j is at the top of its range
            hop = h * n[i] * not_full[j] * P
            dP -= hop
            # the hop moves a state (nᵢ, nⱼ) to (nᵢ − 1, nⱼ + 1)
            tgt = [slice(None)] * (imax + 1)
            srcs = [slice(None)] * (imax + 1)
            tgt[i + 1], srcs[i + 1] = slice(None, -1), slice(1, None)
            tgt[j + 1], srcs[j + 1] = slice(1, None), slice(None, -1)
            dP[tuple(tgt)] += hop[tuple(srcs)]
        P += dt * dP
    total = P.sum(axis=0)
    marginals = [total.sum(axis=tuple(k for k in range(imax) if k != i)) for i in range(imax)]
    return float(drho), marginals


def binned_density(drho: float, marginal: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """A marginal P(ρ = nΔρ) as the mean density in each histogram bin ``edges`` (comparable to a histogram).

    Quantum n stands for ρ ∈ [(n − ½)Δρ, (n + ½)Δρ); its probability is shared between bins by overlap.
    """
    lo = (np.arange(len(marginal)) - 0.5) * drho
    hi = lo + drho
    out = np.zeros(len(edges) - 1)
    for b in range(len(out)):
        overlap = np.clip(np.minimum(hi, edges[b + 1]) - np.maximum(lo, edges[b]), 0, None) / drho
        out[b] = (marginal * overlap).sum() / (edges[b + 1] - edges[b])
    return out


def extrapolate(drhos, values) -> np.ndarray:
    """Values at Δρ → 0 from a series at several Δρ: the constant term of a fitted line (two points) or
    quadratic (three or more), per entry, as in the paper's S2 Text."""
    x = np.asarray(drhos, dtype=float)
    y = np.asarray(values, dtype=float)
    deg = 1 if len(x) == 2 else 2
    coef = np.polyfit(x, y.reshape(len(x), -1), deg)
    return coef[-1].reshape(y.shape[1:])


# ---------------------------------------------------------------- the hybrid model

def channel_model(test: ChannelTest):
    """``(grid, model, element_volumes)`` for the hybrid co-simulation of a test.

    Species: the channel as a particle, closed ``C`` or open ``O`` (D = 0), and the field ``rho``
    (dimensionless ρ, D = d). Reactions: opening at αβ (``open_rest``) plus αβ·ρ (``open_rho``),
    closing at α, influx J = a per open channel, decay at rate 1.

    - Test 3: the 3D cube [0, 1]³ with Δx = 0.1 and the channel at its centre node (the paper's
      [−0.5, 0.5]³, shifted for Smoldyn's placement). |Ω| = 1, so ρ̄ obeys dρ̄/dτ = aξ − ρ̄.
    - Tests 4–5: ``imax`` nodes Δx apart in 1D, the channel at node 0. ``element_volumes`` gives
      every node a full Δx (the reference's equal cells; the node-centred grid would halve the ends),
      so node 0 gets a·ξ/Δx.
    """
    from viva_pde_particle.grid import CartesianGrid
    from viva_pde_particle.model import HybridModel, Reaction, Species
    from viva_pde_particle.units import MOLECULES_PER_UM3_PER_UM as NA

    if test.imax == 0:
        g = CartesianGrid((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (11, 11, 11))
        channel = (5, 5, 5)
        volumes = None
    else:
        g = CartesianGrid((0.0,), ((test.imax - 1) * test.dx,), (test.imax,))
        channel = (0,)
        volumes = np.full(g.shape, test.dx)
    closed = np.zeros(g.shape)
    closed[channel] = 1.0
    ab = test.alpha * test.beta
    model = HybridModel(g, [
        Species("C", 0.0, particle=True, initial=closed),
        Species("O", 0.0, particle=True, initial=np.zeros(g.shape)),
        Species("rho", test.d, initial=0.0),
    ], [
        Reaction("open_rest", {"C": 1}, {"O": 1}, k=ab),
        Reaction("open_rho", {"C": 1, "rho": 1}, {"O": 1, "rho": 1}, k=ab),
        Reaction("close", {"O": 1}, {"C": 1}, k=test.alpha),
        Reaction("influx", {"O": 1}, {"O": 1, "rho": 1}, k=test.a * NA),
        Reaction("decay", {"rho": 1}, {}, k=1.0),
    ])
    return g, model, volumes


def trial_rho(seed: int, test: str = "test3", dtau: float | None = None, tau: float | None = None,
              sample_times=None) -> list[float]:
    """One co-simulation trial of a test: ρ at the sample times (the channel node's cells in Tests 4–5, ρ̄ in
    Test 3), flattened time-major. A trial function for :class:`viva_pde_particle.ensemble.EnsembleRunner`.

    Defaults: the test's own Δτ and τ, sampled at τ only.
    """
    from viva_pde_particle.composites.hybrid import build_hybrid_document, run_document

    base = {"test3": TEST3, "test4": TEST4, "test5": TEST5}[test]
    t = ChannelTest(**{**base.__dict__, **({"dtau": dtau} if dtau else {}), **({"tau": tau} if tau else {})})
    times = [float(x) for x in (sample_times or [t.tau])]
    t = ChannelTest(**{**t.__dict__, "tau": max(times)})
    g, model, volumes = channel_model(t)
    every = float(np.gcd.reduce([int(round(x / t.dtau)) for x in times])) * t.dtau
    tr = run_document(build_hybrid_document(model, t.dtau, 1, seed=int(seed), element_volumes=volumes),
                      t.tau, t.dtau, every)
    out = []
    for target in times:
        rho = np.asarray(tr.fields["rho"][int(np.argmin(np.abs(np.asarray(tr.times) - target)))], dtype=float)
        if t.imax == 0:
            out.append(float((rho * g.element_volumes).sum() / g.element_volumes.sum()))
        else:
            out.extend(float(v) for v in rho.ravel())
    return out
