"""Calcium-spark benchmark of Schaff et al. (2016), PLoS Comput Biol 12:e1005236, Tests 1-2.

Eq. 1:  ∂U/∂t = D∇²U + J·Σᵢ ξᵢ δ(r − rᵢ) − V_p (U − U₀), no-flux boundaries,
with 24 stochastic two-state channels ξᵢ ∈ {0, 1}:
- **opening** at ``k_on`` (Test 1, separable) or ``k_on·U(rᵢ)/U₀`` (Test 2, coupled);
- **closing** at ``k_off``.

As a HybridModel: closed and open channels are immobile particle species ``C`` and ``O``,
placed one per channel node. Calcium ``U`` is a field.
- **Influx:** ``O -> O + U``, with ``k = J·602.214`` so that ``dU/dt = J·n_open/V_i``.
- **Pump:** ``U -> 0`` at ``V_p``, plus a constant source ``0 -> U`` at ``V_p·U₀``.

Geometry: [0, 10.1]×[0, 2.1]×[0, 0.5] µm³, Δx = Δy = 0.1, Δz = 0.5, i.e. 102×22×2 nodes.
The channels are arranged as 4 columns of 6 (Fig 1A). The exact coordinates are not
given in the text; the defaults below are PROVISIONAL (approved by the author for now):
columns at x = 2, 4, 6, 8 µm, rows at y = 0.3 ... 1.8 µm (step 0.3), on the z = 0 layer.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from viva_pde_particle.grid import CartesianGrid
from viva_pde_particle.model import HybridModel, Reaction, Species
from viva_pde_particle.units import MOLECULES_PER_UM3_PER_UM

CHANNEL_X = (2.0, 4.0, 6.0, 8.0)  # provisional, see module docstring
CHANNEL_Y = (0.3, 0.6, 0.9, 1.2, 1.5, 1.8)
CHANNEL_Z = 0.0


@dataclass(frozen=True)
class SparkParams:
    D: float = 1.0          # µm²/s
    J: float = 10.0         # µM·µm³/s per open channel
    U0: float = 0.1         # µM
    Vp: float = 1.0         # 1/s
    k_on: float = 1.0       # 1/s
    k_off: float = 5.0      # 1/s
    coupled: bool = False   # Test 2: opening at k_on·U/U0


TEST1 = SparkParams()
TEST2 = SparkParams(D=1000.0, k_on=0.1, coupled=True)


def spark_grid() -> CartesianGrid:
    return CartesianGrid((0.0, 0.0, 0.0), (10.1, 2.1, 0.5), (102, 22, 2))


def channel_nodes(grid: CartesianGrid) -> tuple[np.ndarray, ...]:
    """Array indices (z, y, x) of the 24 channel nodes."""
    pts = np.array([(x, y, CHANNEL_Z) for x in CHANNEL_X for y in CHANNEL_Y])
    return grid.node_index(pts)


def calcium_sparks_model(p: SparkParams = TEST1, grid: CartesianGrid | None = None) -> HybridModel:
    g = grid or spark_grid()
    closed = np.zeros(g.shape)
    np.add.at(closed, channel_nodes(g), 1.0)
    if p.coupled:
        opening = Reaction("open", {"C": 1, "U": 1}, {"O": 1, "U": 1}, k=p.k_on / p.U0)
    else:
        opening = Reaction("open", {"C": 1}, {"O": 1}, k=p.k_on)
    return HybridModel(
        g,
        [
            Species("C", 0.0, particle=True, initial=closed),
            Species("O", 0.0, particle=True, initial=np.zeros(g.shape)),
            Species("U", p.D, initial=p.U0),
        ],
        [
            opening,
            Reaction("close", {"O": 1}, {"C": 1}, k=p.k_off),
            Reaction("influx", {"O": 1}, {"O": 1, "U": 1}, k=p.J * MOLECULES_PER_UM3_PER_UM),
            Reaction("pump", {"U": 1}, {}, k=p.Vp),
            Reaction("leak", {}, {"U": 1}, k=p.Vp * p.U0),
        ],
    )


def open_probability(t: np.ndarray, p: SparkParams = TEST1) -> np.ndarray:
    """Exact E[ξᵢ](t) for the separable two-state channel started closed."""
    k = p.k_on + p.k_off
    return p.k_on / k * (1.0 - np.exp(-k * np.asarray(t)))


def separable_expectation(times: np.ndarray, p: SparkParams = TEST1, dt_ref: float = 1e-4,
                          grid: CartesianGrid | None = None) -> np.ndarray:
    """E[U](r, t) for Test 1 on the shared grid.

    The system is linear and the channels are independent of U, so E[U] solves Eq. 1
    with ξᵢ replaced by the exact open probability. It is integrated here with the same
    FV operator at a fine ``dt_ref``, which leaves only the hybrid's statistical and Δt
    errors in a comparison. Returns an array of shape (len(times), *grid.shape).
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    g = grid or spark_grid()
    s = g.volume_fraction.ravel()
    K = g.diffusion_matrix()
    # pump treated implicitly (linear), source explicit: exact enough at dt_ref
    A = sp.diags(s * (1 + p.Vp * dt_ref)) + p.D * dt_ref * K
    solve = spla.factorized(A.tocsc())
    src = np.zeros(g.shape)
    np.add.at(src, channel_nodes(g), 1.0)
    src = (p.J * src / g.element_volumes).ravel()  # µM/s per unit open probability

    u = np.full(g.num_elements, p.U0)
    out, t, idx = [], 0.0, 0
    times = np.asarray(times)
    n_total = int(round(times[-1] / dt_ref))
    if np.isclose(times[0], 0.0):
        out.append(u.reshape(g.shape).copy())
        idx = 1
    for n in range(1, n_total + 1):
        t_mid = (n - 0.5) * dt_ref
        rhs = s * (u + dt_ref * (src * open_probability(t_mid, p) + p.Vp * p.U0))
        u = solve(rhs)
        t = n * dt_ref
        while idx < len(times) and np.isclose(t, times[idx], atol=0.5 * dt_ref):
            out.append(u.reshape(g.shape).copy())
            idx += 1
    return np.stack(out)


def spark_error(mean_u: np.ndarray, expected_u: np.ndarray, U0: float) -> float:
    """ε: RMS over space and time of (trial-mean U − E[U]), relative to max(E[U] − U0)."""
    return float(np.sqrt(np.mean((mean_u - expected_u) ** 2)) / np.max(expected_u - U0))
