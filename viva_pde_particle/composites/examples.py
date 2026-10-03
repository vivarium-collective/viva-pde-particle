"""Example hybrid models used by the cosim-vs-embedded-hybrid studies, as composite generators.

Each ``*_model`` function returns a :class:`HybridModel`. The matching
``@composite_generator`` wraps it in a process-bigraph document that the workbench can
run (composite id ``viva_pde_particle.composites.examples.<name>``).
"""
from __future__ import annotations

import numpy as np
from process_bigraph.composite_generator import composite_generator

from viva_pde_particle.composites.hybrid import build_hybrid_document
from viva_pde_particle.grid import CartesianGrid
from viva_pde_particle.model import HybridModel, Reaction, Species

_COUPLING_PARAMS = {
    "dt": {"type": "float", "default": 0.01, "description": "PDE time step (s)"},
    "step_multiplier": {"type": "integer", "default": 1,
                        "description": "Smoldyn steps once per k PDE steps (SMOLDYN_STEP_MULTIPLIER)"},
    "coupling": {"type": "string", "default": "fvsolver",
                 "description": "'fvsolver' (exact vcell-fvsolver lag) or 'start-of-interval'"},
    "seed": {"type": "integer", "default": 1, "description": "Random seed (Smoldyn and initial placement)"},
}


def square_grid(length: float = 10.0, num: int = 11, thickness: float | None = None) -> CartesianGrid:
    """Square 2D grid, or a quasi-2D 3D slab (3 z-nodes, the minimum VCell allows) of the given thickness.

    The VCell native hybrid reference needs 3D, so comparisons use the slab.
    """
    if thickness is None:
        return CartesianGrid((0.0, 0.0), (length, length), (num, num))
    return CartesianGrid((0.0, 0.0, 0.0), (length, length, thickness), (num, num, 3))


# ---------------------------------------------------------------- particle diffusion

def particle_diffusion_model(n_particles: int = 20000, diffusion: float = 1.0,
                             length: float = 40.0, num: int = 41, start: str = "center") -> HybridModel:
    """Particles only (no fields): A diffuses freely. ``start`` is ``center`` or ``uniform``."""
    g = square_grid(length, num)
    if start == "center":
        init = np.zeros(g.shape)
        init[num // 2, num // 2] = n_particles
    else:
        init = n_particles
    return HybridModel(g, [Species("A", diffusion, particle=True, initial=init)])


@composite_generator(
    name="particle_diffusion",
    description="Free Brownian diffusion of Smoldyn particles binned on the shared grid (no fields).",
    parameters={
        "n_particles": {"type": "integer", "default": 20000, "description": "Number of A molecules"},
        "diffusion": {"type": "float", "default": 1.0, "description": "D_A (µm²/s)"},
        "start": {"type": "string", "default": "center", "description": "'center' or 'uniform'"},
        **_COUPLING_PARAMS,
    },
)
def particle_diffusion(core=None, *, n_particles=20000, diffusion=1.0, start="center",
                       dt=0.01, step_multiplier=1, coupling="fvsolver", seed=1):
    model = particle_diffusion_model(n_particles, diffusion, start=start)
    return build_hybrid_document(model, dt, step_multiplier, coupling, seed)


# ---------------------------------------------------------------- field-modulated decay

def linear_field(grid: CartesianGrid, low: float, high: float) -> np.ndarray:
    """A field rising linearly in x from ``low`` to ``high`` (µM)."""
    x = grid.node_coordinates()[0]
    return low + (high - low) * (x - grid.origin[0]) / grid.size[0]


def gaussian_field(grid: CartesianGrid, peak: float, width: float) -> np.ndarray:
    x, y = grid.node_coordinates()[:2]
    cx = grid.origin[0] + 0.5 * grid.size[0]
    cy = grid.origin[1] + 0.5 * grid.size[1]
    return peak * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * width**2))


def field_modulated_decay_model(n_particles: int = 4000, k: float = 1.0, d_a: float = 0.0,
                                d_b: float = 0.0, b_profile: str = "linear",
                                length: float = 10.0, num: int = 11,
                                thickness: float | None = None) -> HybridModel:
    """A_particle -> 0 at rate k·[B](x). B is a catalyst field (unchanged by the reaction).

    ``b_profile``: ``linear`` (0 -> 1 µM across x) or ``gaussian`` (1 µM peak, width L/5).
    """
    g = square_grid(length, num, thickness)
    if b_profile == "linear":
        b, expr = linear_field(g, 0.0, 1.0), f"(x / {length!r})"
    else:
        w, c = length / 5, length / 2
        b, expr = gaussian_field(g, 1.0, w), f"exp(-((x - {c!r})^2 + (y - {c!r})^2) / {2 * w * w!r})"
    return HybridModel(
        g,
        [Species("A", d_a, particle=True, initial=n_particles),
         Species("B", d_b, initial=b, initial_expression=expr)],
        [Reaction("decay", {"A": 1, "B": 1}, {"B": 1}, k=k)],
    )


@composite_generator(
    name="field_modulated_decay",
    description="A_particle -> 0 at rate k·[B](x): Smoldyn reads the PDE field B at each molecule.",
    parameters={
        "n_particles": {"type": "integer", "default": 4000, "description": "Initial A molecules"},
        "k": {"type": "float", "default": 1.0, "description": "Rate constant (1/(µM·s))"},
        "d_a": {"type": "float", "default": 0.0, "description": "D_A (µm²/s)"},
        "d_b": {"type": "float", "default": 0.0, "description": "D_B (µm²/s)"},
        "b_profile": {"type": "string", "default": "linear", "description": "'linear' or 'gaussian'"},
        **_COUPLING_PARAMS,
    },
)
def field_modulated_decay(core=None, *, n_particles=4000, k=1.0, d_a=0.0, d_b=0.0, b_profile="linear",
                          dt=0.01, step_multiplier=1, coupling="fvsolver", seed=1):
    model = field_modulated_decay_model(n_particles, k, d_a, d_b, b_profile)
    return build_hybrid_document(model, dt, step_multiplier, coupling, seed)


# ---------------------------------------------------------------- calcium sparks (paper benchmark)

@composite_generator(
    name="calcium_sparks",
    description="Schaff et al. 2016 calcium sparks: 24 stochastic channels (particles) + diffusing Ca2+ (PDE).",
    parameters={
        "test": {"type": "integer", "default": 1,
                 "description": "1 = separable (Test 1); 2 = coupled k_on·U/U0, fast diffusion (Test 2)"},
        **_COUPLING_PARAMS,
    },
)
def calcium_sparks(core=None, *, test=1, dt=0.01, step_multiplier=1, coupling="fvsolver", seed=1):
    from viva_pde_particle.benchmarks.calcium_sparks import TEST1, TEST2, calcium_sparks_model

    params = {1: TEST1, 2: TEST2}[int(test)]
    return build_hybrid_document(calcium_sparks_model(params), dt, step_multiplier, coupling, seed)


# ---------------------------------------------------------------- two-way exchange

def two_way_exchange_model(k1: float = 1.0, k2: float = 0.5, b0: float = 0.2, d_a: float = 1.0,
                           d_b: float = 1.0, length: float = 10.0, num: int = 11,
                           thickness: float | None = 1.0) -> HybridModel:
    """A_particle -> B_field (k1) and B_field -> A_particle (k2·[B]), conserving A + B.

    B starts as a uniform field ``b0`` (µM) and A is absent. The particle side creates A
    by 0th-order, field-dependent creation (rate k2·[B] per volume), and the PDE side
    gains B from the binned A. At steady state, total A / total B = k2 / k1.
    """
    g = square_grid(length, num, thickness)
    return HybridModel(
        g,
        [Species("A", d_a, particle=True, initial=0), Species("B", d_b, initial=b0)],
        [Reaction("a_to_b", {"A": 1}, {"B": 1}, k=k1), Reaction("b_to_a", {"B": 1}, {"A": 1}, k=k2)],
    )


@composite_generator(
    name="two_way_exchange",
    description="A_particle <-> B_field exchange: particle decay feeds the field; the field creates particles.",
    parameters={
        "k1": {"type": "float", "default": 1.0, "description": "A -> B rate (1/s)"},
        "k2": {"type": "float", "default": 0.5, "description": "B -> A rate (1/s)"},
        "b0": {"type": "float", "default": 0.2, "description": "Initial B (µM)"},
        **_COUPLING_PARAMS,
    },
)
def two_way_exchange(core=None, *, k1=1.0, k2=0.5, b0=0.2, dt=0.01, step_multiplier=1,
                     coupling="fvsolver", seed=1):
    return build_hybrid_document(two_way_exchange_model(k1, k2, b0), dt, step_multiplier, coupling, seed)


# ---------------------------------------------------------------- bimolecular hybrid

def bimolecular_model(n_particles: int = 20000, k: float = 1.0, b0: float = 0.5, d_a: float = 1.0,
                      d_b: float = 1.0, d_c: float = 1.0, length: float = 10.0, num: int = 11,
                      thickness: float | None = 1.0) -> HybridModel:
    """A_particle + B_field -> C_field at k (1/(µM·s)). B is consumed, so the coupling is two-way."""
    g = square_grid(length, num, thickness)
    return HybridModel(
        g,
        [Species("A", d_a, particle=True, initial=n_particles), Species("B", d_b, initial=b0),
         Species("C", d_c, initial=0.0)],
        [Reaction("bind", {"A": 1, "B": 1}, {"C": 1}, k=k)],
    )


@composite_generator(
    name="bimolecular_hybrid",
    description="A_particle + B_field -> C_field: particles consume a field (two-way, nonlinear coupling).",
    parameters={
        "n_particles": {"type": "integer", "default": 20000, "description": "Initial A molecules"},
        "k": {"type": "float", "default": 1.0, "description": "Rate constant (1/(µM·s))"},
        "b0": {"type": "float", "default": 0.5, "description": "Initial B (µM)"},
        **_COUPLING_PARAMS,
    },
)
def bimolecular_hybrid(core=None, *, n_particles=20000, k=1.0, b0=0.5, dt=0.01, step_multiplier=1,
                       coupling="fvsolver", seed=1):
    return build_hybrid_document(bimolecular_model(n_particles, k, b0), dt, step_multiplier, coupling, seed)
