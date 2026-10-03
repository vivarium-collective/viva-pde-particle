"""FEniCSx PDE engine: Q1 on the grid's nodes, interchangeable with the FV engine."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("dolfinx")
from process_bigraph import allocate_core  # noqa: E402

from viva_pde_particle.grid import CartesianGrid  # noqa: E402
from viva_pde_particle.processes.fenicsx_reaction_diffusion import FenicsxReactionDiffusion, q1_operators  # noqa: E402


def _proc(grid, D=1.0, dt=0.01, terms=None, mass="lumped", particle_species=()):
    return FenicsxReactionDiffusion(core=allocate_core(), config={
        "grid": grid.to_config(), "dt": dt, "mass": mass,
        "pde": {"species": {"B": {"diffusion": D}}, "particle_species": list(particle_species), "terms": terms or []},
    })


def test_lumped_mass_equals_fv_control_volumes():
    g = CartesianGrid((0, 0, 0), (10, 10, 1), (11, 11, 3))
    K, M, dof_to_node = q1_operators(g)
    ml = np.asarray(M.sum(axis=1)).ravel()
    np.testing.assert_allclose(ml, g.element_volumes.ravel()[dof_to_node], atol=1e-14)
    np.testing.assert_allclose(np.asarray(K.sum(axis=1)).ravel(), 0, atol=1e-12)  # zero-flux: constants in kernel


@pytest.mark.parametrize("mass", ["lumped", "consistent"])
def test_cosine_mode_matches_continuum(mass):
    L, N, D, t_end = 10.0, 21, 1.0, 1.0
    g = CartesianGrid((0.0, 0.0), (L, L), (N, N))
    x, y = g.node_coordinates()
    u0 = np.cos(np.pi * x / L) * np.cos(np.pi * y / L)
    out = _proc(g, D, 0.01, mass=mass).update({"fields": {"B": u0}, "particle_counts": {}}, t_end)["fields"]["B"]
    exact = u0 * np.exp(-2 * np.pi**2 * D * t_end / L**2)
    assert np.abs(out - exact).max() / np.abs(exact).max() < 0.01


def test_mass_conserved_and_reactions_match_fv():
    g = CartesianGrid((0, 0, 0), (4, 4, 1), (5, 5, 3))
    rng = np.random.default_rng(0)
    u = rng.random(g.shape)
    out = _proc(g, 1.0, 0.1).update({"fields": {"B": u}, "particle_counts": {}}, 1.0)["fields"]["B"]
    assert (out * g.element_volumes).sum() == pytest.approx((u * g.element_volumes).sum(), rel=1e-12)
    k, dt = 0.5, 0.1
    decay = _proc(g, 0.0, dt, terms=[{"species": "B", "coeff": -1, "k": k, "reactants": {"B": 1}}])
    out = decay.update({"fields": {"B": np.ones(g.shape)}, "particle_counts": {}}, dt)["fields"]["B"]
    np.testing.assert_allclose(out, 1 - k * dt)
