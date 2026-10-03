"""Grid conventions (vcell-fvsolver CartesianMesh) and the finite volume PDE step."""
from __future__ import annotations

import numpy as np
import pytest
from process_bigraph import allocate_core

from viva_pde_particle.grid import CartesianGrid
from viva_pde_particle.processes.fv_reaction_diffusion import FVReactionDiffusion
from viva_pde_particle.units import MOLECULES_PER_UM3_PER_UM


def test_geometry_and_volumes_2d():
    g = CartesianGrid((0.0, 0.0), (10.0, 4.0), (11, 5))
    assert g.shape == (5, 11)
    assert g.spacing == (1.0, 1.0)
    frac = g.volume_fraction
    assert frac[0, 0] == 0.25 and frac[0, 5] == 0.5 and frac[2, 5] == 1.0  # corner, edge, interior
    # node-centred cells tile the domain exactly (unit-thickness slab in 2D)
    assert g.element_volumes.sum() == pytest.approx(10.0 * 4.0 * 1.0)


def test_volumes_3d_corner_is_one_eighth():
    g = CartesianGrid((0, 0, 0), (2.0, 2.0, 2.0), (3, 3, 3))
    assert g.volume_fraction[0, 0, 0] == 0.125
    assert g.element_volumes.sum() == pytest.approx(8.0)


def test_nearest_node_binning_and_units():
    g = CartesianGrid((0.0, 0.0), (10.0, 10.0), (11, 11))
    pts = np.array([[0.49, 0.0], [0.51, 0.0], [9.9, 10.0], [5.0, 5.0], [5.0, 5.0]])
    h = g.histogram(pts)
    assert h[0, 0] == 1 and h[0, 1] == 1 and h[10, 10] == 1 and h[5, 5] == 2
    assert h.sum() == len(pts)
    # 1 µM in an interior element of 1 µm³ is 602.2 molecules; round trip
    conc = g.counts_to_uM(g.uM_to_counts(np.full(g.shape, 2.5)))
    np.testing.assert_allclose(conc, 2.5)
    assert g.uM_to_counts(np.ones(g.shape))[5, 5] == pytest.approx(MOLECULES_PER_UM3_PER_UM)


def test_diffusion_matrix_is_symmetric_and_conservative():
    g = CartesianGrid((0.0, 0.0), (3.0, 2.0), (7, 5))
    K = g.diffusion_matrix()
    assert abs(K - K.T).max() < 1e-14
    np.testing.assert_allclose(np.asarray(K.sum(axis=1)).ravel(), 0.0, atol=1e-12)


_CORE = None


def _core():
    global _CORE
    if _CORE is None:
        _CORE = allocate_core()
    return _CORE


def _fv(grid, D, dt, terms=None, particle_species=()):
    return FVReactionDiffusion(
        core=_core(),
        config={
            "grid": grid.to_config(),
            "pde": {"species": {"B": {"diffusion": D}}, "particle_species": list(particle_species),
                    "terms": terms or []},
            "dt": dt,
        }
    )


def test_cosine_mode_decays_at_exact_backward_euler_rate():
    """Neumann node-centred FV: cos(pi*m*i/(N-1)) is an exact discrete eigenmode."""
    L, N, D, dt, m = 10.0, 21, 2.0, 0.05, 2
    g = CartesianGrid((0.0,), (L,), (N,))
    proc = _fv(g, D, dt)
    x = g.axis_nodes(0)
    u = np.cos(np.pi * m * x / L)
    dx = g.spacing[0]
    lam = (2 - 2 * np.cos(np.pi * m * dx / L)) / dx**2
    out = proc.update({"fields": {"B": u}, "particle_counts": {}}, 10 * dt)["fields"]["B"]
    np.testing.assert_allclose(out, u / (1 + D * dt * lam) ** 10, rtol=1e-10, atol=1e-12)


def test_mass_is_conserved_by_diffusion():
    g = CartesianGrid((0.0, 0.0), (5.0, 5.0), (11, 11))
    rng = np.random.default_rng(0)
    u = rng.random(g.shape)
    out = _fv(g, 1.0, 0.1).update({"fields": {"B": u}, "particle_counts": {}}, 1.0)["fields"]["B"]
    assert (out * g.element_volumes).sum() == pytest.approx((u * g.element_volumes).sum(), rel=1e-12)


def test_explicit_reaction_term_and_particle_coupling():
    """dB/dt = -k*B (explicit) and dB/dt = +k*[A_particle] with A binned from counts."""
    g = CartesianGrid((0.0, 0.0), (4.0, 4.0), (5, 5))
    k, dt = 0.5, 0.1
    decay = _fv(g, 0.0, dt, terms=[{"species": "B", "coeff": -1, "k": k, "reactants": {"B": 1}}])
    out = decay.update({"fields": {"B": np.ones(g.shape)}, "particle_counts": {}}, dt)["fields"]["B"]
    np.testing.assert_allclose(out, 1 - k * dt)  # forward Euler reaction

    produce = _fv(g, 0.0, dt, particle_species=["A"],
                  terms=[{"species": "B", "coeff": 1, "k": k, "reactants": {"A": 1}}])
    counts = np.zeros(g.shape)
    counts[2, 2] = 602.214076  # 1 µM in the interior element (volume 1 µm³)
    counts[0, 0] = 602.214076 / 4  # 1 µM in the corner element (volume 1/4)
    out = produce.update({"fields": {"B": np.zeros(g.shape)}, "particle_counts": {"A": counts}}, dt)["fields"]["B"]
    assert out[2, 2] == pytest.approx(k * dt) and out[0, 0] == pytest.approx(k * dt)
    assert out.sum() == pytest.approx(2 * k * dt)
