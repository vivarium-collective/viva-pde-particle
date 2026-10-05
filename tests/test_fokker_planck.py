"""The single-channel Fokker–Planck references of Study C3 (Schaff et al. 2016, Tests 3–5)."""
from __future__ import annotations

import numpy as np
import pytest

from viva_pde_particle.benchmarks.fokker_planck import (
    TEST4,
    ChannelTest,
    binned_density,
    extrapolate,
    fast_steady_state,
    open_channel_reach,
    solve_fast,
    solve_master,
)


def test_fast_transient_reaches_the_exact_steady_state():
    r, p = solve_fast(24.0, 1.0, 1.0, 30.0, n=1200, dtau=0.01)
    assert np.sum(p) * 24.0 / 1200 == pytest.approx(1.0, abs=1e-9)
    exact = fast_steady_state(r, 24.0, 1.0, 1.0)
    assert np.sqrt(np.mean((p - exact) ** 2)) / exact.max() < 0.01


def test_steady_state_formula_balances_eq4():
    """At steady state the flux balance d/dρ[−ρ p₀] = R holds for the closed-form p₀, p₁."""
    a, al, be = 6.0, 2.0, 0.7
    r = np.linspace(0.5, 5.5, 2001)
    p0 = (a - r) ** al * r ** (al * be - 1) * np.exp(al * be * r)
    p1 = r * p0 / (a - r)
    R = al * (p1 - be * (r + 1) * p0)
    lhs = np.gradient(-r * p0, r)
    np.testing.assert_allclose(lhs[5:-5], R[5:-5], rtol=1e-4, atol=1e-4 * np.abs(R).max())  # R crosses 0
    p = fast_steady_state(r, a, al, be)
    np.testing.assert_allclose(p / p.max(), (p0 + p1) / (p0 + p1).max(), rtol=1e-9)


def test_master_equation_without_diffusion_converges_to_eq4_at_first_order():
    t = ChannelTest(alpha=10.0, beta=1.0, a=20.0 / 3.0, d=0.0, imax=1, dx=1.0)
    r, p = solve_fast(t.a, t.alpha, t.beta, 1.0, n=4000, dtau=1e-4)
    edges = np.linspace(0, 4.4, 21)
    ref = np.array([p[(r >= lo) & (r < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])])
    errs, drhos, dens = [], [], []
    for nm in (100, 200, 400):
        drho, m = solve_master(t, nm)
        assert m[0].sum() == pytest.approx(1.0, abs=1e-10)  # probability is conserved
        d = binned_density(drho, m[0], edges)
        errs.append(np.sqrt(np.mean((d - ref) ** 2)) / ref.max())
        drhos.append(drho)
        dens.append(d)
    assert errs[1] < 0.6 * errs[0] and errs[2] < 0.6 * errs[1]  # first order in Δρ
    ext = extrapolate(drhos, dens)
    assert np.sqrt(np.mean((ext - ref) ** 2)) / ref.max() < 0.25 * errs[2]


def test_finite_diffusion_ranges_and_means():
    reach = open_channel_reach(TEST4)
    np.testing.assert_allclose(reach, [1.9167, 0.1903], atol=1e-3)  # inside the paper's [2.0, 0.25]
    drho, m = solve_master(TEST4, 200)
    assert [x.sum() for x in m] == pytest.approx([1.0, 1.0], abs=1e-10)
    mean = [float((np.arange(len(x)) * drho * x).sum()) for x in m]
    assert 0 < mean[1] < mean[0] < reach[0]


def test_channel_model_equal_volumes_feed_node_zero_a_over_dx():
    """element_volumes gives node 0 a full Δx in the PDE and the adapter: an open channel adds a/Δx per τ."""
    pytest.importorskip("smoldyn")
    from process_bigraph import Composite

    from viva_pde_particle.benchmarks.fokker_planck import channel_model
    from viva_pde_particle.composites.hybrid import build_hybrid_document
    from viva_pde_particle.core import build_core

    t = ChannelTest(alpha=1e-6, beta=1e9, a=4.0, d=0.0, imax=2, dx=2.0)  # opens within ~1 ms, then stays open
    g, model, vol = channel_model(t)
    doc = build_hybrid_document(model, 1e-3, 1, seed=1, element_volumes=vol)
    assert (doc["particle_to_field"]["config"]["staircase"]["volumes"] == 2.0).all()
    assert (doc["pde"]["config"]["domain"]["volume_fraction"] == pytest.approx(np.array([2.0, 2.0]) / g.full_volume))
    sim = Composite({"state": doc}, core=build_core())
    sim.run(0.0005)
    sim.run(0.1)
    rho = np.asarray(sim.state["fields"]["rho"])
    # dρ0/dτ = a/Δx − ρ0 with ρ0(0) = 0 → ρ0(0.1) = 2(1 − e^{−0.1}); no diffusion (d = 0)
    assert rho[0] == pytest.approx(2.0 * (1 - np.exp(-0.1)), rel=0.03)  # (half volumes would double it)
    assert rho[1] == pytest.approx(0.0, abs=1e-12)
