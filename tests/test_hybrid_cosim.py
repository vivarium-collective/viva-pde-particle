"""End-to-end PDE + Smoldyn co-simulation through process-bigraph.

Skipped unless the hybrid Smoldyn module is built (``pixi run build-smoldyn``).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

smoldyn = pytest.importorskip("smoldyn")
if not hasattr(smoldyn._smoldyn, "HybridGrid"):
    pytest.skip("Smoldyn built without the hybrid extensions", allow_module_level=True)

from viva_pde_particle.composites import run_hybrid  # noqa: E402
from viva_pde_particle.grid import CartesianGrid  # noqa: E402
from viva_pde_particle.model import HybridModel, Reaction, Species  # noqa: E402
from viva_pde_particle.units import MOLECULES_PER_UM3_PER_UM as NA  # noqa: E402


def _grid():
    return CartesianGrid((0.0, 0.0), (10.0, 10.0), (11, 11))


def test_field_modulated_decay_follows_local_field():
    """A_particle -> 0 at k·[B]: B static, 1 µM on x > 5.5 and 0 elsewhere. A does not diffuse."""
    g = _grid()
    x = g.node_coordinates()[0]
    b = np.where(x > 5.5, 1.0, 0.0)
    model = HybridModel(g, [Species("A", 0.0, particle=True, initial=4000), Species("B", 0.0, initial=b)],
                        [Reaction("decay", {"A": 1, "B": 1}, {"B": 1}, k=1.0)])
    traj = run_hybrid(model, t_end=1.0, dt=0.01, seed=11)
    t, fields, counts = traj.as_arrays()
    a0, a1 = counts["A"][0], counts["A"][-1]
    zero, one = b == 0.0, b == 1.0
    assert a1[zero].sum() == a0[zero].sum()  # no decay where [B] = 0
    expected = a0[one].sum() * math.exp(-1.0)
    sd = math.sqrt(a0[one].sum() * math.exp(-1.0) * (1 - math.exp(-1.0)))
    assert abs(a1[one].sum() - expected) < 4 * sd
    np.testing.assert_array_equal(fields["B"][-1], b)  # catalyst field unchanged


def test_particle_to_field_conversion_conserves_molecules():
    """A_particle -> B_field at rate k: Smoldyn removes A, the PDE adds k·[A]·dt to B.

    Each side applies its own half (no flux exchange). The total A + B (molecules) is
    conserved in expectation, up to the O(dt) lag of the explicit coupling.
    """
    g = _grid()
    k, n0 = 0.5, 5000
    model = HybridModel(g, [Species("A", 1.0, particle=True, initial=n0), Species("B", 1.0)],
                        [Reaction("convert", {"A": 1}, {"B": 1}, k=k)])
    traj = run_hybrid(model, t_end=2.0, dt=0.01, seed=5)
    t, fields, counts = traj.as_arrays()
    a_tot = counts["A"].sum(axis=(1, 2))
    b_tot = (fields["B"] * g.element_volumes * NA).sum(axis=(1, 2))
    assert abs(a_tot[-1] - n0 * math.exp(-k * 2.0)) < 4 * math.sqrt(n0 * 0.25)
    np.testing.assert_allclose(a_tot + b_tot, n0, rtol=0.02)


def test_fvsolver_coupling_holds_particles_for_k_pde_steps():
    """With k = 4, particle counts change only every 4th PDE step (vcell-fvsolver)."""
    g = CartesianGrid((0.0,), (10.0,), (11,))
    model = HybridModel(g, [Species("A", 1.0, particle=True, initial=2000), Species("B", 1.0, initial=1.0)],
                        [Reaction("decay", {"A": 1, "B": 1}, {"B": 1}, k=2.0)])
    traj = run_hybrid(model, t_end=0.4, dt=0.01, step_multiplier=4, record_every=0.01, seed=2)
    t, _, counts = traj.as_arrays()
    totals = counts["A"].sum(axis=1)
    changed = np.nonzero(np.diff(totals))[0] + 1  # record indices where A changed
    assert len(changed) > 0
    assert all(i % 4 == 0 for i in changed), changed


@pytest.mark.parametrize("coupling", ["fvsolver", "start-of-interval"])
def test_both_coupling_modes_agree_for_slow_dynamics(coupling):
    g = _grid()
    model = HybridModel(g, [Species("A", 1.0, particle=True, initial=3000), Species("B", 0.5, initial=0.5)],
                        [Reaction("decay", {"A": 1, "B": 1}, {"B": 1}, k=1.0)])
    traj = run_hybrid(model, t_end=1.0, dt=0.01, step_multiplier=2, coupling=coupling, seed=9)
    _, _, counts = traj.as_arrays()
    n = counts["A"][-1].sum()
    expected = 3000 * math.exp(-0.5)
    assert abs(n - expected) < 4 * math.sqrt(3000 * math.exp(-0.5) * (1 - math.exp(-0.5)))
