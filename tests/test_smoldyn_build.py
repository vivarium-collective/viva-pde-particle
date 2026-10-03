"""The Smoldyn module built by ``pixi run build-smoldyn`` imports and simulates correctly.

Skipped where Smoldyn has not been built (e.g. the uv-based workspace-ci workflow).
"""
from __future__ import annotations

import math

import pytest

smoldyn = pytest.importorskip("smoldyn")
from smoldyn._smoldyn import MolecState  # noqa: E402


def test_first_order_decay_matches_analytic():
    n0, k, t_end = 10000, 0.5, 2.0
    sim = smoldyn.Simulation(low=[0, 0, 0], high=[10, 10, 10], types="r", log_level=5)
    sim.setRandomSeed(1)
    a = sim.addSpecies("A", difc=1.0)
    a.addToSolution(n0)
    sim.addReaction("decay", subs=[a], prds=[], rate=k)
    sim.setGraphics("none")
    sim.setSimTimes(0.0, t_end, 0.01)
    sim.updateSim()
    sim.runUntil(t_end, 0.01, display=False)

    n = sim.getMoleculeCount("A", MolecState.all)
    expected = n0 * math.exp(-k * t_end)
    # Binomial sd ~ 48 molecules; allow 5% for stochastic + time-discretization error.
    assert abs(n - expected) / expected < 0.05, (n, expected)
