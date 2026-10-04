"""Phase 7f: workbench generators that assemble the componentized hybrid."""
from __future__ import annotations

import numpy as np
import pytest
from process_bigraph.composite_generator import _REGISTRY

from viva_pde_particle.composites import examples
from viva_pde_particle.reference.vcell_native import hybrid_support_available


@pytest.mark.parametrize("name", ["ball_netgen_hybrid", "ball_vcell_geometry_hybrid", "exchange_splitting"])
def test_registered(name):
    assert f"viva_pde_particle.composites.examples.{name}" in _REGISTRY


def _run(doc, t):
    from process_bigraph import Composite

    from viva_pde_particle.core import build_core

    sim = Composite({"state": doc}, core=build_core())
    sim.run(0.005)
    sim.run(t)
    return sim


def test_ball_netgen_hybrid_runs():
    pytest.importorskip("smoldyn")
    pytest.importorskip("vcell_fenics")
    doc = examples.ball_netgen_hybrid(case="conversion", radius=1.5, h=0.5)
    assert doc["pde"]["config"]["mesh"]["kind"] == "geometry"
    assert {"particle_to_field", "field_to_particles"} <= set(doc)
    sim = _run(doc, 0.1)
    assert np.asarray(sim.state["particle_counts"]["A"]).sum() < 20000


def test_exchange_splitting_mixes_q1_engine_and_strang():
    """A combination the original code could not express: the FEniCSx Q1 engine under Strang splitting."""
    pytest.importorskip("smoldyn")
    pytest.importorskip("dolfinx")
    from viva_pde_particle.units import MOLECULES_PER_UM3_PER_UM as NA

    doc = examples.exchange_splitting(scheme="strang", pde_engine="fenicsx", step_multiplier=2)
    cfg = doc["coupler"]["config"]
    assert cfg["pde"]["address"] == "local:FenicsxReactionDiffusion" and cfg["scheme"] == "strang"
    g = examples.two_way_exchange_model().grid
    b0 = float((np.asarray(doc["fields"]["B"]) * g.element_volumes).sum() * NA)
    sim = _run(doc, 1.0)
    a = float(np.asarray(sim.state["particle_counts"]["A"]).sum())
    b = float((np.asarray(sim.state["fields"]["B"]) * g.element_volumes).sum() * NA)
    assert a > 0
    assert (a + b) / b0 == pytest.approx(1.0, abs=0.02)


@pytest.mark.skipif(not hybrid_support_available(), reason="pyvcell without spatial-hybrid support")
def test_ball_vcell_geometry_hybrid_builds():
    doc = examples.ball_vcell_geometry_hybrid(case="exchange", radius=1.5, correction="adapters+volumes")
    assert "volume_fraction" in doc["pde"]["config"]["domain"]
    assert doc["particles"]["inputs"] == {"fields": ["lookup_fields"]}
