"""Phase 7d: the SplittingCoordinator sequences any PDE engine, particle engine and adapters."""
from __future__ import annotations

import numpy as np
import pytest

from viva_pde_particle.processes.splitting import schedule


def test_schedules():
    assert schedule("jacobi", 4, 0.01) == [("pde", 0.03), ("particles", 0.04, "defer"), ("pde", 0.01), ("commit",)]
    assert schedule("jacobi", 1, 0.01) == [("particles", 0.01, "defer"), ("pde", 0.01), ("commit",)]
    assert schedule("strang", 4, 0.01) == [("pde", 0.02), ("particles", 0.04), ("pde", 0.02)]
    assert schedule("gs_pde_first", 2, 0.5) == [("pde", 1.0), ("particles", 1.0)]
    with pytest.raises(ValueError, match="even"):
        schedule("strang", 3, 0.01)
    with pytest.raises(ValueError, match="scheme"):
        schedule("leapfrog", 2, 0.01)


def test_coordinator_document_is_engine_agnostic():
    pytest.importorskip("smoldyn")
    from viva_pde_particle.composites.examples import two_way_exchange_model
    from viva_pde_particle.composites.hybrid import build_coupler_document, build_hybrid_document, to_splitting_document

    doc = build_coupler_document(two_way_exchange_model(), 0.01, 2, "strang", seed=1)
    cfg = doc["coupler"]["config"]
    assert doc["coupler"]["address"] == "local:SplittingCoordinator"
    assert cfg["pde"]["address"] == "local:FVReactionDiffusion"
    assert cfg["particles"]["address"] == "local:SmoldynHybrid"
    assert list(cfg["adapters"]) == ["particle_to_field"]
    assert "pde" not in doc and "particle_to_field" not in doc  # all children live in the coordinator
    with pytest.raises(ValueError, match="start-of-interval"):
        to_splitting_document(build_hybrid_document(two_way_exchange_model(), 0.01, 2, "fvsolver"), "strang", 0.01, 2)


def test_strang_on_the_unstructured_mesh_engine():
    """New with 7d: any engine pair; here P1 mesh + Smoldyn with both mesh adapters, Strang order."""
    pytest.importorskip("smoldyn")
    pytest.importorskip("dolfinx")
    from process_bigraph import Composite

    from viva_pde_particle.composites.hybrid import build_mesh_hybrid_document, to_splitting_document
    from viva_pde_particle.core import build_core
    from viva_pde_particle.grid import CartesianGrid
    from viva_pde_particle.model import HybridModel, Reaction, Species
    from viva_pde_particle.units import MOLECULES_PER_UM3_PER_UM as NA

    g = CartesianGrid((0, 0, 0), (4, 4, 4), (17, 17, 17))
    m = HybridModel(g, [Species("A", 1.0, particle=True, initial=2000), Species("B", 1.0, initial=0.0)],
                    [Reaction("convert", {"A": 1}, {"B": 1}, k=1.0)])
    two = build_mesh_hybrid_document(m, {"center": (2.0, 2.0, 2.0), "radius": 2.0, "h": 0.6}, 0.01, step_multiplier=4,
                                     coupling="start-of-interval", seed=4)
    doc = to_splitting_document(two, "strang", 0.01, 4)
    assert set(doc["coupler"]["config"]["adapters"]) == {"particle_to_field", "field_to_particles"}
    sim = Composite({"state": doc}, core=build_core())
    sim.run(0.005)
    sim.run(0.4)
    a = float(np.asarray(sim.state["particle_counts"]["A"]).sum())
    from viva_pde_particle.mesh import mesh_space

    ml = mesh_space(doc["coupler"]["config"]["pde"]["config"]["mesh"]).ml
    b = float((np.asarray(sim.state["field_dofs"]["B"]) * ml).sum() * NA)
    assert a < 2000 and b > 0
    assert (a + b) / 2000 == pytest.approx(1.0, abs=0.02)  # conserved up to the splitting error
