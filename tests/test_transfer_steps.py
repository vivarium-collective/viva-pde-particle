"""Phase 7a: particle → PDE adapters are Steps; the PDE engines take a generic external_conc."""
from __future__ import annotations

import numpy as np
import pytest
from process_bigraph import Composite, allocate_core

from viva_pde_particle.composites.examples import two_way_exchange_model
from viva_pde_particle.grid import CartesianGrid
from viva_pde_particle.processes import FVReactionDiffusion
from viva_pde_particle.steps import GridCountsToConcentration


def test_grid_step_converts_counts_like_the_grid():
    g = CartesianGrid((0, 0), (4, 4), (5, 5))
    step = GridCountsToConcentration(config={"grid": g.to_config(), "species": ["A", "C"]}, core=allocate_core())
    counts = np.arange(25, dtype=float).reshape(g.shape)
    out = step.update({"particle_counts": {"A": counts}})["particle_conc"]
    np.testing.assert_array_equal(out["A"], g.counts_to_uM(counts))
    np.testing.assert_array_equal(out["C"], np.zeros(g.shape))  # species with no counts yet


def test_pde_engine_has_no_particle_ports():
    g = CartesianGrid((0, 0), (4, 4), (5, 5))
    pde = {"species": {"B": {"diffusion": 1.0}},
           "terms": [{"species": "B", "coeff": 1, "k": 1.0, "reactants": {"A": 1}}]}
    proc = FVReactionDiffusion(config={"grid": g.to_config(), "pde": pde, "dt": 0.1}, core=allocate_core())
    assert set(proc.inputs()) == {"fields", "external_conc"}
    assert proc.external_species == ["A"]  # inferred: read in a term, not evolved here


def test_composite_wires_the_step_and_keeps_it_current():
    pytest.importorskip("smoldyn")
    from viva_pde_particle.composites.hybrid import build_hybrid_document
    from viva_pde_particle.core import build_core

    model = two_way_exchange_model()
    doc = build_hybrid_document(model, 0.01, 2, "fvsolver", seed=5)
    assert doc["particle_to_field"]["_type"] == "step"
    assert doc["pde"]["inputs"] == {"fields": ["fields"], "external_conc": ["particle_conc"]}
    sim = Composite({"state": doc}, core=build_core())
    sim.run(0.005)
    sim.run(0.3)
    counts = np.asarray(sim.state["particle_counts"]["A"])
    assert counts.sum() > 0  # creation happened, so the Step had something to convert
    np.testing.assert_array_equal(np.asarray(sim.state["particle_conc"]["A"]), model.grid.counts_to_uM(counts))
