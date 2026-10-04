"""Example composite generators (workbench-runnable) and the derived-scalar evaluator hook."""
from __future__ import annotations

import json

import pytest
from process_bigraph.composite_generator import _REGISTRY

from viva_pde_particle import evaluators
from viva_pde_particle.composites import examples


@pytest.mark.parametrize("name", ["particle_diffusion", "field_modulated_decay", "calcium_sparks"])
def test_generators_registered_under_spec_ids(name):
    spec_id = f"viva_pde_particle.composites.examples.{name}"
    assert spec_id in _REGISTRY


def test_field_modulated_decay_document_shape():
    doc = examples.field_modulated_decay(n_particles=100, k=2.0)
    assert set(doc) >= {"fields", "particle_counts", "pde", "particles"}
    # default coupling "fvsolver": the Smoldyn engine sits inside a Stepper (tick dt, every k, phase k-1)
    assert doc["particles"]["address"] == "local:Stepper"
    engine = doc["particles"]["config"]["process"]
    assert engine["address"] == "local:SmoldynHybrid"
    assert engine["config"]["field_species"] == ["B"]
    assert "reaction decay A -> 0 2.0*B;" in engine["config"]["config_text"]
    assert doc["particle_counts"]["A"].sum() == 100


def test_generator_document_runs():
    pytest.importorskip("smoldyn")
    from process_bigraph import Composite

    from viva_pde_particle.core import build_core

    doc = examples.field_modulated_decay(n_particles=200, dt=0.01)
    sim = Composite({"state": doc}, core=build_core())
    sim.run(0.05)
    assert sim.state["particle_counts"]["A"].sum() <= 200


def test_evaluator_hook_reads_study_metrics(tmp_path, monkeypatch):
    results = tmp_path / "workspace" / "studies" / "demo" / "results"
    results.mkdir(parents=True)
    (results / "metrics.json").write_text(json.dumps({"z_metric": 1.5, "label": "text"}))
    assert evaluators.load_metrics(tmp_path) == {"z_metric": 1.5}

    registry = {}
    monkeypatch.setattr(evaluators, "load_metrics", lambda ws: {"z_metric": 1.5})
    evaluators.register_derived_scalars(registry)
    assert registry["z_metric"](None, {}, tmp_path) == 1.5
