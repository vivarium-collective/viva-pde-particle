"""Ensembles as documents (EnsembleRunner) and the compose-api runner contract (run <omex> -o <dir> -n <t>)."""
from __future__ import annotations

import json

import pytest

from viva_pde_particle import compose_runner
from viva_pde_particle.ensemble import (
    collect,
    ensemble_document,
    ensemble_documents,
    run_trials,
    write_omex,
)


def test_run_trials_in_order_serial_and_parallel():
    serial = run_trials("viva_pde_particle.ensemble:example_trial", {"scale": 2.0}, [3, 1, 2], workers=1)
    assert serial == [[6.0, 9.0], [2.0, 1.0], [4.0, 4.0]]
    assert run_trials("viva_pde_particle.ensemble:example_trial", {"scale": 2.0}, [3, 1, 2], workers=2) == serial


def test_documents_cover_the_seeds_in_blocks():
    docs = ensemble_documents("viva_pde_particle.ensemble:example_trial", {}, seeds=10, block=4)
    blocks = [d["state"]["ensemble"]["config"]["seeds"] for d in docs]
    assert blocks == [[1, 4], [5, 8], [9, 10]]
    assert json.loads(json.dumps(docs[0])) == docs[0]  # plain JSON: the document travels as a .pbg


def test_runner_runs_an_omex_and_writes_emitter_results(tmp_path):
    omex = write_omex(ensemble_document("viva_pde_particle.ensemble:example_trial", {"scale": 0.5}, 4, 6, workers=1),
                      tmp_path / "experiment.omex")
    assert compose_runner.main(["run", str(omex), "-o", str(tmp_path / "output"), "-n", "1"]) == 0
    files = list((tmp_path / "output").glob("results_*.pber"))
    assert len(files) == 1
    seeds, values = collect(files)
    assert seeds == [4, 5, 6] and values == [[2.0, 16.0], [2.5, 25.0], [3.0, 36.0]]


def test_c3_trial_through_the_runner(tmp_path):
    """A real co-simulation trial function (Test 4, short) run as a document."""
    pytest.importorskip("smoldyn")
    doc = ensemble_document("viva_pde_particle.benchmarks.fokker_planck:trial_rho",
                            {"test": "test4", "tau": 0.05, "dtau": 1e-3}, 1, 2, workers=1)
    pbg = tmp_path / "experiment.pbg"
    pbg.write_text(json.dumps(doc))
    compose_runner.main(["run", str(pbg), "-o", str(tmp_path / "out"), "-n", "1"])
    seeds, values = collect((tmp_path / "out").glob("results_*.pber"))
    assert seeds == [1, 2] and all(len(v) == 2 and v[0] >= 0 for v in values)  # ρ at the two cells


def test_runner_accepts_the_document_without_the_run_subcommand(tmp_path):
    """Viva Core's sbatch calls the image as `<doc.pbg> -o <dir> -n <t>`; compose-api adds `run`."""
    pbg = tmp_path / "experiment.pbg"
    pbg.write_text(json.dumps(ensemble_document("viva_pde_particle.ensemble:example_trial", {}, 1, 2, workers=1)))
    assert compose_runner.main([str(pbg), "-o", str(tmp_path / "out"), "-n", "1"]) == 0
    assert collect((tmp_path / "out").glob("results_*.pber"))[0] == [1, 2]
