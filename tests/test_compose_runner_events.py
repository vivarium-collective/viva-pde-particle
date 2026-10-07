"""compose_runner on compose-api: the run is a ``task`` span under the job's traceparent, and the results file is
announced with ``artifact.written`` (compose-api docs/plan-observability.md step 4)."""
from __future__ import annotations

import hashlib
import json

import pytest

events = pytest.importorskip("process_bigraph.events")

from viva_pde_particle import compose_runner  # noqa: E402
from viva_pde_particle.ensemble import ensemble_document, write_omex  # noqa: E402

TRACE, JOB = "a" * 32, "b" * 16


@pytest.fixture
def engine_events(tmp_path, monkeypatch):
    sink = tmp_path / "events" / "engine.jsonl"
    monkeypatch.setenv("PBG_EVENT_SINKS", f"file:{sink}")
    monkeypatch.setenv("PBG_TRACEPARENT", f"00-{TRACE}-{JOB}-01")
    events.set_emitter(None)  # configured afresh from the environment on first use
    yield sink
    events.get_emitter().close()
    events.set_emitter(None)


def test_a_run_is_a_task_span_that_announces_its_results(tmp_path, engine_events):
    omex = write_omex(ensemble_document("viva_pde_particle.ensemble:example_trial", {"scale": 2.0}, 1, 3),
                      tmp_path / "experiment.omex")
    out = compose_runner.run(omex, tmp_path / "output", 1.0)

    records = [json.loads(line) for line in engine_events.read_text().splitlines()]
    assert all(r["trace_id"] == TRACE for r in records)
    starts = [r for r in records if r["event"] == "span.start" and r["payload"]["name"] == "task"]
    ends = [r for r in records if r["event"] == "span.end" and r["payload"]["name"] == "task"]
    assert len(starts) == 1 and starts[0]["parent_span_id"] == JOB  # the job span compose-api's script opened
    assert ends[0]["payload"]["status"] == "ok"

    (artifact,) = [r for r in records if r["event"] == "artifact.written"]
    assert artifact["span_id"] == starts[0]["span_id"]
    payload = artifact["payload"]
    assert payload["uri"] == str(out.resolve()) and payload["kind"] == "results"
    assert payload["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest() and payload["bytes"] == out.stat().st_size
    assert payload["attributes"] == {"format": "pber", "seeds": 3, "first_seed": 1, "last_seed": 3}
    assert payload["name"] == "ensemble results (3 seeds)"


def test_without_compose_api_nothing_is_written(tmp_path, monkeypatch):
    monkeypatch.delenv("PBG_EVENT_SINKS", raising=False)
    events.set_emitter(None)
    omex = write_omex(ensemble_document("viva_pde_particle.ensemble:example_trial", {"scale": 2.0}, 1, 2),
                      tmp_path / "experiment.omex")
    compose_runner.run(omex, tmp_path / "output", 1.0)
    assert not list(tmp_path.rglob("*.jsonl"))
    events.set_emitter(None)
