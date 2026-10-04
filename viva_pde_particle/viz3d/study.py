"""Helpers for study scripts that record runs into bundles and render 3D figures (Phase 8e).

- :func:`logged_run`: a workbench run-log entry (started/completed/failed) around the rendering, so
  every figure can name the run that produced it.
- :func:`write_meta`: the ``<stem>.meta.json`` sidecar the workbench shows with a figure (give each
  figure its own stem).
"""
from __future__ import annotations

import contextlib
import json
import time
import uuid
from pathlib import Path


@contextlib.contextmanager
def logged_run(workspace_root, study_slug: str, investigation_slug: str, spec_id: str, label: str,
               params: dict | None = None):
    from vivarium_workbench.lib.run_log import append_run_event

    run_id = uuid.uuid4().hex
    append_run_event(workspace_root, {
        "run_id": run_id, "event": "started", "spec_id": spec_id, "label": label, "started_at": time.time(),
        "status": "running", "emitter": "none", "origin": "canonical_run", "study_slug": study_slug,
        "investigation_slug": investigation_slug, "params": params or {}})
    try:
        yield run_id
    except BaseException:
        append_run_event(workspace_root, {"run_id": run_id, "event": "completed", "completed_at": time.time(),
                                          "status": "failed"})
        raise
    append_run_event(workspace_root, {"run_id": run_id, "event": "completed", "completed_at": time.time(),
                                      "status": "completed"})


def write_meta(figure, *, title: str, caption: str, simulations: str, interpretation: str, run_id: str) -> Path:
    path = Path(figure).with_suffix(".meta.json")
    path.write_text(json.dumps({"title": title, "caption": caption, "simulations": simulations,
                                "interpretation": interpretation, "source_run_id": run_id}, indent=2) + "\n")
    return path
