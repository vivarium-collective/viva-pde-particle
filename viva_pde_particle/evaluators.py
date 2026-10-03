"""Workbench evaluation hook: expose study metrics as derived scalars.

Each study's ``sims/run.py`` writes ``workspace/studies/<slug>/results/metrics.json``.
``viva_superpowers.study_evaluator`` calls :func:`register_derived_scalars` and grades
a ``behavior_tests`` entry with ``measure: {kind: derived_scalar, field: <name>}``
against the value recorded there. Metric names are prefixed by study (``a1_``,
``a2_``, ...) so they are unique across the workspace.
"""
from __future__ import annotations

import json
from pathlib import Path

STUDIES_DIR = Path("workspace") / "studies"


def _metrics_files(ws_root: Path):
    return sorted((Path(ws_root) / STUDIES_DIR).glob("*/results/metrics.json"))


def load_metrics(ws_root) -> dict[str, float]:
    merged: dict[str, float] = {}
    for path in _metrics_files(ws_root):
        for key, value in json.loads(path.read_text()).items():
            if isinstance(value, (int, float)):
                merged[key] = float(value)
    return merged


def register_derived_scalars(registry: dict) -> None:
    def make(field: str):
        def compute(reader, test, ws_root):
            metrics = load_metrics(ws_root)
            if field not in metrics:
                raise KeyError(f"metric {field!r} not found; run the study's sims/run.py")
            return metrics[field]
        return compute

    ws_root = Path(__file__).resolve().parents[1]
    for field in load_metrics(ws_root):
        registry[field] = make(field)
