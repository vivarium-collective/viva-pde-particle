"""Stochastic ensembles as process-bigraph documents, for batch compute such as compose-api (SLURM).

A study's ensemble is thousands of short, independent trials. Running each trial as its own job would
mean thousands of submissions. :class:`EnsembleRunner` instead runs one *trial function* over a block of
seeds inside a single Process, across the CPUs the job was given. A study then becomes a few dozen
documents, one per block (:func:`ensemble_documents`).

- **Trial function:** an importable callable named ``"module:function"``, called as
  ``fn(seed, **params)`` and returning a list of floats. Example:
  :func:`viva_pde_particle.benchmarks.fokker_planck.trial_rho`.
- **Document:** a single ``ensemble`` Process writes ``seeds`` and ``values``, and a RAM emitter records
  them, so ``gather_emitter_results`` (what compose-api's runtime saves as ``results_*.pber``)
  returns them.
- **Addresses:** fully qualified ``local:viva_pde_particle.ensemble.EnsembleRunner``, the form registry
  checks accept. :func:`build_runner_core` registers our classes under those names.
"""
from __future__ import annotations

import importlib
import json
import multiprocessing as mp
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from process_bigraph import Process

ADDRESS = "local:viva_pde_particle.ensemble.EnsembleRunner"


def resolve(spec: str):
    """The callable named ``"package.module:function"``."""
    module, _, name = spec.partition(":")
    if not name:
        raise ValueError(f"trial must be 'module:function', got {spec!r}")
    return getattr(importlib.import_module(module), name)


def available_cpus() -> int:
    """CPUs this process may use: the affinity mask (what SLURM allocated), not the node's count."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # macOS
        return os.cpu_count() or 1


def example_trial(seed: int, scale: float = 1.0) -> list[float]:
    """A trivial trial function, for tests and smoke runs: ``[scale·seed, seed²]``."""
    return [scale * seed, float(seed) ** 2]


def _call(args):
    spec, seed, params = args
    return [float(v) for v in resolve(spec)(seed, **params)]


def run_trials(trial: str, params: dict, seeds, workers: int = 0) -> list[list[float]]:
    """``trial`` for each seed, in order, on ``workers`` processes (0: every available CPU)."""
    seeds = [int(s) for s in seeds]
    workers = workers or available_cpus()
    if workers <= 1 or len(seeds) <= 1:
        return [_call((trial, s, params)) for s in seeds]
    with ProcessPoolExecutor(max_workers=min(workers, len(seeds)), mp_context=mp.get_context("spawn")) as pool:
        return list(pool.map(_call, [(trial, s, params) for s in seeds], chunksize=max(1, len(seeds) // (4 * workers))))


class EnsembleRunner(Process):
    """Run a trial function over a block of seeds, once, at the first update.

    Config:
    - ``trial``: ``"module:function"``.
    - ``params``: the function's keyword arguments, as a JSON string (any JSON value types).
    - ``seeds``: ``[first, last]``, inclusive.
    - ``workers``: process count, 0 for every CPU available.
    """

    config_schema = {
        "trial": "string",
        "params": {"_type": "string", "_default": "{}"},
        "seeds": "list[integer]",
        "workers": {"_type": "integer", "_default": 0},
    }

    def initialize(self, config):
        first, last = config["seeds"]
        self.seed_list = list(range(int(first), int(last) + 1))
        self.done = False

    def inputs(self):
        return {}

    def outputs(self):
        return {"seeds": "overwrite[list[integer]]", "values": "overwrite[list[list[float]]]"}

    def update(self, state, interval):
        if self.done:
            return {}
        self.done = True
        values = run_trials(self.config["trial"], json.loads(self.config["params"]), self.seed_list,
                            self.config["workers"])
        return {"seeds": self.seed_list, "values": values}


def ensemble_document(trial: str, params: dict, first: int, last: int, workers: int = 0) -> dict:
    """A composite config (``{"state": …}``) running ``trial`` for seeds ``first``…``last`` and emitting the results."""
    return {"state": {
        "seeds": [],
        "values": [],
        "ensemble": {
            "_type": "process",
            "address": ADDRESS,
            "config": {"trial": trial, "params": json.dumps(params), "seeds": [int(first), int(last)],
                       "workers": int(workers)},
            "interval": 1.0,
            "inputs": {},
            "outputs": {"seeds": ["seeds"], "values": ["values"]},
        },
        "emitter": {
            "_type": "step",
            "address": "local:ram-emitter",
            "config": {"emit": {"seeds": "list[integer]", "values": "list[list[float]]"}},
            "inputs": {"seeds": ["seeds"], "values": ["values"]},
        },
    }}


def ensemble_documents(trial: str, params: dict, seeds: int, block: int, workers: int = 0) -> list[dict]:
    """Seeds 1…``seeds`` in blocks of ``block``, one document each."""
    return [ensemble_document(trial, params, s, min(s + block - 1, seeds), workers) for s in range(1, seeds + 1, block)]


def write_omex(document: dict, path, name: str = "experiment") -> Path:
    """The document as a COMBINE-style archive holding ``<name>.pbg`` (what compose-api accepts)."""
    path = Path(path)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{name}.pbg", json.dumps(document))
    return path


def build_runner_core():
    """Our core, with every workspace Process/Step also registered under its fully qualified name."""
    from process_bigraph.emitter import RAMEmitter

    from viva_pde_particle.core import _iter_own_process_classes, build_core

    core = build_core()
    for _, cls in _iter_own_process_classes():
        name = f"{cls.__module__}.{cls.__name__}"
        if name not in core.link_registry:
            core.register_link(name, cls)
    if "viva_pde_particle.ensemble.EnsembleRunner" not in core.link_registry:
        core.register_link("viva_pde_particle.ensemble.EnsembleRunner", EnsembleRunner)
    if "ram-emitter" not in core.link_registry:
        core.register_link("ram-emitter", RAMEmitter)
    return core


def collect(results_files) -> tuple[list[int], list[list[float]]]:
    """Seeds and values from one or more ``results_*.pber`` files (emitter histories), in seed order."""
    rows = {}
    for f in results_files:
        data = json.loads(Path(f).read_text())
        for history in data.values():
            for record in history if isinstance(history, list) else [history]:
                for s, v in zip(record.get("seeds") or [], record.get("values") or []):
                    rows[int(s)] = v
    seeds = sorted(rows)
    return seeds, [rows[s] for s in seeds]
