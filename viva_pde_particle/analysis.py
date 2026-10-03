"""Ensembles, continuum references and metrics for the hybrid studies."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from viva_pde_particle.composites.hybrid import run_hybrid
from viva_pde_particle.model import HybridModel, initial_fields, partition
from viva_pde_particle.processes.fv_reaction_diffusion import FVReactionDiffusion
from viva_pde_particle.units import MOLECULES_PER_UM3_PER_UM


def _run_one(args):
    model, seed, t_end, dt, kwargs = args
    from viva_pde_particle.core import build_core

    return run_hybrid(model, t_end, dt, seed=int(seed), core=build_core(), **kwargs).as_arrays()


def run_ensemble(model: HybridModel, seeds, t_end: float, dt: float, workers: int = 1, **kwargs):
    """Run the co-simulation once per seed, serially or in ``workers`` spawned processes.

    Returns ``(times, fields, counts)``, where fields/counts map species to arrays of
    shape (n_seeds, n_times, *grid.shape).
    """
    from viva_pde_particle.core import build_core

    if workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
            runs = list(pool.map(_run_one, [(model, s, t_end, dt, kwargs) for s in seeds]))
    else:
        core = build_core()
        runs = [run_hybrid(model, t_end, dt, seed=int(s), core=core, **kwargs).as_arrays() for s in seeds]
    times = runs[0][0]
    fields = {s: np.stack([r[1][s] for r in runs]) for s in runs[0][1]}
    counts = {s: np.stack([r[2][s] for r in runs]) for s in runs[0][2]}
    return times, fields, counts


def continuum_model(model: HybridModel, initial_counts: dict[str, np.ndarray] | None = None) -> HybridModel:
    """The same model with every particle species made continuous (the deterministic limit).

    Particle initial conditions become concentrations, either from per-node counts or
    from a uniform total.
    """
    g = model.grid
    species = []
    for s in model.species:
        if not s.particle:
            species.append(s)
            continue
        if initial_counts is not None and s.name in initial_counts:
            conc = g.counts_to_uM(initial_counts[s.name])
        else:
            init = np.asarray(s.initial, dtype=float)
            if init.ndim == 0:
                conc = np.full(g.shape, float(init) / g.element_volumes.sum() / MOLECULES_PER_UM3_PER_UM)
            else:
                conc = g.counts_to_uM(init)
        species.append(replace(s, particle=False, initial=conc))
    return replace(model, species=species)


def run_continuum(model: HybridModel, t_end: float, dt: float, record_every: float) -> tuple[np.ndarray, dict]:
    """Integrate an all-continuous model with the FV process alone (no Smoldyn)."""
    from process_bigraph import allocate_core

    if model.particle_species:
        raise ValueError("run_continuum needs an all-continuous model; see continuum_model()")
    proc = FVReactionDiffusion(
        config={"grid": model.grid.to_config(), "pde": partition(model).pde, "dt": dt},
        core=allocate_core(),
    )
    fields = initial_fields(model)
    n = int(round(t_end / record_every))
    times, out = [0.0], {s: [v.copy()] for s, v in fields.items()}
    for i in range(1, n + 1):
        fields = proc.update({"fields": fields, "particle_counts": {}}, record_every)["fields"]
        times.append(i * record_every)
        for s, v in fields.items():
            out[s].append(np.array(v))
    return np.array(times), {s: np.stack(v) for s, v in out.items()}


def histogram_variance(grid, counts: np.ndarray, axis: int) -> float:
    """Variance of positions along a spatial axis, computed from per-node counts."""
    x = grid.node_coordinates()[axis]
    n = counts.sum()
    mean = (counts * x).sum() / n
    return float((counts * (x - mean) ** 2).sum() / n)


def two_sample_z(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise z of the difference of two ensemble means (arrays shaped (n_runs, ...))."""
    se = np.sqrt(a.var(axis=0, ddof=1) / len(a) + b.var(axis=0, ddof=1) / len(b))
    return np.where(se > 0, (a.mean(axis=0) - b.mean(axis=0)) / np.where(se > 0, se, 1), 0.0)


def paired_z(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise z of the mean paired difference a − b (same seeds: common random numbers)."""
    d = np.asarray(a) - np.asarray(b)
    se = d.std(axis=0, ddof=1) / np.sqrt(len(d))
    return np.where(se > 0, d.mean(axis=0) / np.where(se > 0, se, 1), 0.0)


def molecules(grid, conc_uM: np.ndarray) -> np.ndarray:
    """Total molecules of a µM field; sums the trailing grid axes (any leading axes kept)."""
    vol = grid.element_volumes * MOLECULES_PER_UM3_PER_UM
    return (np.asarray(conc_uM) * vol).sum(axis=tuple(range(-grid.dim, 0)))


def write_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in metrics.items()}
    path.write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n")


__all__ = [
    "molecules",
    "paired_z",
    "two_sample_z",
    "continuum_model",
    "histogram_variance",
    "run_continuum",
    "run_ensemble",
    "write_metrics",
]
