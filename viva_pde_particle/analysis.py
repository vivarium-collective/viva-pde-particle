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


def run_ensemble(model: HybridModel, seeds, t_end: float, dt: float, **kwargs):
    """Run the co-simulation once per seed.

    Returns ``(times, fields, counts)``, where fields/counts map species to arrays of
    shape (n_seeds, n_times, *grid.shape).
    """
    from viva_pde_particle.core import build_core

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


def write_metrics(path: Path, metrics: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in metrics.items()}
    path.write_text(json.dumps(clean, indent=2, sort_keys=True) + "\n")


__all__ = [
    "continuum_model",
    "histogram_variance",
    "run_continuum",
    "run_ensemble",
    "write_metrics",
]
