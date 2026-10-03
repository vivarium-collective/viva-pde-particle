"""Build and run the PDE + Smoldyn hybrid co-simulation as a process-bigraph composite.

Store layout::

    fields           {species: µM array}       written by pde, read by particles
    particle_counts  {species: count array}    written by particles, read by pde
    particle_totals  {species: float}          written by particles
    pde              FVReactionDiffusion       interval dt
    particles        SmoldynHybrid             interval dt (fvsolver) or k·dt
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from process_bigraph import Composite

from viva_pde_particle.model import (
    HybridModel,
    initial_fields,
    initial_particle_counts,
    partition,
    write_smoldyn_config,
)
from viva_pde_particle.processes import interval_for


def build_hybrid_document(
    model: HybridModel,
    dt: float,
    step_multiplier: int = 1,
    coupling: str = "fvsolver",
    seed: int = 1,
    particle_init: str = "exact",
) -> dict:
    parts = partition(model)
    rng = np.random.default_rng(seed)
    counts0 = initial_particle_counts(model, rng, mode=particle_init)
    grid_cfg = model.grid.to_config()
    config_text = write_smoldyn_config(
        parts.particles, model.grid, counts0, time_step=step_multiplier * dt, seed=seed
    )
    return {
        "fields": initial_fields(model),
        "particle_counts": {s: c.copy() for s, c in counts0.items()},
        "particle_totals": {s: float(c.sum()) for s, c in counts0.items()},
        "pde": {
            "_type": "process",
            "address": "local:FVReactionDiffusion",
            "config": {"grid": grid_cfg, "pde": parts.pde, "dt": dt},
            "interval": dt,
            "inputs": {"fields": ["fields"], "particle_counts": ["particle_counts"]},
            "outputs": {"fields": ["fields"]},
        },
        "particles": {
            "_type": "process",
            "address": "local:SmoldynHybrid",
            "config": {
                "grid": grid_cfg,
                "config_text": config_text,
                "particle_species": model.particle_species,
                "field_species": parts.particles["field_species"],
                "dt": dt,
                "step_multiplier": step_multiplier,
                "coupling": coupling,
            },
            "interval": interval_for(coupling, dt, step_multiplier),
            "inputs": {"fields": ["fields"]},
            "outputs": {"particle_counts": ["particle_counts"], "particle_totals": ["particle_totals"]},
        },
    }


@dataclass
class HybridTrajectory:
    times: list[float] = field(default_factory=list)
    fields: dict[str, list[np.ndarray]] = field(default_factory=dict)
    particle_counts: dict[str, list[np.ndarray]] = field(default_factory=dict)

    def record(self, t: float, state: dict):
        self.times.append(t)
        for s, v in state["fields"].items():
            self.fields.setdefault(s, []).append(np.array(v, dtype=float))
        for s, v in state["particle_counts"].items():
            self.particle_counts.setdefault(s, []).append(np.array(v, dtype=float))

    def as_arrays(self):
        return (
            np.array(self.times),
            {s: np.stack(v) for s, v in self.fields.items()},
            {s: np.stack(v) for s, v in self.particle_counts.items()},
        )


def run_hybrid(
    model: HybridModel,
    t_end: float,
    dt: float,
    step_multiplier: int = 1,
    coupling: str = "fvsolver",
    seed: int = 1,
    record_every: float | None = None,
    core=None,
    particle_init: str = "exact",
) -> HybridTrajectory:
    """Run the composite to ``t_end``, recording state every ``record_every`` (default k·dt)."""
    from viva_pde_particle.core import build_core

    core = core or build_core()
    doc = build_hybrid_document(model, dt, step_multiplier, coupling, seed, particle_init)
    sim = Composite({"state": doc}, core=core)
    every = record_every or step_multiplier * dt
    n_records = int(round(t_end / every))
    traj = HybridTrajectory()
    traj.record(0.0, sim.state)
    # Process-bigraph accumulates each process's time as a float sum of its interval, so an
    # update ending exactly at a record time can land just past it (25 × 0.01 > 0.25) and be
    # deferred, leaving the recorded state one step behind. Advancing to half a step past
    # each record time applies every update ending at or before it. The state is piecewise
    # constant between updates, so the recorded time is still exact.
    half = 0.5 * dt
    sim.run(half)
    for i in range(1, n_records + 1):
        sim.run(every)
        traj.record(i * every, sim.state)
    return traj
