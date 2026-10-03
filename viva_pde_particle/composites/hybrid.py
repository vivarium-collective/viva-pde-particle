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
    pde_engine: str = "fv",
    pde_options: dict | None = None,
) -> dict:
    """``pde_engine``: ``"fv"`` (FVReactionDiffusion) or ``"fenicsx"`` (FenicsxReactionDiffusion,
    ``pde_options={"mass": "lumped" | "consistent"}``)."""
    engines = {"fv": "local:FVReactionDiffusion", "fenicsx": "local:FenicsxReactionDiffusion"}
    if pde_engine not in engines:
        raise ValueError(f"pde_engine must be one of {sorted(engines)}, got {pde_engine!r}")
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
            "address": engines[pde_engine],
            "config": {"grid": grid_cfg, "pde": parts.pde, "dt": dt, **(pde_options or {})},
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
    field_dofs: dict[str, list[np.ndarray]] = field(default_factory=dict)  # unstructured-mesh engine only

    def record(self, t: float, state: dict):
        self.times.append(t)
        for s, v in (state.get("field_dofs") or {}).items():
            self.field_dofs.setdefault(s, []).append(np.array(v, dtype=float))
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
    pde_engine: str = "fv",
    pde_options: dict | None = None,
) -> HybridTrajectory:
    """Run the composite to ``t_end``, recording state every ``record_every`` (default k·dt)."""
    doc = build_hybrid_document(model, dt, step_multiplier, coupling, seed, particle_init, pde_engine, pde_options)
    return run_document(doc, t_end, dt, record_every or step_multiplier * dt, core)


def run_document(doc: dict, t_end: float, dt: float, every: float, core=None) -> HybridTrajectory:
    """Run a hybrid composite document, recording ``fields`` and ``particle_counts`` every ``every``."""
    from viva_pde_particle.core import build_core

    core = core or build_core()
    sim = Composite({"state": doc}, core=core)
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


def uniform_in_sphere(n: int, center, radius: float, rng: np.random.Generator) -> np.ndarray:
    """n points uniformly distributed in a ball."""
    pts = rng.normal(size=(n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return np.asarray(center) + pts * radius * rng.random(n)[:, None] ** (1.0 / 3.0)


def build_mesh_hybrid_document(model: HybridModel, sphere: dict, dt: float, step_multiplier: int = 1,
                               coupling: str = "fvsolver", seed: int = 1, particle_transfer: str = "grid",
                               membrane: str = "mesh") -> dict:
    """Hybrid co-simulation in a ball: unstructured FEniCSx PDE + Smoldyn confined to the ball.

    ``model.grid`` is the background Cartesian grid (it must contain the ball). Particle
    species need scalar initial totals, placed uniformly in the particle domain, and continuous
    species scalar initial concentrations. ``sphere``: ``{"center", "radius", "h"}``.

    - ``particle_transfer``: ``"grid"`` (Pᵀ of the grid histogram) or ``"positions"`` (exact P1
      load from molecule positions); see FenicsxMeshReactionDiffusion.
    - ``membrane``: the reflecting surface confining the particles.
      - ``"mesh"`` (default): the PDE mesh's boundary triangles, so the particle and PDE
        domains coincide.
      - ``"sphere"``: the exact sphere, about 1.4% larger than the inscribed mesh at
        h = R/5. Particles in that sliver load the boundary DOFs (+1.3% outer-shell bias,
        and +2.9% exchange mass balance; Study B2d).
    """
    from viva_pde_particle.processes.fenicsx_mesh_reaction_diffusion import FenicsxMeshReactionDiffusion

    if membrane not in ("sphere", "mesh"):
        raise ValueError(f"membrane must be 'sphere' or 'mesh', got {membrane!r}")
    parts = partition(model)
    rng = np.random.default_rng(seed)
    g = model.grid
    mesh_spec = {"kind": "sphere", "center": list(sphere["center"]), "radius": sphere["radius"],
                 "h": sphere.get("h", sphere["radius"] / 6)}
    pde_cfg = {"grid": g.to_config(), "mesh": mesh_spec, "pde": parts.pde, "dt": dt,
               "particle_transfer": particle_transfer}
    from process_bigraph import allocate_core

    probe = FenicsxMeshReactionDiffusion(config=pde_cfg, core=allocate_core())
    if membrane == "mesh":
        from viva_pde_particle.mesh import PointLocator, boundary_triangles, uniform_in_mesh

        msh = probe.transfer.V.mesh
        locator = getattr(probe, "locator", None) or PointLocator.build(msh, probe.transfer.V)
        lo, hi = msh.geometry.x.min(axis=0), msh.geometry.x.max(axis=0)
        positions = {s.name: uniform_in_mesh(int(s.initial), locator, lo, hi, rng)
                     for s in model.species if s.particle}
        geometry = {"kind": "triangles", "triangles": boundary_triangles(msh),
                    "interior_point": list(sphere["center"])}
    else:
        positions = {s.name: uniform_in_sphere(int(s.initial), sphere["center"], sphere["radius"], rng)
                     for s in model.species if s.particle}
        geometry = {"kind": "sphere", "center": list(sphere["center"]), "radius": sphere["radius"]}
    counts0 = {name: g.histogram(p) for name, p in positions.items()}
    # initial field DOFs (scalar initial conditions only)
    dofs0 = probe.initial_dofs({s.name: float(s.initial) for s in model.species if not s.particle})
    config_text = write_smoldyn_config(parts.particles, g, counts0, time_step=step_multiplier * dt, seed=seed,
                                       geometry=geometry, positions=positions)
    by_position = particle_transfer == "positions"
    doc = {
        "field_dofs": dofs0,
        "fields": {s: probe.to_grid(v) for s, v in dofs0.items()},
        "particle_counts": {s: c.copy() for s, c in counts0.items()},
        "particle_totals": {s: float(c.sum()) for s, c in counts0.items()},
        "pde": {
            "_type": "process",
            "address": "local:FenicsxMeshReactionDiffusion",
            "config": pde_cfg,
            "interval": dt,
            "inputs": {"field_dofs": ["field_dofs"], "particle_counts": ["particle_counts"]},
            "outputs": {"field_dofs": ["field_dofs"], "fields": ["fields"]},
        },
        "particles": {
            "_type": "process",
            "address": "local:SmoldynHybrid",
            "config": {
                "grid": g.to_config(), "config_text": config_text, "particle_species": model.particle_species,
                "field_species": parts.particles["field_species"], "dt": dt,
                "step_multiplier": step_multiplier, "coupling": coupling, "emit_positions": by_position,
            },
            "interval": interval_for(coupling, dt, step_multiplier),
            "inputs": {"fields": ["fields"]},
            "outputs": {"particle_counts": ["particle_counts"], "particle_totals": ["particle_totals"]},
        },
    }
    if by_position:
        doc["particle_positions"] = {s: np.asarray(p, dtype=float) for s, p in positions.items()}
        doc["pde"]["inputs"]["particle_positions"] = ["particle_positions"]
        doc["particles"]["outputs"]["particle_positions"] = ["particle_positions"]
    return doc


def build_coupler_document(model: HybridModel, dt: float, step_multiplier: int, scheme: str, seed: int = 1,
                           particle_init: str = "exact") -> dict:
    """One HybridCoupler process (PDE + particles with a chosen splitting) instead of two processes."""
    doc = build_hybrid_document(model, dt, step_multiplier, "start-of-interval", seed, particle_init)
    return {
        "fields": doc["fields"],
        "particle_counts": doc["particle_counts"],
        "particle_totals": doc["particle_totals"],
        "coupler": {
            "_type": "process",
            "address": "local:HybridCoupler",
            "config": {"pde": doc["pde"]["config"], "particles": doc["particles"]["config"], "scheme": scheme},
            "interval": step_multiplier * dt,
            "inputs": {"fields": ["fields"], "particle_counts": ["particle_counts"]},
            "outputs": {"fields": ["fields"], "particle_counts": ["particle_counts"],
                        "particle_totals": ["particle_totals"]},
        },
    }
