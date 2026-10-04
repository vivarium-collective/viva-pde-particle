"""Build and run the PDE + Smoldyn hybrid co-simulation as a process-bigraph composite.

Store layout::

    fields            {species: µM array}       written by pde, read by particles
    particle_counts   {species: count array}    written by particles, read by particle_to_field
    particle_conc     {species: µM array}       written by particle_to_field, read by pde (external_conc)
    particle_totals   {species: float}          written by particles
    pde               FVReactionDiffusion       Process, interval dt
    particles         SmoldynHybrid             Process, interval dt (fvsolver) or k·dt
    particle_to_field GridCountsToConcentration Step, runs whenever particle_counts changes
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
from viva_pde_particle.processes.smoldyn_hybrid import COUPLING_MODES


def particle_node(smoldyn_config: dict, coupling: str, dt: float, step_multiplier: int,
                  inputs: dict, outputs: dict) -> dict:
    """The particle engine node for a coupling mode (the engine itself has no clock of the PDE).

    - ``"start-of-interval"``: SmoldynHybrid with interval k·dt (reads the field at T).
    - ``"fvsolver"``: SmoldynHybrid inside a Stepper (tick dt, every k, phase k−1), which reads
      the field at T+(k−1)·dt and publishes at T+k·dt, as vcell-fvsolver does.
    """
    if coupling not in COUPLING_MODES:
        raise ValueError(f"coupling must be one of {COUPLING_MODES}, got {coupling!r}")
    node = {"_type": "process", "interval": interval_for(coupling, dt, step_multiplier),
            "inputs": inputs, "outputs": outputs}
    if coupling == "start-of-interval":
        return {**node, "address": "local:SmoldynHybrid", "config": smoldyn_config}
    return {**node, "address": "local:Stepper", "config": {
        "process": {"address": "local:SmoldynHybrid", "config": smoldyn_config},
        "dt": dt, "every": step_multiplier, "phase": step_multiplier - 1}}


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
    from process_bigraph import allocate_core

    from viva_pde_particle.steps import GridCountsToConcentration

    to_field_cfg = {"grid": grid_cfg, "species": model.particle_species}
    to_field = GridCountsToConcentration(config=to_field_cfg, core=allocate_core())
    return {
        "fields": initial_fields(model),
        "particle_counts": {s: c.copy() for s, c in counts0.items()},
        "particle_conc": to_field.convert(counts0),  # Steps do not run at initialization
        "particle_totals": {s: float(c.sum()) for s, c in counts0.items()},
        "pde": {
            "_type": "process",
            "address": engines[pde_engine],
            "config": {"grid": grid_cfg, "pde": parts.pde, "dt": dt, **(pde_options or {})},
            "interval": dt,
            "inputs": {"fields": ["fields"], "external_conc": ["particle_conc"]},
            "outputs": {"fields": ["fields"]},
        },
        "particle_to_field": {
            "_type": "step",
            "address": "local:GridCountsToConcentration",
            "config": to_field_cfg,
            "inputs": {"particle_counts": ["particle_counts"]},
            "outputs": {"particle_conc": ["particle_conc"]},
        },
        "particles": particle_node(
            {"grid": grid_cfg, "config_text": config_text, "particle_species": model.particle_species,
             "field_species": parts.particles["field_species"], "dt": dt, "step_multiplier": step_multiplier},
            coupling, dt, step_multiplier,
            inputs={"fields": ["fields"]},
            outputs={"particle_counts": ["particle_counts"], "particle_totals": ["particle_totals"]}),
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


def build_mesh_hybrid_document(model: HybridModel, sphere: dict | None, dt: float, step_multiplier: int = 1,
                               coupling: str = "fvsolver", seed: int = 1, particle_transfer: str = "grid",
                               membrane: str = "mesh", volume_samples: int | None = 2,
                               geometry: dict | None = None) -> dict:
    """Hybrid co-simulation on an unstructured mesh: FEniCSx PDE + Smoldyn confined to the PDE domain.

    The domain is given by one of:
    - ``geometry``: ``{"description": GeometryDescription (or its dict), "region", "h"}``, a
      VCell-style geometry realized by Netgen (Phase 7e; any analytic/CSG/image subvolume);
    - ``sphere``: ``{"center", "radius", "h"}``, a gmsh ball (pre-7e studies).

    ``model.grid`` is the background Cartesian grid (it must contain the domain). Particle
    species need scalar initial totals, placed uniformly in the particle domain, and continuous
    species scalar initial concentrations.

    - ``particle_transfer``: ``"grid"`` (Pᵀ of the grid histogram) or ``"positions"`` (exact P1
      load from molecule positions); see FenicsxMeshReactionDiffusion.
    - ``membrane``: the reflecting surface confining the particles.
      - ``"mesh"`` (default): the PDE mesh's boundary triangles, so the particle and PDE
        domains coincide.
        ``volume_samples`` (default 2) is the number of samples per grid spacing in Smoldyn's
        ``highResVolumeSamples`` compartment map, VCell's acceleration of ``posincompart``.
        It is 1.8× faster for exchange with identical results. None writes no map, so every
        compartment test crosses all boundary triangles.
      - ``"sphere"``: the exact sphere, about 1.4% larger than the inscribed mesh at
        h = R/5. Particles in that sliver load the boundary DOFs (+1.3% outer-shell bias,
        and +2.9% exchange mass balance; Study B2d).
    """
    from viva_pde_particle.processes.fenicsx_mesh_reaction_diffusion import FenicsxMeshReactionDiffusion

    if membrane not in ("sphere", "mesh"):
        raise ValueError(f"membrane must be 'sphere' or 'mesh', got {membrane!r}")
    if (sphere is None) == (geometry is None):
        raise ValueError("give exactly one of sphere= or geometry=")
    parts = partition(model)
    rng = np.random.default_rng(seed)
    g = model.grid
    realization = None
    if geometry is not None:
        if membrane != "mesh":
            raise ValueError("a geometry= domain uses the mesh boundary as the membrane (membrane='mesh')")
        from viva_pde_particle.geometry import description_from_config, description_to_config, realize_fenics

        desc = geometry["description"]
        desc_cfg = desc if isinstance(desc, dict) else description_to_config(desc)
        mesh_spec = {"kind": "geometry", "description": desc_cfg, "region": geometry["region"],
                     "h": float(geometry["h"])}
        realization = realize_fenics(description_from_config(desc_cfg), mesh_spec["h"])
    else:
        mesh_spec = {"kind": "sphere", "center": list(sphere["center"]), "radius": sphere["radius"],
                     "h": sphere.get("h", sphere["radius"] / 6)}
    if particle_transfer not in ("grid", "positions"):
        raise ValueError(f"particle_transfer must be 'grid' or 'positions', got {particle_transfer!r}")
    pde_cfg = {"mesh": mesh_spec, "pde": parts.pde, "dt": dt}
    from process_bigraph import allocate_core

    probe = FenicsxMeshReactionDiffusion(config=pde_cfg, core=allocate_core())
    if membrane == "mesh":
        # Particle domain = PDE domain: everything Smoldyn needs is derived from one realization (7e.3).
        from viva_pde_particle.geometry import MeshRealization
        from viva_pde_particle.geometry import smoldyn_geometry as derive_smoldyn_geometry
        from viva_pde_particle.mesh import mesh_locator

        if realization is None:  # gmsh ball: a one-region realization of the PDE mesh
            realization = MeshRealization(probe.V.mesh, locator=mesh_locator(mesh_spec, g),
                                          interior=list(sphere["center"]))
            region = "domain"
        else:
            region = mesh_spec["region"]
        positions = {s.name: realization.uniform_points(region, int(s.initial), rng)
                     for s in model.species if s.particle}
        smoldyn_geometry = derive_smoldyn_geometry(realization, region, g, volume_samples)
    else:
        positions = {s.name: uniform_in_sphere(int(s.initial), sphere["center"], sphere["radius"], rng)
                     for s in model.species if s.particle}
        smoldyn_geometry = {"kind": "sphere", "center": list(sphere["center"]), "radius": sphere["radius"]}
    counts0 = {name: g.histogram(p) for name, p in positions.items()}
    # initial field DOFs (scalar initial conditions only)
    dofs0 = probe.initial_dofs({s.name: float(s.initial) for s in model.species if not s.particle})
    config_text = write_smoldyn_config(parts.particles, g, counts0, time_step=step_multiplier * dt, seed=seed,
                                       geometry=smoldyn_geometry, positions=positions)
    by_position = particle_transfer == "positions"
    from viva_pde_particle.steps import GridCountsToMeshConcentration, MeshToGridField, PositionsToMeshConcentration

    step_cls = PositionsToMeshConcentration if by_position else GridCountsToMeshConcentration
    to_field_cfg = {"grid": g.to_config(), "mesh": mesh_spec, "species": model.particle_species}
    to_field = step_cls(config=to_field_cfg, core=allocate_core())
    step_input = "particle_positions" if by_position else "particle_counts"
    to_grid_cfg = {"grid": g.to_config(), "mesh": mesh_spec, "species": list(dofs0)}
    to_grid = MeshToGridField(config=to_grid_cfg, core=allocate_core())
    doc = {
        "field_dofs": dofs0,
        "fields": to_grid.convert(dofs0),  # Steps do not run at initialization
        "particle_counts": {s: c.copy() for s, c in counts0.items()},
        "particle_conc": to_field.convert(positions if by_position else counts0),  # Steps don't run at init
        "particle_totals": {s: float(c.sum()) for s, c in counts0.items()},
        "pde": {
            "_type": "process",
            "address": "local:FenicsxMeshReactionDiffusion",
            "config": pde_cfg,
            "interval": dt,
            "inputs": {"field_dofs": ["field_dofs"], "external_conc": ["particle_conc"]},
            "outputs": {"field_dofs": ["field_dofs"]},
        },
        "field_to_particles": {
            "_type": "step",
            "address": "local:MeshToGridField",
            "config": to_grid_cfg,
            "inputs": {"field_dofs": ["field_dofs"]},
            "outputs": {"fields": ["fields"]},
        },
        "particle_to_field": {
            "_type": "step",
            "address": f"local:{step_cls.__name__}",
            "config": to_field_cfg,
            "inputs": {step_input: [step_input]},
            "outputs": {"particle_conc": ["particle_conc"]},
        },
        "particles": particle_node(
            {"grid": g.to_config(), "config_text": config_text, "particle_species": model.particle_species,
             "field_species": parts.particles["field_species"], "dt": dt, "step_multiplier": step_multiplier,
             "emit_positions": by_position},
            coupling, dt, step_multiplier,
            inputs={"fields": ["fields"]},
            outputs={"particle_counts": ["particle_counts"], "particle_totals": ["particle_totals"]}),
    }
    if by_position:
        doc["particle_positions"] = {s: np.asarray(p, dtype=float) for s, p in positions.items()}
        doc["particles"]["outputs"]["particle_positions"] = ["particle_positions"]  # shared dict in both modes
    return doc


def build_vcell_geometry_hybrid_document(model: HybridModel, description, dt: float, step_multiplier: int = 1,
                                         coupling: str = "fvsolver", seed: int = 1, region: str | None = None,
                                         correction: str = "adapters+volumes", volume_samples: int | None = 2,
                                         realization=None) -> dict:
    """FV + Smoldyn on a geometry discretized exactly as native VCell does (Phase 7e.5).

    The geometry (a vcell-fenics ``GeometryDescription``) is realized by VCell itself
    (:class:`~viva_pde_particle.geometry.vcell_fv.VCellFVRealization`, via libvcell):

    - **PDE:** FVReactionDiffusion on VCell's staircase ``region`` (domain mask; zero flux at its
      boundary). With ``correction="adapters+volumes"`` its element volumes are the accessible
      volumes.
    - **Particles:** Smoldyn confined by VCell's smooth membrane, with VCell's compartment points
      and a volume-sample map, all derived from the realization (7e.3).
    - **Adapters:** ``GridCountsToConcentration`` folds counts at exterior nodes into the domain
      (compartment-aware binning) and divides by the ``correction``'s effective volumes.
      ``ExtendGridField`` extends fields into the exterior band for the particle engine's lookups.

    ``correction``: ``"none"`` (VCell-like full voxel volumes), ``"adapters"`` or
    ``"adapters+volumes"`` (default); see :mod:`viva_pde_particle.geometry.accessible`. Particle
    species need scalar initial totals (placed uniformly in the smooth region), and continuous
    species scalar initial concentrations. Needs pyvcell with spatial-hybrid support and libvcell
    (the ``dev`` env), unless ``realization`` is given.
    """
    from process_bigraph import allocate_core

    from viva_pde_particle.geometry import smoldyn_geometry
    from viva_pde_particle.geometry.accessible import accessible_fractions, effective_volumes, fold_map
    from viva_pde_particle.steps import ExtendGridField, GridCountsToConcentration

    g = model.grid
    if realization is None:
        from viva_pde_particle.geometry.vcell_fv import VCellFVRealization

        realization = VCellFVRealization(description, g)
    region = region or realization.regions[0]
    parts = partition(model)
    rng = np.random.default_rng(seed)
    mask = realization.node_mask(region)
    fractions = accessible_fractions(realization, region, g)
    fold = fold_map(mask, g)
    volumes = effective_volumes(g, mask, fractions, fold, correction)
    domain = {"mask": mask}
    if correction == "adapters+volumes":
        domain["volume_fraction"] = volumes / g.full_volume
    positions = {s.name: realization.uniform_points(region, int(s.initial), rng) for s in model.species if s.particle}
    counts0 = {name: g.histogram(p) for name, p in positions.items()}
    config_text = write_smoldyn_config(parts.particles, g, counts0, time_step=step_multiplier * dt, seed=seed,
                                       geometry=smoldyn_geometry(realization, region, g, volume_samples),
                                       positions=positions)
    grid_cfg = g.to_config()
    to_field_cfg = {"grid": grid_cfg, "species": model.particle_species,
                    "staircase": {"fold": fold, "volumes": volumes}}
    to_field = GridCountsToConcentration(config=to_field_cfg, core=allocate_core())
    field_species = list(parts.pde["species"])
    lookup_cfg = {"grid": grid_cfg, "species": field_species, "staircase": {"fold": fold}}
    lookup = ExtendGridField(config=lookup_cfg, core=allocate_core())
    fields0 = initial_fields(model)
    return {
        "fields": fields0,
        "lookup_fields": lookup.convert(fields0),  # Steps do not run at initialization
        "particle_counts": {s: c.copy() for s, c in counts0.items()},
        "particle_conc": to_field.convert(counts0),
        "particle_totals": {s: float(c.sum()) for s, c in counts0.items()},
        "pde": {
            "_type": "process",
            "address": "local:FVReactionDiffusion",
            "config": {"grid": grid_cfg, "pde": parts.pde, "dt": dt, "domain": domain},
            "interval": dt,
            "inputs": {"fields": ["fields"], "external_conc": ["particle_conc"]},
            "outputs": {"fields": ["fields"]},
        },
        "particles": particle_node(
            {"grid": grid_cfg, "config_text": config_text, "particle_species": model.particle_species,
             "field_species": parts.particles["field_species"], "dt": dt, "step_multiplier": step_multiplier},
            coupling, dt, step_multiplier,
            inputs={"fields": ["lookup_fields"]},
            outputs={"particle_counts": ["particle_counts"], "particle_totals": ["particle_totals"]}),
        "particle_to_field": {
            "_type": "step", "address": "local:GridCountsToConcentration", "config": to_field_cfg,
            "inputs": {"particle_counts": ["particle_counts"]}, "outputs": {"particle_conc": ["particle_conc"]},
        },
        "field_to_particles": {
            "_type": "step", "address": "local:ExtendGridField", "config": lookup_cfg,
            "inputs": {"fields": ["fields"]}, "outputs": {"lookup_fields": ["lookup_fields"]},
        },
    }


def to_splitting_document(doc: dict, scheme: str, dt: float, step_multiplier: int) -> dict:
    """Replace the engine and adapter nodes of a two-process document with one SplittingCoordinator.

    ``doc`` must use ``coupling="start-of-interval"``, so its particle node is the bare engine.
    Works for any engine pair (FV, FEniCSx Q1, unstructured mesh); the stores are unchanged.
    """
    if doc["particles"]["address"] == "local:Stepper":
        raise ValueError("build the two-process document with coupling='start-of-interval' (the coordinator schedules)")

    def node(name):
        n = doc[name]
        return {"address": n["address"], "config": n["config"], "inputs": n["inputs"], "outputs": n["outputs"]}

    adapter_names = [name for name in ("particle_to_field", "field_to_particles") if name in doc]
    children = {"pde": node("pde"), "particles": node("particles"), **{a: node(a) for a in adapter_names}}
    reads = {path[0] for c in children.values() for path in c["inputs"].values()}
    writes = {path[0] for c in children.values() for path in c["outputs"].values()}
    out = {k: v for k, v in doc.items() if k not in children}
    out["coupler"] = {
        "_type": "process",
        "address": "local:SplittingCoordinator",
        "config": {"pde": children["pde"], "particles": children["particles"],
                   "adapters": {a: children[a] for a in adapter_names},
                   "scheme": scheme, "dt": dt, "step_multiplier": step_multiplier},
        "interval": step_multiplier * dt,
        "inputs": {s: [s] for s in sorted(reads)},
        "outputs": {s: [s] for s in sorted(writes)},
    }
    return out


def build_coupler_document(model: HybridModel, dt: float, step_multiplier: int, scheme: str, seed: int = 1,
                           particle_init: str = "exact", pde_engine: str = "fv", pde_options: dict | None = None) -> dict:
    """One SplittingCoordinator (PDE + particles + adapters with a chosen splitting) instead of two processes."""
    doc = build_hybrid_document(model, dt, step_multiplier, "start-of-interval", seed, particle_init,
                                pde_engine, pde_options)
    return to_splitting_document(doc, scheme, dt, step_multiplier)

