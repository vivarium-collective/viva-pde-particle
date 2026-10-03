"""Run a HybridModel on VCell's native PDE/particle hybrid solver, headlessly.

The model goes through VCell's own pipeline:

1. **pyvcell:** a spatial, stochastic BioModel application. Continuous species are
   ``force_continuous`` and the rest become particles. The simulation uses the
   ``"Finite Volume Standalone, Regular Grid"`` solver.
2. **libvcell:** VCell math generation (``ParticleMathMapping.combineHybrid``) and the
   writers produce ``.fvinput`` plus ``.smoldynInput``.
3. **pyvcell-fvsolver:** vcell-fvsolver with its embedded Smoldyn 2.38 runs them.
4. The ``.sim`` output is read with pyvcell's PdeDataSet: fields in µM and particle
   variables as counts per element.

This needs a pyvcell with spatial-hybrid support (``HYBRID_SOLVER``); see the ``dev``
pixi environment. pyvcell's zarr-based ``Result`` is not used: it pins zarr 2, and the
workbench needs zarr 3.

Limits of the BioModel route:

- **Uniform particle initial conditions:** a total ``N`` is passed as the equivalent
  concentration, and VCell writes ``compartment_mol Poisson(N)``. Match it in the
  co-simulation with ``particle_init="poisson"``.
- **Per-node particle counts** (e.g. ion channels at fixed sites) are not expressible in a
  BioModel. Such species get zero initial concentration, and their initial-molecule lines
  in the VCell-generated ``.smoldynInput`` are replaced by ``mol 1 X x y z`` at the node
  centres. Only initial positions change; the math and all other inputs remain VCell's.
  (The long-term fix is located initial counts in VCell, or MathModel support in libvcell.)
- **No-op particle reactions** (a particle catalysing a continuous source, ``O -> O + U``)
  are removed from the ``.smoldynInput``. VCell keeps them as destroy/create jump processes,
  which suppress the particle's competing reactions in Smoldyn. See
  :func:`drop_noop_reactions`.
- Spatially varying field initial conditions need ``Species.initial_expression``.
- VCell's spatial stochastic math requires 3D geometry. 2D problems run as quasi-2D slabs
  (``Nz = 3``, the minimum VCell accepts per axis), with the co-simulation on the same grid.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from viva_pde_particle.model import HybridModel
from viva_pde_particle.units import MOLECULES_PER_UM3_PER_UM

COMPARTMENT = "cell"
SUBVOLUME = "cell"


def hybrid_support_available() -> bool:
    try:
        import libvcell  # noqa: F401
        import pyvcell_fvsolver  # noqa: F401
        import pyvcell.vcml.models_app as app
    except Exception:
        return False
    return hasattr(app, "HYBRID_SOLVER")


def validate_for_native(model: HybridModel) -> None:
    """Raise ValueError for models the native VCell hybrid path cannot represent."""
    g = model.grid
    if g.dim != 3:
        raise ValueError("VCell spatial stochastic/hybrid models require 3D geometry; use a quasi-2D slab "
                         "grid (e.g. CartesianGrid((0, 0, 0), (Lx, Ly, Lz), (Nx, Ny, 3)))")
    if any(n < 3 for n in g.num):
        raise ValueError("VCell 3D meshes need at least 3 nodes per axis")
    if any(o != 0 for o in g.origin):
        raise ValueError("native reference assumes a grid origin of 0")
    if not model.particle_species or not model.continuous_species:
        raise ValueError("a hybrid model needs at least one particle and one continuous species")
    for s in model.species:
        init = np.asarray(s.initial)
        if s.particle and init.ndim != 0 and (init.shape != g.shape or np.any(init != np.round(init))):
            raise ValueError(f"species {s.name}: per-node particle counts must be integers of grid shape")
        if not s.particle and init.ndim != 0 and s.initial_expression is None:
            raise ValueError(f"species {s.name}: array initial condition needs Species.initial_expression")


def to_biomodel(model: HybridModel, *, t_end: float, dt: float, output_dt: float,
                step_multiplier: int = 1, seed: int | None = None, name: str = "hybrid"):
    """Build the pyvcell BioModel (one hybrid application, one simulation named ``sim``)."""
    validate_for_native(model)
    import pyvcell.vcml as vc
    from pyvcell.vcml.models import (
        Biomodel,
        Kinetics,
        KineticsParameter,
        Model,
        Reaction,
        SpeciesReference,
        SpeciesRefType,
    )

    g = model.grid

    m = Model(name=name)
    m.add_compartment(COMPARTMENT, dim=3)
    for s in model.species:
        m.add_species(s.name, COMPARTMENT)
    for r in model.reactions:
        if not r.reactants:
            # VCell mass action with no reactants keeps only the reverse term (rate = -Kr·Π products)
            # and discards Kf, so a constant source must be written as General kinetics J = k.
            kin = Kinetics(kinetics_type="GeneralKinetics", kinetics_parameters=[
                KineticsParameter(name="J", value=repr(float(r.k)), role="reaction rate", unit="uM.s-1",
                                  reaction_name=r.name),
            ])
        else:
            kin = Kinetics(kinetics_type="MassAction", kinetics_parameters=[
                KineticsParameter(name="Kf", value=r.k, role="forward rate constant", unit="", reaction_name=r.name),
                KineticsParameter(name="Kr", value=0.0, role="reverse rate constant", unit="", reaction_name=r.name),
            ])
        rx = Reaction(name=r.name, compartment_name=COMPARTMENT, reversible=False, is_flux=False, kinetics=kin)
        for s, n in r.reactants.items():
            rx.reactants.append(SpeciesReference(name=s, stoichiometry=n, species_ref_type=SpeciesRefType.reactant))
        for s, n in r.products.items():
            rx.products.append(SpeciesReference(name=s, stoichiometry=n, species_ref_type=SpeciesRefType.product))
        m.reactions.append(rx)

    geo = vc.Geometry(name=f"{name}_box", dim=3, extent=tuple(g.size), origin=(0.0, 0.0, 0.0))
    geo.add_background(SUBVOLUME)

    bm = Biomodel(name=name, model=m)
    app = bm.add_application("hybrid", geometry=geo, stochastic=True)
    app.map_compartment(COMPARTMENT, SUBVOLUME)
    for s in model.species:
        if s.particle:
            init = np.asarray(s.initial)
            if init.ndim != 0:
                # placed explicitly after input generation (see place_particles)
                app.map_species(s.name, init_conc=0.0, diff_coef=s.diffusion)
                continue
            # Concentration mode (UseConcentration, the default): VCell seeds each node with
            # Poisson(conc·602.214·V_node) molecules, so a total N becomes the equivalent uniform µM.
            conc = float(init) / (g.element_volumes.sum() * MOLECULES_PER_UM3_PER_UM)
            app.map_species(s.name, init_conc=conc, diff_coef=s.diffusion)
        else:
            init = np.asarray(s.initial)
            if s.initial_expression is not None:
                init_conc = s.initial_expression
            elif init.ndim == 0:
                init_conc = float(init)
            else:
                raise ValueError(f"species {s.name}: array initial condition needs Species.initial_expression")
            app.map_species(s.name, init_conc=init_conc, diff_coef=s.diffusion, force_continuous=True)
    for r in model.reactions:
        app.map_reaction(r.name, True)
    mesh = tuple(g.num)
    app.add_hybrid_sim(
        name="sim", duration=t_end, output_time_step=output_dt, mesh_size=mesh, time_step=dt,
        options=vc.SmoldynSimulationOptions(random_seed=seed, step_multiplier=step_multiplier),
    )
    return bm


def particle_positions(model: HybridModel, inset: float = 1e-6) -> dict[str, np.ndarray]:
    """Node-centre positions (n, 3), one row per molecule, for species with per-node counts.

    Positions are nudged ``inset`` (relative to the domain size) inside the domain walls.
    """
    g = model.grid
    coords = [c.ravel() for c in g.node_coordinates()]
    lo = np.array(g.origin) + inset * np.array(g.size)
    hi = np.array(g.origin) + (1 - inset) * np.array(g.size)
    out = {}
    for s in model.species:
        init = np.asarray(s.initial)
        if s.particle and init.ndim != 0:
            reps = init.ravel().astype(int)
            pts = np.stack([np.repeat(c, reps) for c in coords], axis=1)
            out[s.name] = np.clip(pts, lo, hi)
    return out


def drop_noop_reactions(smoldyn_input: Path) -> list[str]:
    """Remove Smoldyn reactions whose particle reactants equal their products; return their names.

    VCell's ParticleMathMapping keeps a particle jump process when a particle is a catalyst of
    a continuous source (e.g. ``O -> O + U``, or O as a modifier): the process destroys and
    re-creates O at rate k. Such a process does nothing to the particles, but Smoldyn gives
    competing first-order reactions probabilities ∝ k_i/Σk, so a fast no-op suppresses the
    particle's real reactions (e.g. channel closing). The continuous source stays in the PDE
    (.fvinput), so dropping the process from the particle side is exact. The proper fix is
    for combineHybrid to drop processes whose actions cancel.
    """
    dropped, kept = [], []
    for line in smoldyn_input.read_text().splitlines():
        words = line.split()
        if words and words[0] in ("reaction", "reaction_cmpt", "reaction_surface") and "->" in words:
            body = words[1:]
            if words[0] != "reaction":
                body = body[1:]  # compartment or surface name
            name, rest = body[0], body[1:]
            arrow = rest.index("->")
            rct = sorted(w for w in rest[:arrow] if w not in ("+", "0"))
            prd_words = rest[arrow + 1:-1]  # last word is the rate
            prd = sorted(w for w in prd_words if w not in ("+", "0"))
            if rct and rct == prd:
                dropped.append(name)
                continue
        kept.append(line)
    if dropped:
        smoldyn_input.write_text("\n".join(kept) + "\n")
    return dropped


def place_particles(smoldyn_input: Path, positions: dict[str, np.ndarray]) -> None:
    """Replace the initial-molecule lines of the given species with explicit ``mol 1`` lines."""
    lines = smoldyn_input.read_text().splitlines()
    kept = []
    for line in lines:
        words = line.split()
        if len(words) >= 3 and words[0] in ("mol", "compartment_mol") and words[2] in positions:
            continue
        kept.append(line)
    end = next(i for i, line in enumerate(kept) if line.strip() == "end_file")
    placed = [f"mol 1 {name} " + " ".join(repr(float(v)) for v in p)
              for name, pts in positions.items() for p in pts]
    smoldyn_input.write_text("\n".join(kept[:end] + placed + kept[end:]) + "\n")


@dataclass
class NativeTrajectory:
    times: np.ndarray
    fields: dict[str, np.ndarray] = field(default_factory=dict)          # (T, *grid.shape), µM
    particle_counts: dict[str, np.ndarray] = field(default_factory=dict)  # (T, *grid.shape), counts
    workdir: Path | None = None
    #: wall-clock seconds per phase: "generate" (pyvcell + libvcell), "solve", "read"
    timings: dict[str, float] = field(default_factory=dict)


def run_native(model: HybridModel, *, t_end: float, dt: float, output_dt: float | None = None,
               step_multiplier: int = 1, seed: int | None = None, workdir: Path | None = None,
               isolate: bool = True) -> NativeTrajectory:
    """Generate inputs with libvcell, solve with pyvcell-fvsolver, and read back fields and counts.

    ``isolate`` (default) runs the solve in a fresh spawned process. vcell-fvsolver cannot run
    two hybrid solves in one process (the second segfaults in its embedded Smoldyn), and a
    child process also keeps a solver crash from taking down the caller.
    """
    kwargs = dict(t_end=t_end, dt=dt, output_dt=output_dt, step_multiplier=step_multiplier, seed=seed,
                  workdir=workdir)
    if not isolate:
        return _run_native_inprocess(model, **kwargs)
    with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as pool:
        return pool.submit(_run_native_inprocess, model, **kwargs).result()


def run_native_ensemble(model: HybridModel, seeds, *, t_end: float, dt: float, output_dt: float | None = None,
                        step_multiplier: int = 1, workers: int | None = None):
    """One native run per seed, in parallel, each in its own process.

    Returns ``(times, fields, counts)`` with arrays of shape (n_seeds, n_times, *grid.shape),
    like :func:`viva_pde_particle.analysis.run_ensemble`.
    """
    seeds = [int(s) for s in seeds]
    workers = workers or min(len(seeds), os.cpu_count() or 1)
    kwargs = dict(t_end=t_end, dt=dt, output_dt=output_dt, step_multiplier=step_multiplier, cleanup=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"),
                             max_tasks_per_child=1) as pool:
        runs = list(pool.map(_run_seed, [(model, s, kwargs) for s in seeds]))
    times = runs[0].times
    fields = {s: np.stack([r.fields[s] for r in runs]) for s in runs[0].fields}
    counts = {s: np.stack([r.particle_counts[s] for r in runs]) for s in runs[0].particle_counts}
    return times, fields, counts


def _run_seed(args):
    model, seed, kwargs = args
    return _run_native_inprocess(model, seed=seed, **kwargs)


def _run_native_inprocess(model: HybridModel, *, t_end: float, dt: float, output_dt: float | None = None,
                          step_multiplier: int = 1, seed: int | None = None, workdir: Path | None = None,
                          cleanup: bool = False) -> NativeTrajectory:
    from libvcell import vcml_to_finite_volume_input
    from pyvcell._internal.simdata.simdata_models import PdeDataSet, VariableType
    from pyvcell._internal.solvers.fvsolver import solve as fvsolve
    from pyvcell.vcml.utils import to_vcml_str

    import time

    t0 = time.perf_counter()
    output_dt = output_dt or step_multiplier * dt
    bm = to_biomodel(model, t_end=t_end, dt=dt, output_dt=output_dt, step_multiplier=step_multiplier, seed=seed)
    vcml = to_vcml_str(bio_model=bm)
    out = Path(workdir or tempfile.mkdtemp(prefix="vcell-native-"))
    out.mkdir(parents=True, exist_ok=True)
    ok, msg = vcml_to_finite_volume_input(vcml_content=vcml, simulation_name="sim", output_dir_path=out)
    if not ok:
        raise RuntimeError(f"libvcell failed to generate hybrid inputs: {msg}")
    files = os.listdir(out)
    fv = out / next(f for f in files if f.endswith(".fvinput"))
    vcg = out / next(f for f in files if f.endswith(".vcg"))
    if "SMOLDYN_BEGIN" not in fv.read_text():
        raise RuntimeError("generated .fvinput is not a hybrid (no SMOLDYN block)")
    smoldyn_input = out / next(f for f in files if f.endswith(".smoldynInput"))
    drop_noop_reactions(smoldyn_input)
    positions = particle_positions(model)
    if positions:
        place_particles(smoldyn_input, positions)
    t1 = time.perf_counter()
    rc = fvsolve(input_file=fv, vcg_file=vcg, output_dir=out)
    t2 = time.perf_counter()
    if rc != 0:
        raise RuntimeError(f"pyvcell-fvsolver returned {rc}")

    sim_id, job_id = fv.name.split("_")[1:3]
    ds = PdeDataSet(base_dir=out, log_filename=f"SimID_{sim_id}_{job_id}_.log")
    ds.read()
    times = np.array(ds.times())
    shape = model.grid.shape
    traj = NativeTrajectory(times=times, workdir=out)
    wanted = {VariableType.VOLUME: traj.fields, VariableType.VOLUME_PARTICLE: traj.particle_counts}
    for header in ds.variables_block_headers():
        info = header.var_info
        target = wanted.get(info.variable_type)
        if target is None or "::" not in info.var_name:
            continue
        name = info.var_name.split("::", 1)[1]
        target[name] = np.stack([np.asarray(ds.get_data(info, t), dtype=float).reshape(shape) for t in times])
    traj.timings = {"generate": t1 - t0, "solve": t2 - t1, "read": time.perf_counter() - t2}
    if cleanup:
        shutil.rmtree(out, ignore_errors=True)
        traj.workdir = None
    return traj
