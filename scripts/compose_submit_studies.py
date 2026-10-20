"""Submit every study's canonical simulation to compose-api, tagged.

One document per study baseline (and variant that the compose image can run): seed 1, the study's model,
time step and end time, and a spatial recorder every ``output_dt`` so the run leaves a web-viewable results
bundle (docs/web-bundle.md). Native VCell runs are skipped: the image has no pyvcell (docs/compose.md).

    pixi run python scripts/compose_submit_studies.py build [NAME ...]    # write runs/compose/<name>.omex
    pixi run -e dev python scripts/compose_submit_studies.py build --vcell # the documents that need libvcell
    pixi run python scripts/compose_submit_studies.py submit [NAME ...]   # POST each, tags + description
    pixi run python scripts/compose_submit_studies.py submit --retry      # and again those that failed
    pixi run python scripts/compose_submit_studies.py status              # refresh the record

Documents on an unstructured mesh (``kind: sphere`` or ``kind: geometry`` in the PDE config) are re-meshed by each
process at run time, and gmsh/Netgen on Linux mesh differently than on macOS, so their initial state only matches
when they are built in the image itself:

    docker run --rm --platform linux/amd64 --entrypoint /bin/bash -v "$PWD:/mnt/repo" -w /tmp \
        ghcr.io/vivarium-collective/viva-pde-particle-compose@sha256:<the digest compose-api runs> \
        -c 'source /app/docker/shell-hook.sh && python /mnt/repo/scripts/compose_submit_studies.py build --mesh'

Every submission carries the tags ``pde-particle``, the study slug and its investigation slug. The record of
what was submitted is ``workspace/reports/compose-submissions.json`` (tracked); the documents go to
``workspace/studies/<slug>/runs/compose/`` (gitignored).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ROOT / "workspace" / "studies"
RECORD = ROOT / "workspace" / "reports" / "compose-submissions.json"
TAG = "pde-particle"


@dataclass
class Doc:
    study: str
    name: str
    build: Callable[[], dict]
    t_end: float
    output_dt: float
    description: str
    vcell: bool = False  # building needs libvcell (the dev env); running does not
    mesh: bool = False  # on an unstructured mesh: build it in the compose image (see the module docstring)
    tags: list[str] = field(default_factory=list)

    @property
    def omex(self) -> Path:
        return STUDIES / self.study / "runs" / "compose" / f"{self.name}.omex"


# ---------------------------------------------------------------- models shared by the geometry studies

def _ball_grid():
    from viva_pde_particle.grid import CartesianGrid

    return CartesianGrid((0, 0, 0), (9, 9, 9), (37, 37, 37))


BALL_CENTER, BALL_R = (4.5, 4.5, 4.5), 4.0
BALL_MESH = {"center": BALL_CENTER, "radius": BALL_R, "h": 0.8}


def _ab_model(grid, which: str, k: float = 0.5, n0: int = 20000):
    """A_p -> B_f (conversion) or A_p <-> B_f (exchange), as the B2 ball and dumbbell studies define them."""
    from viva_pde_particle.model import HybridModel, Reaction, Species

    if which == "conversion":
        return HybridModel(grid, [Species("A", 1.0, particle=True, initial=n0), Species("B", 1.0, initial=0.0)],
                           [Reaction("convert", {"A": 1}, {"B": 1}, k=k)])
    return HybridModel(grid, [Species("A", 1.0, particle=True, initial=0), Species("B", 1.0, initial=0.2)],
                       [Reaction("a_to_b", {"A": 1}, {"B": 1}, k=1.0), Reaction("b_to_a", {"B": 1}, {"A": 1}, k=0.5)])


def _ball_mesh_doc(which, **kw):
    from viva_pde_particle.composites.hybrid import build_mesh_hybrid_document

    return build_mesh_hybrid_document(_ab_model(_ball_grid(), which), BALL_MESH, 0.01, seed=1, **kw)


def _ball_description():
    from viva_pde_particle.geometry import sphere_in_box

    return sphere_in_box(BALL_CENTER, BALL_R, (9.0, 9.0, 9.0))


def _ball_vcell_doc(which, correction):
    from viva_pde_particle.composites.hybrid import build_vcell_geometry_hybrid_document

    return build_vcell_geometry_hybrid_document(_ab_model(_ball_grid(), which), _ball_description(), 0.01, seed=1,
                                                correction=correction)


DUMBBELL_S1 = "(geom.x[0]-2.6)**2 + (geom.x[1]-3)**2 + (geom.x[2]-3)**2 < 2.25"
DUMBBELL_S2 = "(geom.x[0]-5.4)**2 + (geom.x[1]-3)**2 + (geom.x[2]-3)**2 < 2.25"
DUMBBELL_NECK = "(geom.x[1]-3)**2 + (geom.x[2]-3)**2 < 0.36 && geom.x[0] > 2.6 && geom.x[0] < 5.4"


def _dumbbell_grid():
    from viva_pde_particle.grid import CartesianGrid

    return CartesianGrid((0, 0, 0), (8, 6, 6), (33, 25, 25))


def _dumbbell_description():
    from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass

    return GeometryDescription(
        name="dumbbell", dim=3, extent=(8.0, 6.0, 6.0),
        subvolumes=(SubVolume(name="cell", type="analytic",
                              expression=f"({DUMBBELL_S1}) || ({DUMBBELL_S2}) || ({DUMBBELL_NECK})"),
                    SubVolume(name="ec", type="analytic", expression="1.0")),
        surfaces=(SurfaceClass(name="pm", inside="cell", outside="ec"),))


def _dumbbell_mesh_doc(which):
    from viva_pde_particle.composites.hybrid import build_mesh_hybrid_document

    return build_mesh_hybrid_document(_ab_model(_dumbbell_grid(), which), None, 0.01, seed=1,
                                      geometry={"description": _dumbbell_description(), "region": "cell", "h": 0.3})


def _dumbbell_vcell_doc(which, correction):
    from viva_pde_particle.composites.hybrid import build_vcell_geometry_hybrid_document

    return build_vcell_geometry_hybrid_document(_ab_model(_dumbbell_grid(), which), _dumbbell_description(), 0.01,
                                                seed=1, correction=correction)


def _unstructured_doc(which):
    from viva_pde_particle.composites.hybrid import build_mesh_hybrid_document
    from viva_pde_particle.grid import CartesianGrid

    grid = CartesianGrid((0, 0, 0), (8, 8, 8), (33, 33, 33))
    return build_mesh_hybrid_document(_ab_model(grid, which), {"center": (4.0, 4.0, 4.0), "radius": 4.0, "h": 0.8},
                                      0.01, seed=1)


def _fenicsx_doc(mass):
    from viva_pde_particle.composites.examples import bimolecular_model
    from viva_pde_particle.composites.hybrid import build_hybrid_document

    return build_hybrid_document(bimolecular_model(), 0.01, seed=1, particle_init="poisson", pde_engine="fenicsx",
                                 pde_options={"mass": mass})


def _splitting_doc(scheme, k):
    from viva_pde_particle.composites.examples import bimolecular_model
    from viva_pde_particle.composites.hybrid import build_coupler_document

    return build_coupler_document(bimolecular_model(k=2.0), 0.005, k, scheme, seed=1, particle_init="poisson")


def _fokker_planck_doc():
    from viva_pde_particle.benchmarks.fokker_planck import TEST3, channel_model
    from viva_pde_particle.composites.hybrid import build_hybrid_document

    _, model, volumes = channel_model(TEST3)
    return build_hybrid_document(model, TEST3.dtau, 1, seed=1, element_volumes=volumes)


def _ex(name, **params):
    """A composite generator of ``composites.examples`` with the study's parameters."""
    def build():
        from viva_pde_particle.composites import examples

        return getattr(examples, name)(**params)
    return build


T_CONV, T_EXCH = 2.0, 5.0

DOCS = [
    # ---- cosim-vs-embedded-hybrid
    Doc("handoff-units-and-binning", "particle-diffusion", _ex("particle_diffusion", n_particles=20000, diffusion=1.0,
                                                               start="center", dt=0.01), 1.0, 0.1,
        "Study A1 baseline: 20,000 free particles from the centre cell, D = 1, binned onto the 11×11 grid."),
    Doc("field-modulated-decay", "field-modulated-decay", _ex("field_modulated_decay", n_particles=4000, k=1.0, d_a=0.0,
                                                              d_b=0.0, b_profile="linear", dt=0.01), 1.0, 0.1,
        "Study A2 baseline: static particles decaying at k·B(x) with a linear B field."),
    Doc("field-modulated-decay", "field-modulated-decay-continuum", _ex("field_modulated_decay", n_particles=20000, k=1.0,
                                                                        d_a=1.0, d_b=0.5, b_profile="gaussian", dt=0.01),
        1.0, 0.1, "Study A2 continuum-limit variant: 20,000 diffusing particles, Gaussian B, D_B = 0.5."),
    Doc("two-way-exchange", "two-way-exchange", _ex("two_way_exchange", k1=1.0, k2=0.5, b0=0.2, dt=0.01), 5.0, 0.25,
        "Study A3 baseline: A_particle ⇌ B_field (k1 = 1, k2 = 0.5, B₀ = 0.2 µM) on the 11×11×3 slab."),
    Doc("bimolecular-hybrid", "bimolecular-slab", _ex("bimolecular_hybrid", n_particles=20000, k=1.0, b0=0.5, dt=0.01),
        3.0, 0.25, "Study A4 baseline: A_particle + B_field → C_field, k = 1/(µM·s), B₀ = 0.5 µM."),
    Doc("coupling-interval-convergence", "bimolecular-k2-step1", _ex("bimolecular_hybrid", k=2.0, dt=0.005,
                                                                     step_multiplier=1), 1.6, 0.16,
        "Study A5 baseline: bimolecular slab at k = 2 with the Smoldyn step equal to the PDE step."),
    Doc("coupling-interval-convergence", "bimolecular-k2-step16", _ex("bimolecular_hybrid", k=2.0, dt=0.005,
                                                                      step_multiplier=16), 1.6, 0.16,
        "Study A5 variant: the same with Smoldyn stepping once per 16 PDE steps (coupling interval 0.08 s)."),
    Doc("performance-scaling", "bimolecular-perf", _ex("bimolecular_hybrid", n_particles=20000, dt=0.01), 1.0, 0.1,
        "Study A6 baseline: the bimolecular slab timed to t = 1 s."),
    # ---- vcell-hybrid-paper-benchmarks
    Doc("separable-calcium-sparks", "calcium-sparks-test1", _ex("calcium_sparks", test=1, dt=0.005), 5.0, 0.25,
        "Study C1 baseline: Schaff 2016 Test 1, 24 channels on the 4×6 lattice, separable kinetics, dt = 5 ms."),
    Doc("coupled-sparks-fast-diffusion", "calcium-sparks-test2", _ex("calcium_sparks", test=2, dt=0.0002), 3.0, 0.05,
        "Study C2 baseline: Schaff 2016 Test 2, opening rate k_on·U/U0 coupled to the field, fast diffusion, dt = 0.2 ms."),
    Doc("single-channel-fokker-planck", "fokker-planck-test3", _fokker_planck_doc, 30.0, 1.0,
        "Study C3: one trial of Test 3 (fast diffusion, α = β = 1, a = 24, τ = 30, Δτ = 0.002); the ensembles ran "
        "as blocks earlier (runs/compose/)."),
    # ---- hybrid-generalizability
    Doc("sbml-spatial-hybrid", "two-way-exchange-from-sbml", _ex("two_way_exchange", k1=1.0, k2=0.5, b0=0.2, dt=0.01),
        2.0, 0.25, "Study B1 baseline: the two-way exchange model as its SBML round trip defines it, t = 2 s."),
    Doc("fenicsx-same-problem", "bimolecular-fenicsx-lumped", lambda: _fenicsx_doc("lumped"), 1.0, 0.25,
        "Study B2 baseline: the bimolecular slab with the FEniCSx Q1 engine (lumped mass) in place of FV."),
    Doc("fenicsx-same-problem", "bimolecular-fenicsx-consistent", lambda: _fenicsx_doc("consistent"), 1.0, 0.25,
        "Study B2 variant: FEniCSx Q1 with the consistent mass matrix."),
    Doc("unstructured-geometry", "ball-conversion", lambda: _unstructured_doc("conversion"), T_CONV, 0.25,
        "Study B2a baseline: A_p → B_f in a gmsh ball (R = 4 µm, h = 0.8), FEniCSx P1 + Smoldyn confined by the mesh.",
        mesh=True),
    Doc("unstructured-geometry", "ball-exchange", lambda: _unstructured_doc("exchange"), T_EXCH, 0.25,
        "Study B2a variant: A_p ⇌ B_f in the same ball.", mesh=True),
    Doc("sphere-cosim-vs-native", "ball-conversion", lambda: _ball_mesh_doc("conversion"), T_CONV, 0.25,
        "Study B2b co-simulation: A_p → B_f in the ball (centre 4.5, R = 4, h = 0.8) that native VCell also ran.",
        mesh=True),
    Doc("sphere-cosim-vs-native", "ball-exchange", lambda: _ball_mesh_doc("exchange"), T_EXCH, 0.25,
        "Study B2b co-simulation: A_p ⇌ B_f in the same ball.", mesh=True),
    Doc("near-membrane-fields", "ball-conversion", lambda: _ball_mesh_doc("conversion"), T_CONV, 0.25,
        "Study B2c: the B2b ball conversion, whose near-membrane B profile the study measures.", mesh=True),
    Doc("cosim-boundary-transfer", "ball-conversion-grid-sphere", lambda: _ball_mesh_doc("conversion",
                                                                                         particle_transfer="grid",
                                                                                         membrane="sphere"),
        T_CONV, 0.25, "Study B2d variant grid_sphere: Pᵀ grid transfer, particles confined by the analytic sphere.",
        mesh=True),
    Doc("cosim-boundary-transfer", "ball-conversion-positions-sphere", lambda: _ball_mesh_doc("conversion",
                                                                                              particle_transfer="positions",
                                                                                              membrane="sphere"),
        T_CONV, 0.25, "Study B2d variant positions_sphere: exact P1 load from positions, analytic sphere.", mesh=True),
    Doc("cosim-boundary-transfer", "ball-conversion-grid-mesh", lambda: _ball_mesh_doc("conversion",
                                                                                       particle_transfer="grid",
                                                                                       membrane="mesh"),
        T_CONV, 0.25, "Study B2d variant grid_mesh: Pᵀ grid transfer, particles confined by the mesh boundary.",
        mesh=True),
    Doc("cosim-boundary-transfer", "ball-conversion-positions-mesh", lambda: _ball_mesh_doc("conversion",
                                                                                            particle_transfer="positions",
                                                                                            membrane="mesh"),
        T_CONV, 0.25, "Study B2d variant positions_mesh: exact P1 load from positions, mesh boundary.", mesh=True),
    Doc("vcell-geometry-cosim", "ball-conversion-corrected", lambda: _ball_vcell_doc("conversion", "adapters+volumes"),
        T_CONV, 0.25, "Study B2e: A_p → B_f on VCell's own realization of the ball, adapters + accessible volumes.",
        vcell=True),
    Doc("vcell-geometry-cosim", "ball-conversion-none", lambda: _ball_vcell_doc("conversion", "none"), T_CONV, 0.25,
        "Study B2e: the same with no correction (VCell's full voxel volumes).", vcell=True),
    Doc("vcell-geometry-cosim", "ball-exchange-corrected", lambda: _ball_vcell_doc("exchange", "adapters+volumes"),
        T_EXCH, 0.25, "Study B2e: A_p ⇌ B_f on VCell's realization, adapters + accessible volumes.", vcell=True),
    Doc("splitting-schemes", "exchange-strang-k2", lambda: _splitting_doc("strang", 2), 1.6, 0.16,
        "Study B2g: the k = 2 bimolecular slab under the SplittingCoordinator, Strang splitting, coupling 2·dt."),
    Doc("splitting-schemes", "exchange-jacobi-k2", lambda: _splitting_doc("jacobi", 2), 1.6, 0.16,
        "Study B2g: the same under Jacobi splitting."),
    Doc("dumbbell-three-ways", "dumbbell-mesh-conversion", lambda: _dumbbell_mesh_doc("conversion"), T_CONV, 0.25,
        "Study B2f mesh path: A_p → B_f in the dumbbell (two balls and a neck), Netgen h = 0.3.", mesh=True),
    Doc("dumbbell-three-ways", "dumbbell-mesh-exchange", lambda: _dumbbell_mesh_doc("exchange"), T_EXCH, 0.25,
        "Study B2f mesh path: A_p ⇌ B_f in the dumbbell.", mesh=True),
    Doc("dumbbell-three-ways", "dumbbell-vcell-corrected-conversion",
        lambda: _dumbbell_vcell_doc("conversion", "adapters+volumes"), T_CONV, 0.25,
        "Study B2f VCell-realization path: A_p → B_f, adapters + accessible volumes.", vcell=True),
    Doc("dumbbell-three-ways", "dumbbell-vcell-none-conversion", lambda: _dumbbell_vcell_doc("conversion", "none"),
        T_CONV, 0.25, "Study B2f VCell-realization path: A_p → B_f, no correction.", vcell=True),
    Doc("dumbbell-three-ways", "dumbbell-vcell-corrected-exchange",
        lambda: _dumbbell_vcell_doc("exchange", "adapters+volumes"), T_EXCH, 0.25,
        "Study B2f VCell-realization path: A_p ⇌ B_f, adapters + accessible volumes.", vcell=True),
]


def _investigation(study: str) -> str:
    import yaml

    return str(yaml.safe_load((STUDIES / study / "study.yaml").read_text()).get("investigation") or "")


def tags_for(d: Doc) -> list[str]:
    inv = _investigation(d.study)
    return [TAG, d.study, *([inv] if inv else []), *d.tags]


def build(d: Doc) -> Path:
    from viva_pde_particle.compose_runner import _jsonable
    from viva_pde_particle.ensemble import write_omex
    from viva_pde_particle.viz3d import attach_recorder

    doc = attach_recorder(d.build(), f"{d.name}.fenics", d.output_dt, particles=True, source=f"{d.study}/{d.name}")
    d.omex.parent.mkdir(parents=True, exist_ok=True)
    return write_omex({"state": _jsonable(doc)}, d.omex, name=d.name)


def load_record() -> dict:
    return json.loads(RECORD.read_text()) if RECORD.exists() else {"url": None, "simulator": None, "submissions": []}


def save_record(rec: dict) -> None:
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(rec, indent=1) + "\n")


FAILED = {"failed", "cancelled", "timeout", "out_of_memory"}


def submit(docs: list[Doc], url: str, retry: bool = False) -> None:
    """Submit each document once; with ``retry``, submit again those whose run failed (the earlier simulation ids
    stay in the record under ``previous``)."""
    from viva_pde_particle.compose_client import SIMULATOR, ComposeClient

    client = ComposeClient(url)
    rec = load_record()
    rec["url"], rec["simulator"] = client.url, client.simulator
    entries = {(s["study"], s["name"]): s for s in rec["submissions"]}
    for d in docs:
        entry = entries.get((d.study, d.name))
        if entry is not None and not (retry and entry["status"] in FAILED):
            print(f"{d.study}/{d.name}: already submitted as {entry['simulation_id']} ({entry['status']}); skipping")
            continue
        if not d.omex.exists():
            print(f"{d.study}/{d.name}: no document at {d.omex}; build it first", file=sys.stderr)
            continue
        tags = tags_for(d)
        sim_id = client.submit(d.omex, interval=d.t_end, tags=tags, description=d.description)
        if entry is None:
            entry = {"study": d.study, "name": d.name}
            rec["submissions"].append(entry)
        else:
            entry.setdefault("previous", []).append({k: entry.get(k) for k in ("simulation_id", "status", "error")
                                                     if entry.get(k) is not None})
            entry.pop("error", None)
            entry.pop("slurm_job", None)
        entry.update({"simulation_id": sim_id, "tags": tags, "interval": d.t_end, "output_dt": d.output_dt,
                      "description": d.description, "status": "submitting"})
        save_record(rec)
        print(f"{d.study}/{d.name}: simulation {sim_id} ({SIMULATOR}, interval {d.t_end})")


def status(url: str) -> None:
    from viva_pde_particle.compose_client import ComposeClient

    client = ComposeClient(url)
    rec = load_record()
    for s in rec["submissions"]:
        r = client.status(s["simulation_id"])
        s["status"] = r["status"]
        if r.get("slurmjobid"):
            s["slurm_job"] = r["slurmjobid"]
        if r.get("error_message"):
            s["error"] = r["error_message"]
        print(f"{s['simulation_id']:>6}  {s['status']:<12} {s['study']}/{s['name']}")
    save_record(rec)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="compose_submit_studies", description=__doc__.split("\n\n")[0])
    p.add_argument("--url", default=None)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("names", nargs="*", help="<study>/<name> or <name>; default: every document")
    b.add_argument("--vcell", action="store_true", help="only the documents that need libvcell to build (dev env)")
    b.add_argument("--mesh", action="store_true", help="only the unstructured-mesh documents (build in the image)")
    s = sub.add_parser("submit")
    s.add_argument("names", nargs="*")
    s.add_argument("--retry", action="store_true", help="also resubmit documents whose run failed")
    sub.add_parser("status")
    sub.add_parser("list")
    args = p.parse_args(argv)
    from viva_pde_particle.compose_client import DEFAULT_URL

    url = args.url or DEFAULT_URL
    if args.command == "status":
        status(url)
        return 0
    docs = DOCS
    names = getattr(args, "names", [])
    if names:
        docs = [d for d in DOCS if d.name in names or f"{d.study}/{d.name}" in names]
    if args.command == "build":
        docs = [d for d in docs if d.vcell == args.vcell and d.mesh == args.mesh]
    if args.command == "list":
        for d in docs:
            print(f"{d.study}/{d.name}  t_end={d.t_end}  output_dt={d.output_dt}"
                  f"{'  [vcell]' if d.vcell else ''}{'  [mesh]' if d.mesh else ''}")
        return 0
    if args.command == "build":
        for d in docs:
            print(f"building {d.study}/{d.name} ...", flush=True)
            print(f"  wrote {build(d)}")
        return 0
    submit(docs, url, retry=args.retry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
