"""Study B2f 3D figures: one seed per solver path, recorded into results bundles and rendered (Phase 8e).

The four paths of ``run.py`` (native VCell, the co-sim on VCell's geometry with corrections ``none``
and ``adapters+volumes``, and the co-sim on a Netgen mesh) run the conversion case (20,000 A_p → B_f,
k = 0.5) to 2 s with seed 1. Each is recorded every 0.25 s into
``runs/<path>.fenics``, a vcell-fenics results bundle; these are gitignored and regenerated here.

From the bundles:
- ``viz/dumbbell_3d.png``: the four paths at 2 s on one colour range (mid-plane slice and membrane);
- ``viz/dumbbell_3d_movie.gif``: the corrected co-sim over time;
- ``viz/dumbbell_3d_interactive.html``: an interactive three.js page with all four runs.

Each figure gets a ``.meta.json`` sidecar. The native bundle has fields only (VCell writes particle
counts, not positions). To open the runs in VCell's vtk.wasm viewer:

    pixi run view3d workspace/studies/dumbbell-three-ways/runs/*.fenics

Two stages, because native VCell needs the ``dev`` pixi env and PyVista is in ``default``::

    pixi run -e dev python workspace/studies/dumbbell-three-ways/sims/render_3d.py record
    pixi run python workspace/studies/dumbbell-three-ways/sims/render_3d.py render
"""
from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path

SIMS = Path(__file__).resolve().parent
sys.path.insert(0, str(SIMS))

import run as b2f  # noqa: E402

STUDY_DIR, WORKSPACE_ROOT = b2f.STUDY_DIR, b2f.WORKSPACE_ROOT
RUNS, VIZ = STUDY_DIR / "runs", STUDY_DIR / "viz"
T_END, OUT_DT, SEED = 2.0, 0.25, 1
LABELS = {"native": "native VCell", "vcell_none": "co-sim, VCell geometry, none",
          "vcell_corrected": "co-sim, VCell geometry, adapters+volumes", "mesh": "co-sim, Netgen mesh"}


def record_all() -> dict[str, Path]:
    from viva_pde_particle.composites.hybrid import (
        build_mesh_hybrid_document,
        build_vcell_geometry_hybrid_document,
        run_document,
    )
    from viva_pde_particle.geometry import realize_fenics
    from viva_pde_particle.reference.vcell_native import run_native
    from viva_pde_particle.viz3d import attach_recorder, write_native_bundle

    real = b2f.vcell_realization()
    vcell_membrane = {"pm": real.boundary_triangles("cell")}
    model = b2f.model("conversion")
    paths = {k: RUNS / f"{k}.fenics" for k in LABELS}
    # VCell places an initial particle concentration over its staircase volume, so converting the
    # 20,000 molecules with that volume starts native with 20,000 as well (with the smooth volume,
    # as run.py does, it starts with ~22,400; run.py's metrics are relative to each run's own A0).
    nat = run_native(model, t_end=T_END, dt=b2f.DT, output_dt=OUT_DT, seed=SEED, geometry=b2f.description(),
                     domain_volume=real.pde_volume("cell"))
    write_native_bundle(paths["native"], nat, b2f.GRID, mask=real.node_mask("cell"), membranes=vcell_membrane)
    for key, corr in (("vcell_none", "none"), ("vcell_corrected", "adapters+volumes")):
        doc = build_vcell_geometry_hybrid_document(model, b2f.description(), b2f.DT, seed=SEED, correction=corr,
                                                   realization=real)
        attach_recorder(doc, paths[key], OUT_DT, membranes=vcell_membrane, source=LABELS[key])
        run_document(doc, T_END, b2f.DT, OUT_DT)
    netgen = realize_fenics(b2f.description(), b2f.H_MESH)
    doc = build_mesh_hybrid_document(model, None, b2f.DT, seed=SEED,
                                     geometry={"description": b2f.description(), "region": "cell", "h": b2f.H_MESH})
    attach_recorder(doc, paths["mesh"], OUT_DT, membranes={"pm": netgen.boundary_triangles("cell")},
                    source=LABELS["mesh"])
    run_document(doc, T_END, b2f.DT, OUT_DT)
    return paths


def main(stage: str = "all") -> int:
    from vivarium_workbench.lib.run_log import append_run_event

    if stage == "record":
        record_all()
        print(f"[{b2f.STUDY_SLUG}] bundles in {RUNS}")
        return 0
    run_id = uuid.uuid4().hex
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "started", "spec_id": b2f.SPEC_ID, "label": f"{b2f.STUDY_SLUG}-3d",
        "started_at": time.time(), "status": "running", "emitter": "none", "origin": "canonical_run",
        "study_slug": b2f.STUDY_SLUG, "investigation_slug": b2f.INVESTIGATION_SLUG,
        "params": {"seed": SEED, "t_end": T_END, "output_dt": OUT_DT},
    })
    try:
        from viva_pde_particle.viz3d.html import bundle_html
        from viva_pde_particle.viz3d.static import render_gif, render_png

        paths = record_all() if stage == "all" else {k: RUNS / f"{k}.fenics" for k in LABELS}
        bundles = {LABELS[k]: p for k, p in paths.items()}
        render_png(bundles, "B", T_END, VIZ / "dumbbell_3d.png", window=(380, 360),
                   title="B2f: B at t = 2 s, seed 1")
        render_gif(paths["vcell_corrected"], "B", VIZ / "dumbbell_3d_movie.gif")
        bundle_html(bundles, "B", VIZ / "dumbbell_3d_interactive.html", title="B2f dumbbell: B and A particles, four solver paths",
                    max_particles=1500)
        sims = ("One seed per path, conversion (20,000 A → B, k = 0.5), recorded every 0.25 s to 2 s into "
                "vcell-fenics results bundles (sims/render_3d.py); the native bundle has fields only.")
        meta = {
            "dumbbell_3d.png": ("B at 2 s: four solver paths on one colour range",
                                "Top: mid-plane slice with the membrane outline. Bottom: B on the cut-away membrane "
                                "with the A particles.",
                                "Every path converts the same 20,000 molecules, so the level of B shows each path's "
                                "PDE volume: VCell's staircase (27.7 µm³; native and 'none'), the accessible volume "
                                "(24.6 µm³; 'adapters+volumes', which matches the particles' smooth domain) and the "
                                "Netgen mesh (27.3 µm³). The ~1% near-membrane depletion of the uncorrected paths is "
                                "below single-seed noise; the 16-seed metrics resolve it. The staircase panels also "
                                "show the exterior band the particles see (fields extended by the fold map)."),
            "dumbbell_3d_movie.gif": ("The corrected co-sim over time", "Mid-plane slice of B, membrane, A particles.",
                                "B rises uniformly as A converts; no build-up at the membrane."),
            "dumbbell_3d_interactive.html": ("Interactive: four solver paths", "Time slider, run selector, membrane clipping, "
                                 "slices and particles (three.js).", "Explore any path at any recorded time."),
        }
        for name, (title, caption, interp) in meta.items():
            (VIZ / name).with_suffix(".meta.json").write_text(json.dumps(
                {"title": title, "caption": caption, "simulations": sims, "interpretation": interp,
                 "source_run_id": run_id}, indent=2) + "\n")
    except BaseException:
        append_run_event(WORKSPACE_ROOT, {"run_id": run_id, "event": "completed", "completed_at": time.time(),
                                          "status": "failed"})
        raise
    append_run_event(WORKSPACE_ROOT, {"run_id": run_id, "event": "completed", "completed_at": time.time(),
                                      "status": "completed"})
    print(f"[{b2f.STUDY_SLUG}] 3D figures written to {VIZ}; bundles in {RUNS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else "all"))
