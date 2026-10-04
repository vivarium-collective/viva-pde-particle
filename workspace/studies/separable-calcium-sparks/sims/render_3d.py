"""Study C1 3D figures: calcium sparks in the slab, co-simulation vs native VCell (Phase 8e).

One trial (seed 1) of Test 1 per solver at Δt = 0.005, recorded every 0.1 s to 3 s into
``runs/<solver>.fenics`` (vcell-fenics results bundles; gitignored, regenerated here). Recorded per
output: calcium U on the 10.1 × 2.1 × 0.5 µm slab, and the 24 channels as particles, closed (C) or
open (O). Native VCell writes particles as counts per node, which is exact here: the channels sit
on nodes and do not diffuse.

- ``viz/sparks_3d.png``: U in the channel plane (z = 0) at 1, 2 and 3 s, co-sim and native, with
  the channels;
- ``viz/sparks_3d_movie.gif``: the co-sim trial over time;
- ``viz/sparks_3d_interactive.html``: both trials, interactive (three.js).

Two stages (native VCell needs the ``dev`` env; PyVista is in ``default``)::

    pixi run -e dev python workspace/studies/separable-calcium-sparks/sims/render_3d.py record
    pixi run python workspace/studies/separable-calcium-sparks/sims/render_3d.py render

Open the trials in VCell's vtk.wasm viewer: ``pixi run view3d workspace/studies/separable-calcium-sparks/runs/*.fenics``.
"""
from __future__ import annotations

import sys
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[2]
RUNS, VIZ = STUDY_DIR / "runs", STUDY_DIR / "viz"
STUDY_SLUG, INVESTIGATION_SLUG = "separable-calcium-sparks", "vcell-hybrid-paper-benchmarks"
DT, T_END, OUT_DT, SEED = 0.005, 3.0, 0.1, 1
LABELS = {"cosim": "co-simulation", "native": "native VCell"}
COLORS = {"C": "deepskyblue", "O": "lime"}
CHANNEL_PLANE = (5.05, 1.05, 1e-3)  # just inside the slab: a slice exactly on its face is empty


def record() -> None:
    from viva_pde_particle.benchmarks.calcium_sparks import TEST1, calcium_sparks_model
    from viva_pde_particle.composites.hybrid import build_hybrid_document, run_document
    from viva_pde_particle.reference.vcell_native import run_native
    from viva_pde_particle.viz3d import attach_recorder, write_native_bundle

    model = calcium_sparks_model(TEST1)
    doc = attach_recorder(build_hybrid_document(model, DT, seed=SEED), RUNS / "cosim.fenics", OUT_DT,
                          region="cyt", source=LABELS["cosim"])
    run_document(doc, T_END, DT, OUT_DT)
    nat = run_native(model, t_end=T_END, dt=DT, output_dt=OUT_DT, seed=SEED)
    write_native_bundle(RUNS / "native.fenics", nat, model.grid, region="cyt")


def render() -> None:
    from viva_pde_particle.viz3d.html import bundle_html
    from viva_pde_particle.viz3d.static import render_gif, render_png
    from viva_pde_particle.viz3d.study import logged_run, write_meta

    paths = {k: RUNS / f"{k}.fenics" for k in LABELS}
    with logged_run(WORKSPACE_ROOT, STUDY_SLUG, INVESTIGATION_SLUG, "viva_pde_particle.benchmarks.calcium_sparks",
                    f"{STUDY_SLUG}-3d", {"seed": SEED, "dt": DT, "t_end": T_END}) as run_id:
        panels = {f"{LABELS[k]}, t = {t:g} s": (paths[k], t) for k in LABELS for t in (1.0, 2.0, 3.0)}
        png = render_png(panels, "U", None, VIZ / "sparks_3d.png", panels=("slice",), stack="rows", window=(900, 250),
                         colors=COLORS, cmap="inferno", slice_origin=CHANNEL_PLANE, log_scale=True)
        gif = render_gif(paths["cosim"], "U", VIZ / "sparks_3d_movie.gif", view="slice", colors=COLORS,
                         cmap="inferno", window=(900, 300), fps=6, slice_origin=CHANNEL_PLANE, log_scale=True)
        page = bundle_html({LABELS[k]: p for k, p in paths.items()}, "U", VIZ / "sparks_3d_interactive.html",
                           title="C1 calcium sparks: U and the 24 channels (closed C, open O), one trial per solver",
                           colors=COLORS, camera="top", max_frames=16, slice_origin=CHANNEL_PLANE)
        sims = (f"One trial (seed {SEED}) of Test 1 per solver, Δt = {DT}, recorded every {OUT_DT} s to {T_END:g} s "
                "into results bundles (sims/render_3d.py). Native channel states come from VCell's per-node counts "
                "(exact: channels sit on nodes).")
        write_meta(png, title="Calcium sparks in the channel plane", run_id=run_id, simulations=sims,
                   caption="U at z = 0 (log colour scale) at 1, 2 and 3 s; channels closed (blue) or open (green). "
                           "Co-sim channels sit inside their node's voxel (Smoldyn places them there; D = 0), native "
                           "ones at the node (from VCell's per-node counts).",
                   interpretation="Each open channel drives a local calcium spark that decays by diffusion and "
                                  "pumping once it closes. Trials are independent samples, so co-sim and native "
                                  "differ channel by channel; their statistics agree (see the ε(N) figure).")
        write_meta(gif, title="Sparks over time (co-simulation)", run_id=run_id, simulations=sims,
                   caption="U at z = 0 and channel states, every 0.1 s.",
                   interpretation="Sparks appear where channels open and fade after they close.")
        write_meta(page, title="Interactive: sparks, both solvers", run_id=run_id, simulations=sims,
                   caption="Time slider, solver selector, slices and channels (three.js).",
                   interpretation="Explore either trial at any recorded time.")


if __name__ == "__main__":
    {"record": record, "render": render}[sys.argv[1] if len(sys.argv) > 1 else "render"]()
