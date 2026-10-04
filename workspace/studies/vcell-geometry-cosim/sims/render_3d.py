"""Study B2e 3D figures: the ball on VCell's own geometry, native VCell and the three corrections (Phase 8e).

One seed per path runs the conversion case (20,000 A_p → B_f, k = 0.5) to 2 s:
- native VCell;
- the co-simulation on VCell's realization with correction ``none``, ``adapters`` and
  ``adapters+volumes``.

Each is recorded every 0.25 s into ``runs/<path>.fenics`` (vcell-fenics results bundles;
gitignored, regenerated here), with VCell's smooth membrane. Native starts with 20,000 molecules
as well: its initial concentration is converted with VCell's staircase volume, the volume VCell
places particles over.

- ``viz/vcell_geometry_3d.png``: the four paths at 2 s on one colour range;
- ``viz/vcell_geometry_3d_movie.gif``: the corrected co-sim over time;
- ``viz/vcell_geometry_3d_interactive.html``: all four, interactive (three.js).

Two stages (native VCell needs the ``dev`` env; PyVista is in ``default``)::

    pixi run -e dev python workspace/studies/vcell-geometry-cosim/sims/render_3d.py record
    pixi run python workspace/studies/vcell-geometry-cosim/sims/render_3d.py render

Open the runs in VCell's vtk.wasm viewer: ``pixi run view3d workspace/studies/vcell-geometry-cosim/runs/*.fenics``.
"""
from __future__ import annotations

import sys
from pathlib import Path

SIMS = Path(__file__).resolve().parent
sys.path.insert(0, str(SIMS))

import run as b2e  # noqa: E402

RUNS, VIZ = b2e.STUDY_DIR / "runs", b2e.STUDY_DIR / "viz"
T_END, OUT_DT, SEED = 2.0, 0.25, 1
LABELS = {"native": "native VCell", "none": "co-sim, none", "adapters": "co-sim, adapters",
          "adapters_volumes": "co-sim, adapters+volumes"}


def record() -> None:
    from viva_pde_particle.composites.hybrid import build_vcell_geometry_hybrid_document, run_document
    from viva_pde_particle.reference.vcell_native import run_native
    from viva_pde_particle.viz3d import attach_recorder, write_native_bundle

    real = b2e._realization()
    membranes = {"pm": real.boundary_triangles("cell")}
    model = b2e.model("conversion")
    nat = run_native(model, t_end=T_END, dt=b2e.DT, output_dt=OUT_DT, seed=SEED, geometry=b2e.description(),
                     domain_volume=real.pde_volume("cell"))
    write_native_bundle(RUNS / "native.fenics", nat, b2e.GRID, mask=real.node_mask("cell"), membranes=membranes)
    for key in ("none", "adapters", "adapters_volumes"):
        corr = key.replace("_", "+")
        doc = build_vcell_geometry_hybrid_document(model, b2e.description(), b2e.DT, seed=SEED, correction=corr,
                                                   realization=real)
        attach_recorder(doc, RUNS / f"{key}.fenics", OUT_DT, membranes=membranes, source=LABELS[key])
        run_document(doc, T_END, b2e.DT, OUT_DT)


def render() -> None:
    from viva_pde_particle.viz3d.html import bundle_html
    from viva_pde_particle.viz3d.static import render_gif, render_png
    from viva_pde_particle.viz3d.study import logged_run, write_meta

    bundles = {LABELS[k]: RUNS / f"{k}.fenics" for k in LABELS}
    with logged_run(b2e.WORKSPACE_ROOT, b2e.STUDY_SLUG, b2e.INVESTIGATION_SLUG, b2e.SPEC_ID,
                    f"{b2e.STUDY_SLUG}-3d", {"seed": SEED, "t_end": T_END, "output_dt": OUT_DT}) as run_id:
        png = render_png(bundles, "B", T_END, VIZ / "vcell_geometry_3d.png", window=(380, 360),
                         title="B2e: B at t = 2 s, seed 1")
        gif = render_gif(RUNS / "adapters_volumes.fenics", "B", VIZ / "vcell_geometry_3d_movie.gif")
        page = bundle_html(bundles, "B", VIZ / "vcell_geometry_3d_interactive.html",
                           title="B2e ball on VCell's geometry: native and three corrections", max_particles=1500)
        sims = ("One seed per path, conversion (20,000 A → B, k = 0.5), recorded every 0.25 s to 2 s into results "
                "bundles (sims/render_3d.py). Native particles are placed at their nodes from VCell's per-node "
                "counts.")
        write_meta(png, title="B at 2 s: native VCell and the three corrections", run_id=run_id, simulations=sims,
                   caption="Top: mid-plane slice with VCell's smooth membrane. Bottom: B on the cut-away membrane "
                           "with the A particles.",
                   interpretation="Every path converts the same 20,000 molecules. Native and 'none' spread B over "
                                  "VCell's staircase (266.7 µm³); 'adapters+volumes' over the accessible volume "
                                  "(261.9 µm³, the particles' smooth domain), hence its slightly higher level. "
                                  "'adapters' divides by accessible volumes but integrates over full voxels, so "
                                  "its sources are overcounted (+1.3% in the 32-seed metrics). The ~1.4% shell "
                                  "bias of native and 'none' is below single-seed noise; the metrics resolve it.")
        write_meta(gif, title="The corrected co-sim over time", run_id=run_id, simulations=sims,
                   caption="Mid-plane slice of B, membrane, A particles.",
                   interpretation="B rises uniformly as A converts, with no depletion at the membrane.")
        write_meta(page, title="Interactive: native and three corrections", run_id=run_id, simulations=sims,
                   caption="Time slider, run selector, membrane clipping, slices and particles (three.js).",
                   interpretation="Explore any path at any recorded time.")


if __name__ == "__main__":
    {"record": record, "render": render}[sys.argv[1] if len(sys.argv) > 1 else "render"]()
