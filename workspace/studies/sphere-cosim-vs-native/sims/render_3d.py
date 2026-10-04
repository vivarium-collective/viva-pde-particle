"""Study B2b 3D figures: the ball, native VCell vs the co-simulation on a body-fitted mesh (Phase 8e).

One seed per solver and case, recorded every 0.25 s into ``runs/<solver>_<case>.fenics``
(vcell-fenics results bundles; gitignored, regenerated here):
- conversion (20,000 A_p → B_f, k = 0.5) to 2 s;
- exchange (A_p ⇌ B_f, k1 = 1, k2 = 0.5, B₀ = 0.2 µM) to 5 s.

Native VCell runs on its Cartesian grid, as in ``run.py``. Its membrane in the figures is VCell's
own smooth surface for the ball, and its particles are placed at their nodes from VCell's
per-node counts. The co-simulation runs on the gmsh P1 mesh (h = 0.8), with the mesh boundary as
its membrane.

- ``viz/sphere_3d.png``: conversion at 2 s and exchange at 5 s, both solvers;
- ``viz/sphere_3d_movie.gif``: the co-sim exchange over time (A created from B);
- ``viz/sphere_3d_interactive.html``: all four runs, interactive (three.js).

Two stages (native VCell needs the ``dev`` env; PyVista is in ``default``)::

    pixi run -e dev python workspace/studies/sphere-cosim-vs-native/sims/render_3d.py record
    pixi run python workspace/studies/sphere-cosim-vs-native/sims/render_3d.py render

Open the runs in VCell's vtk.wasm viewer: ``pixi run view3d workspace/studies/sphere-cosim-vs-native/runs/*.fenics``.
"""
from __future__ import annotations

import sys
from pathlib import Path

SIMS = Path(__file__).resolve().parent
sys.path.insert(0, str(SIMS))

import run as b2b  # noqa: E402

RUNS, VIZ = b2b.STUDY_DIR / "runs", b2b.STUDY_DIR / "viz"
OUT_DT, SEED = 0.25, 1
T_END = {"conversion": 2.0, "exchange": 5.0}
LABELS = {f"{s}_{c}": f"{'native VCell' if s == 'native' else 'co-sim (mesh)'}, {c}"
          for c in T_END for s in ("native", "cosim")}


def record() -> None:
    from viva_pde_particle.composites.hybrid import build_mesh_hybrid_document, run_document
    from viva_pde_particle.geometry import sphere_in_box
    from viva_pde_particle.geometry.vcell_fv import VCellFVRealization
    from viva_pde_particle.mesh import boundary_triangles, mesh_space
    from viva_pde_particle.reference.vcell_native import run_native
    from viva_pde_particle.viz3d import attach_recorder, write_native_bundle

    real = VCellFVRealization(sphere_in_box(b2b.CENTER, b2b.R, (9.0, 9.0, 9.0)), b2b.GRID)
    for case, t_end in T_END.items():
        model = b2b.model(case)
        nat = run_native(model, t_end=t_end, dt=b2b.DT, output_dt=OUT_DT, seed=SEED, geometry=b2b.NATIVE_GEOMETRY)
        write_native_bundle(RUNS / f"native_{case}.fenics", nat, b2b.GRID, mask=real.node_mask("cell"),
                            membranes={"pm": real.boundary_triangles("cell")})
        doc = build_mesh_hybrid_document(model, b2b.MESH_SPHERE, b2b.DT, seed=SEED)
        membrane = boundary_triangles(mesh_space(doc["pde"]["config"]["mesh"]).V.mesh)
        attach_recorder(doc, RUNS / f"cosim_{case}.fenics", OUT_DT, membranes={"pm": membrane},
                        source=LABELS[f"cosim_{case}"])
        run_document(doc, t_end, b2b.DT, OUT_DT)


def render() -> None:
    from viva_pde_particle.viz3d.html import bundle_html
    from viva_pde_particle.viz3d.static import render_gif, render_png
    from viva_pde_particle.viz3d.study import logged_run, write_meta

    with logged_run(b2b.WORKSPACE_ROOT, b2b.STUDY_SLUG, b2b.INVESTIGATION_SLUG, b2b.SPEC_ID, f"{b2b.STUDY_SLUG}-3d",
                    {"seed": SEED, "output_dt": OUT_DT}) as run_id:
        sims = ("One seed per solver and case, recorded every 0.25 s into results bundles (sims/render_3d.py): "
                "conversion to 2 s, exchange to 5 s. Native particles are placed at their nodes from VCell's "
                "per-node counts; its membrane is VCell's smooth surface.")
        pngs = {}
        for case, t_end in T_END.items():  # one colour range per case
            pngs[case] = render_png({LABELS[f"{s}_{case}"]: RUNS / f"{s}_{case}.fenics" for s in ("native", "cosim")},
                                    "B", t_end, VIZ / f"sphere_3d_{case}.png",
                                    window=(400, 360), title=f"B2b {case}: t = {t_end:g} s, seed 1")
        gif = render_gif(RUNS / "cosim_exchange.fenics", "B", VIZ / "sphere_3d_movie.gif")
        page = bundle_html({LABELS[k]: RUNS / f"{k}.fenics" for k in LABELS}, "B", VIZ / "sphere_3d_interactive.html",
                           title="B2b ball: native VCell vs co-sim on a body-fitted mesh", max_particles=1500)
        write_meta(pngs["conversion"], title="Conversion at 2 s: native VCell vs co-sim", run_id=run_id,
                   simulations=sims,
                   caption="Top: mid-plane slice with the membrane. Bottom: B on the cut-away membrane with the A "
                           "particles.",
                   interpretation="Both solvers spread B uniformly. The staircase panel shows VCell's PDE domain; "
                                  "the co-sim's is the body-fitted mesh inside the same ball.")
        write_meta(pngs["exchange"], title="Exchange at 5 s: native VCell vs co-sim", run_id=run_id,
                   simulations=sims, caption="As above, near the exchange steady state (A/B = 0.5).",
                   interpretation="Native VCell holds fewer A + B molecules by now (its −6.8% exchange balance in "
                                  "the metrics: binning loss at exterior nodes, vcell-fvsolver#25); the co-sim "
                                  "conserves them.")
        write_meta(gif, title="Co-sim exchange over time", run_id=run_id, simulations=sims,
                   caption="Mid-plane slice of B, membrane, A particles created from B.",
                   interpretation="A appears uniformly from B and relaxes to the steady ratio.")
        write_meta(page, title="Interactive: both solvers, both cases", run_id=run_id, simulations=sims,
                   caption="Time slider, run selector, membrane clipping, slices and particles (three.js).",
                   interpretation="Explore either solver at any recorded time.")


if __name__ == "__main__":
    {"record": record, "render": render}[sys.argv[1] if len(sys.argv) > 1 else "render"]()
