"""Study C2 3D figures: coupled channels with fast diffusion, co-simulation vs native VCell (Phase 8e).

One trial (seed 1) of Test 2 per solver (D = 1000 µm²/s, opening at k_on·U/U₀, Δt = 0.2 ms), recorded
every 0.05 s to 3 s into ``runs/<solver>.fenics`` (vcell-fenics results bundles; gitignored,
regenerated here). Recorded per output: calcium U on the slab, and the 24 channels as particles,
closed (C) or open (O). Native channel states come from VCell's per-node counts (exact: channels
sit on nodes and do not diffuse).

At resting calcium a channel opens rarely (k_on = 0.1/s against k_off = 5/s), but opening speeds
up with U (k_on·U/U₀), so openings run away into bursts. In this trial up to 15 (co-sim) and 19
(native) of the 24 channels are open at once and U climbs to ~10 µM. Diffusion keeps U almost
uniform throughout (spread < 0.2 µM across the slab), the well-mixed limit this study tests. The
static figure shows, per solver, the three recorded times with the most open channels.

- ``viz/coupled_sparks_3d.png``: U in the channel plane (z = 0) at those times, both solvers;
- ``viz/coupled_sparks_3d_movie.gif``: the co-sim trial over time;
- ``viz/coupled_sparks_3d_interactive.html``: both trials, interactive (three.js).

Two stages (native VCell needs the ``dev`` env; PyVista is in ``default``)::

    pixi run -e dev python workspace/studies/coupled-sparks-fast-diffusion/sims/render_3d.py record
    pixi run python workspace/studies/coupled-sparks-fast-diffusion/sims/render_3d.py render

Open the trials in VCell's vtk.wasm viewer:
``pixi run view3d workspace/studies/coupled-sparks-fast-diffusion/runs/*.fenics``.
"""
from __future__ import annotations

import sys
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[2]
RUNS, VIZ = STUDY_DIR / "runs", STUDY_DIR / "viz"
STUDY_SLUG, INVESTIGATION_SLUG = "coupled-sparks-fast-diffusion", "vcell-hybrid-paper-benchmarks"
DT, T_END, OUT_DT, SEED = 2e-4, 3.0, 0.05, 1
LABELS = {"cosim": "co-simulation", "native": "native VCell"}
COLORS = {"C": "deepskyblue", "O": "lime"}
CHANNEL_PLANE = (5.05, 1.05, 1e-3)  # just inside the slab: a slice exactly on its face is empty


def record() -> None:
    from viva_pde_particle.benchmarks.calcium_sparks import TEST2, calcium_sparks_model
    from viva_pde_particle.composites.hybrid import build_hybrid_document, run_document
    from viva_pde_particle.reference.vcell_native import run_native
    from viva_pde_particle.viz3d import attach_recorder, write_native_bundle

    model = calcium_sparks_model(TEST2)
    doc = attach_recorder(build_hybrid_document(model, DT, seed=SEED), RUNS / "cosim.fenics", OUT_DT,
                          region="cyt", source=LABELS["cosim"])
    run_document(doc, T_END, DT, OUT_DT)
    nat = run_native(model, t_end=T_END, dt=DT, output_dt=OUT_DT, seed=SEED)
    write_native_bundle(RUNS / "native.fenics", nat, model.grid, region="cyt")


def busiest_times(path, n: int = 3) -> list[float]:
    """The ``n`` recorded times with the most open channels (earliest first among ties), in time order."""
    from vcell_fenics.results import Bundle

    from viva_pde_particle.viz3d import read_particles

    times = Bundle.open(path).times
    open_counts = [len(read_particles(path, "O", r)) for r in range(len(times))]
    rows = sorted(range(len(times)), key=lambda r: (-open_counts[r], r))[:n]
    return [float(times[r]) for r in sorted(rows)]


def render() -> None:
    from viva_pde_particle.viz3d.html import bundle_html
    from viva_pde_particle.viz3d.static import render_gif, render_png
    from viva_pde_particle.viz3d.study import logged_run, write_meta

    paths = {k: RUNS / f"{k}.fenics" for k in LABELS}
    with logged_run(WORKSPACE_ROOT, STUDY_SLUG, INVESTIGATION_SLUG, "viva_pde_particle.composites.examples.calcium_sparks",
                    f"{STUDY_SLUG}-3d", {"seed": SEED, "dt": DT, "t_end": T_END}) as run_id:
        panels = {f"{LABELS[k]}, t = {t:g} s": (paths[k], t) for k in LABELS for t in busiest_times(paths[k])}
        png = render_png(panels, "U", None, VIZ / "coupled_sparks_3d.png", panels=("slice",), stack="rows",
                         window=(900, 250), colors=COLORS, cmap="inferno", slice_origin=CHANNEL_PLANE)
        gif = render_gif(paths["cosim"], "U", VIZ / "coupled_sparks_3d_movie.gif", view="slice", colors=COLORS,
                         cmap="inferno", window=(900, 300), fps=8, slice_origin=CHANNEL_PLANE)
        page = bundle_html({LABELS[k]: p for k, p in paths.items()}, "U", VIZ / "coupled_sparks_3d_interactive.html",
                           title="C2 coupled sparks, fast diffusion: U and the 24 channels (closed C, open O)",
                           colors=COLORS, camera="top", max_frames=16, slice_origin=CHANNEL_PLANE)
        sims = (f"One trial (seed {SEED}) of Test 2 per solver, Δt = {DT:g} s, recorded every {OUT_DT} s to {T_END:g} s "
                "into results bundles (sims/render_3d.py). Native channel states come from VCell's per-node counts "
                "(exact: channels sit on nodes).")
        write_meta(png, title="Coupled channels, fast diffusion: U in the channel plane", run_id=run_id,
                   simulations=sims,
                   caption="U at z = 0 at the three recorded times with the most open channels, per solver; channels "
                           "closed (blue) or open (green).",
                   interpretation="With D = 1000 µm²/s calcium from an open channel spreads over the slab within "
                                  "milliseconds, so U is almost uniform (spread < 0.2 µM at ~10 µM): the well-mixed "
                                  "limit, where the hybrid is compared with the exact piecewise-deterministic "
                                  "process. Opening speeds up with U, so openings run away into bursts in which most "
                                  "channels are open at once. The two trials are independent samples; their "
                                  "distributions agree (see the Fig 2 comparison).")
        write_meta(gif, title="Coupled channels over time (co-simulation)", run_id=run_id, simulations=sims,
                   caption="U at z = 0 and channel states, every 0.05 s.",
                   interpretation="U rises uniformly as channels open, which opens more; it relaxes by pumping once "
                                  "they close.")
        write_meta(page, title="Interactive: coupled sparks, both solvers", run_id=run_id, simulations=sims,
                   caption="Time slider, solver selector, slices and channels (three.js).",
                   interpretation="Explore either trial at any recorded time.")


if __name__ == "__main__":
    {"record": record, "render": render}[sys.argv[1] if len(sys.argv) > 1 else "render"]()
