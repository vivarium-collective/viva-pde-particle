"""Study B2f (dumbbell-three-ways): one non-convex geometry description, three solver paths (Phase 7e.6).

The geometry is a vcell-fenics GeometryDescription: two overlapping balls (R = 1.5 µm, centres
2.8 µm apart) joined by a thin neck (radius 0.6 µm), in an 8 × 6 × 6 µm box on a Δ = 0.25 µm grid.
Each solver path realizes the same description:

- **native:** VCell itself, with the description converted to VCML; vcell-fvsolver with
  embedded Smoldyn;
- **vcell_none / vcell_corrected:** the co-simulation on VCell's own realization
  (VCellFVRealization), with correction ``none`` or ``adapters+volumes``;
- **mesh:** the co-simulation on a Netgen tet mesh (vcell-fenics realize, h = 0.3).

Cases:
- conversion A_p → B_f (k = 0.5, 20,000 A uniform), t = 2 s;
- exchange A_p ⇌ B_f (k1 = 1, k2 = 0.5, B₀ = 0.2 µM), t = 5 s.

Measures:
- A + B balance, with B measured on each method's own PDE volumes;
- **near-membrane bias**: the volume-weighted mean B within 0.5 µm of the membrane against
  deeper B, where E[B] is uniform;
- lobe asymmetry, the mean B of the x > 4 lobe against the x < 4 lobe, which should be 0 by
  symmetry;
- the exchange steady ratio.

Needs the ``dev`` pixi env. Writes results/metrics.json and viz/dumbbell_three_ways.html.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[2]
STUDY_SLUG = "dumbbell-three-ways"
INVESTIGATION_SLUG = "hybrid-generalizability"
SPEC_ID = "viva_pde_particle.geometry"

from vivarium_workbench.lib.run_log import append_run_event  # noqa: E402

from viva_pde_particle.analysis import write_metrics  # noqa: E402
from viva_pde_particle.grid import CartesianGrid  # noqa: E402
from viva_pde_particle.model import HybridModel, Reaction, Species  # noqa: E402
from viva_pde_particle.units import MOLECULES_PER_UM3_PER_UM as NA  # noqa: E402

GRID = CartesianGrid((0, 0, 0), (8, 6, 6), (33, 25, 25))
DT, K, T_CONV, T_EXCH, RECORD = 0.01, 0.5, 2.0, 5.0, 0.25
SEEDS_COSIM, SEEDS_NATIVE = range(1, 17), range(1, 9)
H_MESH, BAND = 0.3, 0.5
S1 = "(geom.x[0]-2.6)**2 + (geom.x[1]-3)**2 + (geom.x[2]-3)**2 < 2.25"
S2 = "(geom.x[0]-5.4)**2 + (geom.x[1]-3)**2 + (geom.x[2]-3)**2 < 2.25"
NECK = "(geom.x[1]-3)**2 + (geom.x[2]-3)**2 < 0.36 && geom.x[0] > 2.6 && geom.x[0] < 5.4"
_REAL: dict = {}


def description():
    from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass

    return GeometryDescription(name="dumbbell", dim=3, extent=(8.0, 6.0, 6.0),
                               subvolumes=(SubVolume(name="cell", type="analytic", expression=f"({S1}) || ({S2}) || ({NECK})"),
                                           SubVolume(name="ec", type="analytic", expression="1.0")),
                               surfaces=(SurfaceClass(name="pm", inside="cell", outside="ec"),))


def analytic_inside(p):
    p = np.asarray(p)
    s1 = ((p - [2.6, 3, 3]) ** 2).sum(1) < 2.25
    s2 = ((p - [5.4, 3, 3]) ** 2).sum(1) < 2.25
    neck = ((p[:, 1] - 3) ** 2 + (p[:, 2] - 3) ** 2 < 0.36) & (p[:, 0] > 2.6) & (p[:, 0] < 5.4)
    return s1 | s2 | neck


def model(which, n0=20000):
    if which == "conversion":
        return HybridModel(GRID, [Species("A", 1.0, particle=True, initial=n0), Species("B", 1.0, initial=0.0)],
                           [Reaction("convert", {"A": 1}, {"B": 1}, k=K)])
    return HybridModel(GRID, [Species("A", 1.0, particle=True, initial=0), Species("B", 1.0, initial=0.2)],
                       [Reaction("a_to_b", {"A": 1}, {"B": 1}, k=1.0), Reaction("b_to_a", {"B": 1}, {"A": 1}, k=0.5)])


def vcell_realization():
    if "vcell" not in _REAL:
        from viva_pde_particle.geometry.vcell_fv import VCellFVRealization

        _REAL["vcell"] = VCellFVRealization(description(), GRID)
    return _REAL["vcell"]


def _cosim(args):
    method, which, seed = args
    from process_bigraph import Composite

    from viva_pde_particle.core import build_core

    if method == "mesh":
        from viva_pde_particle.composites.hybrid import build_mesh_hybrid_document
        from viva_pde_particle.mesh import mesh_space

        doc = build_mesh_hybrid_document(model(which), None, DT, seed=seed,
                                         geometry={"description": description(), "region": "cell", "h": H_MESH})
        space = mesh_space(doc["pde"]["config"]["mesh"])
        w, store, coords = space.ml, "field_dofs", space.V.tabulate_dof_coordinates()
    else:
        from viva_pde_particle.composites.hybrid import build_vcell_geometry_hybrid_document

        corr = "none" if method == "vcell_none" else "adapters+volumes"
        doc = build_vcell_geometry_hybrid_document(model(which), description(), DT, seed=seed, correction=corr,
                                                   realization=vcell_realization())
        store, coords = "fields", None
        w = None
    sim = Composite({"state": doc}, core=build_core())
    if w is None:
        w = (sim.state["pde"]["instance"]._s.reshape(GRID.shape) * GRID.full_volume
             * doc["pde"]["config"]["domain"]["mask"]).ravel()
        x, y, z = GRID.node_coordinates()
        coords = np.stack([x.ravel(), y.ravel(), z.ravel()], 1)
    t_end = T_CONV if which == "conversion" else T_EXCH
    a, b = [], []

    def record():
        a.append(float(np.asarray(sim.state["particle_counts"]["A"]).sum()))
        b.append(float((np.asarray(sim.state[store]["B"]).ravel() * w).sum() * NA))

    record()
    sim.run(0.5 * DT)
    for _ in range(int(round(t_end / RECORD))):
        sim.run(RECORD)
        record()
    return np.array(a), np.array(b), np.asarray(sim.state[store]["B"], dtype=float).ravel().copy(), w, coords


def run_cosim(method, which):
    with ProcessPoolExecutor(max_workers=min(len(SEEDS_COSIM), os.cpu_count() or 1),
                             mp_context=mp.get_context("spawn")) as pool:
        runs = list(pool.map(_cosim, [(method, which, s) for s in SEEDS_COSIM]))
    return (np.stack([r[0] for r in runs]), np.stack([r[1] for r in runs]), np.stack([r[2] for r in runs]),
            runs[0][3], runs[0][4])


def run_native(which):
    from viva_pde_particle.reference.vcell_native import run_native_ensemble

    real = vcell_realization()
    times, f, c = run_native_ensemble(model(which), SEEDS_NATIVE, t_end=T_CONV if which == "conversion" else T_EXCH,
                                      dt=DT, output_dt=RECORD, geometry=description(), domain_volume=real.volume("cell"))
    mask = real.node_mask("cell")
    w = (GRID.element_volumes * mask).ravel()
    a = c["A"].reshape(len(SEEDS_NATIVE), len(times), -1).sum(axis=2)
    bf = f["B"].reshape(len(SEEDS_NATIVE), len(times), -1)
    b = (bf * w).sum(axis=2) * NA
    x, y, z = GRID.node_coordinates()
    return a, b, bf[:, -1], w, np.stack([x.ravel(), y.ravel(), z.ravel()], 1)


def surface_distance(points, surface_points):
    from scipy.spatial import cKDTree

    return cKDTree(surface_points).query(points)[0]


def field_metrics(b_final, w, coords, surf):
    d = surface_distance(coords, surf)
    on = w > 0

    def mean(sel):
        ww = w * sel
        return (b_final * ww).sum(axis=1) / ww.sum()

    deep, band = mean(on & (d > BAND)), mean(on & (d <= BAND))
    left, right = mean(on & (coords[:, 0] < 4.0)), mean(on & (coords[:, 0] > 4.0))
    bias = band / deep - 1
    asym = right / left - 1
    se = lambda v: v.std(ddof=1) / np.sqrt(len(v))  # noqa: E731
    return {"band_bias": float(bias.mean()), "band_bias_z": float(bias.mean() / se(bias)),
            "lobe_asym": float(asym.mean()), "lobe_asym_z": float(asym.mean() / se(asym))}


def main() -> int:
    run_id = uuid.uuid4().hex
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "started", "spec_id": SPEC_ID, "label": STUDY_SLUG,
        "started_at": time.time(), "status": "running", "emitter": "none", "origin": "canonical_run",
        "study_slug": STUDY_SLUG, "investigation_slug": INVESTIGATION_SLUG,
        "params": {"seeds_cosim": len(SEEDS_COSIM), "seeds_native": len(SEEDS_NATIVE), "h_mesh": H_MESH},
    })
    started = time.time()
    try:
        from viva_pde_particle.geometry import realize_fenics
        from viva_pde_particle.reference.vcell_native import hybrid_support_available

        if not hybrid_support_available():
            raise SystemExit("needs pyvcell with spatial-hybrid support (pixi run -e dev ...)")
        real = vcell_realization()
        netgen = realize_fenics(description(), H_MESH)
        mc = np.random.default_rng(0).uniform([0, 0, 0], [8, 6, 6], (4_000_000, 3))
        exact = analytic_inside(mc).mean() * 288.0
        metrics: dict[str, float] = {
            "b2f_volume_analytic": float(exact),
            "b2f_volume_vcell_staircase_rel": real.pde_volume("cell") / exact - 1,
            "b2f_volume_vcell_smooth_rel": real.volume("cell") / exact - 1,
            "b2f_volume_netgen_rel": netgen.volume("cell") / exact - 1,
        }
        tri = netgen.boundary_triangles("cell")
        surf = np.concatenate([tri.reshape(-1, 3), tri.mean(axis=1)])  # reference membrane points (Netgen)
        for method in ("native", "vcell_none", "vcell_corrected", "mesh"):
            for which in ("conversion", "exchange"):
                if method == "native":
                    a, b, bf, w, coords = run_native(which)
                else:
                    a, b, bf, w, coords = run_cosim(method, which)
                total = (a + b).mean(0)
                metrics[f"b2f_{method}_{which}_balance"] = float(total[-1] / total[0] - 1)
                if which == "conversion":
                    for k, v in field_metrics(bf, w, coords, surf).items():
                        metrics[f"b2f_{method}_{k}"] = v
                else:
                    late = np.arange(a.shape[1]) * RECORD >= 4.0
                    metrics[f"b2f_{method}_steady_ratio_rel_err"] = float(abs(a[:, late].mean() / b[:, late].mean() - 0.5) / 0.5)
                print(f"[{STUDY_SLUG}] done {method} {which} at {time.time() - started:.0f}s", flush=True)
        metrics["b2f_wall_time_s"] = time.time() - started
        write_metrics(STUDY_DIR / "results" / "metrics.json", metrics)

        import plotly.graph_objects as go

        methods = ["native", "vcell_none", "vcell_corrected", "mesh"]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="near-membrane bias", x=methods, y=[metrics[f"b2f_{m}_band_bias"] for m in methods]))
        fig.add_trace(go.Bar(name="exchange balance", x=methods, y=[metrics[f"b2f_{m}_exchange_balance"] for m in methods]))
        fig.add_trace(go.Bar(name="conversion balance", x=methods, y=[metrics[f"b2f_{m}_conversion_balance"] for m in methods]))
        fig.update_layout(title="B2f: dumbbell, one GeometryDescription, three solver paths", barmode="group",
                          yaxis_title="relative error", height=440)
        (STUDY_DIR / "viz" / "dumbbell_three_ways.html").write_text(fig.to_html(include_plotlyjs="cdn", full_html=True))
    except BaseException:
        append_run_event(WORKSPACE_ROOT, {"run_id": run_id, "event": "completed",
                                          "completed_at": time.time(), "status": "failed"})
        raise
    append_run_event(WORKSPACE_ROOT, {"run_id": run_id, "event": "completed",
                                      "completed_at": time.time(), "status": "completed"})
    for k, v in metrics.items():
        print(f"[{STUDY_SLUG}] {k} = {v:.6g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
