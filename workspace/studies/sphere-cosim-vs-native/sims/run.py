"""Study B2b (sphere-cosim-vs-native): the same ball, two discretizations.

- **Native VCell:** analytic geometry (``cell`` sphere inside ``ec`` with a membrane), discretized
  on VCell's Cartesian volume grid with its membrane numerics, and solved by vcell-fvsolver with
  embedded Smoldyn (pyvcell → libvcell → pyvcell-fvsolver).
- **Co-simulation:** a body-fitted P1 tetrahedral mesh of the ball (FenicsxMeshReactionDiffusion)
  coupled through the same 37³ background grid to the patched Smoldyn, confined by a reflecting
  sphere.

Box [0, 9]³, Δ = 0.25 µm; ball centre (4.5, 4.5, 4.5), R = 4 µm, 0.5 µm clear of the box
walls (so vcell-fvsolver#24, boundary-node overproduction at the box walls, cannot affect
it); 8 seeds per solver.

- Conversion A_p → B_f (k = 0.5 /s, 20,000 A): decay against exp(−kt), A + B conservation,
  co-sim vs native totals, and the radial B profile.
- Exchange A_p ⇌ B_f with field-dependent creation in the cell (k1 = 1, k2 = 0.5, B₀ = 0.2 µM):
  mass balance, steady ratio, and co-sim vs native totals.
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
STUDY_SLUG = "sphere-cosim-vs-native"
INVESTIGATION_SLUG = "hybrid-generalizability"
SPEC_ID = "viva_pde_particle.composites.hybrid.build_mesh_hybrid_document"

from vivarium_workbench.lib.run_log import append_run_event  # noqa: E402

from viva_pde_particle.analysis import two_sample_z, write_metrics  # noqa: E402
from viva_pde_particle.grid import CartesianGrid  # noqa: E402
from viva_pde_particle.model import HybridModel, Reaction, Species  # noqa: E402
from viva_pde_particle.reference.vcell_native import hybrid_support_available  # noqa: E402
from viva_pde_particle.units import MOLECULES_PER_UM3_PER_UM as NA  # noqa: E402

CENTER, R = (4.5, 4.5, 4.5), 4.0
GRID = CartesianGrid((0, 0, 0), (9, 9, 9), (37, 37, 37))
DT, SEEDS = 0.01, range(1, 9)
NATIVE_GEOMETRY = {"kind": "sphere", "center": list(CENTER), "radius": R}
MESH_SPHERE = {"center": CENTER, "radius": R, "h": 0.8}


def model(which):
    if which == "conversion":
        return HybridModel(GRID, [Species("A", 1.0, particle=True, initial=20000), Species("B", 1.0, initial=0.0)],
                           [Reaction("convert", {"A": 1}, {"B": 1}, k=0.5)])
    return HybridModel(GRID, [Species("A", 1.0, particle=True, initial=0), Species("B", 1.0, initial=0.2)],
                       [Reaction("a_to_b", {"A": 1}, {"B": 1}, k=1.0), Reaction("b_to_a", {"B": 1}, {"A": 1}, k=0.5)])


def _cosim(args):
    which, seed, t_end, every = args
    from viva_pde_particle.composites.hybrid import build_mesh_hybrid_document, run_document

    tr = run_document(build_mesh_hybrid_document(model(which), MESH_SPHERE, DT, seed=seed), t_end, DT, every)
    a = np.stack(tr.particle_counts["A"]).reshape(len(tr.times), -1).sum(axis=1)
    b_grid = np.stack(tr.fields["B"])  # field sampled on the background grid
    return np.array(tr.times), a, b_grid


def run_cosim(which, t_end, every):
    from process_bigraph import allocate_core

    from viva_pde_particle.composites.hybrid import build_mesh_hybrid_document
    from viva_pde_particle.processes import FenicsxMeshReactionDiffusion

    with ProcessPoolExecutor(max_workers=min(len(SEEDS), os.cpu_count() or 1),
                             mp_context=mp.get_context("spawn")) as pool:
        runs = list(pool.map(_cosim, [(which, s, t_end, every) for s in SEEDS]))
    doc = build_mesh_hybrid_document(model(which), MESH_SPHERE, DT, seed=1)
    proc = FenicsxMeshReactionDiffusion(config=doc["pde"]["config"], core=allocate_core())
    times = runs[0][0]
    a = np.stack([r[1] for r in runs])
    b_grid = np.stack([r[2] for r in runs])
    return times, a, b_grid, float(proc.ml.sum())


def run_native(which, t_end, every):
    from viva_pde_particle.reference.vcell_native import run_native_ensemble

    times, f, c = run_native_ensemble(model(which), SEEDS, t_end=t_end, dt=DT, output_dt=every,
                                      geometry=NATIVE_GEOMETRY)
    a = c["A"].reshape(len(SEEDS), len(times), -1).sum(axis=2)
    return times, a, f["B"]


def inside_mask():
    x, y, z = GRID.node_coordinates()
    return (x - CENTER[0]) ** 2 + (y - CENTER[1]) ** 2 + (z - CENTER[2]) ** 2 <= R**2


def b_molecules(b_grid: np.ndarray) -> np.ndarray:
    """∫B over the ball's grid cells (node-centred, nodes with centres inside the ball), in molecules."""
    w = GRID.element_volumes * inside_mask() * NA
    return (b_grid * w).sum(axis=tuple(range(-3, 0)))


def radial_profile(b_grid_mean: np.ndarray, nbins: int = 8) -> np.ndarray:
    x, y, z = GRID.node_coordinates()
    r = np.sqrt((x - CENTER[0]) ** 2 + (y - CENTER[1]) ** 2 + (z - CENTER[2]) ** 2)
    edges = np.linspace(0, R * 0.95, nbins + 1)
    return np.array([b_grid_mean[(r >= lo) & (r < hi)].mean() for lo, hi in zip(edges[:-1], edges[1:])])


def main() -> int:
    run_id = uuid.uuid4().hex
    native = hybrid_support_available()
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "started", "spec_id": SPEC_ID, "label": STUDY_SLUG,
        "started_at": time.time(), "status": "running", "emitter": "none", "origin": "canonical_run",
        "study_slug": STUDY_SLUG, "investigation_slug": INVESTIGATION_SLUG, "params": {"native": native},
    })
    started = time.time()
    try:
        if not native:
            raise SystemExit("needs pyvcell with spatial-hybrid support (pixi run -e dev ...)")
        metrics: dict[str, float] = {}
        exact_volume = 4 / 3 * np.pi * R**3
        metrics["b2b_native_staircase_volume_rel_err"] = float(
            (GRID.element_volumes * inside_mask()).sum() / exact_volume - 1)

        tc, ac, bc, mesh_volume = run_cosim("conversion", 2.0, 0.25)
        tn, an, bn = run_native("conversion", 2.0, 0.25)
        metrics["b2b_cosim_mesh_volume_rel_err"] = float(mesh_volume / exact_volume - 1)
        for name, t, a, bg in (("cosim", tc, ac, bc), ("native", tn, an, bn)):
            exp_ = a[:, :1] * np.exp(-0.5 * t)
            sd = np.sqrt(a[:, 0].mean() * np.exp(-0.5 * t) * (1 - np.exp(-0.5 * t)) / len(a))
            z = np.where(sd > 0, (a.mean(0) - exp_.mean(0)) / np.where(sd > 0, sd, 1), 0)
            metrics[f"b2b_{name}_decay_max_z"] = float(np.abs(z).max())
            bt = b_molecules(bg)
            metrics[f"b2b_{name}_mass_balance_max_rel_dev"] = float(np.abs((a + bt).mean(0) / a[:, 0].mean() - 1).max())
        fc, fn = ac / ac[:, :1], an / an[:, :1]
        metrics["b2b_conversion_survival_max_z"] = float(np.abs(two_sample_z(fc, fn)).max())
        pc, pn = radial_profile(bc[:, -1].mean(0)), radial_profile(bn[:, -1].mean(0))
        metrics["b2b_final_radial_b_rel_diff"] = float(np.abs(pc - pn).max() / np.abs(pn).max())

        tc, ac, bc, _ = run_cosim("exchange", 5.0, 0.25)
        tn, an, bn = run_native("exchange", 5.0, 0.25)
        for name, t, a, bg in (("cosim", tc, ac, bc), ("native", tn, an, bn)):
            bt = b_molecules(bg)
            total0 = bt[:, 0].mean()
            metrics[f"b2b_{name}_exchange_mass_balance_max_rel_dev"] = float(np.abs((a + bt).mean(0) / total0 - 1).max())
            late = t >= 4.0
            metrics[f"b2b_{name}_exchange_steady_ratio_rel_err"] = float(abs(a[:, late].mean() / bt[:, late].mean() - 0.5) / 0.5)
        metrics["b2b_exchange_a_totals_max_z"] = float(np.abs(two_sample_z(ac, an)).max())
        metrics["b2b_exchange_a_final_rel_diff"] = float(abs(ac[:, -1].mean() - an[:, -1].mean()) / an[:, -1].mean())
        metrics["b2b_wall_time_s"] = time.time() - started
        write_metrics(STUDY_DIR / "results" / "metrics.json", metrics)

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Conversion: final radial B profile",
                                                            "Exchange: total A (particles)"))
        edges = np.linspace(0, R * 0.95, 9)
        mids = 0.5 * (edges[1:] + edges[:-1])
        fig.add_trace(go.Scatter(x=mids, y=pn, name="native VCell (Cartesian + membrane numerics)", mode="lines+markers"), 1, 1)
        fig.add_trace(go.Scatter(x=mids, y=pc, name="co-sim (body-fitted P1 mesh)", mode="lines+markers"), 1, 1)
        fig.add_trace(go.Scatter(x=tn, y=an.mean(0), name="native A", mode="lines+markers"), 1, 2)
        fig.add_trace(go.Scatter(x=tc, y=ac.mean(0), name="co-sim A", mode="lines+markers"), 1, 2)
        fig.update_xaxes(title_text="r (µm)", row=1, col=1)
        fig.update_xaxes(title_text="t (s)", row=1, col=2)
        fig.update_layout(title="B2b: ball (R = 4 µm), native VCell vs co-simulation", height=440)
        (STUDY_DIR / "viz").mkdir(parents=True, exist_ok=True)
        (STUDY_DIR / "viz" / "sphere_cosim_vs_native.html").write_text(fig.to_html(include_plotlyjs="cdn", full_html=True))
    except Exception:
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
