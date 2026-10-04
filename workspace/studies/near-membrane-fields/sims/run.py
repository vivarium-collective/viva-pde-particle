"""Study B2c (near-membrane-fields): fields next to a curved membrane, on smoothed and body-fitted grids.

Same ball and solvers as B2b (sphere-cosim-vs-native). Conversion A_p → B_f (k = 0.5 /s) with
20,000 A placed uniformly in the ball. A stays uniform in expectation (reflecting membrane),
so the exact expected B is spatially uniform:

    E[B](t) = A₀ (1 − e^(−kt)) / (V·N_A),

with V each solver's own domain volume. Any radial structure in the ensemble-mean B is a
boundary artifact of particle creation, binning or transfer at the membrane.

- **Native VCell:** node values are mapped onto pyvcell's smoothed unstructured grid of the
  ``cell`` domain (viva_pde_particle.reference.vcell_vtk, with the node-centred correction),
  so boundary elements sit at their smoothed positions rather than on the staircase.
- **Co-simulation:** P1 DOF values on the body-fitted mesh, weighted by lumped mass.

Metrics: smoothed-grid geometry (volume, surface radius), the interior level against E[B],
and the outer-shell bias (shell r > R − 2Δ vs interior r < R/2) for each solver.
Writes results/metrics.json, results/native_mean_B_smoothed.vtu (ParaView) and viz/near_membrane_fields.html.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import tempfile
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[2]
STUDY_SLUG = "near-membrane-fields"
INVESTIGATION_SLUG = "hybrid-generalizability"
SPEC_ID = "viva_pde_particle.composites.hybrid.build_mesh_hybrid_document"

from vivarium_workbench.lib.run_log import append_run_event  # noqa: E402

from viva_pde_particle.analysis import write_metrics  # noqa: E402
from viva_pde_particle.reference.vcell_native import hybrid_support_available  # noqa: E402
from viva_pde_particle.units import MOLECULES_PER_UM3_PER_UM as NA  # noqa: E402

sys.path.insert(0, str(WORKSPACE_ROOT / "workspace" / "studies" / "sphere-cosim-vs-native" / "sims"))
import run as b2b  # noqa: E402  (same geometry, model and solvers as B2b)

CENTER, R, GRID, DT, SEEDS = b2b.CENTER, b2b.R, b2b.GRID, b2b.DT, range(1, 33)
K, T_END = 0.5, 2.0
DELTA = float(GRID.spacing[0])
SHELL, CORE = R - 2 * DELTA, R / 2
BINS = np.arange(0.0, R + 1e-9, 0.2)


def _cosim(seed):
    from process_bigraph import Composite

    from viva_pde_particle.composites.hybrid import build_mesh_hybrid_document
    from viva_pde_particle.core import build_core

    sim = Composite({"state": build_mesh_hybrid_document(b2b.model("conversion"), b2b.MESH_SPHERE, DT, seed=seed)},
                    core=build_core())
    a0 = float(np.asarray(sim.state["particle_counts"]["A"]).sum())
    sim.run(0.5 * DT)  # half-step offset, as in run_document
    sim.run(T_END)
    return a0, np.asarray(sim.state["field_dofs"]["B"], dtype=float).copy()


def run_cosim():
    from process_bigraph import allocate_core

    from viva_pde_particle.composites.hybrid import build_mesh_hybrid_document
    from viva_pde_particle.processes import FenicsxMeshReactionDiffusion

    with ProcessPoolExecutor(max_workers=min(len(SEEDS), os.cpu_count() or 1),
                             mp_context=mp.get_context("spawn")) as pool:
        runs = list(pool.map(_cosim, SEEDS))
    doc = build_mesh_hybrid_document(b2b.model("conversion"), b2b.MESH_SPHERE, DT, seed=1)
    proc = FenicsxMeshReactionDiffusion(config=doc["pde"]["config"], core=allocate_core())
    xyz = proc.V.tabulate_dof_coordinates()
    return np.array([r[0] for r in runs]), np.stack([r[1] for r in runs]), xyz, proc.ml


def run_native(workdir: Path):
    from viva_pde_particle.reference.vcell_native import run_native as run_one
    from viva_pde_particle.reference.vcell_native import run_native_ensemble

    times, f, c = run_native_ensemble(b2b.model("conversion"), SEEDS, t_end=T_END, dt=DT, output_dt=0.5,
                                      geometry=b2b.NATIVE_GEOMETRY)
    a0 = c["A"][:, 0].reshape(len(SEEDS), -1).sum(axis=1)
    # one short solve, kept on disk, for the .mesh file (the mesh does not depend on the seed)
    run_one(b2b.model("conversion"), t_end=2 * DT, dt=DT, output_dt=DT, seed=1, workdir=workdir,
            geometry=b2b.NATIVE_GEOMETRY)
    return a0, f["B"][:, -1], next(workdir.glob("*.mesh"))


def weighted_mean(values, weights, mask):
    """Per-seed weighted mean over the masked points; values (seeds, n)."""
    w = weights * mask
    return (values * w).sum(axis=1) / w.sum()


def shell_stats(values, weights, r, expected):
    """Interior level vs E[B], and outer-shell bias vs interior, with a seed-spread z."""
    core = weighted_mean(values, weights, r < CORE)
    shell = weighted_mean(values, weights, r > SHELL)
    bias = shell / core - 1
    se = bias.std(ddof=1) / np.sqrt(len(bias))
    return {
        "interior_rel": float(core.mean() / expected - 1),
        "shell_bias": float(bias.mean()),
        "shell_bias_z": float(bias.mean() / se) if se > 0 else 0.0,
    }


def profile(values_mean, weights, r):
    idx = np.digitize(r, BINS) - 1
    out = np.full(len(BINS) - 1, np.nan)
    for i in range(len(out)):
        m = idx == i
        if m.any():
            out[i] = (values_mean[m] * weights[m]).sum() / weights[m].sum()
    return out


def main() -> int:
    run_id = uuid.uuid4().hex
    native = hybrid_support_available()
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "started", "spec_id": SPEC_ID, "label": STUDY_SLUG,
        "started_at": time.time(), "status": "running", "emitter": "none", "origin": "canonical_run",
        "study_slug": STUDY_SLUG, "investigation_slug": INVESTIGATION_SLUG,
        "params": {"native": native, "seeds": len(SEEDS), "t_end": T_END},
    })
    started = time.time()
    try:
        if not native:
            raise SystemExit("needs pyvcell with spatial-hybrid support (pixi run -e dev ...)")
        from viva_pde_particle.reference.vcell_vtk import smoothed_domain

        metrics: dict[str, float] = {}
        exact_volume = 4 / 3 * np.pi * R**3
        frac = 1 - np.exp(-K * T_END)
        center = np.array(CENTER)

        with tempfile.TemporaryDirectory(prefix="b2c-native-") as tmp:
            a0_n, b_n, mesh_file = run_native(Path(tmp))
            as_is = smoothed_domain(mesh_file, "cell", node_centred=False)
            dom = smoothed_domain(mesh_file, "cell")
        metrics["b2c_pyvcell_vtk_as_is_volume_rel_err"] = float(as_is.volumes.sum() / exact_volume - 1)
        metrics["b2c_node_centred_smoothed_volume_rel_err"] = float(dom.volumes.sum() / exact_volume - 1)
        r_raw_s = np.linalg.norm(dom.raw_surface_points - center, axis=1)
        r_s = np.linalg.norm(dom.surface_points - center, axis=1)
        metrics["b2c_staircase_surface_r_rms_err"] = float(np.sqrt(((r_raw_s - R) ** 2).mean()))
        metrics["b2c_smoothed_surface_r_rms_err"] = float(np.sqrt(((r_s - R) ** 2).mean()))

        # native: cells of the smoothed grid; staircase (unsmoothed) radius for comparison
        cells_n = np.stack([dom.cell_data(b) for b in b_n])
        v_stair = len(dom.global_index) * DELTA**3  # VCell's cell volume: whole node-centred voxels
        w_n = dom.volumes
        r_n = np.linalg.norm(dom.centroids - center, axis=1)
        r_n_stair = np.linalg.norm(dom.raw_corners.mean(axis=1) - center, axis=1)
        e_n = a0_n.mean() * frac / (v_stair * NA)
        for key, val in shell_stats(cells_n, w_n, r_n, e_n).items():
            metrics[f"b2c_native_{key}"] = val
        metrics["b2c_native_staircase_shell_bias"] = shell_stats(
            cells_n, np.full_like(w_n, DELTA**3), r_n_stair, e_n)["shell_bias"]

        a0_c, b_c, xyz, ml = run_cosim()
        r_c = np.linalg.norm(xyz - center, axis=1)
        e_c = a0_c.mean() * frac / (ml.sum() * NA)
        for key, val in shell_stats(b_c, ml, r_c, e_c).items():
            metrics[f"b2c_cosim_{key}"] = val
        metrics["b2c_wall_time_s"] = time.time() - started
        write_metrics(STUDY_DIR / "results" / "metrics.json", metrics)

        dom.write_vtu(STUDY_DIR / "results" / "native_mean_B_smoothed.vtu",
                      {"B_mean_uM": cells_n.mean(axis=0), "B_over_expected": cells_n.mean(axis=0) / e_n})

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        mids = 0.5 * (BINS[1:] + BINS[:-1])
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            "Ensemble-mean B / E[B] by radius (t = 2 s)", "Boundary surface radius: staircase vs smoothed"))
        series = (("native, smoothed grid (pyvcell, node-centred)", cells_n.mean(0) / e_n, w_n, r_n),
                  ("native, staircase voxels", cells_n.mean(0) / e_n, np.full_like(w_n, DELTA**3), r_n_stair),
                  ("co-sim, body-fitted P1", b_c.mean(0) / e_c, ml, r_c))
        for name, v, w, r in series:
            fig.add_trace(go.Scatter(x=mids, y=profile(v, w, r), name=name, mode="lines+markers"), 1, 1)
        fig.add_hline(y=1.0, line_dash="dash", row=1, col=1)
        fig.add_vline(x=SHELL, line_dash="dot", row=1, col=1)
        for name, rr in (("staircase surface", r_raw_s), ("smoothed surface", r_s)):
            fig.add_trace(go.Histogram(x=rr, name=name, opacity=0.6, nbinsx=40), 1, 2)
        fig.add_vline(x=R, line_dash="dash", row=1, col=2)
        fig.update_xaxes(title_text="r (µm)", row=1, col=1)
        fig.update_xaxes(title_text="surface point radius (µm)", row=1, col=2)
        fig.update_layout(title="B2c: fields next to the membrane (ball, R = 4 µm, 32 seeds per solver)",
                          height=440, barmode="overlay")
        (STUDY_DIR / "viz").mkdir(parents=True, exist_ok=True)
        (STUDY_DIR / "viz" / "near_membrane_fields.html").write_text(fig.to_html(include_plotlyjs="cdn", full_html=True))
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
