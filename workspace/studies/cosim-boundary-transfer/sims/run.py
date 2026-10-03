"""Study B2d (cosim-boundary-transfer): removing the co-simulation's boundary artifact found in B2c.

B2c found the co-sim's ensemble-mean B 1.3% high in the outer shell of the ball. Particles
reach the mesh through the grid histogram. Particles binned to grid nodes outside the mesh
go entirely to the nearest boundary DOF, and particles between the exact sphere (Smoldyn's
membrane) and the inscribed polyhedral mesh (1.4% of the volume) load the boundary as well.

Variants (same ball, mesh, grid and model as B2b/B2c; 32 seeds each):
- **grid / sphere:** Pᵀ of the grid histogram, Smoldyn confined by the exact sphere (B2b/B2c).
- **positions / sphere:** exact P1 load Σ_p φ_j(x_p) from molecule positions.
- **grid / mesh:** grid histogram → Pᵀ, with Smoldyn confined by the mesh's own boundary
  triangles, so the particle and PDE domains coincide.
- **positions / mesh:** exact P1 load and the mesh membrane.

Conversion A_p → B_f (uniform A): interior vs E[B] = A₀(1 − e^(−kt))/(V_mesh·N_A) and the
outer-shell bias (as in B2c). Exchange A_p ⇌ B_f with field-dependent creation: the final
mass balance and the steady ratio.
Writes results/metrics.json and viz/cosim_boundary_transfer.html.
"""
from __future__ import annotations

import importlib.util
import multiprocessing as mp
import os
import sys
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[2]
STUDY_SLUG = "cosim-boundary-transfer"
INVESTIGATION_SLUG = "hybrid-generalizability"
SPEC_ID = "viva_pde_particle.composites.hybrid.build_mesh_hybrid_document"

from vivarium_workbench.lib.run_log import append_run_event  # noqa: E402

from viva_pde_particle.analysis import write_metrics  # noqa: E402
from viva_pde_particle.units import MOLECULES_PER_UM3_PER_UM as NA  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, WORKSPACE_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


b2b = _load("b2b_run", "workspace/studies/sphere-cosim-vs-native/sims/run.py")
b2c = _load("b2c_run", "workspace/studies/near-membrane-fields/sims/run.py")

VARIANTS = {"grid_sphere": ("grid", "sphere"), "positions_sphere": ("positions", "sphere"),
            "grid_mesh": ("grid", "mesh"), "positions_mesh": ("positions", "mesh")}
SEEDS = range(1, 33)
DT, K = b2b.DT, 0.5
T_CONV, T_EXCH, RECORD = 2.0, 5.0, 0.25
CENTER = np.array(b2b.CENTER)


def _run(args):
    variant, which, seed = args
    from process_bigraph import Composite

    from viva_pde_particle.composites.hybrid import build_mesh_hybrid_document
    from viva_pde_particle.core import build_core

    transfer, membrane = VARIANTS[variant]
    doc = build_mesh_hybrid_document(b2b.model(which), b2b.MESH_SPHERE, DT, seed=seed,
                                     particle_transfer=transfer, membrane=membrane)
    sim = Composite({"state": doc}, core=build_core())
    pde = sim.state["pde"]["instance"]
    t_end = T_CONV if which == "conversion" else T_EXCH
    a, b = [], []

    def record():
        a.append(float(np.asarray(sim.state["particle_counts"]["A"]).sum()))
        b.append(float((np.asarray(sim.state["field_dofs"]["B"]) * pde.ml).sum() * NA))

    record()
    started = time.perf_counter()
    sim.run(0.5 * DT)  # half-step offset, as in run_document
    for _ in range(int(round(t_end / RECORD))):
        sim.run(RECORD)
        record()
    wall = time.perf_counter() - started
    return np.array(a), np.array(b), np.asarray(sim.state["field_dofs"]["B"], dtype=float).copy(), wall


def ensemble(variant, which):
    with ProcessPoolExecutor(max_workers=min(len(SEEDS), os.cpu_count() or 1),
                             mp_context=mp.get_context("spawn")) as pool:
        runs = list(pool.map(_run, [(variant, which, s) for s in SEEDS]))
    return (np.stack([r[0] for r in runs]), np.stack([r[1] for r in runs]), np.stack([r[2] for r in runs]),
            float(np.mean([r[3] for r in runs])))


def mesh_geometry():
    from process_bigraph import allocate_core

    from viva_pde_particle.composites.hybrid import build_mesh_hybrid_document
    from viva_pde_particle.processes import FenicsxMeshReactionDiffusion

    doc = build_mesh_hybrid_document(b2b.model("conversion"), b2b.MESH_SPHERE, DT, seed=1)
    proc = FenicsxMeshReactionDiffusion(config=doc["pde"]["config"], core=allocate_core())
    return proc.transfer.V.tabulate_dof_coordinates(), proc.ml


def main() -> int:
    run_id = uuid.uuid4().hex
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "started", "spec_id": SPEC_ID, "label": STUDY_SLUG,
        "started_at": time.time(), "status": "running", "emitter": "none", "origin": "canonical_run",
        "study_slug": STUDY_SLUG, "investigation_slug": INVESTIGATION_SLUG,
        "params": {"variants": list(VARIANTS), "seeds": len(SEEDS)},
    })
    started = time.time()
    try:
        metrics: dict[str, float] = {}
        xyz, ml = mesh_geometry()
        r = np.linalg.norm(xyz - CENTER, axis=1)
        frac = 1 - np.exp(-K * T_CONV)
        profiles = {}
        for name in VARIANTS:
            a, b, b_final, wall = ensemble(name, "conversion")
            expected = a[:, 0].mean() * frac / (ml.sum() * NA)
            for key, val in b2c.shell_stats(b_final, ml, r, expected).items():
                metrics[f"b2d_{name}_{key}"] = val
            metrics[f"b2d_{name}_conversion_mass_balance_final_signed"] = float(((a + b).mean(0) / a[:, 0].mean() - 1)[-1])
            metrics[f"b2d_{name}_wall_s_per_sim_s"] = wall / T_CONV
            profiles[name] = b2c.profile(b_final.mean(0) / expected, ml, r)

            a, b, _, _ = ensemble(name, "exchange")
            total = a + b
            times = np.arange(a.shape[1]) * RECORD
            late = times >= 4.0
            metrics[f"b2d_{name}_exchange_mass_balance_final_signed"] = float(total.mean(0)[-1] / total.mean(0)[0] - 1)
            metrics[f"b2d_{name}_exchange_steady_ratio_rel_err"] = float(abs(a[:, late].mean() / b[:, late].mean() - 0.5) / 0.5)
        metrics["b2d_wall_time_s"] = time.time() - started
        write_metrics(STUDY_DIR / "results" / "metrics.json", metrics)

        import plotly.graph_objects as go

        mids = 0.5 * (b2c.BINS[1:] + b2c.BINS[:-1])
        labels = {"grid_sphere": "grid histogram → Pᵀ, sphere membrane (B2b/B2c)",
                  "positions_sphere": "exact P1 load from positions, sphere membrane",
                  "grid_mesh": "grid histogram → Pᵀ, membrane = mesh boundary",
                  "positions_mesh": "exact P1 load, membrane = mesh boundary"}
        fig = go.Figure()
        for name, prof in profiles.items():
            ok = ~np.isnan(prof)
            fig.add_trace(go.Scatter(x=mids[ok], y=prof[ok], name=labels[name], mode="lines+markers"))
        fig.add_hline(y=1.0, line_dash="dash")
        fig.add_vline(x=b2c.SHELL, line_dash="dot")
        fig.update_layout(title="B2d: co-sim ensemble-mean B / E[B] by radius (ball conversion, t = 2 s, 32 seeds)",
                          xaxis_title="r (µm)", yaxis_title="B / E[B]", height=440)
        (STUDY_DIR / "viz" / "cosim_boundary_transfer.html").write_text(fig.to_html(include_plotlyjs="cdn", full_html=True))
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
