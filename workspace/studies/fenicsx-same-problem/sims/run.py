"""Study B1 (fenicsx-same-problem): swap the PDE engine (FV ↔ FEniCSx Q1) under the same coupling.

Part 1 (discretization): diffusion of cos(πx/L)cos(πy/L) on N×N grids, N ∈ {11, 21, 41, 81},
for three engines: FV (VCell FV_SOLVER scheme), FEniCSx Q1 lumped mass, and FEniCSx Q1
consistent mass. Each is compared to the analytic decay at t = 0.2 s with dt = 1e-4, so the
spatial error dominates; the h-orders are fitted.

Part 2 (coupled): A4's A_p + B_f → C_f on the 11×11×3 slab, 16 seeds per engine, with
SmoldynHybrid unchanged. Survival and produced C are compared between engines (two-sample z)
and against the continuum limit. This separates discretization differences from coupling:
on matched grids the coupled results should agree to within sampling.
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import numpy as np

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[2]
STUDY_SLUG = "fenicsx-same-problem"
INVESTIGATION_SLUG = "hybrid-generalizability"
SPEC_ID = "viva_pde_particle.composites.examples.bimolecular_hybrid"

from process_bigraph import allocate_core  # noqa: E402
from vivarium_workbench.lib.run_log import append_run_event  # noqa: E402

from viva_pde_particle.analysis import (  # noqa: E402
    continuum_model,
    molecules,
    paired_z,
    run_continuum,
    run_ensemble,
    write_metrics,
)
from viva_pde_particle.composites.examples import bimolecular_model  # noqa: E402
from viva_pde_particle.grid import CartesianGrid  # noqa: E402
from viva_pde_particle.processes import FenicsxReactionDiffusion, FVReactionDiffusion  # noqa: E402

ENGINES = {
    "fv": (FVReactionDiffusion, {}),
    "fem_lumped": (FenicsxReactionDiffusion, {"mass": "lumped"}),
    "fem_consistent": (FenicsxReactionDiffusion, {"mass": "consistent"}),
}
NS = [11, 21, 41, 81]
SEEDS = range(1, 17)


def cosine_errors():
    L, D, t_end, dt = 10.0, 1.0, 0.2, 1e-4
    core = allocate_core()
    errs = {name: [] for name in ENGINES}
    for n in NS:
        g = CartesianGrid((0.0, 0.0), (L, L), (n, n))
        x, y = g.node_coordinates()
        u0 = np.cos(np.pi * x / L) * np.cos(np.pi * y / L)
        exact = u0 * np.exp(-2 * np.pi**2 * D * t_end / L**2)
        for name, (cls, opts) in ENGINES.items():
            proc = cls(core=core, config={"grid": g.to_config(), "dt": dt, **opts,
                                          "pde": {"species": {"B": {"diffusion": D}}, "particle_species": [],
                                                  "terms": []}})
            u = proc.update({"fields": {"B": u0}, "particle_counts": {}}, t_end)["fields"]["B"]
            errs[name].append(float(np.abs(u - exact).max() / np.abs(exact).max()))
    h = 10.0 / (np.array(NS) - 1)
    orders = {name: float(np.polyfit(np.log(h), np.log(e), 1)[0]) for name, e in errs.items()}
    return errs, orders, h


def coupled():
    model = bimolecular_model()
    g = model.grid
    workers = os.cpu_count() or 1
    out = {}
    for name, (_, opts) in ENGINES.items():
        engine = "fv" if name == "fv" else "fenicsx"
        times, f, c = run_ensemble(model, SEEDS, 1.0, 0.01, record_every=0.25, workers=workers,
                                   particle_init="poisson", pde_engine=engine, pde_options=opts or None)
        a = c["A"].reshape(len(SEEDS), len(times), -1).sum(axis=2)
        out[name] = (a / a[:, :1], molecules(g, f["C"]) / a[:, :1])
    _, cont = run_continuum(continuum_model(model), 1.0, 0.01, 0.25)
    ac = molecules(g, cont["A"])
    return out, ac / ac[0], times


def main() -> int:
    run_id = uuid.uuid4().hex
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "started", "spec_id": SPEC_ID, "label": STUDY_SLUG,
        "started_at": time.time(), "status": "running", "emitter": "none", "origin": "canonical_run",
        "study_slug": STUDY_SLUG, "investigation_slug": INVESTIGATION_SLUG, "params": {"ns": NS},
    })
    started = time.time()
    try:
        errs, orders, h = cosine_errors()
        runs, cont, times = coupled()
        metrics: dict[str, float] = {}
        for name in ENGINES:
            metrics[f"b1_{name}_h_order"] = orders[name]
            metrics[f"b1_{name}_err_n81"] = errs[name][-1]
            metrics[f"b1_{name}_coupled_vs_continuum_max_abs"] = float(np.abs(runs[name][0].mean(0) - cont).max())
        for name in ("fem_lumped", "fem_consistent"):
            # same seeds across engines = common random numbers: compare paired differences
            metrics[f"b1_{name}_vs_fv_survival_max_paired_z"] = float(np.abs(paired_z(runs[name][0], runs["fv"][0])).max())
            metrics[f"b1_{name}_vs_fv_survival_max_abs_diff"] = float(np.abs(runs[name][0].mean(0) - runs["fv"][0].mean(0)).max())
            metrics[f"b1_{name}_vs_fv_c_max_abs_diff"] = float(np.abs(runs[name][1].mean(0) - runs["fv"][1].mean(0)).max())
        metrics["b1_wall_time_s"] = time.time() - started
        write_metrics(STUDY_DIR / "results" / "metrics.json", metrics)

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Cosine mode: error vs h", "Coupled A_p + B_f → C_f: survival"))
        for name, e in errs.items():
            fig.add_trace(go.Scatter(x=h, y=e, name=name, mode="lines+markers"), 1, 1)
        fig.add_trace(go.Scatter(x=h, y=errs["fv"][0] * (h / h[0]) ** 2, name="O(h²)", mode="lines",
                                 line={"dash": "dash"}), 1, 1)
        fig.add_trace(go.Scatter(x=times, y=cont, name="continuum", mode="lines"), 1, 2)
        for name, (s, _) in runs.items():
            fig.add_trace(go.Scatter(x=times, y=s.mean(0), name=f"co-sim {name}", mode="markers"), 1, 2)
        fig.update_xaxes(type="log", title_text="h (µm)", row=1, col=1)
        fig.update_yaxes(type="log", row=1, col=1)
        fig.update_layout(title="B1: PDE engine swap, FV vs FEniCSx Q1", height=440)
        (STUDY_DIR / "viz").mkdir(parents=True, exist_ok=True)
        (STUDY_DIR / "viz" / "engine_swap.html").write_text(fig.to_html(include_plotlyjs="cdn", full_html=True))
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
