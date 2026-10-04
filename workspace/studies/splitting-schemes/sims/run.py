"""Study B3 (splitting-schemes): coupling error of four operator-splitting schemes.

Model: A5's A_particle + B_field → C_field (k = 2/(µM·s), 20,000 A, 11×11×3 slab),
Δt = 0.005, coupling interval τ = k·Δt with k ∈ {2, 4, 8, 16, 32}, 64 seeds per point.

Schemes (viva_pde_particle.processes.SplittingCoordinator; HybridCoupler before Phase 7d):
- jacobi: vcell-fvsolver's lagged scheme; identical to the two-process composite with
  coupling="fvsolver";
- gs_particles_first and gs_pde_first: Gauss–Seidel orderings;
- strang: symmetric splitting.

Error: max over time of |mean surviving fraction − continuum limit|. Orders are fitted for
k ≥ 4, above the sampling floor (~5·10⁻⁴).
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
STUDY_SLUG = "splitting-schemes"
INVESTIGATION_SLUG = "hybrid-generalizability"
SPEC_ID = "viva_pde_particle.composites.hybrid.build_coupler_document"

from vivarium_workbench.lib.run_log import append_run_event  # noqa: E402

from viva_pde_particle.analysis import continuum_model, molecules, run_continuum, write_metrics  # noqa: E402
from viva_pde_particle.composites.examples import bimolecular_model  # noqa: E402

DT, T_END, RECORD = 0.005, 1.6, 0.16
KS = [2, 4, 8, 16, 32]
SCHEMES = ["jacobi", "gs_particles_first", "gs_pde_first", "strang"]
SEEDS = range(1, 65)
FIT_MIN_K = 4


def _run(args):
    scheme, k, seed = args
    from viva_pde_particle.composites.hybrid import build_coupler_document, run_document

    model = bimolecular_model(k=2.0)
    tr = run_document(build_coupler_document(model, DT, k, scheme, seed=seed, particle_init="poisson"),
                      T_END, DT, RECORD)
    a = np.stack(tr.particle_counts["A"]).reshape(len(tr.times), -1).sum(axis=1)
    return a / a[0]


def main() -> int:
    run_id = uuid.uuid4().hex
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "started", "spec_id": SPEC_ID, "label": STUDY_SLUG,
        "started_at": time.time(), "status": "running", "emitter": "none", "origin": "canonical_run",
        "study_slug": STUDY_SLUG, "investigation_slug": INVESTIGATION_SLUG,
        "params": {"dt": DT, "ks": KS, "schemes": SCHEMES, "seeds": len(SEEDS)},
    })
    started = time.time()
    try:
        model = bimolecular_model(k=2.0)
        _, cont = run_continuum(continuum_model(model), T_END, DT / 10, record_every=RECORD)
        ref = molecules(model.grid, cont["A"])
        ref = ref / ref[0]
        jobs = [(s, k, seed) for s in SCHEMES for k in KS for seed in SEEDS]
        with ProcessPoolExecutor(max_workers=os.cpu_count() or 1, mp_context=mp.get_context("spawn")) as pool:
            results = list(pool.map(_run, jobs, chunksize=8))
        by = {}
        for (s, k, _), surv in zip(jobs, results):
            by.setdefault((s, k), []).append(surv)
        errors = {s: [float(np.abs(np.mean(by[(s, k)], axis=0) - ref).max()) for k in KS] for s in SCHEMES}
        interval = np.array(KS) * DT
        fit = np.array(KS) >= FIT_MIN_K
        metrics: dict[str, float] = {}
        for s, err in errors.items():
            metrics[f"b3_{s}_order"] = float(np.polyfit(np.log(interval[fit]), np.log(err)[fit], 1)[0])
            for k, e in zip(KS, err):
                metrics[f"b3_{s}_err_k{k}"] = e
        best = min(SCHEMES, key=lambda s: errors[s][-1])
        metrics["b3_best_over_jacobi_err_k32"] = errors[best][-1] / errors["jacobi"][-1]
        metrics["b3_strang_over_jacobi_err_k32"] = errors["strang"][-1] / errors["jacobi"][-1]
        metrics["b3_wall_time_s"] = time.time() - started
        write_metrics(STUDY_DIR / "results" / "metrics.json", metrics)
        print(f"[{STUDY_SLUG}] best scheme at k=32: {best}")

        import plotly.graph_objects as go

        fig = go.Figure()
        for s, err in errors.items():
            fig.add_trace(go.Scatter(x=interval, y=err, name=s, mode="lines+markers"))
        e = errors["jacobi"][-1]
        fig.add_trace(go.Scatter(x=interval, y=e * interval / interval[-1], name="first order", mode="lines",
                                 line={"dash": "dash"}))
        fig.add_trace(go.Scatter(x=interval, y=e * (interval / interval[-1]) ** 2, name="second order",
                                 mode="lines", line={"dash": "dot"}))
        fig.update_xaxes(type="log", title_text="coupling interval τ = k·Δt (s)")
        fig.update_yaxes(type="log", title_text="max |survival − continuum|")
        fig.update_layout(title="B3: splitting schemes for the PDE/particle coupling", height=440)
        (STUDY_DIR / "viz").mkdir(parents=True, exist_ok=True)
        (STUDY_DIR / "viz" / "splitting_schemes.html").write_text(fig.to_html(include_plotlyjs="cdn", full_html=True))
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
