"""Study A5 (coupling-interval-convergence): error vs Smoldyn step multiplier k, both solvers.

Model: A4's A_particle + B_field → C_field on the 11×11×3 slab, with 20,000 A. The PDE
step is fixed at Δt = 0.005, and Smoldyn steps once per k PDE steps. The coupling interval
is therefore k·Δt, and the lag grows with it. The error of the ensemble-mean surviving
fraction is measured against the continuum limit (same model, A continuous, fine Δt), and
fitted vs k·Δt on log-log axes. Explicit lagged coupling should give first order (slope 1)
above the statistical/discreteness floor.

Configurations:
- co-sim with coupling="fvsolver" (fvsolver's exact lag);
- co-sim with coupling="start-of-interval" (process-bigraph's plain scheduling);
- native VCell (SMOLDYN_STEP_MULTIPLIER = k).
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import numpy as np

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[2]
STUDY_SLUG = "coupling-interval-convergence"
INVESTIGATION_SLUG = "cosim-vs-embedded-hybrid"
SPEC_ID = "viva_pde_particle.composites.examples.bimolecular_hybrid"

from vivarium_workbench.lib.run_log import append_run_event  # noqa: E402

from viva_pde_particle.analysis import continuum_model, molecules, run_continuum, run_ensemble, write_metrics  # noqa: E402
from viva_pde_particle.composites.examples import bimolecular_model  # noqa: E402
from viva_pde_particle.reference.vcell_native import hybrid_support_available  # noqa: E402

DT = 0.005
T_END = 1.6
RECORD = 0.16  # a multiple of every k*DT below
KS = [1, 2, 4, 8, 16, 32]
NATIVE_MAX_K = 16  # vcell-fvsolver crashes at k = 32 here (Smoldyn step 0.16 s = output interval)
SEEDS = range(1, 65)
FIT_MIN_K = 4  # fit the slope where the coupling error dominates the floor


def survival(counts_a: np.ndarray) -> np.ndarray:
    a = counts_a.reshape(counts_a.shape[0], counts_a.shape[1], -1).sum(axis=2)
    return (a / a[:, :1]).mean(axis=0)


def main() -> int:
    run_id = uuid.uuid4().hex
    native = hybrid_support_available()
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "started", "spec_id": SPEC_ID, "label": STUDY_SLUG,
        "started_at": time.time(), "status": "running", "emitter": "none", "origin": "canonical_run",
        "study_slug": STUDY_SLUG, "investigation_slug": INVESTIGATION_SLUG,
        "params": {"dt": DT, "ks": KS, "seeds": len(SEEDS), "native": native},
    })
    started = time.time()
    try:
        model = bimolecular_model(k=2.0)  # faster reaction makes the lag visible
        g = model.grid
        workers = os.cpu_count() or 1
        _, cont = run_continuum(continuum_model(model), T_END, DT / 10, record_every=RECORD)
        ref = molecules(g, cont["A"])
        ref = ref / ref[0]
        errors: dict[str, list[float]] = {}
        for k in KS:
            for coupling in ("fvsolver", "start-of-interval"):
                _, _, c = run_ensemble(model, SEEDS, T_END, DT, record_every=RECORD, workers=workers,
                                       step_multiplier=k, coupling=coupling, particle_init="poisson")
                errors.setdefault(f"cosim_{coupling}", []).append(float(np.abs(survival(c["A"]) - ref).max()))
            if native and k <= NATIVE_MAX_K:
                from viva_pde_particle.reference.vcell_native import run_native_ensemble

                _, _, c = run_native_ensemble(model, SEEDS, t_end=T_END, dt=DT, output_dt=RECORD,
                                              step_multiplier=k, workers=workers)
                errors.setdefault("native", []).append(float(np.abs(survival(c["A"]) - ref).max()))
        interval = np.array(KS) * DT
        fit = interval >= FIT_MIN_K * DT
        metrics: dict[str, float] = {}
        for name, err in errors.items():
            err = np.array(err)
            x, sel = interval[: len(err)], fit[: len(err)]
            metrics[f"a5_{name.replace('-', '_')}_order"] = float(np.polyfit(np.log(x[sel]), np.log(err[sel]), 1)[0])
            for k, e in zip(KS, err):
                metrics[f"a5_{name.replace('-', '_')}_err_k{k}"] = float(e)
        if "native" in errors:
            n = len(errors["native"])
            ratio = np.array(errors["cosim_fvsolver"][:n]) / np.array(errors["native"])
            metrics[f"a5_fvsolver_mode_vs_native_err_ratio_k{KS[n - 1]}"] = float(ratio[-1])
        metrics["a5_wall_time_s"] = time.time() - started
        write_metrics(STUDY_DIR / "results" / "metrics.json", metrics)

        import plotly.graph_objects as go

        fig = go.Figure()
        for name, err in errors.items():
            fig.add_trace(go.Scatter(x=interval[: len(err)], y=err, name=name, mode="lines+markers"))
        e0 = errors["cosim_fvsolver"][-1]
        fig.add_trace(go.Scatter(x=interval, y=e0 * interval / interval[-1], name="first order", mode="lines",
                                 line={"dash": "dash"}))
        fig.update_xaxes(type="log", title_text="coupling interval k·Δt (s)")
        fig.update_yaxes(type="log", title_text="max |survival − continuum|")
        fig.update_layout(title="A5: error vs coupling interval (A_p + B_f → C_f)", height=440)
        (STUDY_DIR / "viz").mkdir(parents=True, exist_ok=True)
        (STUDY_DIR / "viz" / "coupling_convergence.html").write_text(fig.to_html(include_plotlyjs="cdn", full_html=True))
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
