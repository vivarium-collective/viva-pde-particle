"""Study A4 (bimolecular-hybrid): A_particle + B_field -> C_field on both solvers.

The particle side removes A at rate k·[B](x) per molecule. The PDE side removes k·[A_binned]·[B]
from B and adds it to C. B is substantially depleted (20,000 A vs ~30,000 B molecules), so
each side responds to the other's previous state: a two-way, nonlinear coupling.

Compared:
- co-simulation vs native (per-run totals over time, and final A x-profiles), with
  matched Poisson initial placement;
- both vs the deterministic continuum limit (the same model with A continuous).

Writes results/metrics.json and viz/bimolecular_hybrid.html.
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import numpy as np

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[2]
STUDY_SLUG = "bimolecular-hybrid"
INVESTIGATION_SLUG = "cosim-vs-embedded-hybrid"
SPEC_ID = "viva_pde_particle.composites.examples.bimolecular_hybrid"

from vivarium_workbench.lib.run_log import append_run_event  # noqa: E402

from viva_pde_particle.analysis import (  # noqa: E402
    continuum_model,
    molecules,
    run_continuum,
    run_ensemble,
    two_sample_z,
    write_metrics,
)
from viva_pde_particle.composites.examples import bimolecular_model  # noqa: E402
from viva_pde_particle.reference.vcell_native import hybrid_support_available  # noqa: E402

DT, T_END, RECORD = 0.01, 3.0, 0.25
SEEDS = range(1, 17)


def main() -> int:
    run_id = uuid.uuid4().hex
    native = hybrid_support_available()
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "started", "spec_id": SPEC_ID, "label": STUDY_SLUG,
        "started_at": time.time(), "status": "running", "emitter": "none", "origin": "canonical_run",
        "study_slug": STUDY_SLUG, "investigation_slug": INVESTIGATION_SLUG,
        "params": {"dt": DT, "t_end": T_END, "seeds": len(SEEDS), "native": native},
    })
    started = time.time()
    try:
        model = bimolecular_model()
        g = model.grid
        workers = os.cpu_count() or 1
        runs = {}
        times, f, c = run_ensemble(model, SEEDS, T_END, DT, record_every=RECORD, workers=workers,
                                   particle_init="poisson")
        runs["cosim"] = (c["A"].reshape(len(SEEDS), len(times), -1).sum(axis=2), molecules(g, f["B"]),
                         molecules(g, f["C"]), c["A"][:, -1].sum(axis=(1, 2)))
        if native:
            from viva_pde_particle.reference.vcell_native import run_native_ensemble

            t_n, f, c = run_native_ensemble(model, SEEDS, t_end=T_END, dt=DT, output_dt=RECORD, workers=workers)
            assert np.allclose(t_n, times)
            runs["native"] = (c["A"].reshape(len(SEEDS), len(times), -1).sum(axis=2), molecules(g, f["B"]),
                              molecules(g, f["C"]), c["A"][:, -1].sum(axis=(1, 2)))
        _, cont = run_continuum(continuum_model(model), T_END, DT, record_every=RECORD)
        cont_a = molecules(g, cont["A"])
        cont_c = molecules(g, cont["C"])

        metrics: dict[str, float] = {}
        for name, (a, b, cc, prof) in runs.items():
            frac = a / a[:, :1]  # surviving fraction per run (removes initial Poisson spread)
            frac_cont = cont_a / cont_a[0]
            metrics[f"a4_{name}_vs_continuum_survival_max_abs_diff"] = float(np.abs(frac.mean(axis=0) - frac_cont).max())
            metrics[f"a4_{name}_mass_balance_max_rel_dev"] = float(
                np.abs((cc + b).mean(axis=0) / (cc[:, 0] + b[:, 0]).mean() - 1).max())  # B + C conserved
        if "native" in runs:
            (ac, bc, ccc, pc), (an, bn, ccn, pn) = runs["cosim"], runs["native"]
            fc, fn = ac / ac[:, :1], an / an[:, :1]
            metrics["a4_cosim_vs_native_survival_max_z"] = float(np.abs(two_sample_z(fc, fn)).max())
            metrics["a4_cosim_vs_native_c_max_z"] = float(np.abs(two_sample_z(ccc / ac[:, :1], ccn / an[:, :1])).max())
            pcn, pnn = pc / ac[:, :1], pn / an[:, :1]
            metrics["a4_cosim_vs_native_profile_max_z"] = float(np.abs(two_sample_z(pcn, pnn)).max())
        metrics["a4_wall_time_s"] = time.time() - started
        write_metrics(STUDY_DIR / "results" / "metrics.json", metrics)

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Surviving A fraction", "Final A per x-column (/ initial)"))
        fig.add_trace(go.Scatter(x=times, y=cont_a / cont_a[0], name="continuum", mode="lines"), 1, 1)
        for name, (a, _, _, prof) in runs.items():
            fig.add_trace(go.Scatter(x=times, y=(a / a[:, :1]).mean(axis=0), name=name, mode="markers"), 1, 1)
            fig.add_trace(go.Scatter(y=(prof / a[:, :1]).mean(axis=0), name=f"{name} profile",
                                     mode="lines+markers"), 1, 2)
        fig.update_layout(title="A4: A_particle + B_field → C_field", height=420)
        (STUDY_DIR / "viz").mkdir(parents=True, exist_ok=True)
        (STUDY_DIR / "viz" / "bimolecular_hybrid.html").write_text(fig.to_html(include_plotlyjs="cdn", full_html=True))
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
