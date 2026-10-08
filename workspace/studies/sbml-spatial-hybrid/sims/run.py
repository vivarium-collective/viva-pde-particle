"""Study B6 (sbml-spatial-hybrid): the A3 two-way exchange, described in SBML Spatial and run on both solvers.

The model goes to SBML Spatial with each species' representation annotated (A particle, B continuous) and comes
back two ways:
- **native:** SBML → VCell's SBML import → libvcell → vcell-fvsolver (``run_native(..., sbml=...)``);
- **co-sim:** SBML → ``from_sbml`` → ``HybridModel`` → the co-simulation.

Each is compared, seed for seed, with the same solver run from the original model (``to_biomodel`` VCML, and the
Python ``HybridModel``). Needs libvcell built with VCell's SBML hybrid support (virtualcell/vcell#2175).

Writes results/metrics.json and viz/sbml_spatial_hybrid.html.
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import numpy as np

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[2]
STUDY_SLUG = "sbml-spatial-hybrid"
INVESTIGATION_SLUG = "hybrid-generalizability"
SPEC_ID = "viva_pde_particle.composites.examples.two_way_exchange"

from vivarium_workbench.lib.run_log import append_run_event  # noqa: E402

from viva_pde_particle.analysis import molecules, run_ensemble, two_sample_z, write_metrics  # noqa: E402
from viva_pde_particle.composites.examples import two_way_exchange_model  # noqa: E402
from viva_pde_particle.model.sbml import from_sbml, to_sbml  # noqa: E402
from viva_pde_particle.reference.vcell_native import hybrid_support_available  # noqa: E402

K1, K2, B0 = 1.0, 0.5, 0.2
DT, T_END, RECORD = 0.01, 2.0, 0.25
SEEDS = range(1, 9)


def totals(model, fields, counts):
    """(A particles, B molecules) per seed and time."""
    return counts["A"].sum(axis=(2, 3, 4)), molecules(model.grid, fields["B"])


def main() -> int:
    run_id = uuid.uuid4().hex
    native = hybrid_support_available()
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "started", "spec_id": SPEC_ID, "label": STUDY_SLUG,
        "started_at": time.time(), "status": "running", "emitter": "none", "origin": "canonical_run",
        "study_slug": STUDY_SLUG, "investigation_slug": INVESTIGATION_SLUG,
        "params": {"k1": K1, "k2": K2, "b0": B0, "dt": DT, "t_end": T_END, "native": native},
    })
    started = time.time()
    workers = os.cpu_count() or 1
    try:
        model = two_way_exchange_model(K1, K2, B0)
        sbml = to_sbml(model)
        (STUDY_DIR / "results").mkdir(parents=True, exist_ok=True)
        (STUDY_DIR / "results" / "two_way_exchange.sbml").write_text(sbml)
        loaded = from_sbml(sbml, num=model.grid.num).model
        metrics: dict[str, float] = {
            "b6_model_round_trip_exact": float(loaded == model),
        }

        runs = {}
        t, f, c = run_ensemble(model, SEEDS, T_END, DT, record_every=RECORD, workers=workers)
        runs["cosim"] = totals(model, f, c)
        _, f, c = run_ensemble(loaded, SEEDS, T_END, DT, record_every=RECORD, workers=workers)
        runs["cosim_sbml"] = totals(loaded, f, c)
        if native:
            from viva_pde_particle.reference.vcell_native import run_native_ensemble

            tn, f, c = run_native_ensemble(model, SEEDS, t_end=T_END, dt=DT, output_dt=RECORD, workers=workers)
            assert np.allclose(tn, t)
            runs["native"] = totals(model, f, c)
            _, f, c = run_native_ensemble(loaded, SEEDS, t_end=T_END, dt=DT, output_dt=RECORD, workers=workers,
                                          sbml=sbml)
            runs["native_sbml"] = totals(loaded, f, c)

        pairs = [("cosim", "cosim_sbml")] + ([("native", "native_sbml"), ("cosim_sbml", "native_sbml")] if native else [])
        for x, y in pairs:
            (ax, bx), (ay, by) = runs[x], runs[y]
            metrics[f"b6_{x}_vs_{y}_identical"] = float(np.array_equal(ax, ay) and np.allclose(bx, by, rtol=1e-12))
            metrics[f"b6_{x}_vs_{y}_a_max_z"] = float(np.abs(two_sample_z(ax, ay)).max())
            metrics[f"b6_{x}_vs_{y}_b_max_z"] = float(np.abs(two_sample_z(bx, by)).max())
        for name, (a, b) in runs.items():
            metrics[f"b6_{name}_a_final_mean"] = float(a[:, -1].mean())
            metrics[f"b6_{name}_mass_balance_final_rel"] = float((a + b)[:, -1].mean() / (a + b)[:, 0].mean() - 1)
        metrics["b6_wall_time_s"] = time.time() - started
        write_metrics(STUDY_DIR / "results" / "metrics.json", metrics)
        write_viz(t, runs)
    except Exception:
        append_run_event(WORKSPACE_ROOT, {"run_id": run_id, "event": "completed",
                                          "completed_at": time.time(), "status": "failed"})
        raise
    append_run_event(WORKSPACE_ROOT, {"run_id": run_id, "event": "completed",
                                      "completed_at": time.time(), "status": "completed"})
    for k, v in metrics.items():
        print(f"[{STUDY_SLUG}] {k} = {v:.6g}")
    return 0


def write_viz(times, runs):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Total A (particles)", "Total B (molecules)"))
    styles = {"cosim": "solid", "cosim_sbml": "dot", "native": "solid", "native_sbml": "dot"}
    for name, (a, b) in runs.items():
        line = {"dash": styles[name]}
        fig.add_trace(go.Scatter(x=times, y=a.mean(axis=0), name=f"{name} A", mode="lines+markers", line=line), 1, 1)
        fig.add_trace(go.Scatter(x=times, y=b.mean(axis=0), name=f"{name} B", mode="lines+markers", line=line), 1, 2)
    fig.update_xaxes(title_text="t (s)")
    fig.update_layout(title="B6: the A3 two-way exchange from SBML Spatial (dotted) and from the original model (solid)",
                      height=420)
    path = STUDY_DIR / "viz" / "sbml_spatial_hybrid.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fig.to_html(include_plotlyjs="cdn", full_html=True))


if __name__ == "__main__":
    raise SystemExit(main())
