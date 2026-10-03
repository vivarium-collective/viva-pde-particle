"""Study A3 (two-way-exchange): mass balance across the particle <-> field handoff, both solvers.

A_particle -> B_field (k1 = 1/s) and B_field -> A_particle (k2·[B], k2 = 0.5/s), with B
starting at 0.2 µM and no A, on the 11×11×3 slab:

- Smoldyn removes A (1st order) and creates A per grid node (0th order, rate ∝ local [B]);
- the PDE adds k1·[A_binned] to B and removes k2·[B].

Neither side exchanges fluxes, so total A + B is conserved only in expectation, up to the
explicit coupling's lag. At steady state, total A / total B = k2 / k1.

A creation-only experiment isolates the field -> particle creation path: B is a static
catalyst, so the created A has an exact expectation k2·[B]·V·602.214·t. It measures each
solver's creation against that. vcell-fvsolver's embedded Smoldyn uses the full grid-cell
volume at boundary nodes, whose real cells are 1/2-1/8 size, so it overproduces by
Σ(full cells)/Σ(actual cells). The patched Smoldyn of the co-simulation uses each node's
actual cell.

Writes results/metrics.json and viz/two_way_exchange.html.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[2]
STUDY_SLUG = "two-way-exchange"
INVESTIGATION_SLUG = "cosim-vs-embedded-hybrid"
SPEC_ID = "viva_pde_particle.composites.examples.two_way_exchange"

from vivarium_workbench.lib.run_log import append_run_event  # noqa: E402

from viva_pde_particle.analysis import molecules, run_ensemble, two_sample_z, write_metrics  # noqa: E402
from viva_pde_particle.composites.examples import two_way_exchange_model  # noqa: E402
from viva_pde_particle.model import HybridModel, Reaction, Species  # noqa: E402
from viva_pde_particle.reference.vcell_native import hybrid_support_available  # noqa: E402

K1, K2, B0 = 1.0, 0.5, 0.2
DT, T_END, RECORD = 0.01, 5.0, 0.25
SEEDS = range(1, 17)


def run(native: bool):
    import os

    model = two_way_exchange_model(K1, K2, B0)
    g = model.grid
    total0 = float(molecules(g, np.full(g.shape, B0)))
    workers = os.cpu_count() or 1
    out = {}
    times, f, c = run_ensemble(model, SEEDS, T_END, DT, record_every=RECORD, workers=workers)
    out["cosim"] = (c["A"].sum(axis=(2, 3, 4)), molecules(g, f["B"]), c["A"][:, -1].sum(axis=(1, 2)))
    if native:
        from viva_pde_particle.reference.vcell_native import run_native_ensemble

        t_n, f, c = run_native_ensemble(model, SEEDS, t_end=T_END, dt=DT, output_dt=RECORD, workers=workers)
        assert np.allclose(t_n, times)
        out["native"] = (c["A"].sum(axis=(2, 3, 4)), molecules(g, f["B"]), c["A"][:, -1].sum(axis=(1, 2)))
    return times, total0, out


def creation_only(native: bool):
    """Created A / exact expectation per solver, with B a static catalyst (no decay of A)."""
    import os

    base = two_way_exchange_model(K1, K2, B0)
    g = base.grid
    model = HybridModel(g, [Species("A", 0.0, particle=True, initial=0), Species("B", 0.0, initial=B0)],
                        [Reaction("make", {"B": 1}, {"A": 1, "B": 1}, k=K2)])
    t_end, seeds, workers = 2.0, range(1, 9), os.cpu_count() or 1
    expected = K2 * float(molecules(g, np.full(g.shape, B0))) * t_end
    out = {}
    _, _, c = run_ensemble(model, seeds, t_end, DT, record_every=t_end, workers=workers)
    out["cosim"] = c["A"][:, -1].reshape(len(seeds), -1).sum(axis=1) / expected
    if native:
        from viva_pde_particle.reference.vcell_native import run_native_ensemble

        _, _, c = run_native_ensemble(model, seeds, t_end=t_end, dt=DT, output_dt=t_end, workers=workers)
        out["native"] = c["A"][:, -1].reshape(len(seeds), -1).sum(axis=1) / expected
    full_cells = g.num_elements * g.full_volume
    bug_factor = full_cells / g.element_volumes.sum()
    return out, float(bug_factor)


def write_viz(times, total0, out):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        "Total A (particles) and B (molecules)", "Mass balance (A + B) / initial", "Final A per x-column"))
    a_ss = total0 * K2 / (K1 + K2)
    fig.add_trace(go.Scatter(x=times, y=[a_ss] * len(times), name="A steady state", mode="lines",
                             line={"dash": "dash"}), 1, 1)
    for name, (a, b, prof) in out.items():
        fig.add_trace(go.Scatter(x=times, y=a.mean(axis=0), name=f"{name} A", mode="lines+markers"), 1, 1)
        fig.add_trace(go.Scatter(x=times, y=b.mean(axis=0), name=f"{name} B", mode="lines"), 1, 1)
        fig.add_trace(go.Scatter(x=times, y=(a + b).mean(axis=0) / total0, name=f"{name} A+B", mode="lines"), 1, 2)
        fig.add_trace(go.Scatter(y=prof.mean(axis=0), name=f"{name} profile", mode="lines+markers"), 1, 3)
    fig.update_xaxes(title_text="t (s)", row=1, col=1)
    fig.update_xaxes(title_text="t (s)", row=1, col=2)
    fig.update_xaxes(title_text="x node", row=1, col=3)
    fig.update_layout(title="A3: two-way exchange (particle ↔ field)", height=420)
    path = STUDY_DIR / "viz" / "two_way_exchange.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(fig.to_html(include_plotlyjs="cdn", full_html=True))


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
    try:
        times, total0, out = run(native)
        metrics: dict[str, float] = {}
        target = K2 / K1
        for name, (a, b, _) in out.items():
            balance = (a + b).mean(axis=0) / total0
            metrics[f"a3_{name}_mass_balance_max_rel_dev"] = float(np.abs(balance - 1).max())
            late = times >= 4.0  # ~6 relaxation times
            ratio = a[:, late].mean() / b[:, late].mean()
            metrics[f"a3_{name}_steady_ratio_rel_err"] = float(abs(ratio - target) / target)
        created, bug_factor = creation_only(native)
        metrics["a3_full_to_actual_cell_volume_factor"] = bug_factor
        for name, ratio in created.items():
            metrics[f"a3_{name}_creation_ratio"] = float(ratio.mean())
        if "native" in created:
            metrics["a3_native_creation_vs_bug_factor"] = float(created["native"].mean() / bug_factor)
        if "native" in out:
            (ac, _, pc), (an, _, pn) = out["cosim"], out["native"]
            metrics["a3_cosim_vs_native_totals_max_z"] = float(np.abs(two_sample_z(ac, an)).max())
            metrics["a3_cosim_vs_native_profile_max_z"] = float(np.abs(two_sample_z(pc, pn)).max())
        metrics["a3_wall_time_s"] = time.time() - started
        write_metrics(STUDY_DIR / "results" / "metrics.json", metrics)
        write_viz(times, total0, out)
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
