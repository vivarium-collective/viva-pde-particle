"""Study A2 (field-modulated-decay): A_particle -> 0 at rate k·[B](x), B a PDE field.

Part 1 (exact): static linear B, immobile A. Each x-column survives as
exp(-k·B_col·t), which tests that Smoldyn reads the field at the molecule's node.
Part 2 (continuum limit): diffusing A and a diffusing Gaussian B. The ensemble-mean A
concentration should approach the all-continuous FV solution of the same model.

Writes results/metrics.json and viz/field_modulated_decay.html.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[2]
STUDY_SLUG = "field-modulated-decay"
INVESTIGATION_SLUG = "cosim-vs-embedded-hybrid"
SPEC_ID = "viva_pde_particle.composites.examples.field_modulated_decay"

from vivarium_workbench.lib.run_log import append_run_event  # noqa: E402

from viva_pde_particle.analysis import continuum_model, run_continuum, run_ensemble, write_metrics  # noqa: E402
from viva_pde_particle.composites.examples import field_modulated_decay_model  # noqa: E402
from viva_pde_particle.units import MOLECULES_PER_UM3_PER_UM  # noqa: E402

DT = 0.01
T_END = 1.0
K = 1.0
SEEDS = range(1, 9)


def static_field_columns():
    model = field_modulated_decay_model(n_particles=4000, k=K, d_a=0.0, d_b=0.0, b_profile="linear")
    times, fields, counts = run_ensemble(model, SEEDS, T_END, DT, record_every=T_END)
    b_col = fields["B"][0, 0, 0, :]  # B depends on x only
    n0 = counts["A"][:, 0].sum(axis=(0, 1))  # per column, summed over seeds and rows
    n1 = counts["A"][:, -1].sum(axis=(0, 1))
    p = np.exp(-K * b_col * T_END)
    expected = n0 * p
    sd = np.sqrt(n0 * p * (1 - p))
    z = np.where(sd > 0, (n1 - expected) / np.where(sd > 0, sd, 1), 0.0)
    zero_field_decays = float(n0[b_col == 0].sum() - n1[b_col == 0].sum())
    return float(np.abs(z).max()), zero_field_decays, b_col, n1 / n0, p


def continuum_limit():
    model = field_modulated_decay_model(n_particles=20000, k=K, d_a=1.0, d_b=0.5, b_profile="gaussian")
    g = model.grid
    times, _, counts = run_ensemble(model, SEEDS, T_END, DT, record_every=0.25)
    a_hybrid = np.stack([g.counts_to_uM(c) for c in counts["A"][:, -1]]).mean(axis=0)
    t_c, cont = run_continuum(continuum_model(model), T_END, DT, record_every=0.25)
    a_cont = cont["A"][-1]
    rel_l2 = float(np.sqrt(((a_hybrid - a_cont) ** 2 * g.element_volumes).sum()
                           / (a_cont**2 * g.element_volumes).sum()))
    tot_h = counts["A"].sum(axis=(2, 3)).mean(axis=0)
    tot_c = (cont["A"] * g.element_volumes * MOLECULES_PER_UM3_PER_UM).sum(axis=(1, 2))
    total_rel = float(abs(tot_h[-1] - tot_c[-1]) / tot_c[-1])
    return rel_l2, total_rel, times, tot_h, tot_c, a_hybrid, a_cont


def write_viz(b_col, surv, p, times, tot_h, tot_c, a_h, a_c):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        "Static B: column survival vs exp(-k·B·t)", "Diffusing A and B: total A",
        "Final A (µM), y = L/2: hybrid mean vs continuum"))
    fig.add_trace(go.Scatter(x=b_col, y=p, name="exp(-k B t)", mode="lines"), 1, 1)
    fig.add_trace(go.Scatter(x=b_col, y=surv, name="co-sim", mode="markers"), 1, 1)
    fig.add_trace(go.Scatter(x=times, y=tot_c, name="continuum", mode="lines"), 1, 2)
    fig.add_trace(go.Scatter(x=times, y=tot_h, name="co-sim mean", mode="markers"), 1, 2)
    mid = a_c.shape[0] // 2
    fig.add_trace(go.Scatter(y=a_c[mid], name="continuum", mode="lines"), 1, 3)
    fig.add_trace(go.Scatter(y=a_h[mid], name="co-sim mean", mode="markers"), 1, 3)
    fig.update_xaxes(title_text="B (µM)", row=1, col=1)
    fig.update_xaxes(title_text="t (s)", row=1, col=2)
    fig.update_xaxes(title_text="x node", row=1, col=3)
    fig.update_layout(title="A2: field-modulated decay", height=420)
    out = STUDY_DIR / "viz" / "field_modulated_decay.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(fig.to_html(include_plotlyjs="cdn", full_html=True))


def main() -> int:
    run_id = uuid.uuid4().hex
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "started", "spec_id": SPEC_ID, "label": STUDY_SLUG,
        "started_at": time.time(), "status": "running", "emitter": "none", "origin": "canonical_run",
        "study_slug": STUDY_SLUG, "investigation_slug": INVESTIGATION_SLUG,
        "params": {"dt": DT, "t_end": T_END, "k": K, "seeds": list(SEEDS)},
    })
    started = time.time()
    try:
        zmax, zero_decays, b_col, surv, p = static_field_columns()
        rel_l2, total_rel, times, tot_h, tot_c, a_h, a_c = continuum_limit()
        metrics = {
            "a2_static_field_max_abs_z": zmax,
            "a2_zero_field_decays": zero_decays,
            "a2_continuum_rel_l2_err": rel_l2,
            "a2_continuum_total_rel_err": total_rel,
            "a2_wall_time_s": time.time() - started,
        }
        write_metrics(STUDY_DIR / "results" / "metrics.json", metrics)
        write_viz(b_col, surv, p, times, tot_h, tot_c, a_h, a_c)
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
