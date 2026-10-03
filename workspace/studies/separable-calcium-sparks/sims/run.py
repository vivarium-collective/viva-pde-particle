"""Study C1 (separable-calcium-sparks): Schaff et al. 2016, Test 1 / Fig 1B, on both solvers.

24 two-state channels (opening k_on, closing k_off, independent of U) drive calcium
influx J at their nodes. Diffusion D = 1 and pumps V_p(U − U₀). The trial-mean U is
compared to the exact expectation E[U] (the linear system driven by the exact open
probability; see viva_pde_particle.benchmarks.calcium_sparks.separable_expectation):

- ε(N) at Δt = 0.005 for N = 4 ... 128 trials, with the log-log slope (theory −1/2);
- ε(Δt) at N = 128 for Δt = 0.02, 0.01, 0.005;
- whether the co-simulation and the native VCell hybrid differ by more than sampling explains.

Runs the native solver only where pyvcell has spatial-hybrid support (the ``dev`` pixi env).
Writes results/metrics.json and viz/calcium_sparks_fig1b.html.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[2]
STUDY_SLUG = "separable-calcium-sparks"
INVESTIGATION_SLUG = "vcell-hybrid-paper-benchmarks"
SPEC_ID = "viva_pde_particle.benchmarks.calcium_sparks"

from vivarium_workbench.lib.run_log import append_run_event  # noqa: E402

from viva_pde_particle.analysis import run_ensemble, write_metrics  # noqa: E402
from viva_pde_particle.benchmarks.calcium_sparks import (  # noqa: E402
    TEST1,
    calcium_sparks_model,
    channel_nodes,
    separable_expectation,
    spark_error,
)
from viva_pde_particle.reference.vcell_native import hybrid_support_available  # noqa: E402

T_END = 5.0
RECORD = 0.25
N_MAX = 128
N_SERIES = [4, 8, 16, 32, 64, 128]
DT_SERIES = [0.02, 0.01, 0.005]
DT_N_SERIES = 0.005


def ensembles(dt: float, native: bool):
    model = calcium_sparks_model(TEST1)
    seeds = range(1, N_MAX + 1)
    _, f_c, _ = run_ensemble(model, seeds, T_END, dt, record_every=RECORD)
    out = {"cosim": f_c["U"]}
    if native:
        from viva_pde_particle.reference.vcell_native import run_native_ensemble

        _, f_n, _ = run_native_ensemble(model, seeds, t_end=T_END, dt=dt, output_dt=RECORD)
        out["native"] = f_n["U"]
    return out


def eps_vs_n(u: np.ndarray, ref: np.ndarray) -> list[float]:
    return [spark_error(u[:n].mean(axis=0), ref, TEST1.U0) for n in N_SERIES]


def slope(ns, eps) -> float:
    return float(np.polyfit(np.log(ns), np.log(eps), 1)[0])


def write_viz(eps_n: dict, eps_dt: dict, times, u_mean: dict, ref):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        "ε vs trials N (Δt = 0.005 s)", "ε vs Δt (N = 128)", "U at a channel node: trial mean vs E[U]"))
    ns = np.array(N_SERIES)
    for name, eps in eps_n.items():
        fig.add_trace(go.Scatter(x=ns, y=eps, name=f"{name}", mode="lines+markers"), 1, 1)
    first = next(iter(eps_n.values()))
    fig.add_trace(go.Scatter(x=ns, y=first[0] * np.sqrt(ns[0] / ns), name="∝ N^-1/2",
                             mode="lines", line={"dash": "dash"}), 1, 1)
    for name, eps in eps_dt.items():
        fig.add_trace(go.Scatter(x=DT_SERIES, y=eps, name=f"{name} (Δt)", mode="lines+markers"), 1, 2)
    node = tuple(int(i[0]) for i in channel_nodes(calcium_sparks_model().grid))
    fig.add_trace(go.Scatter(x=times, y=ref[(slice(None),) + node], name="E[U]", mode="lines"), 1, 3)
    for name, u in u_mean.items():
        fig.add_trace(go.Scatter(x=times, y=u[(slice(None),) + node], name=f"{name} mean", mode="markers"), 1, 3)
    fig.update_xaxes(type="log", title_text="N", row=1, col=1)
    fig.update_yaxes(type="log", title_text="ε", row=1, col=1)
    fig.update_xaxes(type="log", title_text="Δt (s)", row=1, col=2)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_xaxes(title_text="t (s)", row=1, col=3)
    fig.update_layout(title="C1: separable calcium sparks (Schaff et al. 2016, Test 1 / Fig 1B)", height=440)
    out = STUDY_DIR / "viz" / "calcium_sparks_fig1b.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(fig.to_html(include_plotlyjs="cdn", full_html=True))


def main() -> int:
    run_id = uuid.uuid4().hex
    native = hybrid_support_available()
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "started", "spec_id": SPEC_ID, "label": STUDY_SLUG,
        "started_at": time.time(), "status": "running", "emitter": "none", "origin": "canonical_run",
        "study_slug": STUDY_SLUG, "investigation_slug": INVESTIGATION_SLUG,
        "params": {"t_end": T_END, "n_max": N_MAX, "dt_series": DT_SERIES, "native": native},
    })
    started = time.time()
    try:
        times = np.arange(0.0, T_END + 1e-9, RECORD)
        ref = separable_expectation(times)
        metrics: dict[str, float] = {}
        eps_dt: dict[str, list[float]] = {}
        eps_n: dict[str, list[float]] = {}
        u_mean: dict[str, np.ndarray] = {}
        for dt in DT_SERIES:
            runs = ensembles(dt, native)
            for name, u in runs.items():
                eps_dt.setdefault(name, []).append(spark_error(u.mean(axis=0), ref, TEST1.U0))
                if dt == DT_N_SERIES:
                    eps_n[name] = eps_vs_n(u, ref)
                    u_mean[name] = u.mean(axis=0)
            if dt == DT_N_SERIES and native:
                between = spark_error(runs["cosim"].mean(axis=0), runs["native"].mean(axis=0), TEST1.U0)
                expected = np.hypot(eps_n["cosim"][-1], eps_n["native"][-1])
                metrics["c1_cosim_vs_native_eps_ratio"] = float(between / expected)
        for name in eps_n:
            metrics[f"c1_{name}_eps_slope"] = slope(N_SERIES, eps_n[name])
            metrics[f"c1_{name}_eps_n128"] = eps_n[name][-1]
            for dt, e in zip(DT_SERIES, eps_dt[name]):
                metrics[f"c1_{name}_eps_dt{dt:g}"] = e
        metrics["c1_wall_time_s"] = time.time() - started
        write_metrics(STUDY_DIR / "results" / "metrics.json", metrics)
        write_viz(eps_n, eps_dt, times, u_mean, ref)
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
