"""Study C2 (coupled-sparks-fast-diffusion): Schaff et al. 2016, Test 2 / Fig 2, on both solvers.

Channel opening is coupled to calcium (k_on·U(rᵢ)/U₀, k_on = 0.1 s⁻¹) with fast diffusion
(D = 1000 µm²/s) and Δt = 0.2 ms. In this limit the spatial hybrid should match the
well-mixed piecewise-deterministic process, sampled exactly by an event-driven method
(viva_pde_particle.benchmarks.pdmp). Compared at t = 1, 2, 3 s:

- the distributions of the volume-averaged calcium U and of the open-channel count n
  (two-sample KS, and the L2 distance between histograms as in the paper);
- mean z-scores against the reference;
- co-simulation vs native.

Writes results/metrics.json and viz/calcium_sparks_fig2.html.
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import numpy as np
from scipy import stats

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[2]
STUDY_SLUG = "coupled-sparks-fast-diffusion"
INVESTIGATION_SLUG = "vcell-hybrid-paper-benchmarks"
SPEC_ID = "viva_pde_particle.composites.examples.calcium_sparks"

from vivarium_workbench.lib.run_log import append_run_event  # noqa: E402

from viva_pde_particle.analysis import run_ensemble, write_metrics  # noqa: E402
from viva_pde_particle.benchmarks.calcium_sparks import TEST2, calcium_sparks_model  # noqa: E402
from viva_pde_particle.benchmarks.pdmp import simulate_well_mixed  # noqa: E402
from viva_pde_particle.reference.vcell_native import hybrid_support_available  # noqa: E402

DT = 2e-4
TIMES = [1.0, 2.0, 3.0]
N_TRIALS = int(os.environ.get("C2_TRIALS", "500"))
N_REFERENCE = 20000
WORKERS = os.cpu_count() or 1


def hybrid_samples(native: bool):
    """(U_mean, n_open) per solver, each shaped (trials, len(TIMES))."""
    model = calcium_sparks_model(TEST2)
    g = model.grid
    w = g.element_volumes / g.element_volumes.sum()
    seeds = range(1, N_TRIALS + 1)
    out = {}
    _, f, c = run_ensemble(model, seeds, TIMES[-1], DT, record_every=1.0, workers=WORKERS)
    out["cosim"] = ((f["U"][:, 1:] * w).sum(axis=(2, 3, 4)), c["O"][:, 1:].sum(axis=(2, 3, 4)))
    if native:
        from viva_pde_particle.reference.vcell_native import run_native_ensemble

        _, f, c = run_native_ensemble(model, seeds, t_end=TIMES[-1], dt=DT, output_dt=1.0, workers=WORKERS)
        out["native"] = ((f["U"][:, 1:] * w).sum(axis=(2, 3, 4)), c["O"][:, 1:].sum(axis=(2, 3, 4)))
    return out


def l2_hist(a: np.ndarray, b: np.ndarray, bins) -> float:
    """L2 distance between normalized histograms (densities) on common bins, as in the paper."""
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    width = np.diff(bins)
    return float(np.sqrt(((ha - hb) ** 2 * width).sum()))


def mean_z(x: np.ndarray, ref: np.ndarray) -> float:
    return float((x.mean() - ref.mean()) / np.sqrt(x.var(ddof=1) / len(x) + ref.var(ddof=1) / len(ref)))


def write_viz(samples, ref_U, ref_n):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=len(TIMES), subplot_titles=[f"p(U, t={t:g} s)" for t in TIMES]
                        + [f"P(n, t={t:g} s)" for t in TIMES])
    n_bins = np.arange(-0.5, 25.5, 1.0)
    for k, t in enumerate(TIMES):
        u_bins = np.linspace(0, max(ref_U[:, k].max(), 1e-3), 40)
        series = {"well-mixed PDMP (exact)": (ref_U[:, k], ref_n[:, k])}
        series.update({name: (u[:, k], n[:, k]) for name, (u, n) in samples.items()})
        for name, (u, n) in series.items():
            hu, _ = np.histogram(u, bins=u_bins, density=True)
            hn, _ = np.histogram(n, bins=n_bins, density=True)
            fig.add_trace(go.Scatter(x=0.5 * (u_bins[1:] + u_bins[:-1]), y=hu, name=name, mode="lines",
                                     legendgroup=name, showlegend=(k == 0)), 1, k + 1)
            fig.add_trace(go.Scatter(x=np.arange(25), y=hn, name=name, mode="lines+markers",
                                     legendgroup=name, showlegend=False), 2, k + 1)
    fig.update_layout(title="C2: coupled calcium sparks, fast diffusion (Schaff et al. 2016, Test 2 / Fig 2)",
                      height=640)
    out = STUDY_DIR / "viz" / "calcium_sparks_fig2.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(fig.to_html(include_plotlyjs="cdn", full_html=True))


def main() -> int:
    run_id = uuid.uuid4().hex
    native = hybrid_support_available()
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "started", "spec_id": SPEC_ID, "label": STUDY_SLUG,
        "started_at": time.time(), "status": "running", "emitter": "none", "origin": "canonical_run",
        "study_slug": STUDY_SLUG, "investigation_slug": INVESTIGATION_SLUG,
        "params": {"dt": DT, "times": TIMES, "trials": N_TRIALS, "native": native},
    })
    started = time.time()
    try:
        g = calcium_sparks_model(TEST2).grid
        volume = float(np.prod(g.size))
        ref_U, ref_n = simulate_well_mixed(TEST2, 24, volume, np.array(TIMES), N_REFERENCE, seed=2016)
        samples = hybrid_samples(native)
        metrics: dict[str, float] = {}
        for name, (u, n) in samples.items():
            ks_u, ks_n, z_u, z_n, l2_u, l2_n = [], [], [], [], [], []
            for k in range(len(TIMES)):
                ks_u.append(stats.ks_2samp(u[:, k], ref_U[:, k]).pvalue)
                ks_n.append(stats.ks_2samp(n[:, k], ref_n[:, k]).pvalue)
                z_u.append(abs(mean_z(u[:, k], ref_U[:, k])))
                z_n.append(abs(mean_z(n[:, k], ref_n[:, k])))
                l2_u.append(l2_hist(u[:, k], ref_U[:, k], np.linspace(0, ref_U[:, k].max(), 40)))
                l2_n.append(l2_hist(n[:, k], ref_n[:, k], np.arange(-0.5, 25.5, 1.0)))
            metrics[f"c2_{name}_min_ks_pvalue"] = float(min(ks_u + ks_n))
            metrics[f"c2_{name}_max_mean_z"] = float(max(z_u + z_n))
            metrics[f"c2_{name}_max_l2_n"] = float(max(l2_n))
            metrics[f"c2_{name}_max_l2_u"] = float(max(l2_u))
        if "native" in samples:
            (uc, nc), (un, nn) = samples["cosim"], samples["native"]
            metrics["c2_cosim_vs_native_min_ks_pvalue"] = float(min(
                [stats.ks_2samp(uc[:, k], un[:, k]).pvalue for k in range(len(TIMES))]
                + [stats.ks_2samp(nc[:, k], nn[:, k]).pvalue for k in range(len(TIMES))]))
        metrics["c2_trials_per_solver"] = float(N_TRIALS)
        metrics["c2_wall_time_s"] = time.time() - started
        write_metrics(STUDY_DIR / "results" / "metrics.json", metrics)
        write_viz(samples, ref_U, ref_n)
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
