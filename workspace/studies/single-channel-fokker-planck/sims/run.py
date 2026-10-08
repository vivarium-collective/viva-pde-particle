"""Study C3 (single-channel-fokker-planck): Schaff et al. 2016, Tests 3–5 / Figs 4–6.

One channel, with opening coupled to calcium (rate αβ(ρ(0) + 1)), against direct solutions of the
Fokker–Planck equations for the joint density of (ρ, channel state)
(viva_pde_particle.benchmarks.fokker_planck):

- **Test 3** (fast diffusion, d = 10⁴, 3D cube, α = β = 1, a = 24):
  - p(ρ̄) at τ = 30 against Eq 4;
  - Eq 4 is solved in time, and checked against its exact steady state;
  - 10,000 co-simulation trials, plus native VCell trials (``dev`` env).
- **Test 4** (finite diffusion, two cells, α = 10, β = 1, a = 20/3, d = 1):
  - p(ρᵢ) at τ = 1 against the functional Fokker–Planck master equation (S2 Text, Eq T2.1);
  - the master equation is solved at Δρ ∝ 1, ½, ¼ and extrapolated to Δρ → 0;
  - 12,500 trials.
- **Test 5** (three cells, α = 20, β = 0.5, a = 20, d = 50): as Test 4, at four Δρ (×1.5 apart);
  12,500 trials.

Densities are 20-bin histograms on [0, ρ_max(i)] (40 bins on [0, a] for Test 3), compared with the
reference averaged over the same bins:
- the relative L2 (RMS over bins, over the reference maximum), as in the paper;
- the reduced χ² against multinomial sampling noise. A value near 1 means the difference is
  sampling noise.

**Deviations from the paper:**
- Test 5 runs at Δτ = 1e-4, not 2e-5 (5× cheaper; switching rates × Δτ ≤ 0.004).
- Tests 4–5 run only the co-simulation. VCell's node-centred mesh gives its end nodes half volumes,
  so its discrete model differs from Eq T2.1's equal cells. The co-sim sets equal volumes
  (``element_volumes``).

Stages (each saves to ``runs/``; ``all`` runs them in order):

    pixi run python sims/run.py reference
    pixi run python sims/run.py cosim
    C3_BACKEND=compose pixi run python sims/run.py cosim   # the same trials on compose-api (SLURM on mantis)
    pixi run -e dev python sims/run.py native     # Test 3 on native VCell
    pixi run python sims/run.py report            # results/metrics.json + viz/fokker_planck.html
    pixi run python sims/run.py dtau              # the Test 3 Δτ check (C3_DTAUS, C3_BACKEND as above)

**Δτ check (Test 3):** the co-sim and native both differ from Eq 4's exact steady state by ~0.6%. The
``dtau`` stage reruns the co-sim at several Δτ and samples ρ̄ at τ = 10, 15, …, 30. Those samples are
stationary (relaxation times ~1) and nearly independent, giving five per trial. ``report`` then gives the
signed bias of the mean and SD against the exact density at each Δτ.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[2]
RUNS = STUDY_DIR / "runs"
STUDY_SLUG = "single-channel-fokker-planck"
INVESTIGATION_SLUG = "vcell-hybrid-paper-benchmarks"
SPEC_ID = "viva_pde_particle.benchmarks.fokker_planck"

from viva_pde_particle.benchmarks.fokker_planck import TEST3, TEST4, TEST5, ChannelTest  # noqa: E402

TESTS = {"test3": TEST3, "test4": TEST4, "test5": ChannelTest(**{**TEST5.__dict__, "dtau": 1e-4})}
LEVELS = {"test4": (800, 1600, 3200), "test5": (45, 68, 101, 152)}
MARGIN = 1.04
WORKERS = int(os.environ.get("C3_WORKERS", os.cpu_count() or 1))
TRIALS = {k: int(os.environ.get("C3_TRIALS", t.trials)) for k, t in TESTS.items()}
NATIVE_TRIALS = int(os.environ.get("C3_NATIVE_TRIALS", TEST3.trials))
BACKEND = os.environ.get("C3_BACKEND", "local")  # "compose": run the trials on compose-api
# Seconds per trial on one core: measured on an M3 Max (8.5, 1.4, 1.5) and padded 2× for mantis.
# compose_client.block_size turns them into blocks that fill half of a 30-minute, 2-CPU job.
SECONDS_PER_TRIAL = {"test3": 17.0, "test4": 3.0, "test5": 3.0}
DTAUS = [float(x) for x in os.environ.get("C3_DTAUS", "0.008,0.004,0.002,0.001").split(",")]
DTAU_TRIALS = int(os.environ.get("C3_DTAU_TRIALS", 4000))
DTAU_TIMES = [10.0, 15.0, 20.0, 25.0, 30.0]


def edges_for(name: str) -> list[np.ndarray]:
    from viva_pde_particle.benchmarks.fokker_planck import open_channel_reach

    t = TESTS[name]
    if t.imax == 0:
        return [np.linspace(0.0, t.a, 41)]
    return [np.linspace(0.0, r, 21) for r in MARGIN * open_channel_reach(t)]


# ---------------------------------------------------------------- reference

def reference() -> None:
    from viva_pde_particle.benchmarks.fokker_planck import binned_density, extrapolate, fast_steady_state, solve_fast, solve_master

    out = {}
    t = TESTS["test3"]
    r, p = solve_fast(t.a, t.alpha, t.beta, t.tau, n=4800, dtau=0.0025)
    e = edges_for("test3")[0]
    out["test3_ref"] = np.array([p[(r >= lo) & (r < hi)].mean() for lo, hi in zip(e[:-1], e[1:])])
    fine = np.linspace(0, t.a, 400001)[1:-1]
    exact = fast_steady_state(fine, t.a, t.alpha, t.beta)
    out["test3_steady"] = np.array([exact[(fine >= lo) & (fine < hi)].mean() for lo, hi in zip(e[:-1], e[1:])])
    for name in ("test4", "test5"):
        t = TESTS[name]
        edges = edges_for(name)
        drhos, dens = [], []
        for nm in LEVELS[name]:
            t0 = time.time()
            drho, marg = solve_master(t, nm, margin=MARGIN)
            drhos.append(drho)
            dens.append(np.concatenate([binned_density(drho, m, e) for m, e in zip(marg, edges)]))
            print(f"[{STUDY_SLUG}] {name} master n0={nm} Δρ={drho:.4g} in {time.time() - t0:.0f}s", flush=True)
        out[f"{name}_levels"] = np.array(dens)
        out[f"{name}_drho"] = np.array(drhos)
        out[f"{name}_ref"] = extrapolate(drhos, dens)
    RUNS.mkdir(parents=True, exist_ok=True)
    np.savez(RUNS / "reference.npz", **out)


# ---------------------------------------------------------------- hybrid ensembles

def _trial(args):
    name, seed = args
    from viva_pde_particle.benchmarks.fokker_planck import channel_model
    from viva_pde_particle.composites.hybrid import build_hybrid_document, run_document

    t = TESTS[name]
    g, model, volumes = channel_model(t)
    tr = run_document(build_hybrid_document(model, t.dtau, 1, seed=seed, element_volumes=volumes), t.tau, t.dtau, t.tau)
    rho = np.asarray(tr.fields["rho"][-1], dtype=float)
    if t.imax == 0:
        return np.array([(rho * g.element_volumes).sum() / g.element_volumes.sum()])  # ρ̄ over Ω
    return rho.ravel()


def _compose(name: str) -> np.ndarray:
    """The trials of one test on compose-api: ``trial_rho`` computes exactly what :func:`_trial` does.
    Resumable: ``runs/compose/<test>/`` keeps the manifest and each block's results."""
    from viva_pde_particle.compose_client import EnsembleRun, block_size

    params = {"test": name, "dtau": TESTS[name].dtau}  # Test 5 runs at Δτ = 1e-4, not its paper value
    run = EnsembleRun(RUNS / "compose" / name, "viva_pde_particle.benchmarks.fokker_planck:trial_rho", params,
                      TRIALS[name], block_size(SECONDS_PER_TRIAL[name]),
                      log=lambda m: print(f"[{STUDY_SLUG}] {name}: {m}", flush=True))
    seeds, values = run.run()
    assert seeds == list(range(1, TRIALS[name] + 1))
    return np.array(values)


def cosim() -> None:
    RUNS.mkdir(parents=True, exist_ok=True)
    only = os.environ.get("C3_TESTS")  # e.g. "test4,test5" to (re)run some tests only
    for name in TESTS if not only else only.split(","):
        t0 = time.time()
        if BACKEND == "compose":
            rho = _compose(name)
        else:
            with ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context("spawn")) as pool:
                rho = np.stack(list(pool.map(_trial, [(name, s) for s in range(1, TRIALS[name] + 1)],
                                             chunksize=25)))
        np.save(RUNS / f"cosim_{name}.npy", rho)
        print(f"[{STUDY_SLUG}] cosim {name} ({BACKEND}): {len(rho)} trials in {time.time() - t0:.0f}s", flush=True)


def _dtau_trial(args):
    from viva_pde_particle.benchmarks.fokker_planck import trial_rho

    dtau, seed = args
    return trial_rho(seed, "test3", dtau=dtau, sample_times=DTAU_TIMES)


def dtau() -> None:
    """Test 3 co-sim at each Δτ in ``C3_DTAUS``: ρ̄ at ``DTAU_TIMES``, one row per trial."""
    RUNS.mkdir(parents=True, exist_ok=True)
    for dt in DTAUS:
        t0 = time.time()
        if BACKEND == "compose":
            from viva_pde_particle.compose_client import EnsembleRun, block_size

            run = EnsembleRun(RUNS / "compose" / f"dtau_{dt:g}", "viva_pde_particle.benchmarks.fokker_planck:trial_rho",
                              {"test": "test3", "dtau": dt, "sample_times": DTAU_TIMES}, DTAU_TRIALS,
                              block_size(SECONDS_PER_TRIAL["test3"] * 0.002 / dt),
                              log=lambda m: print(f"[{STUDY_SLUG}] dtau {dt:g}: {m}", flush=True))
            seeds, values = run.run()
            assert seeds == list(range(1, DTAU_TRIALS + 1))
            rho = np.array(values)
        else:
            with ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context("spawn")) as pool:
                rho = np.array(list(pool.map(_dtau_trial, [(dt, s) for s in range(1, DTAU_TRIALS + 1)], chunksize=10)))
        np.save(RUNS / f"dtau_test3_{dt:g}.npy", rho)
        print(f"[{STUDY_SLUG}] dtau {dt:g} ({BACKEND}): {len(rho)} trials in {time.time() - t0:.0f}s", flush=True)


def native() -> None:
    from viva_pde_particle.benchmarks.fokker_planck import channel_model
    from viva_pde_particle.reference.vcell_native import hybrid_support_available, run_native_ensemble

    if not hybrid_support_available():
        raise SystemExit("needs pyvcell with spatial-hybrid support (pixi run -e dev ...)")
    t = TESTS["test3"]
    g, model, _ = channel_model(t)
    t0 = time.time()
    _, f, _ = run_native_ensemble(model, range(1, NATIVE_TRIALS + 1), t_end=t.tau, dt=t.dtau, output_dt=t.tau,
                                  workers=WORKERS)
    w = g.element_volumes / g.element_volumes.sum()
    rho = (f["rho"][:, -1] * w).sum(axis=(1, 2, 3))[:, None]
    RUNS.mkdir(parents=True, exist_ok=True)
    np.save(RUNS / "native_test3.npy", rho)
    print(f"[{STUDY_SLUG}] native test3: {len(rho)} trials in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- report

def compare(samples: np.ndarray, edges: np.ndarray, ref: np.ndarray) -> dict:
    """Histogram density vs a bin-averaged reference: relative L2, absolute L2 and reduced χ²."""
    n = len(samples)
    w = np.diff(edges)
    counts = np.histogram(np.clip(samples, edges[0], edges[-1]), bins=edges)[0]
    dens = counts / (n * w)
    q = np.clip(ref * w, 0, 1)
    var = q * (1 - q) / (n * w**2)
    ok = q * n >= 5  # bins with enough expected samples for the normal approximation
    chi2 = float(((dens - ref) ** 2 / np.where(ok, var, 1))[ok].sum() / max(ok.sum(), 1))
    return {"density": dens, "rel_l2": float(np.sqrt(np.mean((dens - ref) ** 2)) / ref.max()),
            "abs_l2": float(np.sqrt(((dens - ref) ** 2 * w).sum())), "chi2": chi2,
            "noise_rel_l2": float(np.sqrt(np.mean(var)) / ref.max())}


def _dtau_metrics(edges: np.ndarray, steady: np.ndarray, metrics: dict[str, float]) -> list[dict]:
    """Test 3 at each Δτ run by ``dtau``: the signed bias of ρ̄'s mean and SD against Eq 4's exact steady
    state, and the histogram metrics against it.

    - Standard errors are bootstrapped over trials. ρ̄'s density is skewed (∝ e^ρ for α = β = 1), so the
      normal-theory SE of the SD would be 2–4× too small.
    - The five samples of a trial are nearly independent (lag-5 correlation ≈ 0.01).
    - The mean bias is fitted as offset + slope·Δτ (weighted least squares)."""
    from viva_pde_particle.benchmarks.fokker_planck import fast_steady_state

    t = TESTS["test3"]
    fine = np.linspace(0.0, t.a, 400001)[1:-1]
    p = fast_steady_state(fine, t.a, t.alpha, t.beta)
    p /= np.trapezoid(p, fine)
    mean = float(np.trapezoid(fine * p, fine))
    sd = float(np.sqrt(np.trapezoid((fine - mean) ** 2 * p, fine)))
    rng = np.random.default_rng(0)
    rows = []
    for path in sorted(RUNS.glob("dtau_test3_*.npy"), key=lambda f: float(f.stem.rsplit("_", 1)[1])):
        dt = float(path.stem.rsplit("_", 1)[1])
        trials = np.load(path)
        x = trials.ravel()
        boot = [trials[rng.integers(0, len(trials), len(trials))].ravel() for _ in range(1000)]
        c = compare(x, edges, steady)
        row = {"dtau": dt, "n": len(x), "density": c["density"],
               "mean_bias": (x.mean() - mean) / mean, "mean_bias_se": float(np.std([b.mean() for b in boot])) / mean,
               "sd_bias": (x.std() - sd) / sd, "sd_bias_se": float(np.std([b.std() for b in boot])) / sd,
               "rel_l2": c["rel_l2"], "chi2": c["chi2"]}
        rows.append(row)
        tag = f"{dt:g}".replace(".", "p")
        for k in ("mean_bias", "mean_bias_se", "sd_bias", "sd_bias_se", "rel_l2", "chi2"):
            metrics[f"c3_dtau_{tag}_{k}"] = float(row[k])
    if len(rows) >= 3:
        d = np.array([r["dtau"] for r in rows])
        b = np.array([r["mean_bias"] for r in rows])
        w = 1 / np.array([r["mean_bias_se"] for r in rows]) ** 2
        a = np.stack([np.ones_like(d), d], axis=1)
        cov = np.linalg.inv(a.T @ (w[:, None] * a))
        offset, slope = cov @ a.T @ (w * b)
        metrics["c3_dtau_mean_bias_offset"] = float(offset)
        metrics["c3_dtau_mean_bias_offset_se"] = float(np.sqrt(cov[0, 0]))
        metrics["c3_dtau_mean_bias_slope"] = float(slope)
        metrics["c3_dtau_mean_bias_slope_se"] = float(np.sqrt(cov[1, 1]))
        metrics["c3_dtau_mean_bias_fit_chi2"] = float((w * (b - offset - slope * d) ** 2).sum() / (len(d) - 2))
    return rows


def report() -> None:
    from viva_pde_particle.analysis import write_metrics
    from viva_pde_particle.viz3d.study import logged_run

    ref = dict(np.load(RUNS / "reference.npz"))
    metrics: dict[str, float] = {}
    panels = []
    with logged_run(WORKSPACE_ROOT, STUDY_SLUG, INVESTIGATION_SLUG, SPEC_ID, STUDY_SLUG,
                    {"trials": TRIALS, "native_trials": NATIVE_TRIALS}):
        e3 = edges_for("test3")[0]
        metrics["c3_test3_ref_vs_steady_rel_l2"] = float(np.sqrt(np.mean((ref["test3_ref"] - ref["test3_steady"]) ** 2))
                                                         / ref["test3_steady"].max())
        sources = {"cosim": RUNS / "cosim_test3.npy", "native": RUNS / "native_test3.npy"}
        curves = {}
        for src, path in sources.items():
            if path.exists():
                c = compare(np.load(path)[:, 0], e3, ref["test3_ref"])
                curves[src] = c["density"]
                for k in ("rel_l2", "abs_l2", "chi2", "noise_rel_l2"):
                    metrics[f"c3_test3_{src}_{k}"] = c[k]
        panels.append(("Test 3: ρ̄ at τ = 30 (d = 10⁴)", e3, ref["test3_ref"], curves))
        for name in ("test4", "test5"):
            edges = edges_for(name)
            samples = np.load(RUNS / f"cosim_{name}.npy")
            off = 0
            for i, e in enumerate(edges):
                r = ref[f"{name}_ref"][off:off + len(e) - 1]
                finest = ref[f"{name}_levels"][-1][off:off + len(e) - 1]
                off += len(e) - 1
                c = compare(samples[:, i], e, r)
                for k in ("rel_l2", "abs_l2", "chi2", "noise_rel_l2"):
                    metrics[f"c3_{name}_i{i}_cosim_{k}"] = c[k]
                metrics[f"c3_{name}_i{i}_extrapolation_shift"] = float(np.sqrt(np.mean((finest - r) ** 2)) / r.max())
                panels.append((f"{name.title().replace('t', 'T', 1)}: ρ(x{i}) at τ = 1", e, r, {"cosim": c["density"]}))
        dtau_rows = _dtau_metrics(e3, ref["test3_steady"], metrics)
        write_metrics(STUDY_DIR / "results" / "metrics.json", metrics)

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=2, cols=3, subplot_titles=[p[0] for p in panels])
        colors = {"cosim": "#1f77b4", "native": "#d62728"}
        for k, (title, e, r, curves) in enumerate(panels):
            row, col = divmod(k, 3)
            mid = 0.5 * (e[1:] + e[:-1])
            fig.add_trace(go.Scatter(x=mid, y=r, mode="lines", line=dict(color="black"), name="Fokker–Planck",
                                     showlegend=k == 0), row=row + 1, col=col + 1)
            for src, dens in curves.items():
                fig.add_trace(go.Scatter(x=mid, y=dens, mode="markers", marker=dict(color=colors[src], size=6),
                                         name="co-simulation" if src == "cosim" else "native VCell",
                                         showlegend=k == 0 or (src == "native")), row=row + 1, col=col + 1)
            fig.update_xaxes(title_text="ρ", row=row + 1, col=col + 1)
        fig.update_layout(title="C3: single channel vs direct Fokker–Planck solutions (Schaff et al. 2016, Tests 3–5)",
                          height=720)
        (STUDY_DIR / "viz" / "fokker_planck.html").write_text(fig.to_html(include_plotlyjs="cdn", full_html=True))
        if dtau_rows:
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Bias of ρ̄ against Eq 4's steady state",
                                                                "Test 3 densities by Δτ"])
            d = [r["dtau"] for r in dtau_rows]
            for k, name, color in (("mean_bias", "mean", "#1f77b4"), ("sd_bias", "SD", "#ff7f0e")):
                fig.add_trace(go.Scatter(x=d, y=[100 * r[k] for r in dtau_rows], name=f"{name} (co-sim)",
                                         mode="markers+lines", marker=dict(color=color),
                                         error_y=dict(array=[200 * r[f"{k}_se"] for r in dtau_rows])), row=1, col=1)
            if "c3_dtau_mean_bias_slope" in metrics:
                xs = np.array([0.0, max(d)])
                ys = 100 * (metrics["c3_dtau_mean_bias_offset"] + metrics["c3_dtau_mean_bias_slope"] * xs)
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(dash="dot", color="#1f77b4"),
                                         name="mean: offset + slope·Δτ"), row=1, col=1)
            fig.update_xaxes(title_text="Δτ", rangemode="tozero", row=1, col=1)
            fig.update_yaxes(title_text="bias (%), ±2 SE", row=1, col=1)
            mid = 0.5 * (e3[1:] + e3[:-1])
            fig.add_trace(go.Scatter(x=mid, y=ref["test3_steady"], mode="lines", line=dict(color="black"),
                                     name="Eq 4 steady state"), row=1, col=2)
            for r in dtau_rows:
                fig.add_trace(go.Scatter(x=mid, y=r["density"], mode="markers", marker=dict(size=5),
                                         name=f"Δτ = {r['dtau']:g}"), row=1, col=2)
            fig.update_xaxes(title_text="ρ̄", range=[18, 24], row=1, col=2)
            fig.update_layout(title="C3 Test 3: the hybrid's steady state against Δτ", height=480)
            (STUDY_DIR / "viz" / "dtau.html").write_text(fig.to_html(include_plotlyjs="cdn", full_html=True))
    for k, v in metrics.items():
        print(f"[{STUDY_SLUG}] {k} = {v:.4g}")


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    stages = {"reference": [reference], "cosim": [cosim], "native": [native], "report": [report],
              "dtau": [dtau], "all": [reference, cosim, report]}[stage]
    for f in stages:
        f()
