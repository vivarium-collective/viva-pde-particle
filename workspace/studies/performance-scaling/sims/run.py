"""Study A6 (performance-scaling): wall time of the co-simulation vs the native VCell hybrid.

Model: A4's A_particle + B_field → C_field on N×N×3 slabs; dt = 0.01, t = 1 s (100 steps).

Sweeps:
- grid N ∈ {11, 21, 41, 81} with 20,000 particles;
- particles ∈ {2,000, 20,000, 200,000} on 21×21×3.

Each configuration is timed on one process, without parallel ensembles. The breakdowns:
- co-sim: PDE update (FV solve), particle update (Smoldyn step + field copy + histogram),
  and framework overhead (process-bigraph scheduling and state handling: everything else);
- native: input generation (pyvcell + libvcell math generation), solve, and output read
  (spawned process; spawn time is outside these phases).

Medians of 3 repeats.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[2]
STUDY_SLUG = "performance-scaling"
INVESTIGATION_SLUG = "cosim-vs-embedded-hybrid"
SPEC_ID = "viva_pde_particle.composites.examples.bimolecular_hybrid"

from vivarium_workbench.lib.run_log import append_run_event  # noqa: E402

from viva_pde_particle.analysis import write_metrics  # noqa: E402
from viva_pde_particle.composites.examples import bimolecular_model  # noqa: E402
from viva_pde_particle.composites.hybrid import run_hybrid  # noqa: E402
from viva_pde_particle.core import build_core  # noqa: E402
from viva_pde_particle.processes.fv_reaction_diffusion import FVReactionDiffusion  # noqa: E402
from viva_pde_particle.processes.smoldyn_hybrid import SmoldynHybrid  # noqa: E402
from viva_pde_particle.reference.vcell_native import hybrid_support_available  # noqa: E402

DT, T_END, REPEATS = 0.01, 1.0, 3
GRID_SWEEP = [11, 21, 41, 81]
PARTICLE_SWEEP = [2000, 20000, 200000]

_timers = {"pde": 0.0, "particles": 0.0}


def _instrument():
    for cls, key in ((FVReactionDiffusion, "pde"), (SmoldynHybrid, "particles")):
        original = cls.update

        def timed(self, state, interval, _original=original, _key=key):
            t0 = time.perf_counter()
            try:
                return _original(self, state, interval)
            finally:
                _timers[_key] += time.perf_counter() - t0

        cls.update = timed


def time_cosim(model, core):
    _timers["pde"] = _timers["particles"] = 0.0
    t0 = time.perf_counter()
    run_hybrid(model, T_END, DT, record_every=T_END, core=core, particle_init="poisson")
    total = time.perf_counter() - t0
    return {"total": total, "pde": _timers["pde"], "particles": _timers["particles"],
            "framework": total - _timers["pde"] - _timers["particles"]}


def time_native(model):
    from viva_pde_particle.reference.vcell_native import run_native

    t0 = time.perf_counter()
    tr = run_native(model, t_end=T_END, dt=DT, output_dt=T_END, seed=1)
    out = dict(tr.timings)
    out["total"] = time.perf_counter() - t0
    return out


def median_of(fn, *args):
    runs = [fn(*args) for _ in range(REPEATS)]
    return {k: float(np.median([r[k] for r in runs])) for k in runs[0]}


def main() -> int:
    run_id = uuid.uuid4().hex
    native = hybrid_support_available()
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "started", "spec_id": SPEC_ID, "label": STUDY_SLUG,
        "started_at": time.time(), "status": "running", "emitter": "none", "origin": "canonical_run",
        "study_slug": STUDY_SLUG, "investigation_slug": INVESTIGATION_SLUG,
        "params": {"grids": GRID_SWEEP, "particles": PARTICLE_SWEEP, "native": native},
    })
    started = time.time()
    try:
        _instrument()
        core = build_core()
        rows = []
        for n in GRID_SWEEP:
            model = bimolecular_model(n_particles=20000, num=n)
            rows.append(("grid", n, 20000, median_of(time_cosim, model, core), median_of(time_native, model) if native else None))
            print(f"[{STUDY_SLUG}] grid {n}: {rows[-1][3]} native {rows[-1][4]}", flush=True)
        for p in PARTICLE_SWEEP:
            model = bimolecular_model(n_particles=p, num=21)
            rows.append(("particles", 21, p, median_of(time_cosim, model, core), median_of(time_native, model) if native else None))
            print(f"[{STUDY_SLUG}] particles {p}: {rows[-1][3]} native {rows[-1][4]}", flush=True)

        metrics: dict[str, float] = {}
        for sweep, n, p, cs, nt in rows:
            tag = f"n{n}" if sweep == "grid" else f"p{p}"
            metrics[f"a6_{sweep}_{tag}_cosim_s"] = cs["total"]
            metrics[f"a6_{sweep}_{tag}_cosim_framework_frac"] = cs["framework"] / cs["total"]
            metrics[f"a6_{sweep}_{tag}_cosim_pde_frac"] = cs["pde"] / cs["total"]
            metrics[f"a6_{sweep}_{tag}_cosim_particle_frac"] = cs["particles"] / cs["total"]
            if nt:
                metrics[f"a6_{sweep}_{tag}_native_s"] = nt["total"]
                metrics[f"a6_{sweep}_{tag}_native_solve_s"] = nt["solve"]
                metrics[f"a6_{sweep}_{tag}_cosim_over_native_solve"] = cs["total"] / nt["solve"]
        per_step = [r[3]["framework"] / (T_END / DT) for r in rows]
        metrics["a6_cosim_framework_ms_per_step_max"] = 1000 * float(max(per_step))
        metrics["a6_wall_time_s"] = time.time() - started
        write_metrics(STUDY_DIR / "results" / "metrics.json", metrics)

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Wall time vs grid nodes (20k particles)",
                                                            "Wall time vs particles (21×21×3)"))
        for col, sweep, xs in ((1, "grid", [n * n * 3 for n in GRID_SWEEP]), (2, "particles", PARTICLE_SWEEP)):
            sel = [r for r in rows if r[0] == sweep]
            for comp in ("total", "pde", "particles", "framework"):
                fig.add_trace(go.Scatter(x=xs, y=[r[3][comp] for r in sel], name=f"co-sim {comp}",
                                         mode="lines+markers", legendgroup=comp, showlegend=(col == 1)), 1, col)
            if sel[0][4]:
                for comp in ("total", "solve"):
                    fig.add_trace(go.Scatter(x=xs, y=[r[4][comp] for r in sel], name=f"native {comp}",
                                             mode="lines+markers", line={"dash": "dash"},
                                             legendgroup=f"n{comp}", showlegend=(col == 1)), 1, col)
            fig.update_xaxes(type="log", row=1, col=col)
            fig.update_yaxes(type="log", title_text="s per simulated second", row=1, col=col)
        fig.update_layout(title="A6: performance, co-simulation vs native VCell hybrid", height=460)
        (STUDY_DIR / "viz").mkdir(parents=True, exist_ok=True)
        (STUDY_DIR / "viz" / "performance.html").write_text(fig.to_html(include_plotlyjs="cdn", full_html=True))
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
