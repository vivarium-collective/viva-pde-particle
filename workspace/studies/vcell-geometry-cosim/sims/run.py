"""Study B2e (vcell-geometry-cosim): the co-simulation on VCell's own geometry, with the accessible-volume correction.

The geometry is a vcell-fenics GeometryDescription (the B2b ball, R = 4 µm, in [0, 9]³ on a 37³
grid) realized by VCell itself (VCellFVRealization, through libvcell):

- the PDE runs on VCell's staircase ``cell`` (FV, domain mask);
- Smoldyn is confined by VCell's smooth membrane (VCell's triangulation and compartment points);
- the adapters fold counts at exterior nodes into the domain (the node-level analogue of the
  vcell-fvsolver#25 correction) and divide by effective volumes;
- fields are extended into the exterior band for 0th-order creation.

Correction variants (``geometry.accessible``):
- ``none``: full voxel volumes, as VCell does;
- ``adapters``: accessible volumes in the adapter only;
- ``adapters+volumes`` (default): accessible volumes in the adapter and the FV element volumes.

Cases, 32 seeds each:
- conversion A_p → B_f (k = 0.5, 20,000 A uniform in the smooth cell): interior B vs E[B], the
  outer-shell bias (r > 3.5 vs r < 2) and A + B balance;
- exchange A_p ⇌ B_f (k1 = 1, k2 = 0.5, B₀ = 0.2 µM): A + B balance and the steady ratio.

PDE mass and averages use the PDE's own element volumes. Native VCell on the same ball:
- B2c: shell −2.27% (vcell-fvsolver as released), −1.41% (#25 binning correction re-enabled);
- B2b: exchange −6.8%.

Needs the ``dev`` pixi env (pyvcell with spatial-hybrid support, libvcell).
Writes results/metrics.json and viz/vcell_geometry_cosim.html.
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
STUDY_SLUG = "vcell-geometry-cosim"
INVESTIGATION_SLUG = "hybrid-generalizability"
SPEC_ID = "viva_pde_particle.composites.hybrid.build_vcell_geometry_hybrid_document"

from vivarium_workbench.lib.run_log import append_run_event  # noqa: E402

from viva_pde_particle.analysis import write_metrics  # noqa: E402
from viva_pde_particle.grid import CartesianGrid  # noqa: E402
from viva_pde_particle.model import HybridModel, Reaction, Species  # noqa: E402
from viva_pde_particle.units import MOLECULES_PER_UM3_PER_UM as NA  # noqa: E402

CENTER, R = (4.5, 4.5, 4.5), 4.0
GRID = CartesianGrid((0, 0, 0), (9, 9, 9), (37, 37, 37))
DT, SEEDS = 0.01, range(1, 33)
K, T_CONV, T_EXCH, RECORD = 0.5, 2.0, 5.0, 0.25
CORRECTIONS = ("none", "adapters", "adapters+volumes")
SHELL, CORE = R - 2 * 0.25, R / 2
_REAL = None


def description():
    from viva_pde_particle.geometry import sphere_in_box

    return sphere_in_box(CENTER, R, (9.0, 9.0, 9.0))


def model(which):
    if which == "conversion":
        return HybridModel(GRID, [Species("A", 1.0, particle=True, initial=20000), Species("B", 1.0, initial=0.0)],
                           [Reaction("convert", {"A": 1}, {"B": 1}, k=K)])
    return HybridModel(GRID, [Species("A", 1.0, particle=True, initial=0), Species("B", 1.0, initial=0.2)],
                       [Reaction("a_to_b", {"A": 1}, {"B": 1}, k=1.0), Reaction("b_to_a", {"B": 1}, {"A": 1}, k=0.5)])


def _realization():
    global _REAL
    if _REAL is None:
        from viva_pde_particle.geometry.vcell_fv import VCellFVRealization

        _REAL = VCellFVRealization(description(), GRID)
    return _REAL


def _run(args):
    which, corr, seed = args
    from process_bigraph import Composite

    from viva_pde_particle.composites.hybrid import build_vcell_geometry_hybrid_document
    from viva_pde_particle.core import build_core

    doc = build_vcell_geometry_hybrid_document(model(which), description(), DT, seed=seed, correction=corr,
                                               realization=_realization())
    sim = Composite({"state": doc}, core=build_core())
    w = (sim.state["pde"]["instance"]._s.reshape(GRID.shape) * GRID.full_volume
         * doc["pde"]["config"]["domain"]["mask"])  # the PDE's own element volumes on its domain
    t_end = T_CONV if which == "conversion" else T_EXCH
    a, b = [], []

    def record():
        a.append(float(np.asarray(sim.state["particle_counts"]["A"]).sum()))
        b.append(float((np.asarray(sim.state["fields"]["B"]) * w).sum() * NA))

    record()
    sim.run(0.5 * DT)
    for _ in range(int(round(t_end / RECORD))):
        sim.run(RECORD)
        record()
    return np.array(a), np.array(b), np.asarray(sim.state["fields"]["B"], dtype=float).copy(), w


def ensemble(which, corr):
    with ProcessPoolExecutor(max_workers=min(len(SEEDS), os.cpu_count() or 1),
                             mp_context=mp.get_context("spawn")) as pool:
        runs = list(pool.map(_run, [(which, corr, s) for s in SEEDS]))
    return (np.stack([r[0] for r in runs]), np.stack([r[1] for r in runs]), np.stack([r[2] for r in runs]),
            runs[0][3])


def shell_stats(b_final, w, expected):
    x, y, z = GRID.node_coordinates()
    r = np.sqrt((x - CENTER[0]) ** 2 + (y - CENTER[1]) ** 2 + (z - CENTER[2]) ** 2)

    def mean(mask):
        ww = w * mask
        return (b_final * ww).sum(axis=(1, 2, 3)) / ww.sum()

    core, shell = mean(r < CORE), mean(r > SHELL)
    bias = shell / core - 1
    return {"interior_rel": float(core.mean() / expected - 1), "shell_bias": float(bias.mean()),
            "shell_bias_z": float(bias.mean() / (bias.std(ddof=1) / np.sqrt(len(bias))))}


def main() -> int:
    run_id = uuid.uuid4().hex
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "started", "spec_id": SPEC_ID, "label": STUDY_SLUG,
        "started_at": time.time(), "status": "running", "emitter": "none", "origin": "canonical_run",
        "study_slug": STUDY_SLUG, "investigation_slug": INVESTIGATION_SLUG,
        "params": {"corrections": list(CORRECTIONS), "seeds": len(SEEDS)},
    })
    started = time.time()
    try:
        from viva_pde_particle.reference.vcell_native import hybrid_support_available

        if not hybrid_support_available():
            raise SystemExit("needs pyvcell with spatial-hybrid support (pixi run -e dev ...)")
        real = _realization()
        metrics: dict[str, float] = {"b2e_staircase_volume": real.pde_volume("cell"),
                                     "b2e_smooth_volume": real.volume("cell")}
        frac = 1 - np.exp(-K * T_CONV)
        profiles = {}
        for corr in CORRECTIONS:
            key = corr.replace("+", "_")
            a, b, b_final, w = ensemble("conversion", corr)
            expected = a[:, 0].mean() * frac / (w.sum() * NA)
            for k, v in shell_stats(b_final, w, expected).items():
                metrics[f"b2e_{key}_{k}"] = v
            metrics[f"b2e_{key}_conversion_balance"] = float(((a + b).mean(0) / a[:, 0].mean() - 1)[-1])
            metrics[f"b2e_{key}_pde_volume"] = float(w.sum())
            x, y, z = GRID.node_coordinates()
            r = np.sqrt((x - CENTER[0]) ** 2 + (y - CENTER[1]) ** 2 + (z - CENTER[2]) ** 2)
            bins = np.arange(0, R + 0.25, 0.25)
            mb = b_final.mean(0)
            profiles[corr] = [float((mb * w * ((r >= lo) & (r < hi))).sum() / max((w * ((r >= lo) & (r < hi))).sum(), 1e-30)
                                    / expected) for lo, hi in zip(bins[:-1], bins[1:])]
            a, b, _, _ = ensemble("exchange", corr)
            total = (a + b).mean(0)
            late = np.arange(a.shape[1]) * RECORD >= 4.0
            metrics[f"b2e_{key}_exchange_balance"] = float(total[-1] / total[0] - 1)
            metrics[f"b2e_{key}_exchange_steady_ratio_rel_err"] = float(abs(a[:, late].mean() / b[:, late].mean() - 0.5) / 0.5)
        metrics["b2e_wall_time_s"] = time.time() - started
        write_metrics(STUDY_DIR / "results" / "metrics.json", metrics)

        import plotly.graph_objects as go

        mids = 0.5 * (np.arange(0, R + 0.25, 0.25)[1:] + np.arange(0, R + 0.25, 0.25)[:-1])
        fig = go.Figure()
        for corr, prof in profiles.items():
            fig.add_trace(go.Scatter(x=mids, y=prof, name=corr, mode="lines+markers"))
        fig.add_hline(y=1.0, line_dash="dash")
        fig.add_vline(x=SHELL, line_dash="dot")
        fig.update_layout(title="B2e: co-sim on VCell's geometry, B / E[B] by radius (conversion, t = 2 s, 32 seeds)",
                          xaxis_title="r (µm)", yaxis_title="B / E[B]", height=440)
        (STUDY_DIR / "viz" / "vcell_geometry_cosim.html").write_text(fig.to_html(include_plotlyjs="cdn", full_html=True))
    except BaseException:
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
