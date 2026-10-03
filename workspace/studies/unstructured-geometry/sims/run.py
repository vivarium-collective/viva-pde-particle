"""Study B2 (unstructured-geometry): the hybrid co-simulation in a ball (R = 4 µm).

PDE side: FEniCSx P1 on an unstructured gmsh tetrahedral mesh of the ball
(FenicsxMeshReactionDiffusion). Particle side: Smoldyn confined by a reflecting spherical
surface. They couple through a 33³ background grid (Δ = 0.25 µm): fields are interpolated
onto it, and particle counts are loaded back as Pᵀ·counts (see viva_pde_particle.mesh).

Part 1, conversion: A_p → B_f (k = 0.5 /s), 20,000 A, 8 seeds.
  - A(t) against A₀·e^(−kt);
  - A + ∫B (molecules) conserved;
  - the mean B profile against the continuum FEM solution on the same mesh (A continuous).
Part 2, exchange: A_p → B_f (k1 = 1) and B_f → A_p by field-dependent creation inside the
  spherical compartment (k2 = 0.5), B₀ = 0.2 µM, 8 seeds.
  - mass balance;
  - steady-state total A / total B against k2/k1.
  The residual measures the mismatch between the particle side's geometry (voxel cells
  ∩ ball) and the PDE side's (polyhedral mesh).

The embedded vcell-fvsolver hybrid cannot use an unstructured PDE mesh; native VCell
on a voxelized ball is a possible follow-up comparison.
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
STUDY_SLUG = "unstructured-geometry"
INVESTIGATION_SLUG = "hybrid-generalizability"
SPEC_ID = "viva_pde_particle.composites.hybrid.build_mesh_hybrid_document"

from vivarium_workbench.lib.run_log import append_run_event  # noqa: E402

from viva_pde_particle.analysis import write_metrics  # noqa: E402
from viva_pde_particle.grid import CartesianGrid  # noqa: E402
from viva_pde_particle.model import HybridModel, Reaction, Species  # noqa: E402
from viva_pde_particle.units import MOLECULES_PER_UM3_PER_UM as NA  # noqa: E402

SPHERE = {"center": (4.0, 4.0, 4.0), "radius": 4.0, "h": 0.8}
GRID = CartesianGrid((0, 0, 0), (8, 8, 8), (33, 33, 33))
DT, SEEDS = 0.01, range(1, 9)


def conversion_model():
    return HybridModel(GRID, [Species("A", 1.0, particle=True, initial=20000), Species("B", 1.0, initial=0.0)],
                       [Reaction("convert", {"A": 1}, {"B": 1}, k=0.5)])


def exchange_model():
    return HybridModel(GRID, [Species("A", 1.0, particle=True, initial=0), Species("B", 1.0, initial=0.2)],
                       [Reaction("a_to_b", {"A": 1}, {"B": 1}, k=1.0), Reaction("b_to_a", {"B": 1}, {"A": 1}, k=0.5)])


def _run(args):
    which, seed, t_end, every = args
    from viva_pde_particle.composites.hybrid import build_mesh_hybrid_document, run_document

    model = conversion_model() if which == "conversion" else exchange_model()
    doc = build_mesh_hybrid_document(model, SPHERE, DT, seed=seed)
    tr = run_document(doc, t_end, DT, every)
    return (np.array(tr.times), {s: np.stack(v) for s, v in tr.field_dofs.items()},
            {s: np.stack(v).reshape(len(tr.times), -1).sum(axis=1) for s, v in tr.particle_counts.items()})


def ensemble(which, t_end, every):
    with ProcessPoolExecutor(max_workers=min(len(SEEDS), os.cpu_count() or 1),
                             mp_context=mp.get_context("spawn")) as pool:
        return list(pool.map(_run, [(which, s, t_end, every) for s in SEEDS]))


def lumped_mass():
    from process_bigraph import allocate_core

    from viva_pde_particle.composites.hybrid import build_mesh_hybrid_document
    from viva_pde_particle.processes import FenicsxMeshReactionDiffusion

    doc = build_mesh_hybrid_document(conversion_model(), SPHERE, DT, seed=1)
    return FenicsxMeshReactionDiffusion(config=doc["pde"]["config"], core=allocate_core())


def continuum_b(proc, t_end, every):
    """B on the mesh with A treated as a continuous, uniform field."""
    import scipy.sparse as sp  # noqa: F401

    a = np.full(proc.n_dofs, 20000 / (proc.ml.sum() * NA))
    b = np.zeros(proc.n_dofs)
    out, steps = [b.copy()], int(round(every / DT))
    for _ in range(int(round(t_end / every))):
        for _ in range(steps):
            b, a = proc._solvers["B"](proc.ml * (b + DT * 0.5 * a)), a * (1 - 0.5 * DT)
        out.append(b.copy())
    return np.stack(out)


def main() -> int:
    run_id = uuid.uuid4().hex
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "started", "spec_id": SPEC_ID, "label": STUDY_SLUG,
        "started_at": time.time(), "status": "running", "emitter": "none", "origin": "canonical_run",
        "study_slug": STUDY_SLUG, "investigation_slug": INVESTIGATION_SLUG,
        "params": {"sphere": SPHERE, "grid": GRID.to_config(), "seeds": len(SEEDS)},
    })
    started = time.time()
    try:
        proc = lumped_mass()
        ml = proc.ml
        metrics: dict[str, float] = {"b2_mesh_volume_rel_err": float(ml.sum() / (4 / 3 * np.pi * 4.0**3) - 1)}

        runs = ensemble("conversion", 2.0, 0.25)
        times = runs[0][0]
        a = np.stack([r[2]["A"] for r in runs])
        b = np.stack([(r[1]["B"] * ml).sum(axis=1) * NA for r in runs])
        expected = a[:, :1] * np.exp(-0.5 * times)
        sd = np.sqrt(a[:, :1].mean() * np.exp(-0.5 * times) * (1 - np.exp(-0.5 * times)) / len(a))
        z = np.where(sd > 0, (a.mean(0) - expected.mean(0)) / np.where(sd > 0, sd, 1), 0)
        metrics["b2_conversion_decay_max_z"] = float(np.abs(z).max())
        metrics["b2_conversion_mass_balance_max_rel_dev"] = float(np.abs((a + b).mean(0) / a[:, 0].mean() - 1).max())
        cont = continuum_b(proc, 2.0, 0.25)
        b_mean = np.stack([r[1]["B"] for r in runs]).mean(0)
        metrics["b2_conversion_b_vs_continuum_rel_l2"] = float(
            np.sqrt((((b_mean[-1] - cont[-1]) ** 2) * ml).sum() / ((cont[-1] ** 2) * ml).sum()))

        runs = ensemble("exchange", 5.0, 0.25)
        times_x = runs[0][0]
        ax = np.stack([r[2]["A"] for r in runs])
        bx = np.stack([(r[1]["B"] * ml).sum(axis=1) * NA for r in runs])
        total0 = bx[:, 0].mean()
        metrics["b2_exchange_mass_balance_max_rel_dev"] = float(np.abs((ax + bx).mean(0) / total0 - 1).max())
        late = times_x >= 4.0
        ratio = ax[:, late].mean() / bx[:, late].mean()
        metrics["b2_exchange_steady_ratio_rel_err"] = float(abs(ratio - 0.5) / 0.5)
        metrics["b2_wall_time_s"] = time.time() - started
        write_metrics(STUDY_DIR / "results" / "metrics.json", metrics)

        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Conversion in a ball: A and B (molecules)",
                                                            "Exchange in a ball: A, B and A+B"))
        fig.add_trace(go.Scatter(x=times, y=expected.mean(0), name="A₀·e^(−kt)", mode="lines"), 1, 1)
        fig.add_trace(go.Scatter(x=times, y=a.mean(0), name="A (particles)", mode="markers"), 1, 1)
        fig.add_trace(go.Scatter(x=times, y=b.mean(0), name="B (mesh field)", mode="markers"), 1, 1)
        fig.add_trace(go.Scatter(x=times, y=(a + b).mean(0), name="A + B", mode="lines"), 1, 1)
        fig.add_trace(go.Scatter(x=times_x, y=ax.mean(0), name="A", mode="lines+markers"), 1, 2)
        fig.add_trace(go.Scatter(x=times_x, y=bx.mean(0), name="B", mode="lines+markers"), 1, 2)
        fig.add_trace(go.Scatter(x=times_x, y=(ax + bx).mean(0), name="A + B", mode="lines"), 1, 2)
        fig.update_layout(title="B2: hybrid co-simulation on an unstructured mesh (ball, R = 4 µm)", height=440)
        (STUDY_DIR / "viz").mkdir(parents=True, exist_ok=True)
        (STUDY_DIR / "viz" / "unstructured_geometry.html").write_text(fig.to_html(include_plotlyjs="cdn", full_html=True))
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
