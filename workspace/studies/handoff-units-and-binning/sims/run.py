"""Study A1 (handoff-units-and-binning): each side alone, and the particle -> grid handoff.

Writes results/metrics.json (graded by the workbench via viva_pde_particle.evaluators)
and viz/handoff.html. Run with ``pixi run python workspace/studies/handoff-units-and-binning/sims/run.py``.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[2]
STUDY_SLUG = "handoff-units-and-binning"
INVESTIGATION_SLUG = "cosim-vs-embedded-hybrid"
SPEC_ID = "viva_pde_particle.composites.examples.particle_diffusion"

from vivarium_workbench.lib.run_log import append_run_event  # noqa: E402

from viva_pde_particle.analysis import histogram_variance, run_ensemble, write_metrics  # noqa: E402
from viva_pde_particle.composites.examples import particle_diffusion_model, square_grid  # noqa: E402
from viva_pde_particle.grid import CartesianGrid  # noqa: E402
from viva_pde_particle.processes.fv_reaction_diffusion import FVReactionDiffusion  # noqa: E402

DT = 0.01


def fv_cosine_mode():
    """FV diffusion of cos(pi x/L)cos(pi y/L) vs the continuum decay exp(-2 pi^2 D t / L^2)."""
    from process_bigraph import allocate_core

    L, N, D, t_end = 10.0, 21, 1.0, 1.0
    g = CartesianGrid((0.0, 0.0), (L, L), (N, N))
    x, y = g.node_coordinates()
    u0 = np.cos(np.pi * x / L) * np.cos(np.pi * y / L)
    proc = FVReactionDiffusion(
        config={"grid": g.to_config(), "pde": {"species": {"B": {"diffusion": D}}, "particle_species": [],
                                               "terms": []}, "dt": DT},
        core=allocate_core(),
    )
    u = proc.update({"fields": {"B": u0}, "particle_counts": {}}, t_end)["fields"]["B"]
    exact = u0 * np.exp(-2 * np.pi**2 * D * t_end / L**2)
    return float(np.abs(u - exact).max() / np.abs(exact).max()), x[0], u[N // 2], exact[N // 2]


def particle_variance(seeds):
    """Free diffusion from the centre cell: binned variance per axis vs 2Dt."""
    n, D, t_end = 20000, 1.0, 1.0
    model = particle_diffusion_model(n_particles=n, diffusion=D, start="center")
    times, _, counts = run_ensemble(model, seeds, t_end, DT, record_every=0.25)
    g = model.grid
    dx2 = g.spacing[0] ** 2
    # initial uniform-in-cell variance dx²/12 plus Sheppard's binning correction dx²/12
    offset = dx2 / 6
    var = np.array([[np.mean([histogram_variance(g, c[i], ax) for ax in (0, 1)]) for i in range(len(times))]
                    for c in counts["A"]])
    measured = var.mean(axis=0) - offset
    expected = 2 * D * times
    rel_err = float(abs(measured[-1] - expected[-1]) / expected[-1])
    mismatch = float(np.abs(counts["A"].sum(axis=(2, 3)) - n).max())
    return rel_err, mismatch, times, measured, expected


def boundary_classes(seeds):
    """Uniform particles after diffusion: per-class counts match element volume shares."""
    n, D, t_end = 100000, 1.0, 2.0
    model = particle_diffusion_model(n_particles=n, diffusion=D, length=10.0, num=11, start="uniform")
    _, _, counts = run_ensemble(model, seeds, t_end, DT, record_every=t_end)
    g = model.grid
    frac = g.volume_fraction
    final = counts["A"][:, -1]  # (seeds, Ny, Nx)
    zmax, conc_ratio = 0.0, {}
    conc = np.stack([g.counts_to_uM(c) for c in final]).mean(axis=0)
    interior_conc = conc[frac == 1.0].mean()
    for label, f in (("corner", 0.25), ("edge", 0.5), ("interior", 1.0)):
        mask = frac == f
        p = frac[mask].sum() / frac.sum()
        expected = n * len(seeds) * p
        observed = final[:, mask].sum()
        zmax = max(zmax, abs(observed - expected) / np.sqrt(expected * (1 - p)))
        conc_ratio[label] = float(conc[mask].mean() / interior_conc)
    return float(zmax), conc_ratio


def write_viz(times, measured, expected, x, u_mid, exact_mid, conc_ratio):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=1, cols=3, subplot_titles=(
        "FV diffusion vs analytic (y = L/2, t = 1 s)", "Particle variance per axis vs 2Dt",
        "Binned concentration / interior"))
    fig.add_trace(go.Scatter(x=x, y=exact_mid, name="analytic", mode="lines"), 1, 1)
    fig.add_trace(go.Scatter(x=x, y=u_mid, name="FV", mode="markers"), 1, 1)
    fig.add_trace(go.Scatter(x=times, y=expected, name="2Dt", mode="lines"), 1, 2)
    fig.add_trace(go.Scatter(x=times, y=measured, name="Smoldyn (binned)", mode="markers"), 1, 2)
    fig.add_trace(go.Bar(x=list(conc_ratio), y=list(conc_ratio.values()), name="conc ratio"), 1, 3)
    fig.update_layout(title="A1: handoff units and binning", height=420)
    out = STUDY_DIR / "viz" / "handoff.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(fig.to_html(include_plotlyjs="cdn", full_html=True))


def main() -> int:
    run_id = uuid.uuid4().hex
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "started", "spec_id": SPEC_ID, "label": STUDY_SLUG,
        "started_at": time.time(), "status": "running", "emitter": "none", "origin": "canonical_run",
        "study_slug": STUDY_SLUG, "investigation_slug": INVESTIGATION_SLUG, "params": {"dt": DT},
    })
    started = time.time()
    try:
        fv_err, x, u_mid, exact_mid = fv_cosine_mode()
        var_err, mismatch, times, measured, expected = particle_variance(seeds=range(1, 5))
        zmax, conc_ratio = boundary_classes(seeds=range(1, 5))
        metrics = {
            "a1_fv_cosine_rel_err": fv_err,
            "a1_particle_variance_rel_err": var_err,
            "a1_histogram_count_mismatch": mismatch,
            "a1_boundary_class_max_abs_z": zmax,
            "a1_corner_to_interior_conc_ratio": conc_ratio["corner"],
            "a1_edge_to_interior_conc_ratio": conc_ratio["edge"],
            "a1_wall_time_s": time.time() - started,
        }
        write_metrics(STUDY_DIR / "results" / "metrics.json", metrics)
        write_viz(times, measured, expected, x, u_mid, exact_mid, conc_ratio)
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
