"""Regenerate the illustrative figures in docs/DESIGN.md.

    pixi run python docs/figures/make_design_figures.py

The figures are schematic but computed: binning uses CartesianGrid's nearest-node rule,
the transfer figure uses barycentric weights, and the lookup figure uses the same
bounding-box overlap rule as viva_pde_particle.mesh._voxel_table (in 2D).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle  # noqa: E402
from scipy.spatial import Delaunay  # noqa: E402

OUT = Path(__file__).resolve().parent
PDE_C, PART_C, OLD_C, NEW_C, BAD_C = "#3b6fb6", "#d9822b", "#9fb8de", "#2a8a4a", "#c0392b"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def arrow(ax, a, b, color="k", lw=1.0, style="-|>", ms=8, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=ms, color=color, lw=lw, linestyle=ls))


# ---------------------------------------------------------------------------- 1
def coupling_timelines(k=4):
    """Which field value the particle step reads, and which particle state the PDE reads."""
    rows = [
        ("vcell-fvsolver (SimTool loop)", k - 1),
        ('co-sim, coupling="fvsolver"\n(Stepper ticks every dt,\nruns Smoldyn on tick k−1 of k)', k - 1),
        ('co-sim, coupling="start-of-interval"\n(particles interval k·dt)', 0),
    ]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for r, (label, read_at) in enumerate(rows):
        y = 2 * (len(rows) - 1 - r)
        for i in range(2 * k):  # two coupling intervals
            ax.add_patch(Rectangle((i, y + 0.55), 0.92, 0.45, color=PDE_C, alpha=0.85))
        for j in range(2):
            ax.add_patch(Rectangle((j * k, y - 0.05), k - 0.08, 0.45, color=PART_C, alpha=0.85))
            src = j * k + read_at
            col = BAD_C if read_at else NEW_C
            arrow(ax, (src, y + 0.62), (src, y + 0.4), color=col, lw=1.8, ms=10)
            ax.scatter([src], [y + 0.62], s=18, color=col, zorder=5)
            start = "T" if j == 0 else "T+τ"
            ax.text(j * k + k / 2 + (0.6 if read_at == 0 else -0.4), y + 0.17,
                    f"Smoldyn step from {start}, reads u({start}{'+' + str(read_at) + 'dt' if read_at else ''})",
                    ha="center", va="center", color="white", fontsize=7.5)
        # PDE holds the particle concentrations of the last particle update for k steps
        ax.annotate("", xy=(2 * k - 0.05, y + 1.12), xytext=(k, y + 1.12),
                    arrowprops=dict(arrowstyle="<->", color="0.4", lw=0.8))
        ax.text(1.5 * k, y + 1.22, "PDE reads the counts of the last particle update (held for k steps)",
                ha="center", fontsize=7.5, color="0.35")
        ax.text(-0.3, y + 0.5, label, ha="right", va="center", fontsize=8.5)
    for i in range(2 * k + 1):
        ax.axvline(i, color="0.85", lw=0.6, zorder=0)
    ax.set_xticks(range(2 * k + 1), ["T"] + [f"T+{i}dt" for i in range(1, 2 * k + 1)], fontsize=7.5)
    ax.set_yticks([])
    ax.set_xlim(-0.2, 2 * k + 0.2)
    ax.set_ylim(-0.4, 2 * len(rows) - 0.4)
    ax.spines["left"].set_visible(False)
    ax.set_title(f"Coupling timelines, k = {k}. Blue: PDE steps of dt. Orange: Smoldyn steps of τ = k·dt.\n"
                 "The arrow drops from the time whose field value the Smoldyn step reads.", fontsize=9)
    save(fig, "coupling_timelines.svg")


# ---------------------------------------------------------------------------- 2
def splitting_schemes(k=4):
    schemes = [
        ("jacobi (vcell-fvsolver)", [("pde", k - 1, "old"), ("part", None, k - 1), ("pde", 1, "old")]),
        ("gs_particles_first", [("part", None, 0), ("pde", k, "new")]),
        ("gs_pde_first", [("pde", k, "old"), ("part", None, k)]),
        ("strang", [("pde", k // 2, "old"), ("part", None, k // 2), ("pde", k // 2, "new")]),
    ]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for r, (name, seq) in enumerate(schemes):
        y = len(schemes) - 1 - r
        t = 0
        for kind, n, info in seq:
            if kind == "pde":
                for _ in range(n):
                    ax.add_patch(Rectangle((t, y + 0.1), 0.92, 0.5, color=OLD_C if info == "old" else NEW_C))
                    t += 1
            else:
                ax.add_patch(Rectangle((t - 0.12, y + 0.02), 0.24, 0.66, color=PART_C, zorder=3))
                ax.text(t, y + 0.8, f"particles read u(T+{info}dt)" if info else "particles read u(T)",
                        ha="center", fontsize=7.5, color=PART_C)
        ax.text(-0.3, y + 0.35, name, ha="right", va="center")
    ax.set_xlim(-0.3, k + 0.3)
    ax.set_ylim(-0.1, len(schemes) + 0.1)
    ax.set_xticks(range(k + 1), ["T"] + [f"T+{i}dt" for i in range(1, k + 1)])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    handles = [Rectangle((0, 0), 1, 1, color=OLD_C), Rectangle((0, 0), 1, 1, color=NEW_C),
               Rectangle((0, 0), 1, 1, color=PART_C)]
    ax.legend(handles, ["PDE substep with old particles pⁿ", "PDE substep with new particles pⁿ⁺¹",
                        "one Smoldyn step of τ = k·dt (instantaneous in this picture)"],
              loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=8)
    ax.set_title("HybridCoupler splitting schemes over one coupling interval τ = k·dt (k = 4)")
    save(fig, "splitting_schemes.svg")


# ---------------------------------------------------------------------------- 3
def grid_binning():
    rng = np.random.default_rng(3)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.8))
    fig.subplots_adjust(wspace=0.3)
    # (a) node-centred grid, element volumes, nearest-node binning
    nx, ny, L, H = 6, 5, 5.0, 4.0
    dx, dy = L / (nx - 1), H / (ny - 1)
    for i in range(nx):
        for j in range(ny):
            x0, x1 = max(i * dx - dx / 2, 0), min(i * dx + dx / 2, L)
            y0, y1 = max(j * dy - dy / 2, 0), min(j * dy + dy / 2, H)
            frac = (x1 - x0) * (y1 - y0) / (dx * dy)
            a1.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=True, alpha=0.15 + 0.25 * (1 - frac),
                                   color=PDE_C, ec="0.5", lw=0.6, ls="--"))
    pts = rng.uniform([0, 0], [L, H], (28, 2))
    counts = np.zeros((ny, nx), int)
    for p in pts:
        i, j = int(p[0] / dx + 0.5), int(p[1] / dy + 0.5)  # CartesianGrid's nearest-node rule
        counts[j, i] += 1
        a1.plot([p[0], i * dx], [p[1], j * dy], color=PART_C, lw=0.7)
    a1.scatter(pts[:, 0], pts[:, 1], s=12, color=PART_C, zorder=3, label="particles")
    X, Y = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dy)
    a1.scatter(X, Y, s=22, color=PDE_C, zorder=4, label="grid nodes (field values live here)")
    for j in range(ny):
        for i in range(nx):
            if counts[j, i]:
                a1.text(i * dx + 0.07, j * dy + 0.07, str(counts[j, i]), fontsize=7.5, color="k", zorder=5)
    a1.annotate("corner element: ¼ volume", xy=(0.15, H - 0.15), xytext=(-0.2, H + 0.35), fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color="0.4", lw=0.6))
    a1.annotate("edge element: ½ volume", xy=(2.5, H - 0.15), xytext=(2.3, H + 0.35), fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color="0.4", lw=0.6))
    a1.set_aspect("equal")
    a1.set_xlim(-0.3, L + 0.3)
    a1.set_ylim(-0.3, H + 0.7)
    a1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=8)
    a1.set_title("(a) Node-centred grid: dx = L/(N−1); each particle is\n"
                 "counted at its nearest node; c = count / (V_element · 602.214)")
    # (b) curved membrane: a particle inside can be nearest to an exterior node
    R, c = 3.0, np.array([0.0, 0.0])
    a2.add_patch(Circle(c, R, fill=False, color="k", lw=1.6))
    h = 0.5
    g = np.arange(-4, 4 + 1e-9, h)
    X, Y = np.meshgrid(g, g)
    inside = X**2 + Y**2 <= R**2
    a2.scatter(X[inside], Y[inside], s=10, color=PDE_C, label="node inside the cell: field solved")
    a2.scatter(X[~inside], Y[~inside], s=10, color="0.75", label="node outside: no field for this species")
    th = rng.uniform(0, 2 * np.pi, 400)
    rr = R * np.sqrt(rng.uniform(0.8, 1.0, 400))
    pts = np.stack([rr * np.cos(th), rr * np.sin(th)], 1)
    near = np.round(pts / h) * h
    bad = (near**2).sum(1) > R**2
    a2.scatter(pts[~bad, 0], pts[~bad, 1], s=6, color=PART_C)
    a2.scatter(pts[bad, 0], pts[bad, 1], s=10, color=BAD_C, label="particle inside, nearest node outside")
    for p, q in zip(pts[bad][:40], near[bad][:40]):
        a2.plot([p[0], q[0]], [p[1], q[1]], color=BAD_C, lw=0.7)
    a2.set_aspect("equal")
    a2.set_xlim(-3.6, 3.6)
    a2.set_ylim(-3.6, 3.6)
    a2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.3), ncol=1, frameon=False, fontsize=8)
    a2.set_title("(b) Curved membrane: the source k·[A] of red particles\n"
                 "lands on exterior nodes (native: dropped, B2b/B2c)")
    save(fig, "grid_binning.svg")


def _disk_mesh(R=3.0, h=0.9, seed=0):
    rng = np.random.default_rng(seed)
    n_ring = int(2 * np.pi * R / h)
    ring = R * np.stack([np.cos(t := np.linspace(0, 2 * np.pi, n_ring, endpoint=False)), np.sin(t)], 1)
    inner = []
    for r in np.arange(h, R - 0.5 * h, h):
        m = max(int(2 * np.pi * r / h), 1)
        a = np.linspace(0, 2 * np.pi, m, endpoint=False) + rng.uniform(0, 1)
        inner.append(np.stack([r * np.cos(a), r * np.sin(a)], 1))
    pts = np.vstack([[[0, 0]], *inner, ring])
    return pts, Delaunay(pts)


def _bary(tri, simplex, p):
    T = tri.transform[simplex]
    lam = T[:2] @ (p - T[2])
    return np.array([lam[0], lam[1], 1 - lam.sum()])


# ---------------------------------------------------------------------------- 4
def mesh_grid_transfer():
    pts, tri = _disk_mesh()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 5.2))
    fig.subplots_adjust(wspace=0.35)
    for ax in (a1, a2):
        ax.triplot(pts[:, 0], pts[:, 1], tri.simplices, color="0.6", lw=0.6)
        ax.scatter(pts[:, 0], pts[:, 1], s=14, color=NEW_C, zorder=3, label="mesh DOFs (P1 vertices)")
        ax.add_patch(Circle((0, 0), 3.0, fill=False, color="k", lw=1.2, ls=":"))
        ax.set_aspect("equal")
        ax.set_xlim(-3.7, 3.7)
        ax.set_ylim(-3.7, 3.7)
    g = np.arange(-3.5, 3.6, 0.5)
    X, Y = np.meshgrid(g, g)
    nodes = np.stack([X.ravel(), Y.ravel()], 1)
    a1.scatter(nodes[:, 0], nodes[:, 1], s=4, color=PDE_C, label="background grid nodes")
    s = tri.find_simplex(nodes)
    q = np.array([0.5, 1.0])  # an interior node: row of P = barycentric weights
    sq = tri.find_simplex(q)
    w = _bary(tri, sq, q)
    for v, wv in zip(tri.simplices[sq], w):
        a1.plot([q[0], pts[v, 0]], [q[1], pts[v, 1]], color=PDE_C, lw=1.2)
        a1.text(*(0.3 * q + 0.7 * pts[v] + [0.06, 0.06]), f"{wv:.2f}", fontsize=7.5, color=PDE_C,
                bbox=dict(fc="white", ec="none", pad=0.5))
    a1.scatter(*q, s=40, color=PDE_C, zorder=4)
    o = nodes[(s < 0) & (np.hypot(nodes[:, 0], nodes[:, 1]) < 3.4)][3]
    v = np.argmin(np.linalg.norm(pts - o, axis=1))
    arrow(a1, o, pts[v], color=BAD_C, lw=1.2)
    a1.scatter(*o, s=40, color=BAD_C, zorder=4)
    a1.text(o[0] + 0.25, o[1] - 0.25, "outside node → nearest DOF (weight 1)", fontsize=7.5, color=BAD_C,
            bbox=dict(fc="white", ec="none", pad=0.5))
    a1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=8)
    a1.set_title('(a) particle_transfer="grid": u_grid = P·u, load = Pᵀ·counts\n'
                 "P row = barycentric weights of the node in its triangle")
    rng = np.random.default_rng(5)
    r_ = 3.0 * np.sqrt(rng.uniform(0, 1, 60))
    t_ = rng.uniform(0, 2 * np.pi, 60)
    parts = np.stack([r_ * np.cos(t_), r_ * np.sin(t_)], 1)
    a2.scatter(parts[:, 0], parts[:, 1], s=8, color=PART_C, label="particles", zorder=4)
    p = np.array([-1.2, -0.9])
    sp_ = tri.find_simplex(p)
    w = _bary(tri, sp_, p)
    for v, wv in zip(tri.simplices[sp_], w):
        a2.plot([p[0], pts[v, 0]], [p[1], pts[v, 1]], color=PART_C, lw=1.2)
        a2.text(*(0.3 * p + 0.7 * pts[v] + [0.06, 0.06]), f"{wv:.2f}", fontsize=7.5, color=PART_C,
                bbox=dict(fc="white", ec="none", pad=0.5))
    a2.scatter(*p, s=40, color=PART_C, zorder=5)
    a2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False, fontsize=8)
    a2.set_title('(b) particle_transfer="positions": load_j = Σ_p φ_j(x_p)\n'
                 "each particle splits 1 molecule over its cell's vertices")
    save(fig, "mesh_grid_transfer.svg")


# ---------------------------------------------------------------------------- 5
def voxel_lookup():
    pts, tri = _disk_mesh()
    voxel = 0.75
    lo = pts.min(0) - 1e-9
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    ax.triplot(pts[:, 0], pts[:, 1], tri.simplices, color="0.65", lw=0.6)
    for x in np.arange(lo[0], pts[:, 0].max() + voxel, voxel):
        ax.axvline(x, color=PDE_C, lw=0.4, alpha=0.6)
    for y in np.arange(lo[1], pts[:, 1].max() + voxel, voxel):
        ax.axhline(y, color=PDE_C, lw=0.4, alpha=0.6)
    q = np.array([0.95, 0.35])
    iv = np.floor((q - lo) / voxel).astype(int)
    vlo = lo + iv * voxel
    ax.add_patch(Rectangle(vlo, voxel, voxel, color=PDE_C, alpha=0.25, zorder=2))
    verts = pts[tri.simplices]
    tlo = np.floor((verts.min(1) - lo) / voxel).astype(int)
    thi = np.floor((verts.max(1) - lo) / voxel).astype(int)
    cand = np.nonzero(((tlo <= iv) & (thi >= iv)).all(1))[0]  # bounding box overlaps the voxel
    hit = tri.find_simplex(q)
    for c in cand:
        ax.add_patch(Polygon(verts[c], closed=True, fill=True, color=PART_C, alpha=0.18, ec=PART_C, lw=0.8))
    ax.add_patch(Polygon(verts[hit], closed=True, fill=True, color=NEW_C, alpha=0.55, zorder=3))
    ax.scatter(*q, s=40, color="k", zorder=5)
    ax.text(q[0] + 0.08, q[1] + 0.08, "particle", fontsize=8)
    ax.set_aspect("equal")
    ax.set_xlim(-3.3, 3.3)
    ax.set_ylim(-3.3, 3.3)
    ax.set_title(f"Voxel → tet lookup: voxel index = floor((x − lo)/s) (arithmetic);\n"
                 f"{len(cand)} candidates (orange, bounding box overlaps the voxel),\n"
                 "barycentric test finds the containing cell (green)", fontsize=9)
    save(fig, "voxel_lookup.svg")


if __name__ == "__main__":
    coupling_timelines()
    splitting_schemes()
    grid_binning()
    mesh_grid_transfer()
    voxel_lookup()
