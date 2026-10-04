"""Accessible volumes and compartment-aware folding for a staircase PDE domain (Phase 7e.5).

Native VCell solves the PDE on the voxel staircase but confines particles by a smooth membrane, and
the two domains differ by ~1.8% for the B2b ball. That mismatch caused the outer-shell bias in
Study B2c, and the exterior-node binning loss of vcell-fvsolver#25. Three ingredients make the
co-simulation's particle → field transfer consistent with the smooth membrane:

- **accessible fraction** fᵢ: the share of node i's element (its node-centred control volume)
  inside the smooth region. The accessible volume is Aᵢ = fᵢ·Vᵢ.
- **fold map**: every node outside the PDE domain is mapped to its nearest in-domain neighbour (of
  26), so particles binned to an exterior node still feed the PDE. This is the node-level analogue
  of the same-compartment correction commented out in vcell-fvsolver's ``computeHistogram``. The
  same map extends fields into those exterior nodes for the particle engine's rate lookups.
- **effective volume** V_eff,i: the volume each domain node's counts are divided by, per
  correction variant:
  - ``"none"`` (VCell-like): the full element volume Vᵢ. Mass-conservative, but the shell is
    depleted (B2c).
  - ``"adapters"``: Σ Aⱼ over node i and the exterior nodes folded into it. Unbiased shell, but the
    PDE still integrates over Vᵢ, so particle sources are overcounted by V/A at the membrane.
  - ``"adapters+volumes"`` (default): the same effective volumes, and the FV engine uses them as
    its element volumes (fraction V_eff,i/V_full). Conservative and unbiased.
"""
from __future__ import annotations

import numpy as np

CORRECTIONS = ("none", "adapters", "adapters+volumes")


def _element_samples(grid, idx: np.ndarray, s: int) -> np.ndarray:
    """s³ sample points inside each element ``idx`` (flat indices), shape (len(idx), s³, 3)."""
    k, rem = np.divmod(idx, grid.num[0] * grid.num[1])
    j, i = np.divmod(rem, grid.num[0])
    lohi = [grid.element_bounds(d) for d in range(3)]
    lo = np.stack([lohi[0][0][i], lohi[1][0][j], lohi[2][0][k]], axis=1)
    hi = np.stack([lohi[0][1][i], lohi[1][1][j], lohi[2][1][k]], axis=1)
    u = (np.arange(s) + 0.5) / s
    g = np.stack(np.meshgrid(u, u, u, indexing="ij"), axis=-1).reshape(-1, 3)  # (s³, 3) in the unit cube
    return lo[:, None, :] + (hi - lo)[:, None, :] * g[None, :, :]


def accessible_fractions(realized, region: str, grid, coarse: int = 3, fine: int = 8) -> np.ndarray:
    """fᵢ ∈ [0, 1]: the share of each node's element inside ``region``'s smooth domain (grid-shaped).

    Every element gets ``coarse``³ midpoint samples. Elements whose samples disagree (the membrane
    crosses them) get ``fine``³. Uses the realization's own ``inside``.
    """
    n = grid.num_elements
    allidx = np.arange(n)
    f = np.empty(n)
    pts = _element_samples(grid, allidx, coarse)
    ins = realized.inside(region, pts.reshape(-1, 3)).reshape(n, -1)
    f[:] = ins.mean(axis=1)
    band = np.nonzero((f > 0) & (f < 1))[0]
    if len(band):
        fp = _element_samples(grid, band, fine)
        f[band] = realized.inside(region, fp.reshape(-1, 3)).reshape(len(band), -1).mean(axis=1)
    return f.reshape(grid.shape)


def fold_map(mask: np.ndarray, grid) -> np.ndarray:
    """Flat target index per node: itself in the mask, else its nearest in-mask 26-neighbour (−1 if none)."""
    m = np.asarray(mask, dtype=bool)
    nz, ny, nx = m.shape
    target = np.where(m.ravel(), np.arange(m.size), -1)
    best = np.full(m.size, np.inf)
    coords = np.stack(np.meshgrid(*[grid.axis_nodes(d) for d in (2, 1, 0)], indexing="ij"), axis=-1)  # (z,y,x,3) as (z,y,x)
    flat = np.arange(m.size).reshape(m.shape)
    outside = ~m
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == dy == dx == 0:
                    continue
                src = (slice(max(0, -dz), nz - max(0, dz)), slice(max(0, -dy), ny - max(0, dy)),
                       slice(max(0, -dx), nx - max(0, dx)))
                dst = (slice(max(0, dz), nz - max(0, -dz)), slice(max(0, dy), ny - max(0, -dy)),
                       slice(max(0, dx), nx - max(0, -dx)))
                cand = outside[src] & m[dst]
                if not cand.any():
                    continue
                d2 = ((coords[src] - coords[dst]) ** 2).sum(axis=-1)
                i = flat[src][cand]
                j = flat[dst][cand]
                dd = d2[cand]
                better = dd < best[i] - 1e-12
                target[i[better]] = j[better]
                best[i[better]] = dd[better]
    return target


def effective_volumes(grid, mask: np.ndarray, fractions: np.ndarray, fold: np.ndarray, correction: str) -> np.ndarray:
    """V_eff per node (grid-shaped; 0 outside the mask) for a correction variant (see module docstring)."""
    if correction not in CORRECTIONS:
        raise ValueError(f"correction must be one of {CORRECTIONS}, got {correction!r}")
    v = grid.element_volumes.ravel()
    if correction == "none":
        return np.where(np.asarray(mask, dtype=bool).ravel(), v, 0.0).reshape(grid.shape)
    acc = np.asarray(fractions, dtype=float).ravel() * v
    ok = fold >= 0
    out = np.bincount(fold[ok], weights=acc[ok], minlength=v.size)
    return out.reshape(grid.shape)
