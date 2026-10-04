"""Robust point-in-closed-surface test for triangulated membranes (vertical-ray parity).

Why not VTK: ``vtkSelectEnclosedPoints`` (random-direction ray casting) returned wrong and
state-dependent answers for lattice-aligned query points against VCell's smooth surface, whose
vertices lie on lattice coordinates. Degenerate ray hits through vertices and edges gave inside
fractions of 0.33 and 0.16 for a true 0.36, and reseeding did not fix it.

Method: for a closed triangle surface, a point is inside iff a vertical ray from it crosses the
surface an odd number of times below it. Every triangle is binned by its xy bounding box, and each
point tests only the triangles in its xy cell. The query's xy is shifted by a tiny irrational
offset (~1e-7 of the surface's span), so no ray passes exactly through a vertex or edge. The
classification is unaffected except within that distance of the surface.
"""
from __future__ import annotations

import numpy as np

_JITTER = np.array([np.sqrt(2.0), np.sqrt(3.0)]) * 1e-7


class TriangleInside:
    """Inside test for a closed (watertight) triangulated surface, (m, 3, 3)."""

    def __init__(self, triangles: np.ndarray, cells: int | None = None):
        tri = np.asarray(triangles, dtype=float)
        self.tri = tri
        self.lo = tri.reshape(-1, 3).min(axis=0)
        self.hi = tri.reshape(-1, 3).max(axis=0)
        span = float(max(self.hi[0] - self.lo[0], self.hi[1] - self.lo[1], 1e-12))
        self.eps = _JITTER * span
        n = cells or max(8, int(np.sqrt(len(tri) / 4)))
        self.n = n
        self.cell = (self.hi[:2] - self.lo[:2]) / n + 1e-15
        tlo = np.floor((tri[:, :, :2].min(axis=1) - self.lo[:2]) / self.cell).astype(int).clip(0, n - 1)
        thi = np.floor((tri[:, :, :2].max(axis=1) - self.lo[:2]) / self.cell).astype(int).clip(0, n - 1)
        bins: list[list[int]] = [[] for _ in range(n * n)]
        for t in range(len(tri)):
            for ix in range(tlo[t, 0], thi[t, 0] + 1):
                for iy in range(tlo[t, 1], thi[t, 1] + 1):
                    bins[iy * n + ix].append(t)
        self.bins = [np.asarray(b, dtype=np.int64) for b in bins]
        # per-triangle xy data for the 2D point-in-triangle test and the hit height
        a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
        self._a2 = a[:, :2]
        self._e1 = (b - a)[:, :2]
        self._e2 = (c - a)[:, :2]
        self._det = self._e1[:, 0] * self._e2[:, 1] - self._e1[:, 1] * self._e2[:, 0]
        self._z = np.stack([a[:, 2], b[:, 2], c[:, 2]], axis=1)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        out = np.zeros(len(p), dtype=bool)
        xy = p[:, :2] + self.eps
        inbox = ((p >= self.lo) & (p <= self.hi)).all(axis=1)
        idx = np.nonzero(inbox)[0]
        if not len(idx):
            return out
        cxy = np.floor((xy[idx] - self.lo[:2]) / self.cell).astype(int).clip(0, self.n - 1)
        cell_id = cxy[:, 1] * self.n + cxy[:, 0]
        order = np.argsort(cell_id, kind="stable")
        cid_sorted = cell_id[order]
        starts = np.flatnonzero(np.r_[True, cid_sorted[1:] != cid_sorted[:-1]])
        ends = np.r_[starts[1:], len(order)]
        for s, e in zip(starts, ends):
            tris = self.bins[cid_sorted[s]]
            if not len(tris):
                continue
            pts = idx[order[s:e]]
            q = xy[pts]  # (P, 2)
            d = q[:, None, :] - self._a2[tris][None, :, :]  # (P, T, 2)
            e1, e2, det = self._e1[tris], self._e2[tris], self._det[tris]
            with np.errstate(divide="ignore", invalid="ignore"):
                u = (d[..., 0] * e2[None, :, 1] - d[..., 1] * e2[None, :, 0]) / det[None, :]
                v = (e1[None, :, 0] * d[..., 1] - e1[None, :, 1] * d[..., 0]) / det[None, :]
            hit = (u > 0) & (v > 0) & (u + v < 1) & (det[None, :] != 0)
            z = self._z[tris]
            zh = (1 - u - v) * z[None, :, 0] + u * z[None, :, 1] + v * z[None, :, 2]
            below = hit & (zh < p[pts, 2][:, None])
            out[pts] = (below.sum(axis=1) % 2) == 1
        return out
