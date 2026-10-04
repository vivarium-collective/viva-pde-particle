"""Realized geometry: the interface everything downstream of a PDE geometry compiler relies on.

A :class:`RealizedGeometry` gives:
- the volume regions (subvolume names);
- a vectorized ``locate(points) → region index`` (−1 outside every region);
- each region's boundary as outward-oriented triangles;
- interior points per region;
- each region's volume.

The particle side (Smoldyn membrane, volume samples) and the adapters (accessible volumes, region
labels) are derived from it, so the particle and PDE domains cannot be realized inconsistently.

:class:`FenicsRealization` is the FEniCS geometry compiler: vcell-fenics' Netgen-based ``realize``,
which gives one tetrahedral mesh per subvolume.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod

import numpy as np


class RealizedGeometry(ABC):
    regions: tuple[str, ...]
    lo: np.ndarray
    hi: np.ndarray

    @abstractmethod
    def locate(self, points: np.ndarray) -> np.ndarray:
        """Region index (into ``regions``) per point, −1 outside every region."""

    @abstractmethod
    def boundary_triangles(self, region: str) -> np.ndarray:
        """The region's boundary, (m, 3, 3) outward-oriented triangles."""

    @abstractmethod
    def interior_points(self, region: str, max_points: int = 64) -> np.ndarray:
        """Points strictly inside the region, (n, 3). Smoldyn's compartment test needs a line to *some*
        listed point that crosses no surface, so a non-convex region needs several, spread out."""

    @abstractmethod
    def volume(self, region: str) -> float:
        """The region's volume as discretized (µm³)."""

    @abstractmethod
    def region_bounds(self, region: str) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounding box (lo, hi) of the region as discretized."""

    def inside(self, region: str, points: np.ndarray) -> np.ndarray:
        return self.locate(points) == self.regions.index(region)

    def uniform_points(self, region: str, n: int, rng: np.random.Generator) -> np.ndarray:
        """n points uniformly distributed in the region (rejection sampling in its bounding box)."""
        lo, hi = self.region_bounds(region)
        out = np.empty((0, 3))
        while len(out) < n:
            cand = lo + (hi - lo) * rng.random((max(2 * (n - len(out)), 1000), 3))
            out = np.vstack([out, cand[self.inside(region, cand)]])
        return out[:n]


class FenicsRealization(RealizedGeometry):
    """A GeometryDescription meshed by Netgen (vcell-fenics ``realize``): one tet mesh per volume region."""

    def __init__(self, description, h: float):
        from dolfinx import fem
        from mpi4py import MPI
        from vcell_fenics.backend.realize import realize

        from viva_pde_particle.mesh import PointLocator

        self.description = description
        self.h = float(h)
        self.geometry = realize(description, h=self.h, comm=MPI.COMM_SELF)
        self.regions = tuple(sv.name for sv in description.subvolumes
                             if self.geometry.kind_of(sv.name) == "volume")
        self.meshes = {r: self.geometry.mesh_of(r) for r in self.regions}
        self._locators = {r: PointLocator.build(m, fem.functionspace(m, ("Lagrange", 1)))
                          for r, m in self.meshes.items()}
        self.lo = np.asarray(description.origin, dtype=float)
        self.hi = self.lo + np.asarray(description.extent, dtype=float)

    def locate(self, points):
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        labels = np.full(len(pts), -1, dtype=np.int64)
        for i, r in enumerate(self.regions):
            todo = np.nonzero(labels < 0)[0]
            if not len(todo):
                break
            labels[todo[self._locators[r].inside(pts[todo])]] = i
        return labels

    def inside(self, region, points):
        return self._locators[region].inside(np.asarray(points, dtype=float).reshape(-1, 3))

    def region_bounds(self, region):
        x = self.meshes[region].geometry.x
        return x.min(axis=0), x.max(axis=0)

    def boundary_triangles(self, region):
        from viva_pde_particle.mesh import boundary_triangles

        return boundary_triangles(self.meshes[region])

    def interior_points(self, region, max_points=64):
        c = self._locators[region].centroids  # tet centroids are inside the region by construction
        step = max(1, len(c) // max_points)
        return c[::step][:max_points]

    def volume(self, region):
        import ufl
        from dolfinx import fem

        m = self.meshes[region]
        return float(fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=m))))


class MeshRealization(RealizedGeometry):
    """A single dolfinx tet mesh as a one-region RealizedGeometry (e.g. the pre-7e gmsh ball).

    ``locator`` (a PointLocator on the mesh) and ``interior`` (compartment points) may be given to
    reuse existing objects; otherwise they are built from the mesh.
    """

    def __init__(self, msh, region: str = "domain", locator=None, interior=None):
        self.msh = msh
        self.regions = (region,)
        if locator is None:
            from dolfinx import fem

            from viva_pde_particle.mesh import PointLocator

            locator = PointLocator.build(msh, fem.functionspace(msh, ("Lagrange", 1)))
        self.locator = locator
        self._interior = None if interior is None else np.atleast_2d(np.asarray(interior, dtype=float))
        x = msh.geometry.x
        self.lo, self.hi = x.min(axis=0), x.max(axis=0)

    def locate(self, points):
        return np.where(self.locator.inside(np.asarray(points, dtype=float).reshape(-1, 3)), 0, -1)

    def inside(self, region, points):
        return self.locator.inside(np.asarray(points, dtype=float).reshape(-1, 3))

    def region_bounds(self, region):
        return self.lo, self.hi

    def boundary_triangles(self, region):
        from viva_pde_particle.mesh import boundary_triangles

        return boundary_triangles(self.msh)

    def interior_points(self, region, max_points=64):
        if self._interior is not None:
            return self._interior
        c = self.locator.centroids
        step = max(1, len(c) // max_points)
        return c[::step][:max_points]

    def volume(self, region):
        import ufl
        from dolfinx import fem

        return float(fem.assemble_scalar(fem.form(1.0 * ufl.dx(domain=self.msh))))


_REALIZATIONS: dict = {}


def realize_fenics(description, h: float) -> FenicsRealization:
    """FenicsRealization for (description, h), cached per process (realization is deterministic)."""
    from viva_pde_particle.geometry.description import description_to_config

    key = (json.dumps(description_to_config(description), sort_keys=True), float(h))
    if key not in _REALIZATIONS:
        _REALIZATIONS[key] = FenicsRealization(description, h)
    return _REALIZATIONS[key]
