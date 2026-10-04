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

    def inside(self, region: str, points: np.ndarray) -> np.ndarray:
        return self.locate(points) == self.regions.index(region)


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


_REALIZATIONS: dict = {}


def realize_fenics(description, h: float) -> FenicsRealization:
    """FenicsRealization for (description, h), cached per process (realization is deterministic)."""
    from viva_pde_particle.geometry.description import description_to_config

    key = (json.dumps(description_to_config(description), sort_keys=True), float(h))
    if key not in _REALIZATIONS:
        _REALIZATIONS[key] = FenicsRealization(description, h)
    return _REALIZATIONS[key]
