"""Node-centred Cartesian grid with vcell-fvsolver's CartesianMesh conventions.

The same geometry is used on both sides of the coupling, so fields and particle
histograms line up node for node:

- ``N`` nodes per axis, spacing ``dx = L/(N-1)`` (``L`` when ``N == 1``);
- node ``i`` at ``x0 + i·dx``; a point belongs to the nearest node
  ``(int)((x-x0)·(N-1)/L + 0.5)``;
- element volume ``dx·dy·dz`` scaled by 1/2 for each axis on which the node lies on
  the boundary (faces 1/2, edges 1/4, corners 1/8), as in ``getVolumeOfElement_cu``;
- arrays have shape ``(Nz, Ny, Nx)`` restricted to the model dimension. In C order
  that is VCell's index ``i + Nx·(j + Ny·k)``, the layout ``smoldyn.HybridGrid`` uses.

Unused axes (2D, 1D) have size 1 µm and a single node (unit-thickness slab).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import scipy.sparse as sp

from viva_pde_particle.units import MOLECULES_PER_UM3_PER_UM


@dataclass(frozen=True)
class CartesianGrid:
    origin: tuple[float, ...]
    size: tuple[float, ...]
    num: tuple[int, ...]

    def __post_init__(self):
        if not (len(self.origin) == len(self.size) == len(self.num)) or not 1 <= len(self.num) <= 3:
            raise ValueError("origin, size and num must have equal length 1, 2 or 3")
        if any(n < 1 for n in self.num) or any(s <= 0 for s in self.size):
            raise ValueError("num must be >= 1 and size > 0")
        object.__setattr__(self, "origin", tuple(float(v) for v in self.origin))
        object.__setattr__(self, "size", tuple(float(v) for v in self.size))
        object.__setattr__(self, "num", tuple(int(v) for v in self.num))

    @classmethod
    def from_config(cls, config: dict) -> "CartesianGrid":
        return cls(tuple(config["origin"]), tuple(config["size"]), tuple(config["num"]))

    def to_config(self) -> dict:
        return {"origin": list(self.origin), "size": list(self.size), "num": list(self.num)}

    # ------------------------------------------------------------------ geometry
    @property
    def dim(self) -> int:
        return len(self.num)

    @property
    def shape(self) -> tuple[int, ...]:
        """Array shape (Nz, Ny, Nx) restricted to ``dim``."""
        return tuple(reversed(self.num))

    @property
    def spacing(self) -> tuple[float, ...]:
        return tuple(L / (n - 1) if n > 1 else L for L, n in zip(self.size, self.num))

    @property
    def num_elements(self) -> int:
        return int(np.prod(self.num))

    @cached_property
    def full_volume(self) -> float:
        """Interior element volume dx·dy·dz in µm³ (unused axes have thickness 1)."""
        return float(np.prod(self.spacing))

    def axis_nodes(self, d: int) -> np.ndarray:
        n, x0, L = self.num[d], self.origin[d], self.size[d]
        if n == 1:
            return np.array([x0 + 0.5 * L])
        return x0 + np.arange(n) * self.spacing[d]

    def node_coordinates(self) -> list[np.ndarray]:
        """Coordinate arrays (x, y, z order), each broadcast to ``shape``."""
        axes = [self.axis_nodes(d) for d in range(self.dim)]
        mesh = np.meshgrid(*reversed(axes), indexing="ij")  # shape order (z, y, x)
        return list(reversed(mesh))

    def _boundary_flags(self, d: int) -> np.ndarray:
        """1 where the node lies on the low/high boundary of axis d (axis with >1 node)."""
        n = self.num[d]
        flags = np.zeros(n)
        if n > 1:
            flags[0] = flags[-1] = 1
        return flags

    @cached_property
    def volume_fraction(self) -> np.ndarray:
        """Element volume / full volume: 1/2 per boundary axis (VCell VOLUME_HALF etc.)."""
        frac = np.ones(self.shape)
        for d in range(self.dim):
            factor = np.where(self._boundary_flags(d) == 1, 0.5, 1.0)
            frac = frac * factor.reshape(self._axis_shape(d))
        return frac

    @cached_property
    def element_volumes(self) -> np.ndarray:
        return self.full_volume * self.volume_fraction

    def _axis_shape(self, d: int) -> tuple[int, ...]:
        """Broadcast shape for a 1D array along axis d (x is the last array axis)."""
        shape = [1] * self.dim
        shape[self.dim - 1 - d] = self.num[d]
        return tuple(shape)

    # ------------------------------------------------------------------ particles <-> nodes
    def node_index(self, points: np.ndarray) -> tuple[np.ndarray, ...]:
        """Nearest-node array indices (z, y, x order) for points of shape (n, dim), clamped."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        idx = []
        for d in range(self.dim):
            n = self.num[d]
            if n == 1:
                i = np.zeros(len(points), dtype=int)
            else:
                i = ((points[:, d] - self.origin[d]) * (n - 1) / self.size[d] + 0.5).astype(int)
                i = np.clip(i, 0, n - 1)
            idx.append(i)
        return tuple(reversed(idx))

    def histogram(self, points: np.ndarray) -> np.ndarray:
        counts = np.zeros(self.shape)
        if len(points):
            np.add.at(counts, self.node_index(points), 1)
        return counts

    def counts_to_uM(self, counts: np.ndarray) -> np.ndarray:
        """Binned particle counts -> µM (vcell-fvsolver copyParticleCountsToConcentration)."""
        return np.asarray(counts, dtype=float) / self.element_volumes / MOLECULES_PER_UM3_PER_UM

    def uM_to_counts(self, conc: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
        """Expected molecule counts per element for a µM field; Poisson-sampled if ``rng`` is given."""
        expected = np.asarray(conc, dtype=float) * self.element_volumes * MOLECULES_PER_UM3_PER_UM
        return rng.poisson(expected).astype(float) if rng is not None else expected

    def element_bounds(self, d: int) -> tuple[np.ndarray, np.ndarray]:
        """Low/high extent of each node's cell along axis d, clipped to the domain."""
        x = self.axis_nodes(d)
        if self.num[d] == 1:
            return np.array([self.origin[d]]), np.array([self.origin[d] + self.size[d]])
        h = 0.5 * self.spacing[d]
        lo = np.maximum(x - h, self.origin[d])
        hi = np.minimum(x + h, self.origin[d] + self.size[d])
        return lo, hi

    # ------------------------------------------------------------------ diffusion operator
    def diffusion_matrix(self) -> sp.csr_matrix:
        """Volume-scaled negative Laplacian ``K`` with zero-flux boundaries.

        For element i and neighbour j along axis d: ``K_ij = -a_ij/dx_d²`` and
        ``K_ii = Σ_j a_ij/dx_d²``. The face fraction ``a_ij`` is halved for each
        transverse axis on which the face lies on the boundary. This is the operator
        in vcell-fvsolver's SparseVolumeEqnBuilder, with the system scaled by
        dt/VOLUME. A backward-Euler step is then ``(diag(s) + D·dt·K) uⁿ⁺¹ = s·(uⁿ + R·dt)``,
        with ``s`` the volume fraction.
        """
        n_el = self.num_elements
        flat = np.arange(n_el).reshape(self.shape)
        rows, cols, vals = [], [], []
        for d in range(self.dim):
            if self.num[d] < 2:
                continue
            area = np.ones(self.shape)
            for e in range(self.dim):
                if e != d:
                    area = area * np.where(self._boundary_flags(e) == 1, 0.5, 1.0).reshape(self._axis_shape(e))
            ax = self.dim - 1 - d  # array axis of spatial axis d
            lo = [slice(None)] * self.dim
            hi = [slice(None)] * self.dim
            lo[ax] = slice(0, -1)
            hi[ax] = slice(1, None)
            i_idx = flat[tuple(lo)].ravel()
            j_idx = flat[tuple(hi)].ravel()
            coeff = (area[tuple(lo)] / self.spacing[d] ** 2).ravel()  # a_ij equal on both sides
            rows += [i_idx, j_idx, i_idx, j_idx]
            cols += [j_idx, i_idx, i_idx, j_idx]
            vals += [-coeff, -coeff, coeff, coeff]
        if not rows:
            return sp.csr_matrix((n_el, n_el))
        return sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_el, n_el)
        )
