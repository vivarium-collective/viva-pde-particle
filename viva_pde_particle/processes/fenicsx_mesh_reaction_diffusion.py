"""FEniCSx reaction-diffusion on an unstructured mesh, coupled to particles through a background grid.

Same ports as FVReactionDiffusion: ``particle_counts`` in and ``fields`` out, both on the
Cartesian background grid that SmoldynHybrid uses. Internally the PDE is P1 on an
unstructured mesh (e.g. a gmsh ball), with lumped mass, backward-Euler diffusion and
explicit reactions. The authoritative state is the DOF vector, kept in ``field_dofs``.

- **Grid → mesh:** particle loads b = Pᵀ·counts give nodal concentrations
  b_j / (M_L,j · 602.214).
- **Mesh → grid:** fields are sampled as P·u.

See :mod:`viva_pde_particle.mesh`.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from process_bigraph import Process

from viva_pde_particle.grid import CartesianGrid
from viva_pde_particle.processes.fv_reaction_diffusion import reaction_rates
from viva_pde_particle.units import MOLECULES_PER_UM3_PER_UM

_MESH_CACHE: dict = {}


def build_mesh(spec: dict):
    key = tuple(sorted((k, tuple(v) if isinstance(v, list) else v) for k, v in spec.items()))
    if key not in _MESH_CACHE:
        from viva_pde_particle.mesh import sphere_mesh

        if spec.get("kind") != "sphere":
            raise ValueError(f"unsupported mesh kind {spec.get('kind')!r}")
        _MESH_CACHE[key] = sphere_mesh(spec["radius"], tuple(spec["center"]), spec.get("h"))
    return _MESH_CACHE[key]


class FenicsxMeshReactionDiffusion(Process):
    """PDE half on an unstructured P1 mesh; particle side on ``grid`` (background Cartesian grid).

    Config:
        grid: background grid ``{origin, size, num}`` (covers the mesh; SmoldynHybrid uses the same).
        mesh: ``{"kind": "sphere", "radius", "center", "h"}``.
        pde: the ``pde`` part of a PartitionedModel.
        dt: time step (s).
    """

    config_schema = {
        "grid": "map",
        "mesh": "map",
        "pde": "map",
        "dt": {"_type": "float", "_default": 0.01},
    }

    def initialize(self, config):
        import ufl
        from dolfinx import fem

        from viva_pde_particle.mesh import MeshGridTransfer

        self.grid = CartesianGrid.from_config(config["grid"])
        self.pde = config["pde"]
        self.species = list(self.pde["species"])
        self.particle_species = list(self.pde.get("particle_species", []))
        self.dt = float(config["dt"])
        msh = build_mesh(config["mesh"])
        self.transfer = MeshGridTransfer.build(msh, self.grid)
        V = self.transfer.V
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        K = fem.assemble_matrix(fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)).to_scipy().tocsr()
        M = fem.assemble_matrix(fem.form(u * v * ufl.dx)).to_scipy()
        self.ml = np.asarray(M.sum(axis=1)).ravel()  # lumped mass = nodal control volumes (µm³)
        self.n_dofs = len(self.ml)
        self._solvers = {
            name: spla.factorized((sp.diags(self.ml) + float(spec["diffusion"]) * self.dt * K).tocsc())
            for name, spec in self.pde["species"].items()
        }

    def initial_dofs(self, initial: dict[str, float]) -> dict[str, np.ndarray]:
        return {s: np.full(self.n_dofs, float(initial.get(s, 0.0))) for s in self.species}

    def to_grid(self, dofs: np.ndarray) -> np.ndarray:
        return (self.transfer.P @ dofs).reshape(self.grid.shape)

    def inputs(self):
        return {"field_dofs": "map[array[float]]", "particle_counts": "map[array[float]]"}

    def outputs(self):
        return {"field_dofs": "map[overwrite[array[float]]]", "fields": "map[overwrite[array[float]]]"}

    def update(self, state, interval):
        u = {s: np.asarray(state["field_dofs"][s], dtype=float) for s in self.species}
        counts = state.get("particle_counts") or {}
        pc = {}
        for p in self.particle_species:
            load = self.transfer.P.T @ np.asarray(counts.get(p, np.zeros(self.grid.shape)), dtype=float).ravel()
            pc[p] = load / (self.ml * MOLECULES_PER_UM3_PER_UM)
        for _ in range(max(1, int(round(interval / self.dt)))):
            rates = reaction_rates(self.pde["terms"], self.species, {**pc, **u}, (self.n_dofs,))
            u = {s: self._solvers[s](self.ml * (u[s] + self.dt * rates[s])) for s in self.species}
        return {"field_dofs": u, "fields": {s: self.to_grid(v) for s, v in u.items()}}
