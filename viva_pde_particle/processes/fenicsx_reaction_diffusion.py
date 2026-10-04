"""FEniCSx (dolfinx) reaction-diffusion process, interchangeable with FVReactionDiffusion.

The same PDE (zero-flux diffusion plus mass-action reactions), discretized with Q1
Lagrange finite elements on a structured quadrilateral or hexahedral mesh whose vertices
are the grid nodes. Time stepping matches the FV process: backward Euler for diffusion,
explicit reactions:

    (M + D·dt·K) uⁿ⁺¹ = M (uⁿ + dt·Rⁿ)

- **Mass matrix** ``mass="lumped"`` (default) or ``"consistent"``. On this mesh the Q1
  lumped mass at a vertex equals the FV dual-cell volume exactly, so particle counts
  convert to µM identically. The stiffness K is the Q1 27-point (3D) / 9-point (2D)
  operator, a different second-order discretization from FV's 7/5-point stencil.
- **Interface:** the same ports as FVReactionDiffusion (``fields`` and ``external_conc`` as
  grid-shaped arrays in node order), so the process is a drop-in replacement for it (an engine swap).
  DOFs are permuted to and from grid order internally.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from process_bigraph import Process

from viva_pde_particle.grid import CartesianGrid
from viva_pde_particle.processes.fv_reaction_diffusion import reaction_rates


def q1_operators(grid: CartesianGrid):
    """(K, M, dof_to_node): Q1 stiffness and mass (scipy CSR, DOF order) and each DOF's flat grid index."""
    import ufl
    from dolfinx import fem, mesh
    from mpi4py import MPI

    lo = np.array(grid.origin)
    hi = lo + np.array(grid.size)
    cells = [n - 1 for n in grid.num]
    if grid.dim == 2:
        msh = mesh.create_rectangle(MPI.COMM_SELF, [lo, hi], cells, mesh.CellType.quadrilateral)
    elif grid.dim == 3:
        msh = mesh.create_box(MPI.COMM_SELF, [lo, hi], cells, mesh.CellType.hexahedron)
    else:
        msh = mesh.create_interval(MPI.COMM_SELF, cells[0], [lo[0], hi[0]])
    V = fem.functionspace(msh, ("Lagrange", 1))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    K = fem.assemble_matrix(fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)).to_scipy()
    M = fem.assemble_matrix(fem.form(u * v * ufl.dx)).to_scipy()
    X = V.tabulate_dof_coordinates()[:, : grid.dim]
    idx = grid.node_index(X)  # (z, y, x) array indices
    dof_to_node = np.ravel_multi_index(idx, grid.shape)
    if len(np.unique(dof_to_node)) != grid.num_elements:
        raise RuntimeError("Q1 DOFs do not map one-to-one onto grid nodes")
    return K.tocsr(), M.tocsr(), dof_to_node


class FenicsxReactionDiffusion(Process):
    """PDE half of a hybrid model on FEniCSx (Q1, structured mesh); same ports as FVReactionDiffusion.

    Config:
        grid: ``{origin, size, num}``; mesh vertices are the grid nodes.
        pde: the ``pde`` part of a PartitionedModel.
        dt: time step (s).
        mass: ``"lumped"`` (default) or ``"consistent"``.
    """

    config_schema = {
        "grid": "map",
        "pde": "map",
        "dt": {"_type": "float", "_default": 0.01},
        "mass": {"_type": "string", "_default": "lumped"},
    }

    def initialize(self, config):
        if config["mass"] not in ("lumped", "consistent"):
            raise ValueError(f"mass must be 'lumped' or 'consistent', got {config['mass']!r}")
        self.grid = CartesianGrid.from_config(config["grid"])
        self.pde = config["pde"]
        self.species = list(self.pde["species"])
        # species read but not evolved here (e.g. particle species): an explicit list, or every
        # reactant in the terms that is not one of this engine's species
        self.external_species = list(self.pde.get("external_species") or sorted(
            {s for t in self.pde["terms"] for s in t["reactants"]} - set(self.species)))
        self.particle_species = self.external_species  # alias kept for HybridCoupler (Phase 7d)
        self.dt = float(config["dt"])
        K, M, self._dof_to_node = q1_operators(self.grid)
        if config["mass"] == "lumped":
            M = sp.diags(np.asarray(M.sum(axis=1)).ravel())
        self._M = M.tocsr()
        self._solvers = {
            name: spla.factorized((self._M + float(spec["diffusion"]) * self.dt * K).tocsc())
            for name, spec in self.pde["species"].items()
        }

    def inputs(self):
        return {"fields": "map[array[float]]", "external_conc": "map[array[float]]"}

    def outputs(self):
        return {"fields": "map[overwrite[array[float]]]"}

    def _to_dofs(self, arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=float).ravel()[self._dof_to_node]

    def _to_grid(self, dofs: np.ndarray) -> np.ndarray:
        out = np.empty(self.grid.num_elements)
        out[self._dof_to_node] = dofs
        return out.reshape(self.grid.shape)

    def step(self, fields: dict[str, np.ndarray], external_conc: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        rates = reaction_rates(self.pde["terms"], self.species, {**external_conc, **fields}, self.grid.shape)
        return {
            s: self._to_grid(self._solvers[s](self._M @ (self._to_dofs(fields[s]) + self.dt * self._to_dofs(rates[s]))))
            for s in self.species
        }

    def update(self, state, interval):
        fields = {s: np.asarray(state["fields"][s], dtype=float) for s in self.species}
        ext = state.get("external_conc") or {}
        external_conc = {
            s: np.asarray(ext[s], dtype=float) if s in ext else np.zeros(self.grid.shape)
            for s in self.external_species
        }
        for _ in range(max(1, int(round(interval / self.dt)))):
            fields = self.step(fields, external_conc)
        return {"fields": fields}
