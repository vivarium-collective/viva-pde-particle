"""FEniCSx reaction-diffusion on an unstructured mesh (P1). A general PDE engine: it knows no grid and no particles.

Internally the PDE is P1 on an unstructured mesh (e.g. a gmsh ball), with lumped mass,
backward-Euler diffusion and explicit reactions. The authoritative state is the DOF vector,
kept in ``field_dofs``.
- **Inputs:** species read but not evolved (particle species in a hybrid) come in through
  ``external_conc`` as DOF arrays. A Step fills them from the particle engine:
  :class:`~viva_pde_particle.steps.GridCountsToMeshConcentration` (Pᵀ of the grid histogram)
  or :class:`~viva_pde_particle.steps.PositionsToMeshConcentration` (the exact P1 point load).
- **Outputs:** ``field_dofs`` only. In a hybrid, the
  :class:`~viva_pde_particle.steps.MeshToGridField` Step samples them onto the particle
  engine's lookup grid (P·u).

See :mod:`viva_pde_particle.mesh`.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from process_bigraph import Process

from viva_pde_particle.mesh import build_mesh  # noqa: F401  (re-exported for existing callers)
from viva_pde_particle.processes.fv_reaction_diffusion import reaction_rates


class FenicsxMeshReactionDiffusion(Process):
    """Reaction-diffusion on an unstructured P1 mesh.

    Config:
        mesh: ``{"kind": "sphere", "radius", "center", "h"}``.
        pde: the ``pde`` part of a PartitionedModel.
        dt: time step (s).
    """

    config_schema = {
        "mesh": "map",
        "pde": "map",
        "dt": {"_type": "float", "_default": 0.01},
    }

    def initialize(self, config):
        import ufl
        from dolfinx import fem

        from viva_pde_particle.mesh import mesh_space

        self.pde = config["pde"]
        self.species = list(self.pde["species"])
        self.external_species = list(self.pde.get("external_species") or sorted(
            {s for t in self.pde["terms"] for s in t["reactants"]} - set(self.species)))
        self.dt = float(config["dt"])
        space = mesh_space(config["mesh"])
        self.V = V = space.V
        u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
        K = fem.assemble_matrix(fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)).to_scipy().tocsr()
        self.ml = space.ml  # lumped mass = nodal control volumes (µm³)
        self.n_dofs = len(self.ml)
        self._solvers = {
            name: spla.factorized((sp.diags(self.ml) + float(spec["diffusion"]) * self.dt * K).tocsc())
            for name, spec in self.pde["species"].items()
        }

    def initial_dofs(self, initial: dict[str, float]) -> dict[str, np.ndarray]:
        return {s: np.full(self.n_dofs, float(initial.get(s, 0.0))) for s in self.species}

    def inputs(self):
        return {"field_dofs": "map[array[float]]", "external_conc": "map[array[float]]"}

    def outputs(self):
        return {"field_dofs": "map[overwrite[array[float]]]"}

    def update(self, state, interval):
        u = {s: np.asarray(state["field_dofs"][s], dtype=float) for s in self.species}
        ext = state.get("external_conc") or {}
        pc = {s: np.asarray(ext[s], dtype=float) if s in ext else np.zeros(self.n_dofs) for s in self.external_species}
        for _ in range(max(1, int(round(interval / self.dt)))):
            rates = reaction_rates(self.pde["terms"], self.species, {**pc, **u}, (self.n_dofs,))
            u = {s: self._solvers[s](self.ml * (u[s] + self.dt * rates[s])) for s in self.species}
        return {"field_dofs": u}
