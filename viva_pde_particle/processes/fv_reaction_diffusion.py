"""Finite volume reaction-diffusion process on a VCell-style Cartesian grid.

Reproduces vcell-fvsolver's ``FV_SOLVER`` (SparseVolumeEqnBuilder + PCG), the PDE
solver used in its hybrid runs. Per element ``i`` with volume fraction ``s_i``:

    s_i·uᵢⁿ⁺¹ + D·dt·Σⱼ K_ij·uⱼⁿ⁺¹ = s_i·(uᵢⁿ + Rᵢⁿ·dt)

- **Diffusion** is backward Euler (implicit). ``K`` is the volume-scaled zero-flux
  Laplacian from :meth:`CartesianGrid.diffusion_matrix`.
- **Reactions** are forward Euler (explicit). The rate ``R`` is evaluated from the
  state at the start of the step.

Particle species enter the reaction terms as binned concentrations, computed from
the ``particle_counts`` store (``count / (element volume · 602.214)``) and held
between particle updates.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from process_bigraph import Process

from viva_pde_particle.grid import CartesianGrid


def reaction_rates(terms: list[dict], species: list[str], conc: dict[str, np.ndarray], shape) -> dict[str, np.ndarray]:
    """dC/dt from mass-action terms (µM/s) for each continuous species (shared by the PDE engines)."""
    rates = {s: np.zeros(shape) for s in species}
    for term in terms:
        r = np.full(shape, term["coeff"] * term["k"])
        for s, n in term["reactants"].items():
            r = r * conc[s] ** n
        rates[term["species"]] += r
    return rates


class FVReactionDiffusion(Process):
    """PDE half of a hybrid model (see :func:`viva_pde_particle.model.partition`).

    Config:
        grid: ``{origin, size, num}`` (µm), VCell CartesianMesh conventions.
        pde: the ``pde`` part of a PartitionedModel, i.e. ``{species: {name: {diffusion}},
            particle_species: [...], terms: [{species, coeff, k, reactants}]}``.
        dt: PDE time step (s); each update takes ``round(interval/dt)`` steps.
    """

    config_schema = {
        "grid": "map",
        "pde": "map",
        "dt": {"_type": "float", "_default": 0.01},
    }

    def initialize(self, config):
        self.grid = CartesianGrid.from_config(config["grid"])
        self.pde = config["pde"]
        self.species = list(self.pde["species"])
        self.particle_species = list(self.pde.get("particle_species", []))
        self.dt = float(config["dt"])
        frac = self.grid.volume_fraction.ravel()
        self._s = frac
        K = self.grid.diffusion_matrix()
        self._solvers = {}
        for name, spec in self.pde["species"].items():
            D = float(spec["diffusion"])
            A = sp.diags(frac) + (D * self.dt) * K
            self._solvers[name] = spla.factorized(A.tocsc())

    def inputs(self):
        return {
            "fields": "map[array[float]]",
            "particle_counts": "map[array[float]]",
        }

    def outputs(self):
        return {"fields": "map[overwrite[array[float]]]"}

    def reaction_rates(self, conc: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """dC/dt from reactions (µM/s) for each continuous species."""
        return reaction_rates(self.pde["terms"], self.species, conc, self.grid.shape)

    def step(self, fields: dict[str, np.ndarray], particle_conc: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        conc = {**particle_conc, **fields}
        rates = self.reaction_rates(conc)
        new = {}
        for s in self.species:
            rhs = self._s * (fields[s].ravel() + rates[s].ravel() * self.dt)
            new[s] = self._solvers[s](rhs).reshape(self.grid.shape)
        return new

    def update(self, state, interval):
        fields = {s: np.asarray(state["fields"][s], dtype=float) for s in self.species}
        counts = state.get("particle_counts") or {}
        particle_conc = {
            p: self.grid.counts_to_uM(np.asarray(counts[p], dtype=float))
            if p in counts else np.zeros(self.grid.shape)
            for p in self.particle_species
        }
        n_steps = max(1, int(round(interval / self.dt)))
        for _ in range(n_steps):
            fields = self.step(fields, particle_conc)
        return {"fields": fields}
