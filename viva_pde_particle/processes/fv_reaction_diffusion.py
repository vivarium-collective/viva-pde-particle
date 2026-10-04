"""Finite volume reaction-diffusion process on a VCell-style Cartesian grid.

Reproduces vcell-fvsolver's ``FV_SOLVER`` (SparseVolumeEqnBuilder + PCG), the PDE
solver used in its hybrid runs. Per element ``i`` with volume fraction ``s_i``:

    s_i·uᵢⁿ⁺¹ + D·dt·Σⱼ K_ij·uⱼⁿ⁺¹ = s_i·(uᵢⁿ + Rᵢⁿ·dt)

- **Diffusion** is backward Euler (implicit). ``K`` is the volume-scaled zero-flux
  Laplacian from :meth:`CartesianGrid.diffusion_matrix`.
- **Reactions** are forward Euler (explicit). The rate ``R`` is evaluated from the
  state at the start of the step.

Species the engine reads but does not evolve (particle species in a hybrid) come in through
``external_conc``: {species: µM array on this grid}, held between updates. The engine knows
nothing about particles. In a hybrid composite a Step
(:class:`viva_pde_particle.steps.GridCountsToConcentration`) fills ``external_conc`` from the
particle histogram.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from process_bigraph import Process

from viva_pde_particle.grid import CartesianGrid
from viva_pde_particle.kinetics import reaction_rates  # noqa: F401  (re-exported for existing callers)


class FVReactionDiffusion(Process):
    """PDE half of a hybrid model (see :func:`viva_pde_particle.model.partition`).

    Config:
        grid: ``{origin, size, num}`` (µm), VCell CartesianMesh conventions.
        pde: the ``pde`` part of a PartitionedModel, i.e. ``{species: {name: {diffusion}},
            terms: [{species, coeff, k, reactants}]}``, optionally ``external_species``.
        dt: PDE time step (s); each update takes ``round(interval/dt)`` steps.
        domain: optional ``{"mask", "volume_fraction"}`` (grid-shaped arrays). ``mask`` restricts
            the PDE to a subdomain (zero flux at its boundary; nodes outside are frozen and get no
            reactions), e.g. VCell's staircase region. ``volume_fraction`` replaces the element
            volume fractions ``s`` inside the mask, e.g. accessible volumes from a smooth
            membrane (Phase 7e.5).
    """

    config_schema = {
        "grid": "map",
        "pde": "map",
        "dt": {"_type": "float", "_default": 0.01},
        "domain": {"_type": "map", "_default": {}},
    }

    def initialize(self, config):
        self.grid = CartesianGrid.from_config(config["grid"])
        self.pde = config["pde"]
        self.species = list(self.pde["species"])
        # species read but not evolved here (e.g. particle species): an explicit list, or every
        # reactant in the terms that is not one of this engine's species
        self.external_species = list(self.pde.get("external_species") or sorted(
            {s for t in self.pde["terms"] for s in t["reactants"]} - set(self.species)))
        self.dt = float(config["dt"])
        domain = config.get("domain") or {}
        mask = domain.get("mask")
        frac = self.grid.volume_fraction.ravel()
        self._mask = None
        if mask is not None:
            self._mask = np.asarray(mask, dtype=bool).ravel()
            vf = domain.get("volume_fraction")
            if vf is not None:
                frac = np.asarray(vf, dtype=float).ravel()
            frac = np.where(self._mask, frac, 1.0)  # frozen outside: unit diagonal, no coupling, no reactions
        self._s = frac
        K = self.grid.diffusion_matrix(mask)
        self._solvers = {}
        for name, spec in self.pde["species"].items():
            D = float(spec["diffusion"])
            A = sp.diags(frac) + (D * self.dt) * K
            self._solvers[name] = spla.factorized(A.tocsc())

    def inputs(self):
        return {
            "fields": "map[array[float]]",
            "external_conc": "map[array[float]]",
        }

    def outputs(self):
        return {"fields": "map[overwrite[array[float]]]"}

    def reaction_rates(self, conc: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """dC/dt from reactions (µM/s) for each continuous species."""
        return reaction_rates(self.pde["terms"], self.species, conc, self.grid.shape)

    def step(self, fields: dict[str, np.ndarray], external_conc: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        conc = {**external_conc, **fields}
        rates = self.reaction_rates(conc)
        new = {}
        for s in self.species:
            rate = rates[s].ravel() if self._mask is None else np.where(self._mask, rates[s].ravel(), 0.0)
            rhs = self._s * (fields[s].ravel() + rate * self.dt)
            new[s] = self._solvers[s](rhs).reshape(self.grid.shape)
        return new

    def update(self, state, interval):
        fields = {s: np.asarray(state["fields"][s], dtype=float) for s in self.species}
        ext = state.get("external_conc") or {}
        external_conc = {
            s: np.asarray(ext[s], dtype=float) if s in ext else np.zeros(self.grid.shape)
            for s in self.external_species
        }
        n_steps = max(1, int(round(interval / self.dt)))
        for _ in range(n_steps):
            fields = self.step(fields, external_conc)
        return {"fields": fields}
