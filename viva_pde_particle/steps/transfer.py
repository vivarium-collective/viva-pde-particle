"""Adapters between the particle and PDE engines, as Steps (Phase 7a, 7b).

**PDE → particles (7b).** ``MeshToGridField`` samples a P1 field onto the Cartesian grid that
the particle engine's rate lookups use (u_grid = P·u). Structured engines (FV, Q1) already
produce fields on that grid and need no adapter.

**Particles → PDE (7a).** The steps below turn particle-engine output into concentrations on
a PDE discretization.

The PDE engines no longer know about particles. Each reads an ``external_conc`` map,
{species: µM array on its own discretization}, for species that appear in its reaction
terms but that it does not evolve. One of these Steps fills that map from the particle
engine's output:

| Step | input | output discretization | rule |
|---|---|---|---|
| ``GridCountsToConcentration`` | ``particle_counts`` (grid histogram) | the same Cartesian grid (FV, FEniCSx Q1) | count / (V_element · 602.214) |
| ``GridCountsToMeshConcentration`` | ``particle_counts`` | P1 mesh DOFs | (Pᵀ · counts) / (M_L · 602.214) |
| ``PositionsToMeshConcentration`` | ``particle_positions`` | P1 mesh DOFs | (Σ_p φ_j(x_p)) / (M_L,j · 602.214) |

**Timing.** A Step runs right after the process updates of a time point are applied, before
the next process intervals start. So the converted concentrations are exactly what the PDE
engine used to compute from the raw counts at the start of its interval, and the coupling
timing (docs/DESIGN.md §6.3) is unchanged.

Each Step also exposes ``convert(...)``, used by the composite builders for the initial state,
because Steps do not run at composite initialization.
"""
from __future__ import annotations

import numpy as np
from process_bigraph import Step

from viva_pde_particle.grid import CartesianGrid
from viva_pde_particle.units import MOLECULES_PER_UM3_PER_UM


class GridCountsToConcentration(Step):
    """Per-node particle counts → µM on the same grid (``count / (V_element · 602.214)``).

    Config: ``grid`` (``{origin, size, num}``), ``species`` (particle species to convert;
    missing ones give zeros). Optional ``staircase`` map, for a staircase PDE domain with a smooth
    membrane (7e.5):

    - ``fold``: flat target node per node (``geometry.accessible.fold_map``). Counts at nodes outside
      the PDE domain are added to their in-domain neighbour (compartment-aware binning).
    - ``volumes``: grid-shaped effective volume per node (µM = count / (volume · 602.214); 0 where
      the volume is 0), e.g. accessible volumes (``geometry.accessible.effective_volumes``).
    """

    config_schema = {"grid": "map", "species": "list[string]", "staircase": {"_type": "map", "_default": {}}}

    def initialize(self, config):
        self.grid = CartesianGrid.from_config(config["grid"])
        self.species = list(config["species"])
        st = config.get("staircase") or {}
        self.fold = None if st.get("fold") is None else np.asarray(st["fold"], dtype=np.int64)
        self.volumes = None if st.get("volumes") is None else np.asarray(st["volumes"], dtype=float)

    def inputs(self):
        return {"particle_counts": "map[array[float]]"}

    def outputs(self):
        return {"particle_conc": "map[overwrite[array[float]]]"}

    def convert(self, counts: dict) -> dict[str, np.ndarray]:
        if self.fold is None and self.volumes is None:
            return {
                s: self.grid.counts_to_uM(np.asarray(counts[s], dtype=float)) if s in counts else np.zeros(self.grid.shape)
                for s in self.species
            }
        vol = self.grid.element_volumes if self.volumes is None else self.volumes
        out = {}
        for s in self.species:
            c = np.asarray(counts[s], dtype=float).ravel() if s in counts else np.zeros(self.grid.num_elements)
            if self.fold is not None:
                ok = self.fold >= 0
                c = np.bincount(self.fold[ok], weights=c[ok], minlength=c.size)
            with np.errstate(divide="ignore", invalid="ignore"):
                conc = np.where(vol.ravel() > 0, c / (vol.ravel() * MOLECULES_PER_UM3_PER_UM), 0.0)
            out[s] = conc.reshape(self.grid.shape)
        return out

    def update(self, state):
        return {"particle_conc": self.convert(state.get("particle_counts") or {})}


class ExtendGridField(Step):
    """Grid fields → lookup fields for the particle engine, extended past a staircase PDE domain (7e.5).

    Nodes outside the PDE domain take the value of their fold target (``geometry.accessible.fold_map``),
    so 0th-order creation in elements the smooth membrane cuts, whose centres lie outside the
    staircase, reads the adjacent domain field rather than a frozen or zero one.
    Config: ``grid``, ``species``, ``staircase: {"fold"}``.
    """

    config_schema = {"grid": "map", "species": "list[string]", "staircase": "map"}

    def initialize(self, config):
        self.grid = CartesianGrid.from_config(config["grid"])
        self.species = list(config["species"])
        fold = np.asarray(config["staircase"]["fold"], dtype=np.int64)
        self._src = np.where(fold >= 0, fold, np.arange(fold.size))

    def inputs(self):
        return {"fields": "map[array[float]]"}

    def outputs(self):
        return {"lookup_fields": "map[overwrite[array[float]]]"}

    def convert(self, fields: dict) -> dict[str, np.ndarray]:
        return {s: np.asarray(fields[s], dtype=float).ravel()[self._src].reshape(self.grid.shape)
                for s in self.species if s in fields}

    def update(self, state):
        return {"lookup_fields": self.convert(state.get("fields") or {})}


class _MeshConcentration(Step):
    config_schema = {"grid": "map", "mesh": "map", "species": "list[string]"}

    def initialize(self, config):
        from viva_pde_particle.mesh import mesh_transfer

        self.grid = CartesianGrid.from_config(config["grid"])
        self.species = list(config["species"])
        self.transfer = mesh_transfer(config["mesh"], self.grid)
        self.ml = self.transfer.ml

    def outputs(self):
        return {"particle_conc": "map[overwrite[array[float]]]"}

    def load(self, data) -> np.ndarray:  # molecules per DOF
        raise NotImplementedError

    def convert(self, data: dict) -> dict[str, np.ndarray]:
        return {s: self.load(data.get(s)) / (self.ml * MOLECULES_PER_UM3_PER_UM) for s in self.species}


class GridCountsToMeshConcentration(_MeshConcentration):
    """Grid histogram → P1 DOF concentrations through the transfer matrix: (Pᵀ·counts)/(M_L·602.214).

    Config: ``grid`` (background grid), ``mesh`` (mesh spec), ``species``.
    """

    def inputs(self):
        return {"particle_counts": "map[array[float]]"}

    def load(self, counts) -> np.ndarray:
        h = np.zeros(self.grid.shape) if counts is None else np.asarray(counts, dtype=float)
        return self.transfer.P.T @ h.ravel()

    def update(self, state):
        return {"particle_conc": self.convert(state.get("particle_counts") or {})}


class PositionsToMeshConcentration(_MeshConcentration):
    """Molecule positions → P1 DOF concentrations by the exact point load Σ_p φ_j(x_p)/(M_L,j·602.214).

    In expectation the load is ρ·∫φ_j, so a uniform density gives a uniform concentration up
    to the boundary. Config as GridCountsToMeshConcentration.
    """

    def initialize(self, config):
        super().initialize(config)
        from viva_pde_particle.mesh import mesh_locator

        self.locator = mesh_locator(config["mesh"], self.grid)

    def inputs(self):
        return {"particle_positions": "map[array[float]]"}

    def load(self, positions) -> np.ndarray:
        return self.locator.load(np.zeros((0, 3)) if positions is None else np.asarray(positions, dtype=float))

    def update(self, state):
        return {"particle_conc": self.convert(state.get("particle_positions") or {})}


class MeshToGridField(Step):
    """P1 DOF fields → grid fields for the particle engine's rate lookups: u_grid = P·u.

    Grid nodes inside the mesh take the P1 interpolant; nodes outside take the nearest DOF value.
    Config: ``grid`` (background grid), ``mesh`` (mesh spec), ``species`` (fields to sample).
    """

    config_schema = {"grid": "map", "mesh": "map", "species": "list[string]"}

    def initialize(self, config):
        from viva_pde_particle.mesh import mesh_transfer

        self.grid = CartesianGrid.from_config(config["grid"])
        self.species = list(config["species"])
        self.transfer = mesh_transfer(config["mesh"], self.grid)

    def inputs(self):
        return {"field_dofs": "map[array[float]]"}

    def outputs(self):
        return {"fields": "map[overwrite[array[float]]]"}

    def convert(self, dofs: dict) -> dict[str, np.ndarray]:
        return {s: (self.transfer.P @ np.asarray(dofs[s], dtype=float)).reshape(self.grid.shape)
                for s in self.species if s in dofs}

    def update(self, state):
        return {"fields": self.convert(state.get("field_dofs") or {})}
