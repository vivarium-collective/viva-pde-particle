"""Operator-splitting schemes for the PDE/particle coupling (Investigation B, splitting-schemes).

``HybridCoupler`` composes an FVReactionDiffusion and a SmoldynHybrid instance and advances
them over one coupling interval τ = k·dt per update, with the substeps ordered by
``scheme``. Here uⁿ is the field and pⁿ the particle state at the start of the interval:

- ``jacobi``: vcell-fvsolver's lagged scheme. The PDE takes k substeps with the old
  particle concentrations, and the particles take one τ step reading the field after k−1
  substeps (the same as the two-process composite with coupling="fvsolver").
- ``gs_particles_first``: the particles step reading uⁿ, then the PDE takes k substeps
  with the new particle concentrations.
- ``gs_pde_first``: the PDE takes k substeps with pⁿ, then the particles step reading uⁿ⁺¹.
- ``strang``: the PDE takes k/2 substeps with pⁿ, the particles step reading the
  mid-interval field, then the PDE takes k/2 substeps with the new particles (k even).

The engines are the same classes used in the two-process composites; only the ordering
differs. A different splitting is therefore a different coupler, not a different solver,
which the hard-coded embedded solver cannot offer.
"""
from __future__ import annotations

import numpy as np
from process_bigraph import Process

from viva_pde_particle.processes.fv_reaction_diffusion import FVReactionDiffusion
from viva_pde_particle.processes.smoldyn_hybrid import SmoldynHybrid

SCHEMES = ("jacobi", "gs_particles_first", "gs_pde_first", "strang")


class HybridCoupler(Process):
    """One process advancing PDE + particles with a chosen splitting.

    Config:
        pde: FVReactionDiffusion config (``grid``, ``pde``, ``dt``).
        particles: SmoldynHybrid config (``grid``, ``config_text``, ``particle_species``,
            ``field_species``, ``dt``, ``step_multiplier``); its coupling mode is ignored.
        scheme: one of ``jacobi``, ``gs_particles_first``, ``gs_pde_first``, ``strang``.
    """

    config_schema = {
        "pde": "map",
        "particles": "map",
        "scheme": {"_type": "string", "_default": "jacobi"},
    }

    def initialize(self, config):
        if config["scheme"] not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}, got {config['scheme']!r}")
        self.scheme = config["scheme"]
        self.pde = FVReactionDiffusion(config=config["pde"], core=self.core)
        particle_cfg = dict(config["particles"], coupling="start-of-interval")
        self.particles = SmoldynHybrid(config=particle_cfg, core=self.core)
        self.k = self.particles.k
        if self.scheme == "strang" and self.k % 2:
            raise ValueError("strang splitting needs an even step multiplier")
        self.grid = self.pde.grid

    def inputs(self):
        return {"fields": "map[array[float]]", "particle_counts": "map[array[float]]"}

    def outputs(self):
        return {
            "fields": "map[overwrite[array[float]]]",
            "particle_counts": "map[overwrite[array[float]]]",
            "particle_totals": "map[overwrite[float]]",
        }

    def initial_state(self):
        return self.particles.initial_state()

    def _pde(self, fields, counts, n):
        conc = {p: self.grid.counts_to_uM(counts[p]) for p in self.pde.particle_species}
        for _ in range(n):
            fields = self.pde.step(fields, conc)
        return fields

    def _particles(self, fields):
        self.particles._run(fields, 1)
        return self.particles.counts()

    def update(self, state, interval):
        fields = {s: np.asarray(state["fields"][s], dtype=float) for s in self.pde.species}
        counts = {p: np.asarray(state["particle_counts"][p], dtype=float) for p in self.pde.particle_species}
        n_intervals = max(1, int(round(interval / (self.k * self.pde.dt))))
        for _ in range(n_intervals):
            if self.scheme == "jacobi":
                fields_km1 = self._pde(fields, counts, self.k - 1)
                new_counts = self._particles(fields_km1)
                fields = self._pde(fields_km1, counts, 1)
                counts = new_counts
            elif self.scheme == "gs_particles_first":
                counts = self._particles(fields)
                fields = self._pde(fields, counts, self.k)
            elif self.scheme == "gs_pde_first":
                fields = self._pde(fields, counts, self.k)
                counts = self._particles(fields)
            else:  # strang
                fields = self._pde(fields, counts, self.k // 2)
                counts = self._particles(fields)
                fields = self._pde(fields, counts, self.k // 2)
        all_counts = self.particles.counts()
        return {
            "fields": fields,
            "particle_counts": all_counts,
            "particle_totals": {s: float(c.sum()) for s, c in all_counts.items()},
        }
