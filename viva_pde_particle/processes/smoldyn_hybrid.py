"""Smoldyn particle process with field-dependent rates (virtualcell/Smoldyn ``pyhybrid``).

Each update copies the continuous fields into a ``smoldyn.HybridGrid``. Reaction
rates written ``k*B;`` then read them at each molecule, or at each grid node for
0th-order creation. Smoldyn advances, and the process returns per-node particle
counts binned on the same grid.

Coupling modes (``coupling`` config):

- ``"fvsolver"`` (default) reproduces vcell-fvsolver exactly. The process runs on
  the PDE clock (interval ``dt``) and takes one Smoldyn step of ``k·dt`` on every
  k-th call. That step reads the field at ``T+(k-1)·dt``, as fvsolver's Smoldyn
  step does (it runs after the k-th PDE iterate; see docs/fvsolver-hybrid-notes.md).
- ``"start-of-interval"`` is the plain process-bigraph form: interval ``k·dt``,
  reading the field at the start of the Smoldyn step ``T``.

The composite must set the process interval to match: ``dt`` for ``fvsolver``,
``k·dt`` for ``start-of-interval``. See :func:`interval_for`.
"""
from __future__ import annotations

import contextlib
import os
import tempfile

import numpy as np
from process_bigraph import Process

from viva_pde_particle.grid import CartesianGrid

COUPLING_MODES = ("fvsolver", "start-of-interval")


def interval_for(coupling: str, dt: float, step_multiplier: int) -> float:
    if coupling == "fvsolver":
        return dt
    if coupling == "start-of-interval":
        return step_multiplier * dt
    raise ValueError(f"coupling must be one of {COUPLING_MODES}, got {coupling!r}")


@contextlib.contextmanager
def _quiet_stdout():
    """Silence Smoldyn's C-level stdout chatter (it bypasses sys.stdout)."""
    try:
        fd = os.dup(1)
    except OSError:
        yield
        return
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        os.dup2(fd, 1)
        os.close(fd)
        os.close(devnull)


class SmoldynHybrid(Process):
    """Particle half of a hybrid model.

    Config:
        grid: ``{origin, size, num}`` shared with the PDE process.
        config_text: Smoldyn configuration text (e.g. from
            :func:`viva_pde_particle.model.write_smoldyn_config`), with ``time_step`` = k·dt.
        particle_species: species whose per-node counts are reported.
        field_species: continuous species referenced by rate expressions.
        dt: PDE time step (s).
        step_multiplier: Smoldyn steps once per k PDE steps (vcell-fvsolver
            ``SMOLDYN_STEP_MULTIPLIER``).
        coupling: ``"fvsolver"`` or ``"start-of-interval"``.
    """

    config_schema = {
        "grid": "map",
        "config_text": "string",
        "particle_species": "list[string]",
        "field_species": {"_type": "list[string]", "_default": []},
        "dt": {"_type": "float", "_default": 0.01},
        "step_multiplier": {"_type": "integer", "_default": 1},
        "coupling": {"_type": "string", "_default": "fvsolver"},
        "quiet": {"_type": "boolean", "_default": True},
    }

    def initialize(self, config):
        import smoldyn._smoldyn as _smoldyn

        if config["coupling"] not in COUPLING_MODES:
            raise ValueError(f"coupling must be one of {COUPLING_MODES}, got {config['coupling']!r}")
        self.grid = CartesianGrid.from_config(config["grid"])
        self.k = int(config["step_multiplier"])
        self.dt = float(config["dt"])
        self.smoldyn_dt = self.k * self.dt
        self.coupling = config["coupling"]
        self.particle_species = list(config["particle_species"])
        self.field_species = list(config["field_species"])
        self._calls = 0
        self.time = 0.0

        g = self.grid
        self.hybrid_grid = _smoldyn.HybridGrid(list(g.origin), list(g.size), list(g.num))
        self._workdir = tempfile.TemporaryDirectory(prefix="smoldyn-hybrid-")
        path = os.path.join(self._workdir.name, "model.txt")
        with open(path, "w") as f:
            f.write(config["config_text"])
        quiet = _quiet_stdout() if config["quiet"] else contextlib.nullcontext()
        with quiet:
            self.sim = _smoldyn.Simulation(path, "q", self.hybrid_grid)
        missing = set(self.hybrid_grid.requiredFields()) - set(self.field_species)
        if missing:
            raise ValueError(f"rate expressions reference fields not in field_species: {sorted(missing)}")

    def inputs(self):
        return {"fields": "map[array[float]]"}

    def outputs(self):
        return {
            "particle_counts": "map[overwrite[array[float]]]",
            "particle_totals": "map[overwrite[float]]",
        }

    def counts(self) -> dict[str, np.ndarray]:
        return {
            s: self.sim.getMoleculeHistogram(s, self.hybrid_grid).astype(float)
            for s in self.particle_species
        }

    def initial_state(self):
        counts = self.counts()
        return {
            "particle_counts": counts,
            "particle_totals": {s: float(c.sum()) for s, c in counts.items()},
        }

    def _run(self, fields, n_steps: int):
        for name in self.field_species:
            self.hybrid_grid.setField(name, np.asarray(fields[name], dtype=float))
        # exactly n Smoldyn steps: a step is taken while time < breaktime
        breaktime = self.time + (n_steps - 0.5) * self.smoldyn_dt
        quiet = _quiet_stdout() if self.config["quiet"] else contextlib.nullcontext()
        with quiet:
            self.sim.runUntil(breaktime, self.smoldyn_dt, False)
        self.time += n_steps * self.smoldyn_dt

    def update(self, state, interval):
        if self.coupling == "fvsolver":
            self._calls += 1
            if self._calls % self.k != 0:
                return {}
            n_steps = 1
        else:
            n_steps = max(1, int(round(interval / self.smoldyn_dt)))
        self._run(state.get("fields", {}), n_steps)
        counts = self.counts()
        return {
            "particle_counts": counts,
            "particle_totals": {s: float(c.sum()) for s, c in counts.items()},
        }
