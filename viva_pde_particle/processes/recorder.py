"""Recorders: Processes that write the run into a spatial results bundle (Phase 8).

:class:`GridRecorder` records fields on a Cartesian grid (port ``fields``); :class:`MeshRecorder`
records P1 DOF fields on a mesh (port ``field_dofs``). Both read ``particle_positions`` when it is
wired. The interface of each is fixed by its class, not by its config, so a composite can be checked
against the declared ports without instantiating anything.

A recorder ticks on the composite's shortest process interval (``base_dt``), so its clock is the PDE's
float for float, and reads state at the start of each tick, after every update ending at that time
has been applied. Every ``every``-th tick it writes a bundle row for ``t = n·base_dt``. It has no
outputs, so the simulation is unchanged.

The tick starting at the run's end time is not executed by the composite, so callers record the
final state with :meth:`GridRecorder.close` (``run_document`` does). Configure one with
:func:`viva_pde_particle.viz3d.record.attach_recorder`.
"""
from __future__ import annotations

from process_bigraph import Process


class _RecorderBase(Process):
    """Base of the recorders; ``FIELD_PORT`` and ``LAYOUT`` are fixed by the subclass."""

    FIELD_PORT: str | None = None
    LAYOUT: str | None = None

    config_schema = {
        "out_dir": "string",
        "base_dt": "float",
        "every": "integer",
        "layout": "map",
        "species": "list[string]",
        "particle_species": {"_type": "list[string]", "_default": []},
        "membranes": {"_type": "map", "_default": {}},
        "region": {"_type": "string", "_default": "cell"},
        "source": {"_type": "string", "_default": ""},
    }

    def initialize(self, config):
        from viva_pde_particle.viz3d.record import BundleRecorder

        if self.FIELD_PORT is None:
            raise TypeError("use GridRecorder or MeshRecorder")
        kind = config["layout"].get("kind")
        if kind != self.LAYOUT:
            raise ValueError(f"{type(self).__name__} records a {self.LAYOUT!r} layout, got {kind!r}")
        self.recorder = BundleRecorder(config["out_dir"], config["layout"], config["species"],
                                       config["particle_species"], config["membranes"], config["region"],
                                       config["source"])
        self.tick = 0
        self.closed = False

    def inputs(self):
        if self.FIELD_PORT is None:
            return {}
        return {self.FIELD_PORT: "map[array[float]]", "particle_positions": "map[array[float]]"}

    def outputs(self):
        return {}

    def _record(self, t, state):
        self.recorder.record(t, state[self.FIELD_PORT], state.get("particle_positions"))

    def update(self, state, interval):
        if self.tick % self.config["every"] == 0 and not self.closed:
            self._record(self.tick * self.config["base_dt"], state)
        self.tick += 1
        return {}

    def close(self, state: dict | None = None, t: float | None = None, status: str = "completed") -> None:
        """Record ``state`` at ``t`` unless that time is already recorded, then finalize the bundle."""
        if self.closed:
            return
        last = self.recorder.last_time
        if state is not None and t is not None and (last is None or t > last + 1e-9 * max(1.0, abs(t))):
            self._record(t, state)
        self.recorder.close(status)
        self.closed = True


class GridRecorder(_RecorderBase):
    """Records ``fields`` on a Cartesian grid (``layout.kind == "grid"``)."""

    FIELD_PORT = "fields"
    LAYOUT = "grid"


class MeshRecorder(_RecorderBase):
    """Records ``field_dofs`` on a P1 mesh (``layout.kind == "mesh"``)."""

    FIELD_PORT = "field_dofs"
    LAYOUT = "mesh"
