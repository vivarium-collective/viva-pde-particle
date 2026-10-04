"""SpatialRecorder: a Process that writes the run into a spatial results bundle (Phase 8).

It ticks on the composite's shortest process interval (``base_dt``), so its clock is the PDE's
float for float, and reads state at the start of each tick, after every update ending at that time
has been applied. Every ``every``-th tick it writes a bundle row for ``t = n·base_dt``. It has no
outputs, so the simulation is unchanged.

The tick starting at the run's end time is not executed by the composite, so callers record the
final state with :meth:`close` (``run_document`` does). Configure it with
:func:`viva_pde_particle.viz3d.record.attach_recorder`.
"""
from __future__ import annotations

from process_bigraph import Process


class SpatialRecorder(Process):
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

        self.port = "field_dofs" if config["layout"]["kind"] == "mesh" else "fields"
        self.recorder = BundleRecorder(config["out_dir"], config["layout"], config["species"],
                                       config["particle_species"], config["membranes"], config["region"],
                                       config["source"])
        self.tick = 0
        self.closed = False

    def inputs(self):
        ports = {self.port: "map[array[float]]"}
        if self.config["particle_species"]:
            ports["particle_positions"] = "map[array[float]]"
        return ports

    def outputs(self):
        return {}

    def _record(self, t, state):
        self.recorder.record(t, state[self.port], state.get("particle_positions"))

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
