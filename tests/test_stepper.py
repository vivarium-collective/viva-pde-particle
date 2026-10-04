"""Phase 7c: the Stepper runs any child process on a coarser, phase-shifted clock."""
from __future__ import annotations

import pytest
from process_bigraph import Composite, Process, allocate_core

from viva_pde_particle.processes import Stepper

LOG: list[tuple[str, float, float, float]] = []


class Clock(Process):
    """Records (name, start time, value read, interval) and publishes its own end time."""

    config_schema = {"name": "string"}

    def initialize(self, config):
        self.t = 0.0

    def inputs(self):
        return {"other": "float"}

    def outputs(self):
        return {"mine": "overwrite[float]"}

    def update(self, state, interval):
        LOG.append((self.config["name"], self.t, state["other"], interval))
        self.t += interval
        return {"mine": self.t}


def _run(k, dt=1.0, t_end=6.0):
    LOG.clear()
    core = allocate_core()
    core.register_link("Clock", Clock)
    core.register_link("Stepper", Stepper)
    doc = {
        "pde_time": 0.0, "part_time": 0.0,
        "pde": {"_type": "process", "address": "local:Clock", "config": {"name": "pde"}, "interval": dt,
                "inputs": {"other": ["part_time"]}, "outputs": {"mine": ["pde_time"]}},
        "part": {"_type": "process", "address": "local:Stepper", "interval": dt,
                 "config": {"process": {"address": "local:Clock", "config": {"name": "part"}},
                            "dt": dt, "every": k, "phase": k - 1},
                 "inputs": {"other": ["pde_time"]}, "outputs": {"mine": ["part_time"]}},
    }
    Composite({"state": doc}, core=core).run(t_end + 0.5 * dt)
    return [entry for entry in LOG if entry[0] == "part"], [entry for entry in LOG if entry[0] == "pde"]


@pytest.mark.parametrize("k", [1, 2, 3])
def test_child_reads_at_phase_and_publishes_one_tick_later(k):
    part, pde = _run(k)
    # the child runs once per k ticks with interval k·dt
    assert [e[3] for e in part] == [float(k)] * len(part) and len(part) == 6 // k
    # active tick n·k + (k−1): it reads the PDE time stamp of that tick's start, i.e. T + (k−1)·dt
    assert [e[2] for e in part] == [float(n * k + k - 1) for n in range(len(part))]
    # its output is first seen by the PDE step that starts at T + k·dt (fvsolver's timing);
    # the child's own clock reads T + k after each run because it advances by k·dt
    pde_reads = {e[1]: e[2] for e in pde}
    for n in range(len(part) - 1):
        assert pde_reads[float((n + 1) * k)] == float((n + 1) * k)


def test_rejects_bad_phase():
    core = allocate_core()
    core.register_link("Clock", Clock)
    with pytest.raises(ValueError, match="phase"):
        Stepper(config={"process": {"address": "local:Clock", "config": {"name": "x"}}, "dt": 1.0,
                        "every": 2, "phase": 2}, core=core)
