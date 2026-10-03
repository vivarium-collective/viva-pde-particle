"""Phase 0 spike: pin down process-bigraph's scheduling semantics for coupled processes.

The hybrid coupling we replicate (vcell-fvsolver ``SimTool.cpp:871-935``) is explicit
and lagged. Each process reads the other's state from *before* the current step, then
both write. This module checks that process-bigraph gives the same "read old, then
write" behaviour, using toy processes that log every value they read. It also records
how two processes with different intervals (PDE ``dt``, particles ``k·dt``) are
interleaved.

Toy model: ``Field`` and ``Particles`` each write their own *end-of-interval time* into
their store (overwrite) and record the other store's value at the time of reading.
With "read old" semantics the value a process sees is a time stamp, which tells us
exactly which update it saw.
"""
from __future__ import annotations

import pytest
from process_bigraph import Composite, Process, allocate_core

LOG: dict[str, list[tuple[float, float]]] = {}


class _Recorder(Process):
    """Writes its own end-of-interval time; records (start_time, value_read)."""

    config_schema = {
        "interval": {"_type": "float", "_default": 1.0},
        "name": {"_type": "string", "_default": "x"},
    }

    def initialize(self, config):
        self.t = 0.0

    def inputs(self):
        return {"other": "float"}

    def outputs(self):
        return {"mine": "overwrite[float]"}

    def interval(self):
        return self.config["interval"]

    def update(self, state, interval):
        LOG.setdefault(self.config["name"], []).append((self.t, state["other"]))
        self.t += interval
        return {"mine": self.t}


class FieldRecorder(_Recorder):
    pass


class ParticleRecorder(_Recorder):
    pass


def _run(dt: float, k: int, t_end: float):
    LOG.clear()
    core = allocate_core()
    core.register_link("FieldRecorder", FieldRecorder)
    core.register_link("ParticleRecorder", ParticleRecorder)
    doc = {
        "field_time": 0.0,
        "particle_time": 0.0,
        "field": {
            "_type": "process",
            "address": "local:FieldRecorder",
            "config": {"name": "field", "interval": dt},
            "interval": dt,
            "inputs": {"other": ["particle_time"]},
            "outputs": {"mine": ["field_time"]},
        },
        "particles": {
            "_type": "process",
            "address": "local:ParticleRecorder",
            "config": {"name": "particles", "interval": k * dt},
            "interval": k * dt,
            "inputs": {"other": ["field_time"]},
            "outputs": {"mine": ["particle_time"]},
        },
    }
    sim = Composite({"state": doc}, core=core)
    sim.run(t_end)
    return {name: [(round(t, 9), round(v, 9)) for t, v in rows] for name, rows in LOG.items()}


def test_equal_intervals_are_jacobi():
    """k=1: at every step, each process sees the other's value from the *start* of the step."""
    log = _run(dt=1.0, k=1, t_end=4.0)
    for name in ("field", "particles"):
        assert log[name] == [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)], (name, log[name])


@pytest.mark.parametrize("k", [2, 4])
def test_particle_interval_multiple(k):
    """Interval k·dt for particles: record exactly what each side sees.

    Expected (start-of-interval reads):
      * field step [t, t+dt] sees particle_time = the last particle update <= t
        (piecewise-constant, held for k field steps) -- same as fvsolver;
      * particle step [T, T+k·dt] sees field_time = T (start of its own interval).
        fvsolver instead shows Smoldyn the field at T+(k-1)·dt; see
        docs/fvsolver-hybrid-notes.md "Time stepping".
    """
    dt, t_end = 1.0, 4.0 * k
    log = _run(dt=dt, k=k, t_end=t_end)

    field_expected = [(float(n), float((n // k) * k)) for n in range(int(t_end))]
    assert log["field"] == field_expected, log["field"]

    particle_expected = [(float(m * k), float(m * k)) for m in range(int(t_end) // k)]
    assert log["particles"] == particle_expected, log["particles"]
