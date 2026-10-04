"""Stepper: run any child process on a coarser, phase-shifted clock (Phase 7c).

A Stepper ticks every ``dt``. On the ticks with ``tick % every == phase`` (0-based) it calls
its child with interval ``every·dt``, and on other ticks it does nothing. The child's update
is applied at the end of that tick, so the child

- reads its inputs at ``T + phase·dt``, the start of the active tick, and
- publishes its outputs at ``T + (phase+1)·dt``,

while advancing its own state by ``every·dt``. This is vcell-fvsolver's hybrid timing with
``every = k`` and ``phase = k−1`` (docs/fvsolver-hybrid-notes.md): the Smoldyn step over
[T, T+k·dt] reads the field at T+(k−1)·dt, after the k-th PDE iterate, and its counts are
first used by the PDE step starting at T+k·dt. Plain process-bigraph scheduling cannot express
this, because a process reads at the start of its interval and publishes at the end.

The Stepper knows nothing about its child beyond the process interface, so the child keeps no
clock of another process. Ports and initial state are the child's.

Config:
    process: ``{"address": "local:<Process>", "config": {...}}``, the child.
    dt: tick length (s). Also the interval the composite must give the Stepper.
    every: run the child once per ``every`` ticks.
    phase: active tick within each group of ``every`` (0 … every−1).
"""
from __future__ import annotations

from process_bigraph import Process


class Stepper(Process):
    config_schema = {
        "process": "map",
        "dt": "float",
        "every": {"_type": "integer", "_default": 1},
        "phase": {"_type": "integer", "_default": 0},
    }

    def initialize(self, config):
        address = config["process"]["address"]
        name = address.split(":", 1)[1] if ":" in address else address
        cls = self.core.link_registry.get(name)
        if cls is None:
            raise ValueError(f"Stepper: no process registered at {address!r}")
        self.child = cls(config=config["process"].get("config", {}), core=self.core)
        self.dt = float(config["dt"])
        self.every = int(config["every"])
        self.phase = int(config["phase"])
        if self.every < 1 or not 0 <= self.phase < self.every:
            raise ValueError(f"Stepper: need every >= 1 and 0 <= phase < every, got {self.every}, {self.phase}")
        self._tick = 0

    def inputs(self):
        return self.child.inputs()

    def outputs(self):
        return self.child.outputs()

    def initial_state(self):
        return self.child.initial_state() if hasattr(self.child, "initial_state") else {}

    def update(self, state, interval):
        if abs(interval - self.dt) > 1e-9 * self.dt:
            raise ValueError(f"Stepper: interval {interval} must equal its tick dt {self.dt}")
        tick, self._tick = self._tick, self._tick + 1
        if tick % self.every != self.phase:
            return {}
        return self.child.update(state, self.every * self.dt)
