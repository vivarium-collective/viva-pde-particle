"""Coupling glue for the hybrid model's particle side.

The field-coupled Smoldyn process itself now lives in **viva-smoldyn** as
:class:`viva_smoldyn.SmoldynHybridProcess` — the Smoldyn ``HybridGrid`` engine
logic (copy fields in, field-dependent rates, per-node grid-binned counts out) is
pure Smoldyn and belongs in the wrapper, not vendored here. This module keeps only
pde-particle's COUPLING glue: the scheduling helper the composite uses, and a thin
:class:`SmoldynHybrid` alias so the ``local:SmoldynHybrid`` address and every
composite/study/test that references it keep working unchanged.

The engine has no notion of the PDE clock; each update reads the fields at its
start and advances Smoldyn by the update's interval. **Scheduling is the
composite's job:**

- ``coupling="start-of-interval"``: the composite gives the engine interval k·dt,
  so a step over [T, T+k·dt] reads the field at T.
- ``coupling="fvsolver"``: vcell-fvsolver's exact timing (the step reads the field
  at T+(k−1)·dt, see docs/fvsolver-hybrid-notes.md). The composite wraps the
  engine in a :class:`~viva_pde_particle.processes.stepper.Stepper` that ticks
  every dt and runs it on every k-th tick with interval k·dt.

:func:`interval_for` gives the particle node's interval for each mode.
"""
from __future__ import annotations

from viva_smoldyn import SmoldynHybridProcess

COUPLING_MODES = ("fvsolver", "start-of-interval")


def interval_for(coupling: str, dt: float, step_multiplier: int) -> float:
    if coupling == "fvsolver":
        return dt
    if coupling == "start-of-interval":
        return step_multiplier * dt
    raise ValueError(f"coupling must be one of {COUPLING_MODES}, got {coupling!r}")


class SmoldynHybrid(SmoldynHybridProcess):
    """pde-particle's binding of viva-smoldyn's field-coupled Smoldyn process.

    A thin subclass so the class is *defined in this package*: that keeps the
    ``local:SmoldynHybrid`` link (registered by ``build_core``'s own-package scan)
    and every composite / study / test that addresses it working unchanged, while
    the engine implementation is reused from viva-smoldyn rather than vendored.
    The config, ports and behavior are exactly viva-smoldyn's ``SmoldynHybridProcess``.
    """
