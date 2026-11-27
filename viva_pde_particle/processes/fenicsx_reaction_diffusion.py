"""FEniCSx structured-grid reaction-diffusion — now reused from viva-fenics.

The dolfinx Q1 engine (operator assembly + scipy stepping) moved to **viva-fenics**
as :class:`viva_fenics.FenicsxReactionDiffusionProcess` — it is genuine FEniCSx-engine
functionality and belongs in the wrapper, not vendored here. ``FenicsxReactionDiffusion``
is now a thin subclass, kept *defined in this package* so ``build_core``'s own-package
scan still registers ``local:FenicsxReactionDiffusion`` — so every composite, study and
test that addresses it is untouched (identical config, ports and behavior).

``q1_operators`` is re-exported for callers/tests; this workspace's :class:`CartesianGrid`
is duck-compatible with viva-fenics's ``_StructuredGrid`` (same ``origin``/``size``/``num``/
``dim``/``shape``/``num_elements``/``node_index``), and the ``{origin, size, num}`` config
is the identical grid shape the wrapper accepts.
"""
from __future__ import annotations

from viva_fenics import FenicsxReactionDiffusionProcess
from viva_fenics.processes.reaction_diffusion import q1_operators  # re-exported for callers/tests

__all__ = ["FenicsxReactionDiffusion", "q1_operators"]


class FenicsxReactionDiffusion(FenicsxReactionDiffusionProcess):
    """pde-particle's binding of viva-fenics's structured-grid RD process.

    A thin subclass so the class is defined in this package (keeps the
    ``local:FenicsxReactionDiffusion`` link via build_core's own-package scan), while the
    engine implementation is reused from viva-fenics instead of vendoring dolfinx here.
    """
