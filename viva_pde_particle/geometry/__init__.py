"""Geometry compilers (Phase 7e).

One geometry description — vcell-fenics' VCell-style ``GeometryDescription`` (analytic / CSG /
image / compartmental subvolumes, surface classes) — is realized by a PDE geometry compiler into
a :class:`~viva_pde_particle.geometry.realize.RealizedGeometry`; everything on the particle side
(Smoldyn membrane, volume samples, adapters' accessible volumes) is derived from that realization,
never from the description independently (docs/PLAN.md, "Phase 7e plan").
"""
from viva_pde_particle.geometry.description import (
    description_from_config,
    description_to_config,
    sphere_in_box,
)
from viva_pde_particle.geometry.realize import FenicsRealization, RealizedGeometry, realize_fenics

__all__ = ["FenicsRealization", "RealizedGeometry", "description_from_config", "description_to_config",
           "realize_fenics", "sphere_in_box"]
