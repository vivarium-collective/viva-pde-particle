"""General-mesh FEniCSx reaction-diffusion — now reused from viva-fenics.

The unstructured-mesh P1 RD engine moved to viva-fenics as
:class:`viva_fenics.FenicsxMeshReactionDiffusionProcess`. ``FenicsxMeshReactionDiffusion``
is now a thin subclass that overrides :meth:`build_space` to use this workspace's
``mesh_space`` (the gmsh ball + the VCell-style geometry realization), so the mesh /
geometry / grid↔mesh-transfer framework stays here — it legitimately uses dolfinx as a FEM
toolkit to implement coupling logic that is pde-particle-specific — while the RD solver is
reused. Kept *defined in this package* so ``build_core``'s own-package scan still registers
``local:FenicsxMeshReactionDiffusion`` — every composite/study/test is untouched.
"""
from __future__ import annotations

from viva_fenics import FenicsxMeshReactionDiffusionProcess

from viva_pde_particle.mesh import build_mesh  # noqa: F401  (re-exported for existing callers)

__all__ = ["FenicsxMeshReactionDiffusion", "build_mesh"]


class FenicsxMeshReactionDiffusion(FenicsxMeshReactionDiffusionProcess):
    """pde-particle's binding of viva-fenics's unstructured-mesh RD engine.

    Overrides ``build_space`` to supply this workspace's mesh/geometry (``mesh_space``:
    gmsh sphere + VCell geometry realization, with grid↔mesh transfer); the RD assembly +
    stepping are reused from viva-fenics unchanged.
    """

    def build_space(self, mesh_config):
        from viva_pde_particle.mesh import mesh_space
        return mesh_space(mesh_config)
