"""3D storage and visualization of spatial runs (Phase 8).

- :mod:`.bundle`: vcell-fenics results bundles (VTU + zarr) from numpy, plus a particle extension.
- :mod:`.record`: record a hybrid document or a native VCell trajectory into a bundle.
- :mod:`.static`: PyVista off-screen PNG/GIF figures (pixi env).
- :mod:`.html`: a self-contained three.js page for the workbench.
"""
from viva_pde_particle.viz3d.bundle import (
    SpatialBundleWriter,
    grid_domain,
    membrane_domain,
    membrane_variable,
    mesh_domain,
    particle_species,
    read_particles,
)
from viva_pde_particle.viz3d.record import BundleRecorder, attach_recorder, write_native_bundle

__all__ = ["BundleRecorder", "SpatialBundleWriter", "attach_recorder", "grid_domain", "membrane_domain", "membrane_variable",
           "mesh_domain", "particle_species", "read_particles", "write_native_bundle"]
