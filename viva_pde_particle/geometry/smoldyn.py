"""The Smoldyn geometry of a hybrid, derived from a RealizedGeometry (Phase 7e.3).

The particle domain is never realized on its own: the membrane, compartment points and the
``highResVolumeSamples`` compartment map all come from the same realization the PDE runs on, so
the two domains coincide (the lesson of Studies B2b–B2d). The derived pieces are:

- **membrane:** the region's smooth boundary as triangles (``panel tri``);
- **compartment points:** several interior points, so non-convex regions work with Smoldyn's
  line-of-sight compartment test;
- **volume samples:** the region rasterized with the realization's own ``inside``, the map
  VCell's ``posincompart`` acceleration uses (inside/outside from a 3×3×3 neighbourhood, exact
  panel test only near the surface);
- **initial positions:** uniform in the region, by the realization's own ``inside``.
"""
from __future__ import annotations

import numpy as np


def compartment_samples(inside, origin, size, num) -> np.ndarray:
    """Smoldyn ``highResVolumeSamples`` ids (0 inside, 1 outside), cell-centred, x fastest.

    ``inside(points) -> bool array``. Sample (i, j, k) covers ``[o + i·h, o + (i+1)·h)`` per axis
    with ``h = size/num`` and is classified at its centre, as ``posincompart`` reads it.
    """
    o, s, n = (np.asarray(v, dtype=float) for v in (origin, size, num))
    n = n.astype(int)
    h = s / n
    axes = [o[d] + (np.arange(n[d]) + 0.5) * h[d] for d in range(3)]
    out = np.ones(int(n.prod()), dtype=np.uint8)
    nxy = n[0] * n[1]
    gx, gy = np.meshgrid(axes[0], axes[1], indexing="xy")  # (ny, nx): x fastest when raveled
    xy = np.stack([gx.ravel(), gy.ravel()], axis=1)
    for k, z in enumerate(axes[2]):
        pts = np.column_stack([xy, np.full(len(xy), z)])
        out[k * nxy:(k + 1) * nxy] = np.where(inside(pts), 0, 1)
    return out


def smoldyn_geometry(realized, region: str, grid, volume_samples: int | None = 2) -> dict:
    """``write_smoldyn_config(geometry=...)`` dict for particles confined to ``region``.

    ``volume_samples``: samples per grid spacing in the compartment map (None: no map).
    """
    geometry = {"kind": "triangles", "triangles": realized.boundary_triangles(region),
                "interior_point": realized.interior_points(region)}
    if volume_samples:
        num = [int(round((n - 1) * volume_samples)) or 1 for n in grid.num]
        geometry["volume_samples"] = {
            "origin": list(grid.origin), "size": list(grid.size), "num": num,
            "ids": compartment_samples(lambda p: realized.inside(region, p), grid.origin, grid.size, num)}
    return geometry
