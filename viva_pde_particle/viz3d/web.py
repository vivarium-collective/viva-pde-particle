"""The bundle's *web* extension: a precomputed triangle surface per domain, so a browser draws a results bundle with a
standard renderer (vtk.js PolyData) straight from arrays. No VTU parsing, no tetrahedra (compose-api
docs/plan-viewers.md, B1; profile in docs/web-bundle.md).

Additive, like the particle extension: readers that do not know it ignore it.

- ``web/<domain>/points``: (N, 3) float32, the domain's mesh points in mesh order, so a field row (N,) colours the
  surface directly.
- ``web/<domain>/triangles``: (M, 3) uint32 into those points. A tetra domain's boundary is the faces that belong to
  exactly one tet. A triangle domain (a membrane, a 2D grid) is its own cells.
- the root ``.zattrs["web"] = {"schema": 1, "surfaces": {domain: {"points": path, "triangles": path}}}``.

Remeshed bundles get the first segment's surface.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

WEB_KEY = "web"
WEB_SCHEMA = 1
VTK_TRIANGLE, VTK_TETRA = 5, 10


def boundary_triangles(tets: np.ndarray) -> np.ndarray:
    """The faces of ``tets`` (M, 4) that belong to exactly one tet, (K, 3)."""
    tets = np.asarray(tets, dtype=np.int64)
    faces = np.concatenate([tets[:, [0, 2, 1]], tets[:, [0, 1, 3]], tets[:, [1, 2, 3]], tets[:, [0, 3, 2]]])
    key = np.sort(faces, axis=1)
    _, first, counts = np.unique(key, axis=0, return_index=True, return_counts=True)
    return faces[np.sort(first[counts == 1])]


def surface_of(points: np.ndarray, cells: np.ndarray, vtk_type: int) -> tuple[np.ndarray, np.ndarray] | None:
    """(points (N, 3) float32, triangles (K, 3) uint32) for a domain, or None if it has no surface (lines)."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape[1] < 3:
        pts = np.pad(pts, ((0, 0), (0, 3 - pts.shape[1])))
    if vtk_type == VTK_TETRA:
        tris = boundary_triangles(cells)
    elif vtk_type == VTK_TRIANGLE:
        tris = np.asarray(cells)
    else:
        return None
    return pts, tris.astype(np.uint32)


def write_surfaces(path, surfaces: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """Write ``surfaces`` (domain -> (points, triangles)) into the bundle at ``path``; the ``.zattrs['web']`` entry."""
    import numcodecs
    import zarr

    group = zarr.open_group(str(path), mode="r+", zarr_format=2)
    entry: dict = {"schema": WEB_SCHEMA, "surfaces": {}}
    for domain, (pts, tris) in surfaces.items():
        names = {"points": f"web/{domain}/points", "triangles": f"web/{domain}/triangles"}
        for kind, data in (("points", pts), ("triangles", tris)):
            if names[kind] in group:
                del group[names[kind]]
            array = group.create_array(names[kind], shape=data.shape, chunks=data.shape, dtype=data.dtype,
                                       compressors=numcodecs.Zlib(level=1), order="C")
            array[...] = data
            array.attrs["_ARRAY_DIMENSIONS"] = ["point", "xyz"] if kind == "points" else ["triangle", "vertex"]
        entry["surfaces"][domain] = names
    return entry


def write_attrs_atomically(root, attrs: dict) -> None:
    """Replace ``root/.zattrs`` atomically (a reader may be polling it), readable like every other file in the bundle.

    ``mkstemp`` creates its file 0600 and ``os.replace`` keeps that mode, which left ``.zattrs`` unreadable to anyone
    but the writer: compose-api, serving the bundle as another user, could not read it (compose-api simulation 4570).
    """
    root = Path(root)
    handle, temp = tempfile.mkstemp(dir=root, prefix=".zattrs.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as stream:
            stream.write(json.dumps(attrs, indent=1, sort_keys=True))
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(temp, 0o666 & ~umask)
        os.replace(temp, root / ".zattrs")
    except BaseException:
        Path(temp).unlink(missing_ok=True)
        raise


def publish_web_attrs(path, entry: dict) -> None:
    """Merge ``entry`` into the root ``.zattrs``."""
    root = Path(path)
    attrs = json.loads((root / ".zattrs").read_text())
    attrs[WEB_KEY] = entry
    write_attrs_atomically(root, attrs)


def add_web_extension(path) -> dict:
    """Add the web extension to any results bundle on disk (e.g. one vcell-fenics wrote), reading its meshes back."""
    from vcell_fenics.results.reader import Bundle

    bundle = Bundle.open(path)
    surfaces = {}
    for name, info in bundle.manifest.domains.items():
        mesh = bundle.mesh(name)
        surface = surface_of(mesh.points, mesh.cells, int(info.cell_type))
        if surface is not None:
            surfaces[name] = surface
    entry = write_surfaces(path, surfaces)
    publish_web_attrs(path, entry)
    return entry
