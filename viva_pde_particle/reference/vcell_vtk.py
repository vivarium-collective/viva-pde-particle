"""Native VCell finite volume results on pyvcell's smoothed unstructured (VTK) grid.

pyvcell (``pyvcell._internal.simdata.vtk``), like VCell's Java exporter, turns a finite volume
mesh into an unstructured grid of the volume elements in one domain and smooths its boundary
surface (windowed-sinc). Each cell keeps the global index of its mesh node, so node data maps
onto the grid directly.

The exporter draws each element as the box ``[i·L/N, (i+1)·L/N]``, which tiles the domain with
N boxes per axis. vcell-fvsolver's mesh is node-centred, though: node i sits at ``i·L/(N−1)``
and its element is ``[(i−½)·L/(N−1), (i+½)·L/(N−1)]``. For display the difference is slight.
For quantitative work near a membrane it is not: on a 37³ mesh it shifts elements by up to
~0.1 µm and shrinks the domain volume by ``((N−1)/N)³`` (−8%). ``node_centred=True`` (the
default) maps the box corners affinely onto the node-centred elements before smoothing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# six tetrahedra around the p0–p7 diagonal, for VTK voxel corner order (x fastest)
_HEX_TETS = [(0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7), (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7)]


def hex_volumes(corners: np.ndarray) -> np.ndarray:
    """Volumes of (possibly distorted) hexahedra given as (n, 8, 3) corners in voxel order."""
    p = corners
    return sum(np.einsum("ij,ij->i", np.cross(p[:, b] - p[:, a], p[:, c] - p[:, a]), p[:, d] - p[:, a])
               for a, b, c, d in _HEX_TETS) / 6.0


@dataclass
class SmoothedDomain:
    """One volume domain on the smoothed grid; arrays are per cell."""

    global_index: np.ndarray   # mesh node (x-fastest) whose value the cell carries
    corners: np.ndarray        # (n, 8, 3) smoothed corners
    raw_corners: np.ndarray    # (n, 8, 3) before smoothing
    grid: object               # the vtkUnstructuredGrid

    @property
    def centroids(self) -> np.ndarray:
        return self.corners.mean(axis=1)

    @property
    def volumes(self) -> np.ndarray:
        return hex_volumes(self.corners)

    @property
    def surface_points(self) -> np.ndarray:
        """Corners moved by smoothing (the domain's boundary), smoothed positions, unique."""
        moved = np.abs(self.corners - self.raw_corners).max(axis=2) > 1e-9
        return np.unique(self.corners[moved], axis=0)

    @property
    def raw_surface_points(self) -> np.ndarray:
        moved = np.abs(self.corners - self.raw_corners).max(axis=2) > 1e-9
        return np.unique(self.raw_corners[moved], axis=0)

    def cell_data(self, node_values: np.ndarray) -> np.ndarray:
        """Per-cell values from a node array (any shape whose C-order ravel is x-fastest)."""
        return np.asarray(node_values).reshape(-1)[self.global_index]

    def write_vtu(self, path: Path, arrays: dict[str, np.ndarray]) -> None:
        """Write the grid with per-cell arrays (e.g. from :meth:`cell_data`)."""
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk

        for name, values in arrays.items():
            arr = numpy_to_vtk(np.ascontiguousarray(values, dtype=float), deep=True)
            arr.SetName(name)
            self.grid.GetCellData().AddArray(arr)
        w = vtk.vtkXMLUnstructuredGridWriter()
        w.SetFileName(str(path))
        w.SetInputData(self.grid)
        w.SetDataModeToBinary()
        w.SetCompressorTypeToZLib()
        w.Write()


def smoothed_domain(mesh_file: Path, domain: str, node_centred: bool = True) -> SmoothedDomain:
    """Build pyvcell's smoothed grid for ``domain`` from a VCell ``.mesh`` file."""
    from pyvcell._internal.simdata.mesh import CartesianMesh
    from pyvcell._internal.simdata.vtk.fv_mesh_mapping import from_mesh3d_volume
    from pyvcell._internal.simdata.vtk.vtkmesh_utils import get_volume_vtk_grid, smooth_unstructured_grid_surface
    from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

    mesh = CartesianMesh(mesh_file=Path(mesh_file))
    mesh.read()
    if mesh.dimension != 3:
        raise ValueError("smoothed_domain supports 3D meshes only")
    vis = from_mesh3d_volume(mesh, domain)
    gidx = np.array([v.finiteVolumeIndex.globalIndex for v in vis.visVoxels], dtype=np.int64)
    grid = get_volume_vtk_grid(vis)
    if node_centred:
        n, ext, org = np.array(mesh.size), np.array(mesh.extent), np.array(mesh.origin)
        p = vtk_to_numpy(grid.GetPoints().GetData()).astype(float)
        p = org + ((p - org) / (ext / n) - 0.5) * (ext / (n - 1))
        grid.GetPoints().SetData(numpy_to_vtk(p, deep=True))
    raw = vtk_to_numpy(grid.GetPoints().GetData()).astype(float).copy()
    grid = smooth_unstructured_grid_surface(grid)
    pts = vtk_to_numpy(grid.GetPoints().GetData()).astype(float)
    ncell = grid.GetNumberOfCells()
    conn = np.array([[grid.GetCell(i).GetPointId(j) for j in range(8)] for i in range(ncell)], dtype=np.int64)
    return SmoothedDomain(global_index=gidx, corners=pts[conn], raw_corners=raw[conn], grid=grid)
