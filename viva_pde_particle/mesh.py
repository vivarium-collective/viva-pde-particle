"""Unstructured meshes and the mesh ↔ grid transfer used for non-box geometries (Investigation B).

The particle side keeps working on a Cartesian background ``HybridGrid``: fields for rate
lookups, histograms for counts. The PDE side solves on an unstructured P1 mesh. Data
crosses between them through one sparse matrix ``P`` (grid nodes × mesh DOFs):

- **mesh → grid:** ``u_grid = P @ u_dofs``. Each grid node takes the P1 interpolant at its
  coordinates, or the nearest DOF value for nodes outside the mesh.
- **grid → mesh:** ``b = Pᵀ @ h`` turns per-node particle counts h into P1 loads
  (b_j = Σ_n φ_j(x_n)·h_n). P's rows sum to 1, so Σb = Σh and molecules are conserved.
  With grid spacing below the mesh size, this approximates the exact point-particle load
  Σ_p φ_j(x_p).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from viva_pde_particle.grid import CartesianGrid


def sphere_mesh(radius: float, center=(0.0, 0.0, 0.0), h: float | None = None):
    """Unstructured tetrahedral mesh of a ball (gmsh), as a dolfinx Mesh on COMM_SELF."""
    import gmsh
    from dolfinx.io import gmsh as gmshio
    from mpi4py import MPI

    h = h or radius / 6
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("ball")
        tag = gmsh.model.occ.addSphere(*center, radius)
        gmsh.model.occ.synchronize()
        gmsh.model.addPhysicalGroup(3, [tag], 1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h / 2)
        gmsh.model.mesh.generate(3)
        data = gmshio.model_to_mesh(gmsh.model, MPI.COMM_SELF, 0, gdim=3)
    finally:
        gmsh.finalize()
    return data.mesh if hasattr(data, "mesh") else data[0]


def _locate(msh, points: np.ndarray) -> np.ndarray:
    """Index of a cell containing each point, or -1."""
    from dolfinx import geometry

    tree = geometry.bb_tree(msh, msh.topology.dim)
    cand = geometry.compute_collisions_points(tree, points)
    coll = geometry.compute_colliding_cells(msh, cand, points)
    cells = np.full(len(points), -1, dtype=np.int64)
    for i in range(len(points)):
        links = coll.links(i)
        if len(links):
            cells[i] = links[0]
    return cells


@dataclass
class MeshGridTransfer:
    """P1 mesh space plus the sparse transfer matrix to and from a background grid."""

    V: object                 # dolfinx FunctionSpace (P1)
    P: sp.csr_matrix          # (grid nodes, dofs)
    inside: np.ndarray        # bool per grid node (node lies in the mesh)

    @classmethod
    def build(cls, msh, grid: CartesianGrid) -> "MeshGridTransfer":
        from dolfinx import fem
        from scipy.spatial import cKDTree

        V = fem.functionspace(msh, ("Lagrange", 1))
        nodes = np.stack([c.ravel() for c in grid.node_coordinates()], axis=1)
        if nodes.shape[1] < 3:
            nodes = np.hstack([nodes, np.zeros((len(nodes), 3 - nodes.shape[1]))])
        cells = _locate(msh, nodes)
        x = msh.geometry.x
        gdofs = msh.geometry.dofmap
        rows, cols, vals = [], [], []
        inside = cells >= 0
        for n in np.nonzero(inside)[0]:
            c = cells[n]
            verts = x[gdofs[c]]                       # (4, 3) tetrahedron vertices
            T = (verts[1:] - verts[0]).T              # barycentric solve
            lam = np.linalg.solve(T, nodes[n] - verts[0])
            bary = np.concatenate([[1 - lam.sum()], lam])
            dofs = V.dofmap.cell_dofs(c)
            rows += [n] * len(dofs)
            cols += list(dofs)
            vals += list(np.clip(bary, 0.0, 1.0))
        dof_xyz = V.tabulate_dof_coordinates()
        tree = cKDTree(dof_xyz)
        outside = np.nonzero(~inside)[0]
        if len(outside):
            _, near = tree.query(nodes[outside])
            rows += list(outside)
            cols += list(near)
            vals += [1.0] * len(outside)
        n_dofs = V.dofmap.index_map.size_local
        P = sp.csr_matrix((vals, (rows, cols)), shape=(len(nodes), n_dofs))
        rowsum = np.asarray(P.sum(axis=1)).ravel()
        P = sp.diags(1.0 / rowsum) @ P               # exact partition of unity per node
        return cls(V=V, P=P.tocsr(), inside=inside)


@dataclass
class PointLocator:
    """Vectorized point location and P1 barycentric weights on a tetrahedral mesh.

    Used for the exact point-particle load b_j = Σ_p φ_j(x_p). It bypasses the grid
    histogram, so in expectation b_j = ∫ φ_j ρ.

    Location uses a voxel → tet lookup table on a uniform grid over the mesh's bounding
    box (built once). Each tet is listed in every voxel its bounding box overlaps. That is
    a conservative superset: any tet containing a point is listed in the point's voxel, so
    a point is located by one arithmetic voxel lookup plus barycentric tests of that voxel's
    candidates, nearest-centroid first, and location is exact without a tree search. Points outside the mesh, e.g. between a
    curved membrane and the inscribed polyhedral mesh, go to the cell with the nearest
    centroid. Their barycentric weights are clipped to the cell and renormalized, which
    projects them onto its closest face.
    """

    msh: object
    cell_dofs: np.ndarray      # (n_cells, 4)
    v0: np.ndarray             # (n_cells, 3)
    tinv: np.ndarray           # (n_cells, 3, 3)
    centroids: np.ndarray      # (n_cells, 3)
    n_dofs: int
    voxel_lo: np.ndarray       # (3,) lookup grid origin
    voxel_size: float
    voxel_num: np.ndarray      # (3,) lookup voxels per axis, x fastest
    voxel_table: np.ndarray    # (n_voxels, width) candidate tets per voxel, -1 padded

    @classmethod
    def build(cls, msh, V, voxel_size: float | None = None) -> "PointLocator":
        """``voxel_size`` defaults to a third of the mean tet edge (≈ 10–15 candidates per voxel)."""
        x = msh.geometry.x
        gdofs = np.asarray(msh.geometry.dofmap)
        verts = x[gdofs]                                       # (n_cells, 4, 3)
        T = np.transpose(verts[:, 1:] - verts[:, :1], (0, 2, 1))
        cell_dofs = np.stack([V.dofmap.cell_dofs(c) for c in range(len(gdofs))])
        centroids = verts.mean(axis=1)
        if voxel_size is None:
            edges = verts[:, [0, 0, 0, 1, 1, 2]] - verts[:, [1, 2, 3, 2, 3, 3]]
            voxel_size = float(np.linalg.norm(edges, axis=2).mean()) / 3
        lo, num, table = _voxel_table(verts, centroids, voxel_size)
        return cls(msh=msh, cell_dofs=cell_dofs, v0=verts[:, 0], tinv=np.linalg.inv(T), centroids=centroids,
                   n_dofs=V.dofmap.index_map.size_local, voxel_lo=lo, voxel_size=voxel_size, voxel_num=num,
                   voxel_table=table)

    def locate(self, points: np.ndarray, tol: float = 1e-10) -> np.ndarray:
        """Containing cell per point, or -1 (exact; see the class docstring)."""
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        cells = np.full(len(pts), -1, dtype=np.int64)
        idx = np.floor((pts - self.voxel_lo) / self.voxel_size).astype(np.int64)
        inbox = np.nonzero(((idx >= 0) & (idx < self.voxel_num)).all(axis=1))[0]
        i = idx[inbox]
        n = self.voxel_num
        rows = self.voxel_table[(i[:, 2] * n[1] + i[:, 1]) * n[0] + i[:, 0]]
        todo = np.arange(len(inbox))
        for j in range(self.voxel_table.shape[1]):  # candidate column j, only for unresolved points
            c = rows[todo, j]
            keep = c >= 0
            todo, c = todo[keep], c[keep]
            if not len(todo):
                break
            lam = np.einsum("nij,nj->ni", self.tinv[c], pts[inbox[todo]] - self.v0[c])
            hit = (lam >= -tol).all(axis=1) & (lam.sum(axis=1) <= 1 + tol)
            cells[inbox[todo[hit]]] = c[hit]
            todo = todo[~hit]
        return cells

    def _locate_dolfinx(self, pts: np.ndarray) -> np.ndarray:
        """Reference location by dolfinx's bounding-box tree (used in tests)."""
        from dolfinx import geometry

        if not hasattr(self, "_tree"):
            self._tree = geometry.bb_tree(self.msh, self.msh.topology.dim)
        coll = geometry.compute_colliding_cells(self.msh, geometry.compute_collisions_points(self._tree, pts), pts)
        off = np.asarray(coll.offsets)
        arr = np.asarray(coll.array)
        has = off[1:] > off[:-1]
        cells = np.full(len(pts), -1, dtype=np.int64)
        cells[has] = arr[off[:-1][has]]
        return cells

    def weights(self, points: np.ndarray):
        """(cells, (n, 4) barycentric weights) for every point (outside points projected)."""
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        cells = self.locate(pts)
        out = cells < 0
        if out.any():
            if not hasattr(self, "_ctree"):
                from scipy.spatial import cKDTree

                self._ctree = cKDTree(self.centroids)
            cells[out] = self._ctree.query(pts[out])[1]
        lam = np.einsum("cij,cj->ci", self.tinv[cells], pts - self.v0[cells])
        bary = np.concatenate([1 - lam.sum(axis=1, keepdims=True), lam], axis=1)
        bary = np.clip(bary, 0.0, None)
        bary /= bary.sum(axis=1, keepdims=True)
        return cells, bary

    def load(self, points: np.ndarray) -> np.ndarray:
        """P1 load b_j = Σ_p φ_j(x_p); Σ b = number of points."""
        if len(points) == 0:
            return np.zeros(self.n_dofs)
        cells, bary = self.weights(points)
        return np.bincount(self.cell_dofs[cells].ravel(), weights=bary.ravel(), minlength=self.n_dofs)

    def inside(self, points: np.ndarray) -> np.ndarray:
        return self.locate(np.asarray(points, dtype=float).reshape(-1, 3)) >= 0


def _voxel_table(verts: np.ndarray, centroids: np.ndarray, voxel: float):
    """Uniform voxel grid over the tets' bounding box, listing each tet in every voxel its
    bounding box overlaps. Candidates in a voxel are ordered by centroid distance to the
    voxel centre. Returns (origin, voxels per axis, (n_voxels, width) table padded with -1)."""
    pts = verts.reshape(-1, 3)
    lo = pts.min(axis=0) - 1e-9
    num = np.maximum(np.ceil((pts.max(axis=0) + 1e-9 - lo) / voxel).astype(np.int64), 1)
    tlo = np.floor((verts.min(axis=1) - lo) / voxel).astype(np.int64).clip(0, num - 1)
    thi = np.floor((verts.max(axis=1) - lo) / voxel).astype(np.int64).clip(0, num - 1)
    span = thi - tlo + 1
    counts = span.prod(axis=1)
    tet = np.repeat(np.arange(len(verts)), counts)
    off = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
    sx, sy = np.repeat(span[:, 0], counts), np.repeat(span[:, 1], counts)
    ix = np.repeat(tlo[:, 0], counts) + off % sx
    iy = np.repeat(tlo[:, 1], counts) + (off // sx) % sy
    iz = np.repeat(tlo[:, 2], counts) + off // (sx * sy)
    vox = (iz * num[1] + iy) * num[0] + ix
    centre = lo + (np.stack([ix, iy, iz], axis=1) + 0.5) * voxel
    dist = np.linalg.norm(centroids[tet] - centre, axis=1)
    order = np.lexsort((dist, vox))
    vox, tet = vox[order], tet[order]
    per = np.bincount(vox, minlength=int(num.prod()))
    table = np.full((int(num.prod()), int(per.max())), -1, dtype=np.int64)
    start = np.cumsum(per) - per
    table[vox, np.arange(len(vox)) - start[vox]] = tet
    return lo, num, table


def boundary_triangles(msh) -> np.ndarray:
    """Exterior facets of a tetrahedral mesh as (m, 3, 3) vertex coordinates, outward-oriented."""
    from dolfinx import mesh as dmesh

    tdim = msh.topology.dim
    msh.topology.create_connectivity(tdim - 1, tdim)
    facets = dmesh.exterior_facet_indices(msh.topology)
    tri = np.asarray(dmesh.entities_to_geometry(msh, tdim - 1, facets))
    x = msh.geometry.x
    pts = x[tri]                                               # (m, 3, 3)
    f2c = msh.topology.connectivity(tdim - 1, tdim)
    cells = np.array([f2c.links(f)[0] for f in facets])
    cell_centroid = x[np.asarray(msh.geometry.dofmap)[cells]].mean(axis=1)
    normal = np.cross(pts[:, 1] - pts[:, 0], pts[:, 2] - pts[:, 0])
    inward = np.einsum("ij,ij->i", normal, cell_centroid - pts.mean(axis=1)) > 0
    pts[inward] = pts[inward][:, [0, 2, 1]]
    return pts


def uniform_in_mesh(n: int, locator: PointLocator, bbox_lo, bbox_hi, rng: np.random.Generator) -> np.ndarray:
    """n points uniformly distributed in the mesh (rejection sampling in its bounding box)."""
    lo, hi = np.asarray(bbox_lo, float), np.asarray(bbox_hi, float)
    out = np.empty((0, 3))
    while len(out) < n:
        cand = lo + (hi - lo) * rng.random((max(2 * (n - len(out)), 1000), 3))
        out = np.vstack([out, cand[locator.inside(cand)]])
    return out[:n]


def volume_samples(locator: PointLocator, origin, size, num) -> np.ndarray:
    """Compartment map for Smoldyn's ``highResVolumeSamples``: 0 inside the mesh, 1 outside.

    Samples are cell-centred, as Smoldyn's ``posincompart`` reads them: sample (i, j, k)
    covers ``[o + i·h, o + (i+1)·h)`` per axis with ``h = size/num``. It is classified at
    its centre. Returns uint8 of length ``prod(num)`` in x-fastest order.
    """
    o, s, n = (np.asarray(v, dtype=float) for v in (origin, size, num))
    n = n.astype(int)
    h = s / n
    axes = [o[d] + (np.arange(n[d]) + 0.5) * h[d] for d in range(3)]
    out = np.ones(int(n.prod()), dtype=np.uint8)
    nxy = n[0] * n[1]
    gx, gy = np.meshgrid(axes[0], axes[1], indexing="xy")      # (ny, nx): x fastest when raveled
    xy = np.stack([gx.ravel(), gy.ravel()], axis=1)
    for k, z in enumerate(axes[2]):
        pts = np.column_stack([xy, np.full(len(xy), z)])
        out[k * nxy:(k + 1) * nxy] = np.where(locator.inside(pts), 0, 1)
    return out
