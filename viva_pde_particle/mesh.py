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
