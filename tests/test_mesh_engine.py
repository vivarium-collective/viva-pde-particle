"""Unstructured mesh engine: gmsh ball, mesh <-> grid transfer, and the mesh PDE process."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("dolfinx")
pytest.importorskip("gmsh")
from process_bigraph import allocate_core  # noqa: E402

from viva_pde_particle.grid import CartesianGrid  # noqa: E402
from viva_pde_particle.mesh import MeshGridTransfer, sphere_mesh  # noqa: E402
from viva_pde_particle.processes import FenicsxMeshReactionDiffusion  # noqa: E402

GRID = CartesianGrid((0, 0, 0), (4, 4, 4), (17, 17, 17))
MESH = {"kind": "sphere", "center": [2.0, 2.0, 2.0], "radius": 2.0, "h": 0.6}


@pytest.fixture(scope="module")
def transfer():
    return MeshGridTransfer.build(sphere_mesh(2.0, (2.0, 2.0, 2.0), 0.6), GRID)


def test_transfer_reproduces_linear_fields_and_conserves_counts(transfer):
    X = transfer.V.tabulate_dof_coordinates()
    u = 3.0 * X[:, 0] - X[:, 2]
    nodes = np.stack([c.ravel() for c in GRID.node_coordinates()], axis=1)
    np.testing.assert_allclose((transfer.P @ u)[transfer.inside], (3.0 * nodes[:, 0] - nodes[:, 2])[transfer.inside],
                               atol=1e-12)
    h = np.random.default_rng(1).poisson(3.0, GRID.num_elements).astype(float)
    assert (transfer.P.T @ h).sum() == pytest.approx(h.sum())
    np.testing.assert_allclose(np.asarray(transfer.P.sum(axis=1)).ravel(), 1.0)


def test_mesh_process_conserves_and_sources_from_particles():
    proc = FenicsxMeshReactionDiffusion(core=allocate_core(), config={
        "grid": GRID.to_config(), "mesh": MESH, "dt": 0.01,
        "pde": {"species": {"B": {"diffusion": 1.0}}, "particle_species": ["A"],
                "terms": [{"species": "B", "coeff": 1, "k": 1.0, "reactants": {"A": 1}}]},
    })
    counts = np.zeros(GRID.shape)
    counts[8, 8, 8] = 6022.14076  # 10 µM·µm³ worth of A at the centre
    dofs0 = proc.initial_dofs({"B": 0.0})
    out = proc.update({"field_dofs": dofs0, "particle_counts": {"A": counts}}, 0.1)
    produced = (out["field_dofs"]["B"] * proc.ml).sum()  # µM·µm³ of B made in 0.1 s at rate k·[A]
    assert produced == pytest.approx(10.0 * 1.0 * 0.1, rel=1e-10)
    assert out["fields"]["B"].shape == GRID.shape
