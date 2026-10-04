"""Particle → mesh transfer by molecule positions, and a membrane that is the mesh boundary."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("dolfinx")
pytest.importorskip("gmsh")
from dolfinx import fem  # noqa: E402

from viva_pde_particle.mesh import PointLocator, boundary_triangles, sphere_mesh, uniform_in_mesh  # noqa: E402
from viva_pde_particle.model.smoldyn_config import write_smoldyn_config  # noqa: E402

C, R = (2.0, 2.0, 2.0), 2.0


@pytest.fixture(scope="module")
def mesh_and_locator():
    msh = sphere_mesh(R, C, 0.6)
    V = fem.functionspace(msh, ("Lagrange", 1))
    return msh, V, PointLocator.build(msh, V)


def test_locate_agrees_with_dolfinx(mesh_and_locator):
    _, _, loc = mesh_and_locator
    pts = np.asarray(C) + np.random.default_rng(0).uniform(-2.1, 2.1, (3000, 3))
    fast = loc.locate(pts)
    slow = loc._locate_dolfinx(pts)
    np.testing.assert_array_equal(fast >= 0, slow >= 0)
    _, w = loc.weights(pts)
    assert (w >= 0).all() and np.allclose(w.sum(axis=1), 1)


def test_load_reproduces_p1_interpolation_and_conserves(mesh_and_locator):
    _, V, loc = mesh_and_locator
    pts = np.asarray(C) + np.random.default_rng(1).uniform(-1.0, 1.0, (500, 3))  # inside the ball
    load = loc.load(pts)
    assert load.sum() == pytest.approx(len(pts))
    X = V.tabulate_dof_coordinates()
    u = 2.0 * X[:, 0] - X[:, 1] + 0.5
    # Σ_j u_j b_j = Σ_p u_h(x_p), and u_h is exact for linear u
    assert u @ load == pytest.approx((2.0 * pts[:, 0] - pts[:, 1] + 0.5).sum())


def test_uniform_density_loads_the_lumped_mass(mesh_and_locator):
    msh, V, loc = mesh_and_locator
    import ufl

    ml = fem.assemble_vector(fem.form(ufl.TestFunction(V) * ufl.dx)).array
    pts = uniform_in_mesh(200_000, loc, msh.geometry.x.min(0), msh.geometry.x.max(0), np.random.default_rng(2))
    assert loc.inside(pts).all()
    load = loc.load(pts)
    # E[b_j] = ρ ∫φ_j: the exact-position load has no boundary bias
    rel = load / (len(pts) * ml / ml.sum()) - 1
    boundary = np.linalg.norm(V.tabulate_dof_coordinates() - C, axis=1) > R - 0.05
    assert abs(rel[boundary].mean()) < 0.01 and abs(rel[~boundary].mean()) < 0.01


def test_boundary_triangles_close_the_ball_outward(mesh_and_locator):
    msh, _, _ = mesh_and_locator
    tri = boundary_triangles(msh)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    assert (np.einsum("ij,ij->i", n, tri.mean(axis=1) - C) > 0).all()  # outward
    # divergence theorem: (1/3)∮ x·n dA = enclosed volume, the mesh's volume (just under the ball's)
    vol = np.einsum("ij,ij->i", tri.mean(axis=1) - C, n).sum() / 6
    assert vol == pytest.approx(4 / 3 * np.pi * R**3, rel=0.05) and vol < 4 / 3 * np.pi * R**3


def test_config_writes_triangle_membrane():
    from viva_pde_particle.grid import CartesianGrid

    g = CartesianGrid((0, 0, 0), (4, 4, 4), (5, 5, 5))
    tri = np.array([[[1, 1, 1], [3, 1, 1], [1, 3, 1]], [[1, 1, 1], [1, 3, 1], [1, 1, 3]]], dtype=float)
    parts = {"species": {"A": {"diffusion": 1.0}}, "reactions": [], "field_species": []}
    text = write_smoldyn_config(parts, g, {"A": np.zeros(g.shape)}, 0.01, 1,
                                geometry={"kind": "triangles", "triangles": tri, "interior_point": [1.5, 1.5, 1.5]})
    assert text.count("panel tri ") == 2
    assert "point 1.5 1.5 1.5" in text
    with pytest.raises(ValueError, match="unsupported geometry"):
        write_smoldyn_config(parts, g, {"A": np.zeros(g.shape)}, 0.01, 1, geometry={"kind": "cube"})


@pytest.mark.parametrize("membrane", ["sphere", "mesh"])
def test_positions_hybrid_conserves_and_confines(membrane):
    pytest.importorskip("smoldyn")
    from process_bigraph import Composite

    from viva_pde_particle.composites.hybrid import build_mesh_hybrid_document
    from viva_pde_particle.core import build_core
    from viva_pde_particle.grid import CartesianGrid
    from viva_pde_particle.model import HybridModel, Reaction, Species
    from viva_pde_particle.units import MOLECULES_PER_UM3_PER_UM as NA

    g = CartesianGrid((0, 0, 0), (4, 4, 4), (17, 17, 17))
    m = HybridModel(g, [Species("A", 1.0, particle=True, initial=2000), Species("B", 1.0, initial=0.0)],
                    [Reaction("convert", {"A": 1}, {"B": 1}, k=1.0)])
    doc = build_mesh_hybrid_document(m, {"center": C, "radius": R, "h": 0.6}, 0.01, seed=3,
                                     particle_transfer="positions", membrane=membrane)
    sim = Composite({"state": doc}, core=build_core())
    sim.run(0.105)
    pos = np.asarray(sim.state["particle_positions"]["A"])
    counts = np.asarray(sim.state["particle_counts"]["A"])
    assert len(pos) == counts.sum() < 2000
    pde = sim.state["pde"]["instance"]
    b = (np.asarray(sim.state["field_dofs"]["B"]) * pde.ml).sum() * NA
    # first-order lag lets B run slightly ahead of the A lost (as in B2b); within ~2%
    assert (len(pos) + b) / 2000 == pytest.approx(1.0, abs=0.02)
    if membrane == "mesh":
        from viva_pde_particle.mesh import mesh_locator

        loc = mesh_locator(doc["pde"]["config"]["mesh"], g)
        assert loc.inside(pos).all()  # particles never leave the PDE domain
    else:
        assert (np.linalg.norm(pos - C, axis=1) <= R + 1e-9).all()


def test_volume_samples_map_matches_mesh(mesh_and_locator):
    from viva_pde_particle.mesh import volume_samples

    _, _, loc = mesh_and_locator
    ids = volume_samples(loc, (0, 0, 0), (4, 4, 4), (16, 16, 16))
    h = 4 / 16
    k, rem = np.divmod(np.arange(ids.size), 16 * 16)
    j, i = np.divmod(rem, 16)
    centres = (np.stack([i, j, k], axis=1) + 0.5) * h      # x fastest, cell-centred
    np.testing.assert_array_equal(ids == 0, loc.inside(centres))
    assert ((ids == 0).sum() * h**3) == pytest.approx(4 / 3 * np.pi * R**3, rel=0.08)


def test_config_writes_boxsize_and_volume_samples():
    import zlib

    from viva_pde_particle.grid import CartesianGrid

    g = CartesianGrid((0, 0, 0), (4, 4, 4), (9, 9, 9))
    parts = {"species": {"A": {"diffusion": 1.0}}, "reactions": [], "field_species": []}
    ids = np.zeros(8 * 8 * 8, dtype=np.uint8)
    ids[::3] = 1
    tri = np.array([[[1, 1, 1], [3, 1, 1], [1, 3, 1]]], dtype=float)
    text = write_smoldyn_config(parts, g, {"A": np.zeros(g.shape)}, 0.01, 1, geometry={
        "kind": "triangles", "triangles": tri, "interior_point": [1.5, 1.5, 1.5],
        "volume_samples": {"origin": [0, 0, 0], "size": [4, 4, 4], "num": [8, 8, 8], "ids": ids}})
    assert "boxsize 1.0" in text  # two grid spacings of 0.5
    lines = text.splitlines()
    a, b = lines.index("start_highResVolumeSamples"), lines.index("end_highResVolumeSamples")
    assert "VolumeSamples 8 8 8" in lines[a:b] and "domain 0" in lines[a:b]
    hexdata = "".join(lines[lines.index("VolumeSamples 8 8 8") + 1:b])
    np.testing.assert_array_equal(np.frombuffer(zlib.decompress(bytes.fromhex(hexdata)), np.uint8), ids)
    assert lines.index("end_highResVolumeSamples") < lines.index("start_surface walls")
