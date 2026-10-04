"""Phase 7e.3: the Smoldyn geometry is derived from the same realization the PDE uses."""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

pytest.importorskip("vcell_fenics")
pytest.importorskip("netgen")

from viva_pde_particle.geometry import compartment_samples, realize_fenics, smoldyn_geometry  # noqa: E402
from viva_pde_particle.grid import CartesianGrid  # noqa: E402

GRID = CartesianGrid((0, 0, 0), (8, 6, 6), (33, 25, 25))


@pytest.fixture(scope="module")
def dumbbell():
    """A non-convex region: two overlapping balls joined by a neck (VCell-style analytic union)."""
    from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass

    s1 = "(geom.x[0]-2.6)**2 + (geom.x[1]-3)**2 + (geom.x[2]-3)**2 < 2.25"
    s2 = "(geom.x[0]-5.4)**2 + (geom.x[1]-3)**2 + (geom.x[2]-3)**2 < 2.25"
    neck = "(geom.x[1]-3)**2 + (geom.x[2]-3)**2 < 0.36 && geom.x[0] > 2.6 && geom.x[0] < 5.4"
    d = GeometryDescription(name="dumbbell", dim=3, extent=(8.0, 6.0, 6.0),
                            subvolumes=(SubVolume(name="cell", type="analytic", expression=f"({s1}) || ({s2}) || ({neck})"),
                                        SubVolume(name="ec", type="analytic", expression="1.0")),
                            surfaces=(SurfaceClass(name="pm", inside="cell", outside="ec"),))
    return realize_fenics(d, 0.4)


def test_derived_pieces_are_consistent_with_the_realization(dumbbell):
    geom = smoldyn_geometry(dumbbell, "cell", GRID, volume_samples=2)
    assert geom["kind"] == "triangles"
    assert dumbbell.inside("cell", geom["interior_point"]).all() and len(geom["interior_point"]) > 1
    tri = geom["triangles"]
    vol = np.einsum("ij,ij->i", tri.mean(axis=1), np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])).sum() / 6
    assert vol == pytest.approx(dumbbell.volume("cell"), rel=1e-9)  # closed, outward surface of the same region
    vs = geom["volume_samples"]
    assert vs["num"] == [64, 48, 48]
    h = np.asarray(vs["size"]) / np.asarray(vs["num"])
    k, rem = np.divmod(np.arange(vs["ids"].size), vs["num"][0] * vs["num"][1])
    j, i = np.divmod(rem, vs["num"][0])
    centres = np.asarray(vs["origin"]) + (np.stack([i, j, k], axis=1) + 0.5) * h
    np.testing.assert_array_equal(vs["ids"] == 0, dumbbell.inside("cell", centres))
    np.testing.assert_array_equal(compartment_samples(lambda p: dumbbell.inside("cell", p), (0, 0, 0), (8, 6, 6), (8, 6, 6)),
                                  vs["ids"].reshape(48, 48, 64)[4::8, 4::8, 4::8].ravel())


def _create(dumbbell, geom, k=100.0, t=1.0, seed=11):
    import smoldyn._smoldyn as sm

    from viva_pde_particle.model.smoldyn_config import write_smoldyn_config

    parts = {"species": {"A": {"diffusion": 0.0}}, "field_species": ["B"],
             "reactions": [{"name": "make", "reactants": [], "products": ["A"], "order": 0,
                            "rate_constant": k, "fields": ["B"]}]}
    text = write_smoldyn_config(parts, GRID, {"A": np.zeros(GRID.shape)}, 0.01, seed, geometry=geom)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "model.txt")
        with open(path, "w") as f:
            f.write(text)
        hg = sm.HybridGrid(list(GRID.origin), list(GRID.size), list(GRID.num))
        sim = sm.Simulation(path, "q", hg)
        hg.setField("B", np.ones(GRID.shape))
        sim.runUntil(t - 0.005, 0.01, False)
        return np.array(sim.getMoleculePositions("A"))


def test_creation_fills_a_non_convex_compartment_with_several_points(dumbbell):
    pytest.importorskip("smoldyn")
    expected = 100.0 * dumbbell.volume("cell")
    pos = _create(dumbbell, smoldyn_geometry(dumbbell, "cell", GRID))
    assert abs(len(pos) - expected) < 4 * np.sqrt(expected), (len(pos), expected)
    assert dumbbell.inside("cell", pos).all()
    assert abs((pos[:, 0] > 4.0).mean() - 0.5) < 0.04  # both lobes, evenly
    # one compartment point cannot see all of a non-convex region: creation falls short
    single = smoldyn_geometry(dumbbell, "cell", GRID)
    single["interior_point"] = single["interior_point"][:1]
    assert len(_create(dumbbell, single)) < expected - 4 * np.sqrt(expected)
