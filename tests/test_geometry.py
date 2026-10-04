"""Phase 7e: VCell-style geometry descriptions realized by Netgen (vcell-fenics) and used by the hybrid."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("vcell_fenics")
pytest.importorskip("netgen")

from viva_pde_particle.geometry import (  # noqa: E402
    description_from_config,
    description_to_config,
    realize_fenics,
    sphere_in_box,
)

C, R = (2.0, 2.0, 2.0), 1.5


@pytest.fixture(scope="module")
def ball():
    return sphere_in_box(C, R, (4.0, 4.0, 4.0))


def test_description_round_trips_through_a_plain_dict(ball):
    cfg = description_to_config(ball)
    assert isinstance(cfg, dict) and "geometry_description" in cfg
    assert description_from_config(cfg) == ball
    assert [sv.name for sv in ball.subvolumes] == ["cell", "ec"] and ball.surfaces[0].name == "pm"


def test_realization_locates_measures_and_bounds_the_regions(ball):
    real = realize_fenics(ball, 0.4)
    assert real.regions == ("cell", "ec")
    exact = 4 / 3 * np.pi * R**3
    assert real.volume("cell") == pytest.approx(exact, rel=0.05)  # ~4 elements across R: faceting error
    assert real.volume("cell") + real.volume("ec") == pytest.approx(64.0, rel=1e-6)  # the regions tile the box
    pts = np.array([C, [0.1, 0.1, 0.1], [2.0, 2.0, 2.0 + 0.5 * R]])
    np.testing.assert_array_equal(real.locate(pts), [0, 1, 0])
    tri = real.boundary_triangles("cell")
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    assert (np.einsum("ij,ij->i", n, tri.mean(axis=1) - C) > 0).all()  # outward
    ip = real.interior_points("cell")
    assert real.inside("cell", ip).all() and len(ip) > 1
    assert realize_fenics(ball, 0.4) is real  # cached per process


def test_hybrid_on_a_netgen_geometry_conserves_and_confines(ball):
    pytest.importorskip("smoldyn")
    from process_bigraph import Composite

    from viva_pde_particle.composites.hybrid import build_mesh_hybrid_document
    from viva_pde_particle.core import build_core
    from viva_pde_particle.grid import CartesianGrid
    from viva_pde_particle.mesh import mesh_space
    from viva_pde_particle.model import HybridModel, Reaction, Species
    from viva_pde_particle.units import MOLECULES_PER_UM3_PER_UM as NA

    g = CartesianGrid((0, 0, 0), (4, 4, 4), (17, 17, 17))
    m = HybridModel(g, [Species("A", 1.0, particle=True, initial=2000), Species("B", 1.0, initial=0.0)],
                    [Reaction("convert", {"A": 1}, {"B": 1}, k=1.0)])
    doc = build_mesh_hybrid_document(m, None, 0.01, seed=3, geometry={"description": ball, "region": "cell", "h": 0.4})
    assert doc["pde"]["config"]["mesh"]["kind"] == "geometry"
    with pytest.raises(ValueError, match="exactly one"):
        build_mesh_hybrid_document(m, {"center": C, "radius": R}, 0.01, geometry={"description": ball, "region": "cell", "h": 0.4})
    sim = Composite({"state": doc}, core=build_core())
    sim.run(0.005)
    sim.run(0.2)
    engine = sim.state["particles"]["instance"].child  # SmoldynHybrid inside the fvsolver Stepper
    pos = engine.positions()["A"]
    assert realize_fenics(ball, 0.4).inside("cell", pos).all()  # particles stay in the PDE domain
    ml = mesh_space(doc["pde"]["config"]["mesh"]).ml
    b = float((np.asarray(sim.state["field_dofs"]["B"]) * ml).sum() * NA)
    assert len(pos) < 2000 and b > 0
    assert (len(pos) + b) / 2000 == pytest.approx(1.0, abs=0.02)
