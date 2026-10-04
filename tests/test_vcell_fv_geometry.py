"""Phase 7e.4: the VCell-FV geometry compiler (VCell's own discretization of a GeometryDescription)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("vcell_fenics")

from viva_pde_particle.geometry import sphere_in_box  # noqa: E402
from viva_pde_particle.geometry.vcml import vcell_expression  # noqa: E402
from viva_pde_particle.grid import CartesianGrid  # noqa: E402
from viva_pde_particle.reference.vcell_native import hybrid_support_available  # noqa: E402

needs_hybrid = pytest.mark.skipif(not hybrid_support_available(), reason="pyvcell without spatial-hybrid support")


def test_vcell_expression_translation():
    assert vcell_expression("(geom.x[0]-1.5)**2 + geom.x[ 2 ] < 4 && geom.x[1] > 0") == "(x-1.5)^2 + z < 4 && y > 0"


def test_last_subvolume_owns_the_remainder():
    pytest.importorskip("pyvcell")
    from viva_pde_particle.geometry.vcml import to_vcml_geometry

    geo = to_vcml_geometry(sphere_in_box((4.5, 4.5, 4.5), 4.0, (9, 9, 9)))
    assert [sv.name for sv in geo.subvolumes] == ["cell", "ec"]
    assert geo.subvolumes[0].analytic_expr.startswith("(x-4.5)^2")
    assert geo.subvolumes[1].analytic_expr == "1.0"  # owns interface points the strict predicates leave unowned
    assert [(sc.name, sc.subvolume_ref_1, sc.subvolume_ref_2) for sc in geo.surface_classes] == [("pm", "cell", "ec")]


@pytest.fixture(scope="module")
def vcell_ball():
    from viva_pde_particle.geometry.vcell_fv import VCellFVRealization

    g = CartesianGrid((0, 0, 0), (9, 9, 9), (37, 37, 37))
    return g, VCellFVRealization(sphere_in_box((4.5, 4.5, 4.5), 4.0, (9, 9, 9)), g)


@needs_hybrid
def test_staircase_and_smooth_domains_match_native_vcell(vcell_ball):
    g, r = vcell_ball
    x, y, z = g.node_coordinates()
    rr = np.sqrt((x - 4.5) ** 2 + (y - 4.5) ** 2 + (z - 4.5) ** 2)
    # the PDE domain: VCell's staircase (nodes strictly inside; whole node-centred voxels)
    np.testing.assert_array_equal(r.node_mask("cell"), rr < 4.0)
    assert r.pde_volume("cell") == pytest.approx(r.node_mask("cell").sum() * 0.25**3)
    assert r.pde_volume("cell") + r.pde_volume("ec") == pytest.approx(729.0)
    # the particle domain: VCell's smooth membrane triangulation
    exact = 4 / 3 * np.pi * 64
    assert len(r.boundary_triangles("cell")) > 5000
    assert r.volume("cell") == pytest.approx(exact, rel=0.03)
    assert r.volume("cell") < r.pde_volume("cell")  # the B2c mismatch (~1.8%)
    assert r.inside("cell", r.interior_points("cell")).all()
    np.testing.assert_array_equal(r.locate(np.array([[4.5, 4.5, 4.5], [0.2, 0.2, 0.2], [4.5, 4.5, 8.8]])), [0, 1, 1])


@needs_hybrid
def test_smoldyn_geometry_derives_from_the_vcell_realization(vcell_ball):
    from viva_pde_particle.geometry import smoldyn_geometry

    g, r = vcell_ball
    geom = smoldyn_geometry(r, "cell", g, volume_samples=1)
    assert len(geom["triangles"]) == len(r.boundary_triangles("cell"))
    np.testing.assert_array_equal(geom["interior_point"], r.interior_points("cell"))  # VCell's own points
    inside_frac = (geom["volume_samples"]["ids"] == 0).mean()
    assert inside_frac * 729.0 == pytest.approx(r.volume("cell"), rel=0.03)
