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


@needs_hybrid
def test_cosim_on_vcell_geometry_is_wired_and_conservative(vcell_ball):
    pytest.importorskip("smoldyn")
    from process_bigraph import Composite

    from viva_pde_particle.composites.hybrid import build_vcell_geometry_hybrid_document
    from viva_pde_particle.core import build_core
    from viva_pde_particle.model import HybridModel, Reaction, Species
    from viva_pde_particle.units import MOLECULES_PER_UM3_PER_UM as NA

    g, r = vcell_ball
    m = HybridModel(g, [Species("A", 1.0, particle=True, initial=0), Species("B", 1.0, initial=0.2)],
                    [Reaction("a_to_b", {"A": 1}, {"B": 1}, k=1.0), Reaction("b_to_a", {"B": 1}, {"A": 1}, k=0.5)])
    doc = build_vcell_geometry_hybrid_document(m, sphere_in_box((4.5, 4.5, 4.5), 4.0, (9, 9, 9)), 0.01, seed=2,
                                               realization=r)
    dom = doc["pde"]["config"]["domain"]
    assert (dom["mask"] == r.node_mask("cell")).all()
    assert (dom["volume_fraction"] * g.full_volume)[dom["mask"]].sum() == pytest.approx(r.volume("cell"), rel=0.01)
    assert doc["particles"]["inputs"] == {"fields": ["lookup_fields"]}  # extended fields for rate lookups
    sim = Composite({"state": doc}, core=build_core())
    w = sim.state["pde"]["instance"]._s.reshape(g.shape) * g.full_volume * dom["mask"]
    b0 = float((np.asarray(sim.state["fields"]["B"]) * w).sum() * NA)
    sim.run(0.005)
    sim.run(1.0)
    a = float(np.asarray(sim.state["particle_counts"]["A"]).sum())
    b = float((np.asarray(sim.state["fields"]["B"]) * w).sum() * NA)
    assert a > 1000
    assert (a + b) / b0 == pytest.approx(1.0, abs=0.01)  # creation volume = PDE volume: no exchange drift


@needs_hybrid
def test_recorded_vcell_geometry_run_and_native_bundle(vcell_ball, tmp_path):
    """A co-sim on VCell's geometry and a native VCell run, recorded into bundles on the same lattice."""
    pytest.importorskip("smoldyn")
    from vcell_fenics.results import Bundle

    from viva_pde_particle.composites.hybrid import build_vcell_geometry_hybrid_document, run_document
    from viva_pde_particle.model import HybridModel, Reaction, Species
    from viva_pde_particle.reference.vcell_native import run_native
    from viva_pde_particle.viz3d import attach_recorder, read_particles, write_native_bundle

    g, r = vcell_ball
    m = HybridModel(g, [Species("A", 1.0, particle=True, initial=2000), Species("B", 1.0, initial=0.0)],
                    [Reaction("convert", {"A": 1}, {"B": 1}, k=0.5)])
    desc = sphere_in_box((4.5, 4.5, 4.5), 4.0, (9, 9, 9))
    membranes = {"pm": r.boundary_triangles("cell")}
    doc = build_vcell_geometry_hybrid_document(m, desc, 0.01, realization=r)
    attach_recorder(doc, tmp_path / "cosim.fenics", 0.05, membranes=membranes)
    traj = run_document(doc, 0.1, 0.01, 0.05)
    b = Bundle.open(tmp_path / "cosim.fenics")
    mask = r.node_mask("cell")
    assert b.manifest.domains["cell"].n_points < mask.size  # only the hexes touching the domain
    vol = b.mesh("cell")
    assert len(read_particles(tmp_path / "cosim.fenics", "A", 0)) == 2000
    pm = b.field("pm", "B", 2)
    assert np.isfinite(pm).all() and pm.min() > 0  # membrane values come from the extended field
    assert np.allclose(b.times, traj.times)
    nat = run_native(m, t_end=0.1, dt=0.01, output_dt=0.05, seed=1, geometry=desc, domain_volume=r.volume("cell"))
    write_native_bundle(tmp_path / "native.fenics", nat, g, mask=mask, membranes=membranes)
    n = Bundle.open(tmp_path / "native.fenics")
    np.testing.assert_allclose(n.mesh("cell").points, vol.points)  # same lattice: viewable side by side
    assert len(n.times) == 3 and n.field("cell", "B", 2).max() > 0
