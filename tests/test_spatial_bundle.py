"""Phase 8a: spatial results bundles (vcell-fenics format) and the SpatialRecorder."""
from __future__ import annotations

import numpy as np
import pytest

from viva_pde_particle.grid import CartesianGrid
from viva_pde_particle.viz3d import SpatialBundleWriter, grid_domain, membrane_domain, read_particles

pytest.importorskip("vcell_fenics")
pytest.importorskip("zarr")


def test_grid_domain_tets_fill_the_box_and_map_to_nodes():
    g = CartesianGrid((0, 0, 0), (4, 3, 2), (5, 4, 3))
    d = grid_domain("cell", g)
    assert d.cells.shape[1] == 4 and len(d.cells) == 6 * 4 * 3 * 2
    assert d.point_weights().sum() == pytest.approx(24.0)
    x, y, z = g.node_coordinates()
    np.testing.assert_allclose(d.points[:, 0], x.ravel()[d.node_index])
    np.testing.assert_allclose(d.points[:, 2], z.ravel()[d.node_index])


def test_grid_domain_keeps_hexes_touching_the_mask():
    g = CartesianGrid((0, 0, 0), (4, 4, 4), (5, 5, 5))
    mask = np.zeros(g.shape, bool)
    mask[2, 2, 2] = True
    d = grid_domain("cell", g, mask)
    assert len(d.cells) == 6 * 8 and d.point_weights().sum() == pytest.approx(8.0)
    assert len(d.points) == 27


def test_writer_round_trips_through_the_vcell_fenics_reader(tmp_path):
    from vcell_fenics.results import Bundle

    g = CartesianGrid((0, 0, 0), (2, 2, 2), (3, 3, 3))
    vol = grid_domain("cell", g)
    tri = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[1, 0, 0], [1, 1, 0], [0, 1, 0]]], float)
    mem = membrane_domain("pm", tri)
    w = SpatialBundleWriter(tmp_path / "b.fenics", source="test")
    w.add_domain(vol, ["B"])
    w.add_domain(mem, ["B"])
    w.add_particles(["A"])
    w.open()
    rng = np.random.default_rng(0)
    rows = []
    for t in (0.0, 0.5, 1.0, 1.5):  # more rows than planned: the arrays grow
        v, m, p = rng.random(len(vol.points)), rng.random(len(mem.points)), rng.random((int(10 + 2000 * t), 3))
        w.write(t, {("cell", "B"): v, ("pm", "B"): m}, particles={"A": p})
        rows.append((v, m, p))
    w.finalize("completed")
    b = Bundle.open(tmp_path / "b.fenics")
    assert b.times == (0.0, 0.5, 1.0, 1.5) and b.status == "completed"
    assert b.manifest.domains["pm"].kind == "membrane" and b.manifest.domains["cell"].cell_type == 10
    for i, (v, m, p) in enumerate(rows):
        np.testing.assert_array_equal(b.field("cell", "B", i), v)
        np.testing.assert_array_equal(b.field("pm", "B", i), m)
        np.testing.assert_array_equal(read_particles(tmp_path / "b.fenics", "A", i), p)
    st = b.stats("cell", "B")
    assert st.shape == (4, 4)
    assert st[0, 2] == pytest.approx(rows[0][0].min()) and st[0, 3] == pytest.approx(rows[0][0].max())
    mesh = b.mesh("cell")
    np.testing.assert_allclose(mesh.points, vol.points)


def _run(doc, t_end=0.1, dt=0.01, every=0.05):
    from viva_pde_particle.composites.hybrid import run_document

    return run_document(doc, t_end, dt, every)


def test_recorder_records_the_run_without_changing_it(tmp_path):
    pytest.importorskip("smoldyn")
    from vcell_fenics.results import Bundle

    from viva_pde_particle.composites import examples
    from viva_pde_particle.composites.hybrid import build_hybrid_document
    from viva_pde_particle.viz3d import attach_recorder

    m = examples.bimolecular_model(n_particles=2000)
    plain = _run(build_hybrid_document(m, 0.01, 2, seed=3))
    doc = attach_recorder(build_hybrid_document(m, 0.01, 2, seed=3), tmp_path / "r.fenics", 0.05)
    rec = _run(doc)
    for s in plain.fields:
        np.testing.assert_array_equal(np.stack(plain.fields[s]), np.stack(rec.fields[s]))
    for s in plain.particle_counts:
        np.testing.assert_array_equal(np.stack(plain.particle_counts[s]), np.stack(rec.particle_counts[s]))
    b = Bundle.open(tmp_path / "r.fenics")
    assert np.allclose(b.times, rec.times) and b.status == "completed"
    idx = grid_domain("cell", m.grid).node_index
    for i in range(len(rec.times)):
        for s, v in rec.fields.items():
            np.testing.assert_array_equal(b.field("cell", s, i), np.asarray(v[i]).ravel()[idx])
        for s, c in rec.particle_counts.items():
            assert len(read_particles(tmp_path / "r.fenics", s, i)) == int(np.asarray(c[i]).sum())


def test_recorder_on_a_mesh_document_with_membrane(tmp_path):
    pytest.importorskip("smoldyn")
    pytest.importorskip("dolfinx")
    from vcell_fenics.results import Bundle

    from viva_pde_particle.composites import examples
    from viva_pde_particle.mesh import mesh_space
    from viva_pde_particle.viz3d import attach_recorder

    doc = examples.ball_netgen_hybrid(case="conversion", radius=1.5, h=0.5)
    from viva_pde_particle.geometry import realize_fenics

    real = realize_fenics(examples._ball_description(examples.ball_model("conversion", 1.5), 1.5), 0.5)
    attach_recorder(doc, tmp_path / "m.fenics", 0.05, membranes={"pm": real.boundary_triangles("cell")})
    traj = _run(doc)
    b = Bundle.open(tmp_path / "m.fenics")
    assert set(b.manifest.domains) == {"cell", "pm"}
    n_dofs = len(mesh_space(doc["pde"]["config"]["mesh"]).V.tabulate_dof_coordinates())
    assert b.manifest.domains["cell"].n_points == n_dofs
    np.testing.assert_array_equal(b.field("cell", "B", 2), np.asarray(traj.field_dofs["B"][2]))
    assert len(read_particles(tmp_path / "m.fenics", "A", 0)) == 20000  # initial positions recorded
    pm = b.field("pm", "B", 2)
    assert np.isfinite(pm).all() and pm.max() > 0


def test_recorder_inside_a_splitting_coordinator(tmp_path):
    pytest.importorskip("smoldyn")
    from vcell_fenics.results import Bundle

    from viva_pde_particle.composites import examples
    from viva_pde_particle.viz3d import attach_recorder

    doc = attach_recorder(examples.exchange_splitting(scheme="strang", step_multiplier=2), tmp_path / "s.fenics", 0.04)
    traj = _run(doc, t_end=0.08, every=0.04)
    b = Bundle.open(tmp_path / "s.fenics")
    assert np.allclose(b.times, [0.0, 0.04, 0.08])
    for i in range(3):
        assert len(read_particles(tmp_path / "s.fenics", "A", i)) == int(np.asarray(traj.particle_counts["A"][i]).sum())
