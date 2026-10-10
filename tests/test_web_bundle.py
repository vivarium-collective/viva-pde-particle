"""The bundle's web extension (viz3d/web.py), the SpatialExport step, and the compose runner announcing a recorded
run's bundle (compose-api docs/plan-viewers.md, B1)."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

pytest.importorskip("vcell_fenics")
pytest.importorskip("zarr")

from viva_pde_particle.grid import CartesianGrid  # noqa: E402
from viva_pde_particle.viz3d import SpatialBundleWriter, grid_domain  # noqa: E402
from viva_pde_particle.viz3d.web import WEB_KEY, add_web_extension, boundary_triangles  # noqa: E402


def _cube(n=3):
    return grid_domain("cell", CartesianGrid((0, 0, 0), (1, 1, 1), (n, n, n)))


def test_boundary_of_a_tet_cube_is_a_closed_surface():
    dom = _cube(3)  # 2x2x2 hexes, 6 tets each
    tris = boundary_triangles(dom.cells)
    assert len(tris) == 6 * (2 * 2) * 2  # 6 faces of 2x2 squares, 2 triangles each
    edges = {tuple(sorted(e)) for t in tris for e in ((t[0], t[1]), (t[1], t[2]), (t[0], t[2]))}
    vertices = set(tris.ravel())
    assert len(vertices) - len(edges) + len(tris) == 2  # Euler: a closed genus-0 surface
    on_boundary = np.any((dom.points == 0) | (dom.points == 1), axis=1)
    assert vertices == set(np.flatnonzero(on_boundary))  # every boundary point, and no interior one


def _bundle(path, rows=2):
    dom = _cube(3)
    w = SpatialBundleWriter(path, source="test")
    w.add_domain(dom, ["u"])
    w.open()
    for k in range(rows):
        w.write(float(k), {("cell", "u"): np.full(len(dom.points), float(k))})
    w.finalize()
    return dom


def _web_arrays(path, domain="cell"):
    import zarr

    attrs = json.loads((path / ".zattrs").read_text())
    entry = attrs[WEB_KEY]["surfaces"][domain]
    group = zarr.open_group(str(path), mode="r", zarr_format=2)
    return attrs, np.asarray(group[entry["points"]][:]), np.asarray(group[entry["triangles"]][:])


def test_the_writer_adds_the_web_extension_from_the_start(tmp_path):
    dom = _bundle(tmp_path / "b.fenics")
    attrs, points, tris = _web_arrays(tmp_path / "b.fenics")
    assert attrs[WEB_KEY]["schema"] == 1 and attrs["vcell_fenics"]["status"] == "completed"
    assert points.dtype == np.float32 and points.shape == (len(dom.points), 3)
    np.testing.assert_allclose(points, dom.points, rtol=1e-6)  # mesh order: a field row colours it directly
    assert tris.dtype == np.uint32 and len(tris) == 48 and tris.max() < len(points)


def test_export_adds_the_extension_to_a_bundle_without_one(tmp_path):
    from viva_pde_particle.steps.spatial_export import export

    path = tmp_path / "b.fenics"
    _bundle(path)
    attrs = json.loads((path / ".zattrs").read_text())
    del attrs[WEB_KEY]  # as vcell-fenics writes it
    (path / ".zattrs").write_text(json.dumps(attrs))
    shutil.rmtree(path / "web")
    (record,) = export([path])
    assert record["web"] == 1 and record["variables"] == ["u"] and record["times"] == 2
    _, points, tris = _web_arrays(path)
    assert len(tris) == 48
    assert add_web_extension(path)["surfaces"]["cell"]["triangles"] == "web/cell/triangles"  # idempotent


def test_the_compose_runner_announces_a_recorded_bundle(tmp_path, monkeypatch):
    pytest.importorskip("smoldyn")
    from process_bigraph import events

    from viva_pde_particle import compose_runner
    from viva_pde_particle.composites import examples
    from viva_pde_particle.composites.hybrid import build_hybrid_document
    from viva_pde_particle.viz3d import attach_recorder

    doc = attach_recorder(build_hybrid_document(examples.bimolecular_model(n_particles=200), 0.01, 2, seed=3),
                          "/somewhere/else/run.fenics", 0.02)
    pbg = tmp_path / "doc.pbg"
    pbg.write_text(json.dumps(compose_runner._jsonable(doc)))
    announced = []
    real = events.get_emitter()
    monkeypatch.setattr(real, "event", lambda event, **payload: announced.append((event, payload)), raising=False)
    monkeypatch.setattr(events, "get_emitter", lambda: real)
    compose_runner.run(pbg, tmp_path / "output", 0.04)

    bundle = tmp_path / "output" / "run.fenics"  # the document's out_dir, moved under the job's output
    attrs = json.loads((bundle / ".zattrs").read_text())
    assert attrs["vcell_fenics"]["status"] == "completed" and WEB_KEY in attrs
    assert attrs["vcell_fenics"]["times"][-1] == pytest.approx(0.04)  # the final state is recorded at close()
    bundles = [p for name, p in announced if name == "artifact.written" and p.get("kind") == "results-bundle"]
    assert len(bundles) == 1
    assert bundles[0]["uri"] == str(bundle.resolve())
    assert bundles[0]["attributes"]["format"] == "vcell-fenics-bundle" and bundles[0]["attributes"]["web"] == 1
    assert bundles[0]["bytes"] > 0 and "sha256" not in bundles[0]
