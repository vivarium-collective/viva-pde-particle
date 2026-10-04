"""Phase 8d: the bundle server for VCell's vtk.wasm field viewer (the FieldViewerServer JSON contract)."""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

pytest.importorskip("vcell_fenics")
pv = pytest.importorskip("pyvista")

from viva_pde_particle.grid import CartesianGrid  # noqa: E402
from viva_pde_particle.viz3d import SpatialBundleWriter, grid_domain, membrane_domain  # noqa: E402
from viva_pde_particle.viz3d.viewer_server import BundleViews, make_server  # noqa: E402


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    path = tmp_path_factory.mktemp("srv") / "run.fenics"
    g = CartesianGrid((0, 0, 0), (2, 2, 2), (5, 5, 5))
    vol = grid_domain("cell", g)
    s = pv.Sphere(radius=0.8, center=(1, 1, 1), theta_resolution=12, phi_resolution=12).triangulate()
    mem = membrane_domain("pm", s.points[s.regular_faces])
    w = SpatialBundleWriter(path)
    w.add_domain(vol, ["B"])
    w.add_domain(mem, ["B"])
    w.add_particles(["A"])
    w.open()
    for t in (0.0, 1.0, 2.0):  # B = t·(x + 2y + 3z): linear, so P1 interpolation is exact
        w.write(t, {("cell", "B"): t * (vol.points @ [1, 2, 3]), ("pm", "B"): t * (mem.points @ [1, 2, 3])},
                particles={"A": np.full((int(3 - t), 3), 1.0)})
    w.finalize()
    views = BundleViews()
    views.register(path)
    srv = make_server(views, None, port=0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", vol, mem
    srv.shutdown()


def get(base, path):
    try:
        with urllib.request.urlopen(base + path) as r:
            return r.status, json.loads(r.read()) if path != "/health" else r.read()
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_info_grid_field(server):
    base, vol, mem = server
    status, info = get(base, "/info?sim=run&job=0")
    assert status == 200 and info["solver"] == "FEniCSx" and info["times"] == [0.0, 1.0, 2.0]
    assert info["particleSpecies"] == ["A"]
    assert info["status"] == "completed" and info["domains"] == ["cell", "pm"]
    assert {"name": "B", "domain": "pm", "location": "point", "isFunction": False} in info["variables"]
    _, grid = get(base, "/grid?sim=run&job=0&domain=cell&time=1")
    assert grid["geometryId"] == "run/cell" and grid["bodyFitted"] and grid["dimension"] == 3 and grid["cellType"] == 10
    assert grid["numPoints"] == len(vol.points) and len(grid["points"]) == 3 * len(vol.points)
    assert np.asarray(grid["cells"]).shape == vol.cells.shape and grid["timeIndex"] == 1
    _, field = get(base, "/field?sim=run&job=0&var=B&domain=pm&time=2.2")
    assert field["time"] == 2.0 and field["geometryId"] == "run/pm" and field["location"] == "point"
    np.testing.assert_allclose(field["values"], 2 * (mem.points @ [1, 2, 3]))
    assert field["range"] == pytest.approx([min(field["values"]), max(field["values"])])


def test_stats_timeseries_particles(server):
    base, vol, _ = server
    _, st = get(base, "/stats?sim=run&job=0&var=B")
    assert st["weighting"] == "integral" and len(st["series"]) == 2  # B on cell and on pm
    cell = next(s for s in st["series"] if s["domain"] == "cell")
    assert cell["measure"][0] == pytest.approx(8.0) and cell["mean"][1] == pytest.approx(6.0)
    _, ts = get(base, "/timeseries?sim=run&job=0&var=B&domain=cell&points=0.3,0.7,1.1;9,9,9")
    inside, outside = ts["series"]
    assert inside["inDomain"] and outside["inDomain"] is False and outside["values"] == [None] * 3
    np.testing.assert_allclose(inside["values"], np.array([0, 1, 2]) * (0.3 + 1.4 + 3.3))
    _, one = get(base, "/timeseries?sim=run&job=0&var=B&domain=cell&x=1&y=1&z=1")
    np.testing.assert_allclose(one["values"], [0, 6, 12])
    _, parts = get(base, "/particles?sim=run&job=0&time=1")
    assert parts["species"][0]["count"] == 2 and len(parts["species"][0]["points"]) == 6
    _, capped = get(base, "/particles?sim=run&job=0&time=0&max=2")
    assert capped["species"][0]["count"] == 3 and capped["species"][0]["shown"] == 2


def test_errors_and_health(server):
    base, _, _ = server
    assert get(base, "/health") == (200, b"ok")
    assert get(base, "/info?sim=nope&job=0")[0] == 404
    assert get(base, "/field?sim=run&job=0")[0] == 400  # missing var
    assert get(base, "/grid?sim=run&job=0&domain=nope")[0] == 400
