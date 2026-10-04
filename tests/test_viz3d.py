"""Phase 8b/8c: static PyVista figures and the self-contained three.js page, from a recorded bundle."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("vcell_fenics")
pv = pytest.importorskip("pyvista")

from viva_pde_particle.grid import CartesianGrid  # noqa: E402
from viva_pde_particle.viz3d import SpatialBundleWriter, grid_domain, membrane_domain  # noqa: E402


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    path = tmp_path_factory.mktemp("viz") / "ball.fenics"
    g = CartesianGrid((0, 0, 0), (4, 4, 4), (17, 17, 17))
    x, y, z = g.node_coordinates()
    r = np.sqrt((x - 2) ** 2 + (y - 2) ** 2 + (z - 2) ** 2)
    mask = r < 1.6
    vol = grid_domain("cell", g, mask)
    sphere = pv.Sphere(radius=1.5, center=(2, 2, 2), theta_resolution=24, phi_resolution=24).triangulate()
    mem = membrane_domain("pm", sphere.points[sphere.regular_faces])
    w = SpatialBundleWriter(path)
    w.add_domain(vol, ["B"])
    w.add_domain(mem, ["B"])
    w.add_particles(["A"])
    w.open()
    rng = np.random.default_rng(0)
    for k, t in enumerate(np.linspace(0, 1, 5)):
        field = np.exp(-r ** 2) * t
        mr = np.linalg.norm(mem.points - 2, axis=1)
        w.write(t, {("cell", "B"): field.ravel()[vol.node_index], ("pm", "B"): np.exp(-mr ** 2) * t},
                particles={"A": 2 + rng.normal(scale=0.4, size=(500 - 50 * k, 3))})
    w.finalize()
    return path


def _can_render():
    import os
    import sys

    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return False  # headless Linux: VTK may abort instead of raising
    try:
        pv.OFF_SCREEN = True
        p = pv.Plotter(off_screen=True, window_size=(64, 64))
        p.add_mesh(pv.Sphere())
        p.screenshot(None, return_img=True)
        p.close()
        return True
    except Exception:  # noqa: BLE001
        return False


def test_png_and_gif(bundle, tmp_path):
    if not _can_render():
        pytest.skip("no off-screen render backend")
    from viva_pde_particle.viz3d.static import render_gif, render_png

    png = render_png({"a": bundle, "b": bundle}, "B", 1.0, tmp_path / "f.png")
    assert png.stat().st_size > 10_000
    gif = render_gif(bundle, "B", tmp_path / "f.gif")
    assert gif.read_bytes()[:3] == b"GIF"


def test_html_page_is_self_contained_and_small(bundle, tmp_path):
    from viva_pde_particle.viz3d.html import bundle_html

    page = bundle_html({"run": bundle}, "B", tmp_path / "p.html", max_frames=3)
    text = page.read_text()
    assert "three.min.js" in text and '"run"' in text
    assert page.stat().st_size < 2_000_000
    import json
    import re

    data = json.loads(re.search(r"const DATA = (\{.*?\}), VAR", text, re.S).group(1))
    run = data["run"]
    assert run["times"] == [0.0, 0.5, 1.0]  # 5 rows subsampled to 3, first and last kept
    assert set(run["slices"]) == {"x", "y", "z"} and len(run["particles"]["A"]) == 3
    assert run["range"][1] == pytest.approx(1.0, rel=1e-6)
