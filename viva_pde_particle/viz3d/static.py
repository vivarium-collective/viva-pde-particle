"""Static 3D figures from spatial results bundles, rendered off-screen with PyVista (PNG, GIF).

- :func:`render_png`: one column per bundle (e.g. native VCell | co-sim | mesh co-sim) on a shared
  colour range. Top row: the mid-plane slice of the volume field with the membrane's outline. Bottom
  row: the field on the membrane, with particles as points.
- :func:`render_gif`: one bundle over time (slice, translucent membrane, particles).

The workbench embeds PNG and GIF study figures directly. PyVista is in the pixi env only.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from viva_pde_particle.viz3d.bundle import membrane_variable, particle_species, read_particles

_PARTICLE_COLORS = ("orangered", "crimson", "purple", "dimgray")
_BAR = {"vertical": False, "position_x": 0.15, "position_y": 0.03, "width": 0.7, "height": 0.07,
        "fmt": "%.3g", "n_labels": 3, "label_font_size": 11, "title_font_size": 12, "color": "black"}


def _cutaway(surface):
    """The surface with the octant facing an isometric camera removed, so the inside is visible."""
    hi = np.asarray(surface.bounds)[1::2]
    c = np.asarray(surface.center)
    return surface.clip_box([c[0], hi[0] + 1, c[1], hi[1] + 1, c[2], hi[2] + 1], invert=True)


def _pv():
    import pyvista as pv

    pv.OFF_SCREEN = True
    return pv


def _domains(bundle):
    vol = [n for n, d in bundle.manifest.domains.items() if d.kind == "volume"]
    mem = [n for n, d in bundle.manifest.domains.items() if d.kind == "membrane"]
    return vol, mem


def domain_mesh(bundle, domain: str, var: str | None = None, row: int | None = None):
    """A bundle domain as a pyvista UnstructuredGrid, with ``var`` at ``row`` as point data."""
    pv = _pv()
    grid = bundle.mesh(domain)
    cells = np.asarray(grid.cells, dtype=np.int64)
    ug = pv.UnstructuredGrid({int(grid.cell_types[0]): cells}, np.asarray(grid.points, dtype=float))
    if var is not None and row is not None:
        names = {v.name for v in bundle.manifest.variables if v.domain == domain}
        name = var if var in names else membrane_variable(var, domain)
        if name in names:
            ug.point_data[var] = bundle.field(domain, name, row)
    return ug


def nearest_row(bundle, t: float) -> int:
    return int(np.argmin(np.abs(np.asarray(bundle.times) - t)))


def _particles(path, row: int, max_points: int, rng):
    out = {}
    for s in particle_species(path):
        p = read_particles(path, s, row)
        if len(p) > max_points:
            p = p[rng.choice(len(p), max_points, replace=False)]
        if len(p):
            out[s] = p
    return out


def _open(path):
    from vcell_fenics.results import Bundle

    return Bundle.open(path)


def render_png(bundles: dict, var: str, t: float, out, *, clim=None, normal: str = "z",
               max_particles: int = 1500, window=(420, 380), title: str | None = None, seed: int = 0) -> Path:
    """Side-by-side panels per bundle (label → path) of ``var`` at the output nearest ``t``."""
    pv = _pv()
    rng = np.random.default_rng(seed)
    opened = {label: _open(path) for label, path in bundles.items()}
    scenes = {}
    lo, hi = np.inf, -np.inf
    for label, b in opened.items():
        row = nearest_row(b, t)
        vol, mem = _domains(b)
        v = domain_mesh(b, vol[0], var, row)
        m = domain_mesh(b, mem[0], var, row) if mem else v.extract_surface(algorithm=None)
        scenes[label] = (b.times[row], v, m, _particles(bundles[label], row, max_points=max_particles, rng=rng))
        vals = v.point_data[var]
        lo, hi = min(lo, float(np.nanmin(vals))), max(hi, float(np.nanmax(vals)))
    clim = clim or (lo, hi if hi > lo else lo + 1e-12)
    n = len(scenes)
    p = pv.Plotter(shape=(2, n), off_screen=True, window_size=(window[0] * n, window[1] * 2), border=False)
    for j, (label, (tt, v, m, parts)) in enumerate(scenes.items()):
        center = np.asarray(v.center)
        sl = v.slice(normal=normal, origin=center)
        p.subplot(0, j)
        p.add_mesh(sl, scalars=var, clim=clim, cmap="viridis", show_scalar_bar=True,
                   scalar_bar_args={"title": f"{var} (µM)", **_BAR})
        outline = m.slice(normal=normal, origin=center) if m.n_cells else None
        if outline is not None and outline.n_points:
            p.add_mesh(outline, color="black", line_width=2)
        p.add_text(f"{label}\nt = {tt:g} s (mid-plane {normal})", font_size=9)
        {"x": p.view_yz, "y": p.view_xz, "z": p.view_xy}[normal]()
        p.subplot(1, j)
        cut = _cutaway(m)
        if var in m.point_data:
            p.add_mesh(cut, scalars=var, clim=clim, cmap="viridis", show_scalar_bar=False)
        else:
            p.add_mesh(cut, color="lightgrey")
        for k, (s, pts) in enumerate(parts.items()):
            p.add_mesh(pv.PolyData(pts), color=_PARTICLE_COLORS[k % len(_PARTICLE_COLORS)], point_size=4,
                       render_points_as_spheres=True)
        names = ", ".join(f"{s} ({_PARTICLE_COLORS[k % len(_PARTICLE_COLORS)]})" for k, s in enumerate(parts))
        p.add_text(f"{var} on the membrane (cutaway)" + (f"\nparticles: {names}" if parts else ""), font_size=9)
        p.view_isometric()
    if title:
        p.subplot(0, 0)
        p.add_text(title, position="upper_right", font_size=10)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    p.screenshot(str(out))
    p.close()
    return out


def render_gif(path, var: str, out, *, rows=None, clim=None, normal: str = "z", max_particles: int = 3000,
               window=(640, 520), fps: int = 4, seed: int = 0) -> Path:
    """A time animation of one bundle: mid-plane slice, translucent membrane, particles."""
    pv = _pv()
    rng = np.random.default_rng(seed)
    b = _open(path)
    vol, mem = _domains(b)
    rows = list(range(len(b.times))) if rows is None else list(rows)
    if clim is None:
        st = b.stats(vol[0], var)
        clim = (float(np.nanmin(st[:, 2])), float(np.nanmax(st[:, 3])))
        if clim[1] <= clim[0]:
            clim = (clim[0], clim[0] + 1e-12)
    v = domain_mesh(b, vol[0])
    m = domain_mesh(b, mem[0]) if mem else v.extract_surface(algorithm=None)
    center = np.asarray(v.center)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    p = pv.Plotter(off_screen=True, window_size=window)
    p.open_gif(str(out), fps=fps)
    for row in rows:
        p.clear()
        v.point_data[var] = b.field(vol[0], var, row)
        p.add_mesh(v.slice(normal=normal, origin=center), scalars=var, clim=clim, cmap="viridis",
                   scalar_bar_args={"title": f"{var} (µM)", **_BAR})
        p.add_mesh(_cutaway(m), color="lightgrey", opacity=0.25)
        for k, (s, pts) in enumerate(_particles(path, row, max_particles, rng).items()):
            p.add_mesh(pv.PolyData(pts), color=_PARTICLE_COLORS[k % len(_PARTICLE_COLORS)], point_size=3,
                       render_points_as_spheres=True)
        p.add_text(f"t = {b.times[row]:g} s", font_size=11)
        p.view_isometric()
        p.write_frame()
    p.close()
    return out
