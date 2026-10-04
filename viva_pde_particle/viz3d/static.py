"""Static 3D figures from spatial results bundles, rendered off-screen with PyVista (PNG, GIF).

- :func:`render_png`: one column per bundle (e.g. native VCell | co-sim | mesh co-sim), or per
  (bundle, time), on a shared colour range. Panel ``"slice"``: the mid-plane slice of the volume field
  with the membrane's outline. Panel ``"membrane"``: the field on the cut-away membrane, with particles
  as points. Without a membrane panel the particles are drawn on the slice (e.g. channels in a slab).
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


def _face_on(p, normal: str, bounds, window) -> None:
    """Look at the ``normal`` plane face-on, filling the panel (parallel projection, fitted to ``bounds``)."""
    {"x": p.view_yz, "y": p.view_xz, "z": p.view_xy}[normal]()
    b = np.asarray(bounds, dtype=float).reshape(3, 2)
    axes = {"x": (1, 2), "y": (0, 2), "z": (0, 1)}[normal]
    w, h = (b[axes[0], 1] - b[axes[0], 0]), (b[axes[1], 1] - b[axes[1], 0])
    p.enable_parallel_projection()
    p.camera.parallel_scale = 0.5 * max(1.45 * h, 1.05 * w * window[1] / window[0])  # room for the labels above


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


def _color(species: str, k: int, colors: dict | None) -> str:
    return (colors or {}).get(species, _PARTICLE_COLORS[k % len(_PARTICLE_COLORS)])


def _add_particles(p, pv, parts: dict, colors: dict | None, size: int) -> str:
    for k, (s, pts) in enumerate(parts.items()):
        p.add_mesh(pv.PolyData(pts), color=_color(s, k, colors), point_size=size, render_points_as_spheres=True)
    return ", ".join(f"{s} ({_color(s, k, colors)})" for k, s in enumerate(parts))


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


def render_png(bundles: dict, var: str, t: float | None, out, *, clim=None, normal: str = "z",
               panels=("slice", "membrane"), colors: dict | None = None, cmap: str = "viridis", units: str = "µM",
               max_particles: int = 1500, window=(420, 380), title: str | None = None, seed: int = 0,
               stack: str = "columns", slice_origin=None, log_scale: bool = False) -> Path:
    """Panels per bundle of ``var`` at the output nearest ``t``.

    ``stack="rows"`` puts each bundle on its own row (wide domains such as a slab). ``slice_origin``
    places the slice (default: the domain's centre). ``log_scale`` colours on a log scale (sparks).

    ``bundles`` maps a label to a bundle path, or to ``(path, time)`` for one column per time.
    ``colors`` maps particle species to colours.
    """
    pv = _pv()
    rng = np.random.default_rng(seed)
    scenes = {}
    lo, hi = np.inf, -np.inf
    for label, spec in bundles.items():
        path, tt = (spec, t) if not isinstance(spec, tuple) else spec
        b = _open(path)
        row = nearest_row(b, tt)
        vol, mem = _domains(b)
        v = domain_mesh(b, vol[0], var, row)
        m = domain_mesh(b, mem[0], var, row) if mem else v.extract_surface(algorithm=None)
        scenes[label] = (b.times[row], v, m if mem else None, _particles(path, row, max_points=max_particles, rng=rng))
        vals = v.point_data[var]
        lo, hi = min(lo, float(np.nanmin(vals))), max(hi, float(np.nanmax(vals)))
    clim = clim or (lo, hi if hi > lo else lo + 1e-12)
    n, rows = len(scenes), list(panels)
    shape = (len(rows), n) if stack == "columns" else (n, len(rows))
    p = pv.Plotter(shape=shape, off_screen=True, window_size=(window[0] * shape[1], window[1] * shape[0]),
                   border=False)
    for j, (label, (tt, v, m, parts)) in enumerate(scenes.items()):
        center = np.asarray(v.center if slice_origin is None else slice_origin, dtype=float)
        for i, panel in enumerate(rows):
            p.subplot(i, j) if stack == "columns" else p.subplot(j, i)
            if panel == "slice":
                p.add_mesh(v.slice(normal=normal, origin=center), scalars=var, clim=clim, cmap=cmap, log_scale=log_scale,
                           show_scalar_bar=True, scalar_bar_args={"title": f"{var} ({units})", **_BAR})
                outline = m.slice(normal=normal, origin=center) if m is not None and m.n_cells else None
                if outline is not None and outline.n_points:
                    p.add_mesh(outline, color="black", line_width=2)
                names = "" if "membrane" in rows else _add_particles(p, pv, parts, colors, 6)
                where = f"mid-plane {normal}" if slice_origin is None else f"{normal} = {center['xyz'.index(normal)]:g}"
                when = "" if "t = " in label else f"t = {tt:g} s, "
                p.add_text(f"{label}\n{when}{where}" + (f"; particles: {names}" if names else ""), font_size=9)
                _face_on(p, normal, v.bounds, window)
            elif panel == "membrane":
                surface = m if m is not None else v.extract_surface(algorithm=None)
                cut = _cutaway(surface)
                if var in surface.point_data:
                    p.add_mesh(cut, scalars=var, clim=clim, cmap=cmap, show_scalar_bar=False)
                else:
                    p.add_mesh(cut, color="lightgrey")
                names = _add_particles(p, pv, parts, colors, 4)
                p.add_text(f"{var} on the membrane (cutaway)" + (f"\nparticles: {names}" if names else ""),
                           font_size=9)
                p.view_isometric()
            else:
                raise ValueError(f"unknown panel {panel!r} (use 'slice' or 'membrane')")
    if title:
        p.subplot(0, 0)
        p.add_text(title, position="upper_right", font_size=10)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    p.screenshot(str(out))
    p.close()
    return out


def render_gif(path, var: str, out, *, rows=None, clim=None, normal: str = "z", max_particles: int = 3000,
               window=(640, 520), fps: int = 4, seed: int = 0, view: str = "isometric", colors: dict | None = None,
               cmap: str = "viridis", units: str = "µM", slice_origin=None, log_scale: bool = False) -> Path:
    """A time animation of one bundle: mid-plane slice, translucent membrane, particles.

    ``view="slice"`` looks at the slice face-on (a slab), without the membrane.
    """
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
    center = np.asarray(v.center if slice_origin is None else slice_origin, dtype=float)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    p = pv.Plotter(off_screen=True, window_size=window)
    p.open_gif(str(out), fps=fps)
    for row in rows:
        p.clear()
        v.point_data[var] = b.field(vol[0], var, row)
        p.add_mesh(v.slice(normal=normal, origin=center), scalars=var, clim=clim, cmap=cmap, log_scale=log_scale,
                   scalar_bar_args={"title": f"{var} ({units})", **_BAR})
        if view == "isometric":
            p.add_mesh(_cutaway(m), color="lightgrey", opacity=0.25)
        names = _add_particles(p, pv, _particles(path, row, max_particles, rng), colors, 6 if view == "slice" else 3)
        p.add_text(f"t = {b.times[row]:g} s" + (f"\nparticles: {names}" if names else ""), font_size=11)
        if view == "slice":
            _face_on(p, normal, v.bounds, window)
        else:
            p.view_isometric()
        p.write_frame()
    p.close()
    return out
