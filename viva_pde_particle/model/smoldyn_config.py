"""Write a Smoldyn configuration file for the particle half of a partitioned model.

Field-dependent rates use the VCell hybrid syntax that ``smoldyn.HybridGrid`` reads
(``reaction r A -> 0 0.5*B;``). The domain has reflective boundaries, matching the
PDE side's zero-flux boundaries. A surface and compartment covering the domain are
added when a 0th-order reaction needs one: Smoldyn's grid-based 0th-order creation
only runs for compartment reactions (``reaction_cmpt``).
"""
from __future__ import annotations

import numpy as np

from viva_pde_particle.grid import CartesianGrid

DOMAIN = "domain"


def _fmt(x: float) -> str:
    return repr(float(x))


def _rate(rx: dict) -> str:
    if not rx["fields"]:
        return _fmt(rx["rate_constant"])
    return "*".join([_fmt(rx["rate_constant"])] + rx["fields"]) + ";"


def _domain_surface(grid: CartesianGrid) -> list[str]:
    lines = ["start_surface walls", "action both all reflect"]
    lo = grid.origin
    hi = [o + s for o, s in zip(grid.origin, grid.size)]
    for d in range(grid.dim):
        # rect panel normal to axis d at lo/hi; its corner is the domain's low corner
        # (high face shifted to hi[d]) and it spans the other axes.
        others = [e for e in range(grid.dim) if e != d]
        for sign, pos in (("+", lo[d]), ("-", hi[d])):
            corner = list(lo)
            corner[d] = pos
            extent = [grid.size[e] for e in others]
            lines.append(f"panel rect {sign}{d} " + " ".join(_fmt(c) for c in corner) + " "
                         + " ".join(_fmt(x) for x in extent))
    lines.append("end_surface")
    center = [o + 0.5 * s for o, s in zip(grid.origin, grid.size)]
    lines += [f"start_compartment {DOMAIN}", "surface walls",
              "point " + " ".join(_fmt(c) for c in center), "end_compartment"]
    return lines


def _molecules(grid: CartesianGrid, name: str, counts: np.ndarray) -> list[str]:
    """`mol n A lo-hi ...` per grid node, placing molecules uniformly in each node's cell."""
    bounds = [grid.element_bounds(d) for d in range(grid.dim)]
    lines = []
    for idx in zip(*np.nonzero(counts)):
        n = int(counts[idx])
        ranges = []
        for d in range(grid.dim):
            i = idx[grid.dim - 1 - d]
            lo, hi = bounds[d][0][i], bounds[d][1][i]
            ranges.append(f"{_fmt(lo)}-{_fmt(hi)}" if hi > lo else _fmt(lo))
        lines.append(f"mol {n} {name} " + " ".join(ranges))
    return lines


def _sphere_surface(center, radius) -> list[str]:
    c = " ".join(_fmt(v) for v in center)
    return ["start_surface walls", "action both all reflect", f"panel sph {c} {_fmt(radius)} 30 30",
            "end_surface", f"start_compartment {DOMAIN}", "surface walls", f"point {c}", "end_compartment"]


def _volume_samples_block(vs: dict) -> list[str]:
    """``highResVolumeSamples``: a voxel map of compartment IDs (VCell's acceleration in OPTION_VCELL).

    ``vs`` has ``origin``, ``size``, ``num`` (3 each) and ``ids`` (uint8, x fastest), with 0 for
    ``domain`` and 1 for outside. ``posincompart`` decides membership from the 3×3×3 sample
    neighbourhood and runs the exact panel-crossing test only next to the surface.
    """
    import zlib

    ids = np.ascontiguousarray(vs["ids"], dtype=np.uint8)
    if ids.size != int(np.prod(vs["num"])):
        raise ValueError("volume_samples: ids size does not match num")
    hexdata = zlib.compress(ids.tobytes(), 9).hex().upper()
    lines = ["start_highResVolumeSamples",
             "Origin " + " ".join(_fmt(v) for v in vs["origin"]),
             "Size " + " ".join(_fmt(v) for v in vs["size"]),
             "CompartmentHighResPixelMap 2", f"{DOMAIN} 0", "outside 1",
             "VolumeSamples " + " ".join(str(int(v)) for v in vs["num"])]
    lines += [hexdata[i:i + 200] for i in range(0, len(hexdata), 200)]
    return lines + ["end_highResVolumeSamples"]


def _triangle_surface(triangles, interior_point) -> list[str]:
    """A closed reflecting surface of triangle panels (e.g. a mesh boundary) and the domain inside it."""
    lines = ["start_surface walls", "action both all reflect"]
    lines += ["panel tri " + " ".join(_fmt(v) for v in np.asarray(t, dtype=float).ravel()) for t in triangles]
    pts = np.atleast_2d(np.asarray(interior_point, dtype=float))  # one point, or several for non-convex domains
    points = ["point " + " ".join(_fmt(v) for v in p) for p in pts]
    return lines + ["end_surface", f"start_compartment {DOMAIN}", "surface walls", *points, "end_compartment"]


def write_smoldyn_config(
    particles: dict,
    grid: CartesianGrid,
    initial_counts: dict[str, np.ndarray],
    time_step: float,
    seed: int,
    geometry: dict | None = None,
    positions: dict[str, np.ndarray] | None = None,
    box_size: float | None = None,
) -> str:
    """Return the configuration text for the particle half (from :func:`partition`).

    ``geometry``: None (the grid box), ``{"kind": "sphere", "center", "radius"}`` or
    ``{"kind": "triangles", "triangles": (m, 3, 3), "interior_point"}`` (one point, or an (n, 3) array
    for non-convex domains) (e.g. the boundary of the
    PDE mesh, so particles and fields share one domain). Either adds a reflecting surface
    and the ``domain`` compartment inside it. ``triangles`` may add ``"volume_samples"``
    (see :func:`_volume_samples_block`) to accelerate compartment tests.
    ``positions``: explicit initial molecule positions per species (n, dim), which replace
    the per-node placement from ``initial_counts``.
    ``box_size``: edge of Smoldyn's virtual boxes (µm). The default is two grid spacings.
    Without it Smoldyn sizes boxes from the *initial* molecule count (``molperbox``), and a
    model that starts with no particles gets a single box. Every molecule then tests every
    surface panel every step: N×M collision checks, e.g. 23× slower with an 806-triangle
    membrane.
    """
    if any(o < 0 for o in grid.origin):
        # Smoldyn writes position ranges as 'lo-hi', which is ambiguous for negative values.
        raise ValueError("grid origin must be non-negative for Smoldyn 'mol lo-hi' placement")
    species = list(particles["species"])
    lines = [
        "# generated by viva_pde_particle.model.smoldyn_config",
        f"dim {grid.dim}",
    ]
    for d in range(grid.dim):
        lines.append(f"boundaries {d} {_fmt(grid.origin[d])} {_fmt(grid.origin[d] + grid.size[d])} r")
    if species:
        lines.append("species " + " ".join(species))
        for s, spec in particles["species"].items():
            lines.append(f"difc {s} {_fmt(spec['diffusion'])}")
    lines += [
        "time_start 0",
        "time_stop 1e30",
        f"time_step {_fmt(time_step)}",
        f"random_seed {int(seed)}",
        f"boxsize {_fmt(box_size if box_size is not None else 2 * max(grid.spacing))}",
    ]
    reactions = particles["reactions"]
    if geometry is not None:
        kind = geometry.get("kind")
        if kind == "sphere":
            lines += _sphere_surface(geometry["center"], geometry["radius"])
        elif kind == "triangles":
            if geometry.get("volume_samples") is not None:
                lines += _volume_samples_block(geometry["volume_samples"])
            lines += _triangle_surface(geometry["triangles"], geometry["interior_point"])
        else:
            raise ValueError(f"unsupported geometry kind {kind!r}")
    elif any(rx["order"] == 0 and rx["fields"] for rx in reactions):
        lines += _domain_surface(grid)
    for rx in reactions:
        rct = " + ".join(rx["reactants"]) or "0"
        prd = " + ".join(rx["products"]) or "0"
        keyword = f"reaction_cmpt {DOMAIN}" if (rx["order"] == 0 and (rx["fields"] or geometry)) else "reaction"
        lines.append(f"{keyword} {rx['name']} {rct} -> {prd} {_rate(rx)}")
    for s in species:
        if positions is not None and s in positions:
            lines += [f"mol 1 {s} " + " ".join(_fmt(v) for v in p) for p in positions[s]]
        else:
            lines += _molecules(grid, s, initial_counts.get(s, np.zeros(grid.shape)))
    lines.append("end_file")
    return "\n".join(lines) + "\n"
