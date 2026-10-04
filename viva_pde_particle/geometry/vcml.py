"""Adapters between vcell-fenics ``GeometryDescription`` and VCell's ``Geometry`` (pyvcell VCML).

- :func:`to_vcml_geometry`: GeometryDescription → ``pyvcell.vcml.Geometry``, so the native VCell
  solver and the VCell-FV geometry compiler see the same geometry as the FEniCS one.
- :func:`from_vcml_geometry`: the reverse, through vcell-fenics' own importer.

Analytic subvolumes are supported; CSG and image subvolumes come later. Subvolume order is kept,
and it is VCell's priority order (the first subvolume whose predicate holds owns a point).

**Coverage.** VCell requires every point of the domain to belong to some subvolume. A
description written with complementary strict predicates (``r² < R²`` / ``r² > R²``) leaves
points exactly on the interface unowned, and grid nodes can land there (libvcell then fails with
``null``). The last, lowest-priority subvolume is therefore written as ``1.0``. Under VCell's
priority rule this owns exactly the points no earlier subvolume owns: its own region plus any
gaps. Every earlier subvolume is unchanged.
"""
from __future__ import annotations

import re

_COORD = {"0": "x", "1": "y", "2": "z"}


def vcell_expression(expression: str) -> str:
    """A GeometryDescription predicate in VCell expression syntax.

    ``geom.x[i]`` becomes ``x``/``y``/``z`` and ``**`` becomes ``^``. The comparison and logic
    operators (``<``, ``&&``, ``||``) are already shared.
    """
    out = re.sub(r"geom\.x\[\s*([012])\s*\]", lambda m: _COORD[m.group(1)], expression)
    return out.replace("**", "^")


def to_vcml_geometry(description, name: str | None = None):
    """A ``pyvcell.vcml.Geometry`` for a 3D GeometryDescription with analytic subvolumes."""
    import pyvcell.vcml as vc
    from pyvcell.vcml.models_geometry import SubVolume, SubVolumeType

    if description.dim != 3:
        raise NotImplementedError("only 3D geometries are supported (VCell spatial stochastic needs 3D)")
    geo = vc.Geometry(name=name or description.name, dim=3, extent=tuple(description.extent),
                      origin=tuple(description.origin))
    last = len(description.subvolumes) - 1
    for handle, sv in enumerate(description.subvolumes):
        if sv.type != "analytic":
            raise NotImplementedError(f"subvolume {sv.name!r}: {sv.type} subvolumes are not supported yet")
        expr = "1.0" if handle == last else vcell_expression(sv.expression)  # see "Coverage" above
        geo.subvolumes.append(SubVolume(name=sv.name, handle=handle, subvolume_type=SubVolumeType.analytic,
                                        analytic_expr=expr))
    for sc in description.surfaces:
        geo.add_surface(sc.name, sc.inside, sc.outside)
    return geo


def from_vcml_geometry(geometry, name: str | None = None):
    """A GeometryDescription from a ``pyvcell.vcml.Geometry`` (vcell-fenics ``import_geometry``)."""
    from vcell_fenics.pyvcell_bridge.geometry import import_geometry

    return import_geometry(geometry, name=name)
