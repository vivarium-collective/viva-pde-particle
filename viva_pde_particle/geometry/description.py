"""Geometry descriptions: vcell-fenics ``GeometryDescription`` behind thin adapters.

The format is vcell-fenics' (``vcell_fenics.formalism.geometry_schema``), a faithful mirror of
VCell's ``Geometry``. It is new, so this module is the only place that touches it directly:
builders, a plain-dict carrier for process configs, and (later) adapters from VCML ``Geometry``
(pyvcell) and SBML-Spatial.
"""
from __future__ import annotations

from typing import Any


def sphere_in_box(center, radius: float, extent, origin=(0.0, 0.0, 0.0), *, inside: str = "cell",
                  outside: str = "ec", membrane: str = "pm", name: str = "sphere_in_box"):
    """An analytic sphere ``inside`` a box ``outside``, separated by surface class ``membrane``."""
    from vcell_fenics.formalism.geometry_schema import GeometryDescription, SubVolume, SurfaceClass

    cx, cy, cz = (float(c) for c in center)
    r2 = float(radius) ** 2
    q = f"(geom.x[0]-{cx})**2 + (geom.x[1]-{cy})**2 + (geom.x[2]-{cz})**2"
    return GeometryDescription(
        name=name, dim=3, extent=tuple(float(e) for e in extent), origin=tuple(float(o) for o in origin),
        subvolumes=(SubVolume(name=inside, type="analytic", expression=f"{q} < {r2}"),
                    SubVolume(name=outside, type="analytic", expression=f"{q} > {r2}")),
        surfaces=(SurfaceClass(name=membrane, inside=inside, outside=outside),),
    )


def description_to_config(description) -> dict[str, Any]:
    """Plain-dict carrier (vcell-fenics ``geometry_to_dict``), safe in a process config."""
    from vcell_fenics.formalism.geometry_io import geometry_to_dict

    return geometry_to_dict(description)


def description_from_config(config: dict[str, Any]):
    from vcell_fenics.formalism.geometry_io import load_geometry_dict

    return load_geometry_dict(config)
