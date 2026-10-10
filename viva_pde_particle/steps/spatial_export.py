"""SpatialExport: the ad hoc step that leaves a run's spatial output viewable in a browser (compose-api
docs/plan-viewers.md, B1).

The results bundle (vcell-fenics ADR 010) is the format; this step's job is translation into it as *we* define it for
the web: it adds the web extension (``viz3d/web.py``, a precomputed triangle surface per domain) to each bundle that
lacks it, e.g. one vcell-fenics wrote. Bundles from :class:`~viva_pde_particle.viz3d.bundle.SpatialBundleWriter`
already carry it. Its output lists the bundles, for the runner to announce.

It is not a general converter. A translation between two standard formats, or a figure/view step, belongs in its own
published component once a second producer needs one (plan-viewers.md, "Later").

process-bigraph has no run-once-at-the-end hook, so :mod:`viva_pde_particle.compose_runner` calls :func:`export`
after the run and the recorder's ``close()``; in a composite, wire ``bundles`` to a path that changes once.
"""
from __future__ import annotations

import json
from pathlib import Path

from process_bigraph import Step

from viva_pde_particle.viz3d.web import WEB_KEY, add_web_extension


def export(bundles) -> list[dict]:
    """Make each bundle web-viewable; one record per bundle: ``{path, domains, variables, particles, web}``."""
    records = []
    for b in bundles:
        path = Path(b)
        attrs = json.loads((path / ".zattrs").read_text())
        if WEB_KEY not in attrs:
            add_web_extension(path)
            attrs = json.loads((path / ".zattrs").read_text())
        manifest = attrs["vcell_fenics"]
        records.append({
            "path": str(path),
            "status": manifest["status"],
            "times": len(manifest["times"]),
            "domains": sorted(manifest["domains"]),
            "variables": sorted({v["name"] for v in manifest["variables"]}),
            "particles": sorted((attrs.get("particles") or {}).get("species", {})),
            "web": attrs[WEB_KEY]["schema"],
        })
    return records


class SpatialExport(Step):
    config_schema = {}

    def inputs(self):
        return {"bundles": "list[string]"}

    def outputs(self):
        return {"exported": "list[map]"}

    def update(self, state):
        return {"exported": export(state.get("bundles") or [])}
