"""Serve spatial results bundles to VCell's vtk.wasm field viewer (``vcell/webapp-viewer``).

The viewer is a static page that fetches its data as JSON from the server that served it. In VCell
that is the desktop client's ``FieldViewerServer``. This module is a stdlib HTTP server on loopback
that implements the same contract for results bundles, ported from ``FenicsBundleViews.java``
(the FEniCSx-bundle branch):

``/health``, ``/info``, ``/grid``, ``/field``, ``/stats``, ``/timeseries``
    as VCell serves them (point data, ``"solver": "FEniCSx"``);
``/particles`` (extension)
    the recorded particle positions at a time, for a viewer that draws them (not yet upstream).

Each bundle is registered under a name and opened as ``/?sim=<name>&job=0``. The page itself comes
from a ``webapp-viewer`` checkout, whose ``npm run fetch:vtk-wasm`` must have been run once.

    pixi run view3d path/to/run.fenics [more.fenics ...] [--viewer ../vcell/webapp-viewer] [--port 9125]
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np

DEFAULT_VIEWER = Path(__file__).resolve().parents[2].parent / "vcell" / "webapp-viewer"
WASM_ASSET = Path("assets") / "vtk-wasm" / "vcell-vtk-wasm32-emscripten.tar.gz"


class BadRequest(ValueError):
    pass


class NoSuchDataset(LookupError):
    pass


def _floats(values) -> list:
    """JSON has no NaN/Infinity: non-finite values go out as null (as FieldViewerServer.appendDoubles)."""
    a = np.asarray(values, dtype=float).ravel()
    return [float(v) if np.isfinite(v) else None for v in a]


class _Source:
    """One registered bundle: re-opened per request (a running simulation's times grow), meshes cached."""

    def __init__(self, name: str, path: Path):
        self.name = name
        self.path = Path(path)
        self._meshes: dict = {}
        self._locators: dict = {}
        self._lock = threading.RLock()

    def bundle(self):
        from vcell_fenics.results import Bundle

        return Bundle.open(self.path)

    def mesh(self, bundle, domain):
        with self._lock:
            if domain not in self._meshes:
                self._meshes[domain] = bundle.mesh(domain)
            return self._meshes[domain]

    def measure(self, bundle, domain) -> float:
        from viva_pde_particle.viz3d.bundle import Domain

        m = self.mesh(bundle, domain)
        d = bundle.manifest.domains[domain]
        return float(Domain(domain, d.kind, np.asarray(m.points), np.asarray(m.cells), d.cell_type)
                     .point_weights().sum())

    def locator(self, bundle, domain):
        with self._lock:
            if domain not in self._locators:
                import pyvista as pv

                m = self.mesh(bundle, domain)
                self._locators[domain] = pv.UnstructuredGrid({int(m.cell_types[0]): np.asarray(m.cells)},
                                                             np.asarray(m.points, dtype=float))
            return self._locators[domain]


class BundleViews:
    """The JSON contract over registered bundles (port of ``FenicsBundleViews``)."""

    def __init__(self):
        self.sources: dict[str, _Source] = {}

    def register(self, path, name: str | None = None) -> str:
        path = Path(path)
        name = name or path.name.removesuffix(".fenics")
        self.sources[name] = _Source(name, path)
        return name

    def source(self, q) -> _Source:
        sim = q.get("sim")
        if not sim:
            raise BadRequest("missing required query parameter 'sim'")
        if q.get("job", "0") != "0" or sim not in self.sources:
            raise NoSuchDataset(f"no dataset {sim}:{q.get('job', '0')} is registered with this server")
        return self.sources[sim]

    @staticmethod
    def _open(src):
        b = src.bundle()
        if not b.times:
            raise BadRequest(f"the run {src.name} has not written any output yet")
        return b

    @staticmethod
    def _row(b, q) -> int:
        t = q.get("time")
        if not t:
            return len(b.times) - 1
        return int(np.argmin(np.abs(np.asarray(b.times) - float(t))))

    @staticmethod
    def _domain_of(b, q) -> str:
        d = q.get("domain")
        if not d:
            return b.manifest.variables[0].domain if b.manifest.variables else next(iter(b.manifest.domains))
        if d not in b.manifest.domains:
            raise BadRequest(f"unknown domain '{d}'")
        return d

    @staticmethod
    def _domain_of_variable(b, var) -> str:
        for v in b.manifest.variables:
            if v.name == var:
                return v.domain
        raise BadRequest(f"unknown variable '{var}'")

    @staticmethod
    def _var(q) -> str:
        if not q.get("var"):
            raise BadRequest("missing required query parameter 'var'")
        return q["var"]

    def info(self, q):
        src = self.source(q)
        b = src.bundle()
        m = b.manifest
        return {"simId": src.name, "simName": src.name, "jobIndex": 0, "solver": "FEniCSx", "status": m.status,
                "progress": m.progress, "times": _floats(m.times), "domains": list(m.domains),
                "variables": [{"name": v.name, "domain": v.domain, "location": "point", "isFunction": False}
                              for v in m.variables]}

    def grid(self, q):
        src = self.source(q)
        b = self._open(src)
        domain = self._domain_of(b, q)
        row = self._row(b, q)
        mesh = src.mesh(b, domain)
        d = b.manifest.domains[domain]
        return {"geometryId": f"{src.name}/{domain}", "dimension": 3 if d.gdim >= 3 else 2, "bodyFitted": True,
                "timeIndex": row, "numPoints": int(len(mesh.points)), "points": _floats(mesh.points),
                "cellType": int(mesh.cell_types[0]) if len(mesh.cell_types) else d.cell_type,
                "cells": np.asarray(mesh.cells, dtype=int).tolist(), "domain": domain}

    def field(self, q):
        src = self.source(q)
        b = self._open(src)
        var = self._var(q)
        domain = q.get("domain") or self._domain_of_variable(b, var)
        row = self._row(b, q)
        values = np.asarray(b.field(domain, var, row), dtype=float)
        fin = values[np.isfinite(values)]
        rng = [float(fin.min()), float(fin.max())] if fin.size else [0.0, 0.0]
        return {"geometryId": f"{src.name}/{domain}", "name": var, "domain": domain, "time": float(b.times[row]),
                "location": "point", "values": _floats(values), "range": rng}

    def stats(self, q):
        src = self.source(q)
        b = self._open(src)
        wanted = set(q["var"].split(",")) if q.get("var") else None
        chosen = [v for v in b.manifest.variables if wanted is None or v.name in wanted]
        if not chosen:
            raise BadRequest(f"no variables match {q.get('var')}")
        cols = list(b.manifest.stats_columns)
        series = []
        for v in chosen:
            st = b.stats(v.domain, v.name)
            measure = src.measure(b, v.domain)
            series.append({"name": v.name, "domain": v.domain,
                           **{k: _floats(st[:, cols.index(k)]) for k in ("min", "max", "mean", "total")},
                           "measure": [measure] * len(b.times)})
        return {"times": _floats(b.times), "weighting": "integral", "series": series}

    def _sample(self, src, b, domain, var, points, snap):
        """Per point: (values over time, inDomain, cell, snapped point) by P1 interpolation."""
        mesh = src.mesh(b, domain)
        pts = np.asarray(mesh.points, dtype=float)
        cells = np.asarray(mesh.cells, dtype=int)
        series = b.series(domain, var)  # (T, N)
        out = []
        membrane = b.manifest.domains[domain].kind == "membrane"
        for p in points:
            if membrane:
                k = int(np.argmin(np.linalg.norm(pts - p, axis=1)))
                exact = bool(np.linalg.norm(pts[k] - p) < 1e-9)
                if snap or exact:
                    out.append((series[:, k], True, None, pts[k] if snap and not exact else None))
                else:
                    out.append((np.full(len(series), np.nan), False, None, None))
                continue
            cid = int(np.asarray(src.locator(b, domain).find_containing_cell(p)).ravel()[0])
            if cid < 0:
                out.append((np.full(len(series), np.nan), False, None, None))
                continue
            v = pts[cells[cid]]
            T = (v[1:] - v[0]).T
            lam = np.linalg.solve(T, p - v[0])
            w = np.r_[1 - lam.sum(), lam]
            out.append((series[:, cells[cid]] @ w, True, cid, None))
        return out

    def timeseries(self, q):
        src = self.source(q)
        b = self._open(src)
        var = self._var(q)
        domain = q.get("domain") or self._domain_of_variable(b, var)
        times = _floats(b.times)
        if q.get("points") is not None:
            pts = []
            for entry in q["points"].split(";"):
                if not entry.strip():
                    continue
                parts = entry.split(",")
                if not 2 <= len(parts) <= 3:
                    raise BadRequest(f"malformed point '{entry}' in 'points'; expected x,y,z or, in 2D, x,y")
                pts.append([float(x) for x in parts] + [0.0] * (3 - len(parts)))
            if not pts:
                raise BadRequest("'points' is empty; expected x,y,z;x,y,z;…")
            snap = q.get("snap") == "nearest"
            series = []
            for p, (vals, inside, cell, snapped) in zip(pts, self._sample(src, b, domain, var, np.array(pts), snap)):
                s = {"point": _floats(p)}
                if snapped is not None:
                    s["snapped"] = _floats(snapped)
                if cell is not None:
                    s["cell"] = cell
                s.update({"inDomain": inside, "values": _floats(vals)})
                series.append(s)
            return {"name": var, "domain": domain, "times": times, "location": "point", "series": series}
        if not q.get("x") or not q.get("y"):
            raise BadRequest("a FEniCSx time series is addressed by lab-frame point: 'x' and 'y' are required")
        p = np.array([float(q["x"]), float(q["y"]), float(q.get("z") or 0.0)])
        vals, inside, _, _ = self._sample(src, b, domain, var, p[None, :], False)[0]
        return {"name": var, "domain": domain, "x": p[0], "y": p[1], "z": p[2], "insideCount": int(inside) * len(times),
                "times": times, "values": _floats(vals)}

    def particles(self, q):
        from viva_pde_particle.viz3d.bundle import particle_species, read_particles

        src = self.source(q)
        b = self._open(src)
        row = self._row(b, q)
        species = []
        for s in particle_species(src.path):
            p = read_particles(src.path, s, row)
            species.append({"name": s, "count": int(len(p)), "points": _floats(p)})
        return {"time": float(b.times[row]), "timeIndex": row, "species": species}


ENDPOINTS = ("info", "grid", "field", "stats", "timeseries", "particles")


def make_server(views: BundleViews, viewer_dir: Path | None, host: str = "127.0.0.1", port: int = 9125):
    viewer_dir = Path(viewer_dir) if viewer_dir else None

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, status, body: bytes, ctype: str):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status, obj):
            self._send(status, json.dumps(obj, allow_nan=False).encode(), "application/json")

        def do_GET(self):  # noqa: N802
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query, keep_blank_values=True).items()}
            name = url.path.strip("/")
            if name == "health":
                return self._send(200, b"ok", "text/plain")
            if name in ENDPOINTS:
                try:
                    return self._json(200, getattr(views, name)(q))
                except NoSuchDataset as e:
                    return self._json(404, {"error": str(e)})
                except (BadRequest, ValueError, KeyError, IndexError) as e:
                    return self._json(400, {"error": str(e)})
                except Exception as e:  # noqa: BLE001
                    return self._json(500, {"error": f"{type(e).__name__}: {e}"})
            return self._static(url.path)

        def _static(self, path):
            if viewer_dir is None:
                return self._send(404, b"no viewer directory configured", "text/plain")
            rel = path.lstrip("/") or "index.html"
            target = (viewer_dir / rel).resolve()
            root = viewer_dir.resolve()
            if root not in target.parents or not target.is_file():
                return self._send(HTTPStatus.NOT_FOUND, b"not found", "text/plain")
            ctype = {".mjs": "text/javascript", ".js": "text/javascript", ".wasm": "application/wasm",
                     ".gz": "application/gzip"}.get(target.suffix) or mimetypes.guess_type(target.name)[0]
            return self._send(200, target.read_bytes(), ctype or "application/octet-stream")

    return ThreadingHTTPServer((host, port), Handler)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Serve results bundles to VCell's vtk.wasm field viewer.")
    ap.add_argument("bundles", nargs="+", type=Path)
    ap.add_argument("--viewer", type=Path, default=Path(os.environ.get("VCELL_WEBAPP_VIEWER", DEFAULT_VIEWER)))
    ap.add_argument("--port", type=int, default=9125)
    args = ap.parse_args(argv)
    if not (args.viewer / "index.html").is_file():
        raise SystemExit(f"no webapp-viewer at {args.viewer} (set --viewer or VCELL_WEBAPP_VIEWER)")
    if not (args.viewer / WASM_ASSET).is_file():
        raise SystemExit(f"the vtk.wasm bundle is missing: run `npm run fetch:vtk-wasm` in {args.viewer}")
    views = BundleViews()
    names = [views.register(p) for p in args.bundles]
    server = make_server(views, args.viewer, port=args.port)
    for n in names:
        print(f"http://127.0.0.1:{server.server_address[1]}/?sim={n}&job=0")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
