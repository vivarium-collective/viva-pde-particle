"""Spatial results as vcell-fenics results bundles (ADR 010): VTU meshes + zarr point arrays + a manifest.

The bundle format is the one VCell's field viewer already reads (``FenicsBundle.java``), so any run
written here opens in VCell's vtk.wasm viewer, in pyvista/ParaView (``vcell_fenics.results.export``)
and with :class:`vcell_fenics.results.Bundle`. vcell-fenics' own ``BundleWriter`` gathers from a
dolfinx FunctionSpace; :class:`SpatialBundleWriter` writes the same schema from numpy arrays, so
grid (FV/Q1) fields, P1 mesh fields and native VCell results all land in one format.

Domains are simplicial meshes with P1 point data:

- a **grid** domain is the node lattice cut into 6 tetrahedra per hex (2 triangles per quad in 2D),
  keeping each hex with at least one node in the PDE domain; point values are the node values
  (:func:`grid_domain`);
- a **mesh** domain is the P1 space's own cells, with points in DOF order (:func:`mesh_domain`);
- a **membrane** domain is a triangulated surface (:func:`membrane_domain`).

Particle positions are an extension outside the vcell-fenics manifest (readers ignore it): the root
``.zattrs`` key ``viva_pde_particle.particles`` maps each species to ``particles/<species>/xyz``
``(T, cap, 3)`` (NaN-padded) and ``particles/<species>/count`` ``(T,)``.

Write discipline (as vcell-fenics): a row's arrays are written before its time is appended to the
manifest's ``times``, and the manifest is replaced atomically, so a reader polling a running
simulation only sees complete rows.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

EXTENSION_KEY = "viva_pde_particle"
VTK_TRIANGLE, VTK_TETRA = 5, 10
# Kuhn (Freudenthal) split of a hex into 6 tets along its main diagonal: conforming across neighbours.
_PERMUTATIONS = ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0))


@dataclass
class Domain:
    """A simplicial mesh with P1 point data: points (N, 3), cells (M, k), and its VTK cell type."""

    name: str
    kind: str  # "volume" | "membrane"
    points: np.ndarray
    cells: np.ndarray
    vtk_type: int
    node_index: np.ndarray | None = None  # grid domains: flat grid node of each point

    @property
    def dim(self) -> int:
        return {VTK_TETRA: 3, VTK_TRIANGLE: 2}[self.vtk_type]

    def point_weights(self) -> np.ndarray:
        """Lumped measure per point (tet volume / 4, triangle area / 3), for totals and means."""
        p = self.points[self.cells]
        if self.vtk_type == VTK_TETRA:
            m = np.abs(np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]))) / 6
        else:
            m = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
        w = np.zeros(len(self.points))
        np.add.at(w, self.cells.ravel(), np.repeat(m / self.cells.shape[1], self.cells.shape[1]))
        return w


def _pad3(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    out = np.zeros((len(points), 3))
    out[:, : points.shape[1]] = points
    return out


def grid_domain(name: str, grid, mask: np.ndarray | None = None) -> Domain:
    """The grid's node lattice as tets (3D) or triangles (2D), restricted to hexes touching ``mask``.

    Point values for this domain are ``field.ravel()[domain.node_index]`` (C order = VCell's x-fastest
    global index).
    """
    if grid.dim not in (2, 3):
        raise NotImplementedError("grid domains need a 2D or 3D grid")
    num = np.array(grid.num)  # (nx, ny[, nz])
    flat = np.arange(int(np.prod(num))).reshape(tuple(num[::-1]))  # (nz, ny, nx) / (ny, nx)
    m = np.ones(flat.shape, bool) if mask is None else np.asarray(mask, bool).reshape(flat.shape)
    if grid.dim == 3:
        corner = lambda dx, dy, dz: flat[dz:dz + flat.shape[0] - 1, dy:dy + flat.shape[1] - 1, dx:dx + flat.shape[2] - 1]  # noqa: E731
        mcorner = lambda dx, dy, dz: m[dz:dz + m.shape[0] - 1, dy:dy + m.shape[1] - 1, dx:dx + m.shape[2] - 1]  # noqa: E731
        offs = [(dx, dy, dz) for dz in (0, 1) for dy in (0, 1) for dx in (0, 1)]
        keep = np.zeros(corner(0, 0, 0).shape, bool)
        for o in offs:
            keep |= mcorner(*o)
        tets = []
        for perm in _PERMUTATIONS:
            path, step = [(0, 0, 0)], [0, 0, 0]
            for axis in perm:
                step = list(step)
                step[axis] = 1
                path.append(tuple(step))
            tets.append(np.stack([corner(*p)[keep] for p in path], axis=1))
        cells, vtk_type = np.concatenate(tets), VTK_TETRA
    else:
        corner = lambda dx, dy: flat[dy:dy + flat.shape[0] - 1, dx:dx + flat.shape[1] - 1]  # noqa: E731
        mcorner = lambda dx, dy: m[dy:dy + m.shape[0] - 1, dx:dx + m.shape[1] - 1]  # noqa: E731
        keep = mcorner(0, 0) | mcorner(1, 0) | mcorner(0, 1) | mcorner(1, 1)
        c = {o: corner(*o)[keep] for o in ((0, 0), (1, 0), (0, 1), (1, 1))}
        cells = np.concatenate([np.stack([c[(0, 0)], c[(1, 0)], c[(1, 1)]], 1),
                                np.stack([c[(0, 0)], c[(1, 1)], c[(0, 1)]], 1)])
        vtk_type = VTK_TRIANGLE
    used, cells = np.unique(cells, return_inverse=True)
    cells = cells.reshape(-1, 4 if vtk_type == VTK_TETRA else 3)
    coords = np.stack([c.ravel() for c in grid.node_coordinates()], axis=1)[used]
    dom = Domain(name, "volume", _pad3(coords), cells.astype(np.int64), vtk_type, node_index=used)
    if vtk_type == VTK_TETRA:  # positive orientation
        p = dom.points[dom.cells]
        neg = np.einsum("ij,ij->i", p[:, 1] - p[:, 0], np.cross(p[:, 2] - p[:, 0], p[:, 3] - p[:, 0])) < 0
        dom.cells[neg] = dom.cells[neg][:, [0, 2, 1, 3]]
    return dom


def mesh_domain(name: str, mesh_spec: dict) -> Domain:
    """The P1 mesh engine's domain: its cells over DOFs, points in DOF order (point values = field_dofs)."""
    from viva_pde_particle.mesh import mesh_space

    V = mesh_space(mesh_spec).V
    cells = np.asarray(V.dofmap.list, dtype=np.int64)
    points = _pad3(V.tabulate_dof_coordinates())
    vtk_type = VTK_TETRA if cells.shape[1] == 4 else VTK_TRIANGLE
    return Domain(name, "volume", points, cells, vtk_type)


def membrane_domain(name: str, triangles: np.ndarray, decimals: int = 9) -> Domain:
    """A triangulated surface (m, 3, 3) as a membrane domain with shared vertices."""
    tri = np.asarray(triangles, dtype=float).reshape(-1, 3)
    points, inverse = np.unique(np.round(tri, decimals), axis=0, return_inverse=True)
    return Domain(name, "membrane", points, inverse.reshape(-1, 3).astype(np.int64), VTK_TRIANGLE)


def membrane_variable(var: str, membrane: str) -> str:
    """The bundle name of ``var`` sampled on ``membrane``: ``<var>_<membrane>``, VCell's convention, since
    viewers select variables by name and a volume and a membrane variable must not share one."""
    return f"{var}_{membrane}"


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class SpatialBundleWriter:
    """Write a vcell-fenics schema-1 bundle (fixed meshes) from numpy arrays.

    ::

        w = SpatialBundleWriter(path, source="co-sim B2f")
        w.add_domain(grid_domain("cell", grid, mask), ["B"])
        w.add_domain(membrane_domain("pm", triangles), ["B"])
        w.add_particles(["A"])
        w.open()
        w.write(t, {("cell", "B"): values, ("pm", "B"): values}, particles={"A": xyz})
        w.finalize("completed")
    """

    def __init__(self, path, *, source: str = "", planned_times=(), solver_options: dict | None = None):
        self.path = Path(path)
        self.source = source
        self.planned = tuple(float(t) for t in planned_times)
        self.solver_options = dict(solver_options or {})
        self.domains: dict[str, tuple[Domain, list[str], np.ndarray]] = {}
        self.particle_species: list[str] = []
        self._manifest = None
        self._arrays: dict = {}
        self._rows = 0

    def add_domain(self, domain: Domain, variables) -> None:
        from vcell_fenics.results.writer import _check_name

        _check_name(domain.name)
        if domain.name == "particles":
            raise ValueError("'particles' is reserved for the particle extension")
        for v in variables:
            _check_name(v)
        self.domains[domain.name] = (domain, list(variables), domain.point_weights())

    def add_particles(self, species) -> None:
        self.particle_species = list(species)

    def open(self) -> None:
        import shutil

        import zarr
        from vcell_fenics.results.schema import (
            SCHEMA_VERSION,
            DomainInfo,
            Manifest,
            Segment,
            SolverInfo,
            SourceInfo,
            VariableInfo,
        )
        from vcell_fenics.results.vtu import write_vtu

        if self.path.exists():
            shutil.rmtree(self.path)
        group = zarr.open_group(str(self.path), mode="w", zarr_format=2)  # mode "w" clears the directory
        (self.path / "mesh").mkdir(parents=True)
        rows = max(1, len(self.planned))
        for name, (dom, variables, _) in self.domains.items():
            write_vtu(self.path / "mesh" / f"{name}.vtu", dom.points, dom.cells, dom.vtk_type)
            for v in variables:
                self._arrays[(name, v)] = self._create(group, f"{name}/{v}", (rows, len(dom.points)), ("time", "point"))
                self._arrays[("stats", name, v)] = self._create(group, f"stats/{name}/{v}", (rows, 4), ("time", "stat"))
        for s in self.particle_species:
            self._arrays[("xyz", s)] = self._create(group, f"particles/{s}/xyz", (rows, 1024, 3), ("time", "molecule", "xyz"))
            self._arrays[("count", s)] = self._create(group, f"particles/{s}/count", (rows,), ("time",))
        try:
            from importlib.metadata import version

            ours = version("viva-pde-particle")
        except Exception:  # noqa: BLE001
            ours = "dev"
        self._manifest = Manifest(
            schema=SCHEMA_VERSION, status="running", times=(), planned_times=self.planned,
            segments=(Segment(index=0, t0=self.planned[0] if self.planned else 0.0, count=0),),
            domains={name: DomainInfo(kind=dom.kind, dim=dom.dim, gdim=3, mesh=f"mesh/{name}.vtu",
                                      n_points=len(dom.points), n_cells=len(dom.cells), cell_type=dom.vtk_type)
                     for name, (dom, _, _) in self.domains.items()},
            variables=tuple(VariableInfo(name=v, domain=d, path=f"{d}/{v}", stats=f"stats/{d}/{v}")
                            for d, (_, variables, _) in self.domains.items() for v in variables),
            solver=SolverInfo(version=f"viva-pde-particle {ours}", dolfinx="", mpi_ranks=1,
                              options=self.solver_options),
            source=SourceInfo(kind="viva-pde-particle", file=self.source or None),
            updated=_now(),
        )
        self._publish()

    @staticmethod
    def _create(group, path, shape, dims):
        import numcodecs

        array = group.create_array(path, shape=shape, chunks=(1, *shape[1:]), dtype="<f8",
                                   compressors=numcodecs.Zlib(level=1), fill_value=np.nan, order="C")
        array.attrs["_ARRAY_DIMENSIONS"] = list(dims)
        return array

    def write(self, t: float, values: dict, particles: dict | None = None, progress: float | None = None) -> int:
        """One output row: ``values[(domain, variable)]`` point arrays, ``particles[species]`` (n, ≤3)."""
        if self._manifest is None:
            raise RuntimeError("open() first")
        expected = {(d, v) for d, (_, vs, _) in self.domains.items() for v in vs}
        if set(values) != expected:
            raise ValueError(f"write() needs every (domain, variable) exactly once: {sorted(expected)}")
        row = self._rows
        first = next(iter(self._arrays.values()), None)
        if first is not None and row >= first.shape[0]:
            for a in self._arrays.values():
                a.resize((max(2 * first.shape[0], row + 1), *a.shape[1:]))
        for (d, v), vals in values.items():
            vals = np.asarray(vals, dtype=float).ravel()
            w = self.domains[d][2]
            total = float((w * vals).sum())
            self._arrays[(d, v)][row, :] = vals
            self._arrays[("stats", d, v)][row, :] = (total / w.sum(), total, float(vals.min()), float(vals.max()))
        for s in self.particle_species:
            raw = np.asarray((particles or {}).get(s, np.empty((0, 3))), dtype=float)
            xyz = _pad3(raw.reshape(len(raw), -1)) if raw.size else np.empty((0, 3))
            a = self._arrays[("xyz", s)]
            if len(xyz) > a.shape[1]:
                a.resize((a.shape[0], max(2 * a.shape[1], len(xyz)), 3))
            buf = np.full((a.shape[1], 3), np.nan)
            buf[: len(xyz)] = xyz
            a[row] = buf
            self._arrays[("count", s)][row] = float(len(xyz))
        self._rows += 1
        seg = self._manifest.segments[0]
        self._manifest = replace(self._manifest, times=(*self._manifest.times, float(t)),
                                 segments=(replace(seg, count=self._rows, t0=float(t) if row == 0 else seg.t0),),
                                 progress=self._manifest.progress if progress is None else float(progress),
                                 updated=_now())
        self._publish()
        return row

    def finalize(self, status: str = "completed", message: str | None = None) -> None:
        if self._manifest is None:
            return
        self._manifest = replace(self._manifest, status=status, message=message,
                                 progress=1.0 if status == "completed" else self._manifest.progress, updated=_now())
        self._publish()

    def _publish(self) -> None:
        from vcell_fenics.results.schema import manifest_to_attrs

        attrs = manifest_to_attrs(self._manifest)
        if self.particle_species:
            attrs[EXTENSION_KEY] = {"schema": 1, "particles": {
                s: {"xyz": f"particles/{s}/xyz", "count": f"particles/{s}/count"} for s in self.particle_species}}
        handle, temp = tempfile.mkstemp(dir=self.path, prefix=".zattrs.", suffix=".tmp")
        try:
            with os.fdopen(handle, "w") as stream:
                stream.write(json.dumps(attrs, indent=1, sort_keys=True))
            os.replace(temp, self.path / ".zattrs")
        except BaseException:
            Path(temp).unlink(missing_ok=True)
            raise


def read_particles(path, species: str, row: int) -> np.ndarray:
    """Positions (n, 3) of ``species`` at output row ``row`` of a bundle written here."""
    import zarr

    attrs = json.loads((Path(path) / ".zattrs").read_text())
    ext = attrs.get(EXTENSION_KEY, {}).get("particles", {})
    if species not in ext:
        raise KeyError(f"no particles recorded for {species!r}")
    g = zarr.open_group(str(path), mode="r", zarr_format=2)
    n = int(g[ext[species]["count"]][row])
    return np.asarray(g[ext[species]["xyz"]][row, :n, :])


def particle_species(path) -> list[str]:
    attrs = json.loads((Path(path) / ".zattrs").read_text())
    return sorted(attrs.get(EXTENSION_KEY, {}).get("particles", {}))
