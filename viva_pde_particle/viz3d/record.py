"""Record a hybrid run (or a native VCell trajectory) into a spatial results bundle.

:class:`BundleRecorder` turns store snapshots into bundle rows for one layout:

- ``{"kind": "grid", "grid": cfg, "mask": bool array | None, "fold": int array | None}``: FV/Q1
  fields on the grid, written on the lattice tets (:func:`~viva_pde_particle.viz3d.bundle.grid_domain`).
  Exterior band nodes take their fold target's value, as the particle engine sees them
  (``ExtendGridField``).
- ``{"kind": "mesh", "mesh": spec}``: P1 ``field_dofs`` on the mesh engine's own cells.

Each membrane (name → (m, 3, 3) triangles) gets every field, sampled at its vertices and named
``<field>_<membrane>`` (VCell's convention): trilinear interpolation on the grid, the coincident DOF on
a mesh. Particle positions are recorded when given.

:func:`attach_recorder` adds a :class:`~viva_pde_particle.processes.recorder.SpatialRecorder` to a
hybrid document; :func:`write_native_bundle` writes a native VCell trajectory.
"""
from __future__ import annotations

import numpy as np

from viva_pde_particle.viz3d.bundle import (
    SpatialBundleWriter,
    grid_domain,
    membrane_domain,
    membrane_variable,
    mesh_domain,
)


class BundleRecorder:
    def __init__(self, path, layout: dict, species, particle_species=(), membranes: dict | None = None,
                 region: str = "cell", source: str = "", planned_times=()):
        self.layout = layout
        self.species = list(species)
        self.kind = layout["kind"]
        self.writer = SpatialBundleWriter(path, source=source, planned_times=planned_times,
                                          solver_options={"layout": self.kind})
        if self.kind == "grid":
            from viva_pde_particle.grid import CartesianGrid

            self.grid = CartesianGrid.from_config(layout["grid"])
            mask = layout.get("mask")
            self.volume = grid_domain(region, self.grid, mask)
            fold = layout.get("fold")
            if mask is not None and fold is None:
                from viva_pde_particle.geometry.accessible import fold_map

                fold = fold_map(np.asarray(mask, bool), self.grid)
            n = int(np.prod(self.grid.num))
            self._src = np.arange(n) if fold is None else np.where(np.asarray(fold) >= 0, np.asarray(fold), np.arange(n))
        elif self.kind == "mesh":
            self.volume = mesh_domain(region, layout["mesh"])
        else:
            raise ValueError(f"layout kind must be 'grid' or 'mesh', got {self.kind!r}")
        self.writer.add_domain(self.volume, self.species)
        self.membranes = {}
        for name, tri in (membranes or {}).items():
            dom = membrane_domain(name, tri)
            self.membranes[name] = (dom, self._sampler(dom.points))
            self.writer.add_domain(dom, [membrane_variable(s, name) for s in self.species])
        self.particle_species = list(particle_species)
        if self.particle_species:
            self.writer.add_particles(self.particle_species)
        self.writer.open()
        self.last_time: float | None = None

    def _sampler(self, points):
        if self.kind == "grid":
            from scipy.interpolate import RegularGridInterpolator

            axes = [self.grid.axis_nodes(d) for d in reversed(range(self.grid.dim))]  # array axis order (z, y, x)
            q = np.asarray(points)[:, : self.grid.dim][:, ::-1]
            lo = np.array([a[0] for a in axes])
            hi = np.array([a[-1] for a in axes])
            q = np.clip(q, lo, hi)
            shape = self.grid.shape

            def sample(ext_flat):
                return RegularGridInterpolator(axes, ext_flat.reshape(shape))(q)

            return sample
        from scipy.spatial import cKDTree

        _, idx = cKDTree(self.volume.points).query(np.asarray(points))
        return lambda dofs: np.asarray(dofs)[idx]

    def record(self, t: float, fields: dict, positions: dict | None = None) -> None:
        values = {}
        for s in self.species:
            arr = np.asarray(fields[s], dtype=float).ravel()
            ext = arr[self._src] if self.kind == "grid" else arr
            values[(self.volume.name, s)] = ext[self.volume.node_index] if self.kind == "grid" else ext
            for name, (_, sample) in self.membranes.items():
                values[(name, membrane_variable(s, name))] = sample(ext)
        self.writer.write(t, values, particles=positions)
        self.last_time = float(t)

    def close(self, status: str = "completed", message: str | None = None) -> None:
        self.writer.finalize(status, message)


# ---------------------------------------------------------------- documents

def _particle_smoldyn_node(doc: dict) -> dict:
    """The particle node (bare engine, Stepper-wrapped, or inside a SplittingCoordinator)."""
    node = doc["coupler"]["config"]["particles"] if "coupler" in doc else doc["particles"]
    return node


def _smoldyn_config(node: dict) -> dict:
    return node["config"]["process"]["config"] if node["address"] == "local:Stepper" else node["config"]


def attach_recorder(doc: dict, out_dir, output_dt: float, *, membranes: dict | None = None, region: str = "cell",
                    particles: bool = True, source: str = "") -> dict:
    """Add a ``recorder`` (SpatialRecorder) to a hybrid document, recording every ``output_dt``.

    Works for the grid, mesh, VCell-geometry and splitting documents of ``composites.hybrid``.
    With ``particles``, the particle engine also emits positions, recorded into the bundle.
    """
    pde = doc["coupler"]["config"]["pde"] if "coupler" in doc else doc["pde"]
    cfg = pde["config"]
    if "mesh" in cfg:
        layout = {"kind": "mesh", "mesh": cfg["mesh"]}
        port = "field_dofs"
    else:
        layout = {"kind": "grid", "grid": cfg["grid"]}
        mask = (cfg.get("domain") or {}).get("mask")
        if mask is not None:
            layout["mask"] = np.asarray(mask, bool)
            adapters = doc["coupler"]["config"]["adapters"] if "coupler" in doc else doc
            lookup = adapters.get("field_to_particles", {}).get("config", {})
            if "staircase" in lookup:
                layout["fold"] = np.asarray(lookup["staircase"]["fold"])
        port = "fields"
    intervals = [n["interval"] for n in doc.values() if isinstance(n, dict) and n.get("_type") == "process"]
    base = min(intervals)
    every = int(round(output_dt / base))
    if every < 1 or abs(every * base - output_dt) > 1e-9 * output_dt:
        raise ValueError(f"output_dt {output_dt} must be a multiple of the shortest process interval {base}")
    pnode = _particle_smoldyn_node(doc)
    pspecies = list(_smoldyn_config(pnode)["particle_species"]) if particles else []
    inputs = {port: [port]}
    if pspecies:
        _smoldyn_config(pnode)["emit_positions"] = True
        pnode["outputs"]["particle_positions"] = ["particle_positions"]
        if "coupler" in doc:
            doc["coupler"]["outputs"]["particle_positions"] = ["particle_positions"]
        inputs["particle_positions"] = ["particle_positions"]
    doc["recorder"] = {
        "_type": "process",
        "address": "local:SpatialRecorder",
        "config": {"out_dir": str(out_dir), "base_dt": base, "every": every, "layout": layout,
                   "species": list(cfg["pde"]["species"]), "particle_species": pspecies,
                   "membranes": {k: np.asarray(v, dtype=float) for k, v in (membranes or {}).items()},
                   "region": region, "source": source},
        "interval": base,
        "inputs": inputs,
        "outputs": {},
    }
    return doc


def write_native_bundle(path, trajectory, grid, *, mask=None, membranes: dict | None = None, region: str = "cell",
                        source: str = "native VCell (vcell-fvsolver)") -> None:
    """A native VCell ``NativeTrajectory`` (fields on the grid) as a bundle, for side-by-side viewing."""
    layout = {"kind": "grid", "grid": grid.to_config()}
    if mask is not None:
        layout["mask"] = np.asarray(mask, bool)
    rec = BundleRecorder(path, layout, list(trajectory.fields), (), membranes, region, source,
                         planned_times=trajectory.times)
    for i, t in enumerate(trajectory.times):
        rec.record(float(t), {s: v[i] for s, v in trajectory.fields.items()})
    rec.close()
