"""The VCell-FV geometry compiler (Phase 7e.4): a GeometryDescription realized exactly as native VCell does.

VCell's own Java (through pyvcell + libvcell) discretizes the geometry; this module only reads the
result, so a co-simulation on this realization runs on the same geometry as vcell-fvsolver. No
solve is needed: libvcell writes a hybrid application's inputs for the geometry alone.

- **The PDE domain** is the staircase: VCell's node → region map and region volumes (the ``.vcg``).
  VCell corrects the staircase numerically with implicit-surface membrane areas and normals, but
  keeps full voxel volumes.
- **The smooth particle domain** is VCell's surface triangulation, plus its compartment points
  (from the ``.smoldynInput`` that VCell writes for Smoldyn).

As decided for Phase 7e, ``inside``/``locate`` (and so the Smoldyn geometry derived from this
realization) follow the **smooth** surface, tested by robust vertical-ray parity
(:class:`~viva_pde_particle.geometry.inside.TriangleInside`; VTK's enclosed-points filter misclassifies
lattice-aligned points against VCell's surface). The staircase is exposed separately
(``node_regions``, ``pde_volume``) for the FV engine and the accessible-volume correction (7e.5).
"""
from __future__ import annotations

import os
import tempfile
import zlib
from pathlib import Path

import numpy as np

from viva_pde_particle.geometry.realize import RealizedGeometry


def _parse_vcg(path: Path):
    """(region → (subvolume handle, volume)) and the node region-index array (x fastest)."""
    lines = path.read_text().splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith("volumeRegions"))
    regions = {}
    for k in range(int(lines[i].split()[1])):
        name, volume, handle = lines[i + 1 + k].split()
        regions[k] = (int(handle), float(volume))
    j = next(k for k, line in enumerate(lines) if line.startswith("volumeSamples"))
    num = [int(v) for v in lines[j].split()[1:4]]
    hexdata = "".join(line.strip() for line in lines[j + 1:]
                      if line.strip() and all(c in "0123456789ABCDEFabcdef" for c in line.strip()))
    raw = zlib.decompress(bytes.fromhex(hexdata))
    dtype = np.uint8 if len(raw) == int(np.prod(num)) else np.uint16
    return regions, num, np.frombuffer(raw, dtype=dtype).astype(np.int64)


def _parse_smoldyn(path: Path):
    """VCell's Smoldyn surfaces ({name: (m, 3, 3)}) and compartment points ({name: (n, 3)})."""
    surfaces: dict[str, list] = {}
    points: dict[str, list] = {}
    surface = compartment = None
    for line in path.read_text().splitlines():
        words = line.split()
        if not words:
            continue
        if words[0] == "start_surface":
            surface = words[1]
            surfaces.setdefault(surface, [])
        elif words[0] == "end_surface":
            surface = None
        elif words[0] == "panel" and surface is not None and len(words) >= 11 and words[1] == "tri":
            surfaces[surface].append(np.array(words[2:11], dtype=float).reshape(3, 3))
        elif words[0] == "start_compartment":
            compartment = words[1]
            points.setdefault(compartment, [])
        elif words[0] == "end_compartment":
            compartment = None
        elif words[0] == "point" and compartment is not None:
            points[compartment].append([float(v) for v in words[1:4]])
    return ({k: np.array(v) for k, v in surfaces.items() if v},
            {k: np.array(v) for k, v in points.items() if v})


def _closed_volume(tri: np.ndarray) -> float:
    """Signed volume enclosed by triangles (divergence theorem); positive when outward."""
    return float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0)


class VCellFVRealization(RealizedGeometry):
    """A 3D GeometryDescription (analytic subvolumes) realized by VCell on ``grid``."""

    def __init__(self, description, grid, keep_dir: Path | None = None):
        self.description = description
        self.grid = grid
        self.regions = tuple(sv.name for sv in description.subvolumes)
        self.lo = np.asarray(description.origin, dtype=float)
        self.hi = self.lo + np.asarray(description.extent, dtype=float)
        work = Path(keep_dir) if keep_dir else Path(tempfile.mkdtemp(prefix="vcell-fv-geometry-"))
        work.mkdir(parents=True, exist_ok=True)
        vcg, smoldyn = self._generate(work)
        regions, num, samples = _parse_vcg(vcg)
        if num != list(grid.num):
            raise ValueError(f"VCell mesh {num} != grid {list(grid.num)}")
        handle_of_region = np.array([regions[k][0] for k in range(len(regions))])
        self.node_regions = handle_of_region[samples].reshape(grid.shape)  # subvolume index per node
        self._pde_volume = {}
        for k, (handle, volume) in regions.items():
            name = self.regions[handle]
            self._pde_volume[name] = self._pde_volume.get(name, 0.0) + volume
        surfaces, points = _parse_smoldyn(smoldyn)
        self.surface_triangles = {}
        for sc in description.surfaces:
            tri = surfaces.get(sc.name)
            if tri is None:
                raise ValueError(f"VCell wrote no surface named {sc.name!r} (has {sorted(surfaces)})")
            self.surface_triangles[sc.name] = tri if _closed_volume(tri) > 0 else tri[:, [0, 2, 1]]
        self.compartment_points = points
        self._surfaces_of = {r: [sc.name for sc in description.surfaces if sc.inside == r] for r in self.regions}
        self._enclosers: dict = {}
        if not keep_dir:
            import shutil

            shutil.rmtree(work, ignore_errors=True)

    def _generate(self, work: Path) -> tuple[Path, Path]:
        """libvcell inputs for a minimal hybrid model on this geometry (one empty particle species)."""
        from libvcell import vcml_to_finite_volume_input
        from pyvcell.vcml.utils import to_vcml_str

        from viva_pde_particle.model import HybridModel, Species
        from viva_pde_particle.reference.vcell_native import to_biomodel

        model = HybridModel(self.grid, [Species("P", 1.0, particle=True, initial=0),
                                        Species("F", 1.0, initial=0.0)], [])
        bm = to_biomodel(model, t_end=0.02, dt=0.01, output_dt=0.01, geometry=self.description, seed=1)
        ok, msg = vcml_to_finite_volume_input(vcml_content=to_vcml_str(bio_model=bm), simulation_name="sim",
                                              output_dir_path=work)
        if not ok:
            raise RuntimeError(f"libvcell failed to generate inputs: {msg}")
        files = os.listdir(work)
        return (work / next(f for f in files if f.endswith(".vcg")),
                work / next(f for f in files if f.endswith(".smoldynInput")))

    # the smooth (particle) domain
    def _enclosed(self, region: str, points: np.ndarray) -> np.ndarray:
        """Inside the region's closed smooth surface (robust vertical-ray parity; see geometry.inside)."""
        from viva_pde_particle.geometry.inside import TriangleInside

        if region not in self._enclosers:
            self._enclosers[region] = TriangleInside(np.concatenate([self.surface_triangles[s]
                                                                     for s in self._surfaces_of[region]]))
        return self._enclosers[region](points)

    def _in_box(self, pts):
        return ((pts >= self.lo) & (pts <= self.hi)).all(axis=1)

    def inside(self, region, points):
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        if self._surfaces_of[region]:
            return self._enclosed(region, pts)
        out = self._in_box(pts)  # a background region: the box minus every enclosed region
        for other in self.regions:
            if other != region and self._surfaces_of[other]:
                out &= ~self._enclosed(other, pts)
        return out

    def locate(self, points):
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        labels = np.full(len(pts), -1, dtype=np.int64)
        order = sorted(range(len(self.regions)), key=lambda i: not self._surfaces_of[self.regions[i]])
        for i in order:  # enclosed regions first, then backgrounds
            todo = np.nonzero(labels < 0)[0]
            if len(todo):
                labels[todo[self.inside(self.regions[i], pts[todo])]] = i
        return labels

    def boundary_triangles(self, region):
        if not self._surfaces_of[region]:
            raise NotImplementedError(f"{region!r} is a background region; its boundary includes the box walls")
        return np.concatenate([self.surface_triangles[s] for s in self._surfaces_of[region]])

    def interior_points(self, region, max_points=64):
        pts = self.compartment_points.get(region)
        if pts is None:
            raise KeyError(f"VCell wrote no compartment points for {region!r}")
        step = max(1, len(pts) // max_points)
        return pts[::step][:max_points]

    def volume(self, region):
        if self._surfaces_of[region]:
            return sum(_closed_volume(self.surface_triangles[s]) for s in self._surfaces_of[region])
        box = float(np.prod(self.hi - self.lo))
        return box - sum(self.volume(r) for r in self.regions if r != region and self._surfaces_of[r])

    def region_bounds(self, region):
        if self._surfaces_of[region]:
            tri = self.boundary_triangles(region).reshape(-1, 3)
            return tri.min(axis=0), tri.max(axis=0)
        return self.lo, self.hi

    # the staircase (PDE) domain
    def pde_volume(self, region: str) -> float:
        """VCell's FV volume of the region: whole node-centred voxels (µm³)."""
        return self._pde_volume.get(region, 0.0)

    def node_mask(self, region: str) -> np.ndarray:
        """Grid-shaped bool: nodes whose VCell volume element belongs to the region."""
        return self.node_regions == self.regions.index(region)
