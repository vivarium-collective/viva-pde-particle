# The web bundle profile

What a viva-pde-particle run leaves so a browser can view it **with no data or render service**: the vcell-fenics
results bundle (ADR 010) plus a small **web extension**. compose-api serves the bundle file by file
(`GET /datasets/{id}/files/{path}`), and its web UI's viewers read it chunk by chunk (compose-api
`docs/plan-viewers.md`). This document is the contract between the two.

## The bundle (unchanged, vcell-fenics ADR 010)
A zarr v2 group, `<name>.fenics/`:
- the manifest is in `.zattrs["vcell_fenics"]` (`status`, `times` — authoritative; arrays are preallocated — `domains`,
  `variables`). It is replaced atomically after every row, so a reader may poll it while the run is live.
- `mesh/<domain>.vtu` holds the meshes.
- `<domain>/<var>` arrays are (T, N) float64, one zlib chunk per time row. `stats/<domain>/<var>` arrays are (T, 4):
  mean, total, min, max.
- The particle extension is `.zattrs["particles"]`, plus `particles/<sp>/xyz` (T, cap, 3) NaN-padded and `count` (T, 1).

## The web extension (`viz3d/web.py`, schema 1)
- `.zattrs["web"] = {"schema": 1, "surfaces": {"<domain>": {"points": "web/<domain>/points", "triangles": "web/<domain>/triangles"}}}`
- `web/<domain>/points` is (N, 3) float32: the domain's mesh points **in mesh order**, so a row of `<domain>/<var>`
  colours the surface directly.
- `web/<domain>/triangles` is (M, 3) uint32 indices into `points`:
  - a tetra domain contributes its boundary, i.e. the faces belonging to exactly one tet
  - a triangle domain (a membrane, a 2D grid) contributes its own cells
  - line domains have no surface
- Chunks are whole arrays, zlib level 1. Remeshed bundles carry the first segment's surface.

Additive, like the particle extension: readers that do not know it ignore it.

## Who writes it
- **`SpatialBundleWriter`:** the web extension is written in `open()`, so a live run is viewable from its first row.
- **`steps/spatial_export.py` (`SpatialExport`, `export()`):** adds the extension to any bundle that lacks it (e.g. one
  vcell-fenics wrote). It is the ad hoc translation step for this project's output. A general converter between two
  standard formats, or a figure/view step, belongs in its own published component once a second producer needs one.
- **`compose_runner`:**
  - points every `SpatialRecorder`'s `out_dir` at `<output>/<name>.fenics`
  - calls the recorder's `close()` after the run (its tick at the end time never executes)
  - runs `export()`
  - announces each bundle:
    - `artifact.written` with `kind="results-bundle"` and `media_type="application/vnd.vcell.results-bundle+zarr"`
    - `attributes={format:"vcell-fenics-bundle", web, status, times, domains, variables, particles}`
    - no checksum, since the bundle is a directory
