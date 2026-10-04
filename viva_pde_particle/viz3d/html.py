"""A self-contained interactive 3D page (three.js) for one or more spatial results bundles.

The page needs no server, so the workbench can show it as a study figure (``address:
threejs:viz/<name>.html``). It shows:

- the membrane, coloured by the field, with a movable clipping plane that opens it up;
- the field on a mid-plane slice through the volume (x, y or z);
- particles as points, one colour per species;

with a time slider, play button, bundle selector (e.g. native VCell vs co-sim), layer toggles and a
colour bar on one shared range. Data are embedded as base64 float32 arrays, subsampled in time and
particles to keep the page small. Slices are computed with PyVista (pixi env).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np

from viva_pde_particle.viz3d.bundle import membrane_variable, particle_species, read_particles

THREE_CDN = "https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"
ORBIT_CDN = "https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"
_VIRIDIS = ["#440154", "#482878", "#3e4989", "#31688e", "#26828e", "#1f9e89", "#35b779", "#6ece58", "#b5de2b",
            "#fde725"]
_SPECIES_COLORS = ["#ff4500", "#dc143c", "#800080", "#696969"]


def _b64(a, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def _frame_rows(n: int, max_frames: int) -> list[int]:
    if n <= max_frames:
        return list(range(n))
    return sorted({int(round(i)) for i in np.linspace(0, n - 1, max_frames)})


def _surface(points, cells):
    return {"points": _b64(points, "<f4"), "cells": _b64(np.asarray(cells).ravel(), "<u4"), "n": int(len(points))}


def _bundle_payload(path, var: str, max_frames: int, max_particles: int, rng, slice_origin=None) -> dict:
    from vcell_fenics.results import Bundle

    from viva_pde_particle.viz3d.static import domain_mesh

    b = Bundle.open(path)
    vol = [n for n, d in b.manifest.domains.items() if d.kind == "volume"][0]
    mem = [n for n, d in b.manifest.domains.items() if d.kind == "membrane"]
    rows = _frame_rows(len(b.times), max_frames)
    v = domain_mesh(b, vol)
    for k, r in enumerate(rows):
        v.point_data[f"f{k}"] = b.field(vol, var, r)
    out = {"times": [float(b.times[r]) for r in rows], "slices": {}, "particles": {}}
    center = np.asarray(v.center if slice_origin is None else slice_origin, dtype=float)
    for axis in "xyz":
        sl = v.slice(normal=axis, origin=center).triangulate()
        if not sl.n_cells:
            continue
        tri = sl.regular_faces
        out["slices"][axis] = {**_surface(sl.points, tri),
                               "values": [_b64(sl.point_data[f"f{k}"], "<f4") for k in range(len(rows))]}
    if mem:
        m = b.mesh(mem[0])
        names = {x.name for x in b.manifest.variables if x.domain == mem[0]}
        mvar = var if var in names else membrane_variable(var, mem[0])
        out["membrane"] = {**_surface(m.points, m.cells),
                           "values": [_b64(b.field(mem[0], mvar, r), "<f4") for r in rows] if mvar in names else None}
    else:
        surf = v.extract_surface(algorithm=None).triangulate()
        out["membrane"] = {**_surface(surf.points, surf.regular_faces),
                           "values": [_b64(surf.point_data[f"f{k}"], "<f4") for k in range(len(rows))]}
    for s in particle_species(path):
        frames = []
        for r in rows:
            p = read_particles(path, s, r)
            if len(p) > max_particles:
                p = p[rng.choice(len(p), max_particles, replace=False)]
            frames.append(_b64(p, "<f4"))
        out["particles"][s] = frames
    st = b.stats(vol, var)
    out["range"] = [float(np.nanmin(st[:, 2])), float(np.nanmax(st[:, 3]))]
    out["frame_ranges"] = [[float(st[r, 2]), float(st[r, 3])] for r in rows]
    out["bounds"] = [float(x) for x in v.bounds]
    return out


def bundle_html(bundles: dict, var: str, out, *, title: str | None = None, max_frames: int = 20,
                max_particles: int = 2000, seed: int = 0, colors: dict | None = None, camera: str = "side",
                units: str = "µM", slice_origin=None) -> Path:
    """Write a self-contained three.js page for ``bundles`` (label → bundle path) showing ``var``.

    ``colors`` maps particle species to CSS colours; ``camera`` is ``"side"`` or ``"top"`` (a slab);
    ``slice_origin`` places the slices (default: the domain's centre).
    """
    rng = np.random.default_rng(seed)
    data = {label: _bundle_payload(path, var, max_frames, max_particles, rng, slice_origin)
            for label, path in bundles.items()}
    species = list(next(iter(data.values()))["particles"])
    spc = [(colors or {}).get(s, _SPECIES_COLORS[k % len(_SPECIES_COLORS)]) for k, s in enumerate(species)]
    lo = min(d["range"][0] for d in data.values())
    hi = max(d["range"][1] for d in data.values())
    page = (_TEMPLATE.replace("__TITLE__", title or f"{var}: {', '.join(bundles)}")
            .replace("__THREE__", THREE_CDN).replace("__ORBIT__", ORBIT_CDN)
            .replace("__DATA__", json.dumps(data)).replace("__VAR__", json.dumps(var))
            .replace("__RANGE__", json.dumps([lo, hi if hi > lo else lo + 1e-12]))
            .replace("__CMAP__", json.dumps(_VIRIDIS)).replace("__SPCOLORS__", json.dumps(spc))
            .replace("__CAMERA__", json.dumps(camera)).replace("__UNITS__", units))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return out


_TEMPLATE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>__TITLE__</title>
<style>
 body{margin:0;font:13px system-ui,sans-serif;background:#fafafa;color:#222}
 #bar{display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:8px 12px;background:#fff;border-bottom:1px solid #ddd}
 #bar label{display:flex;gap:4px;align-items:center}
 #view{position:relative;width:100%;height:560px}
 #cbar{position:absolute;right:14px;bottom:14px;background:#fffd;padding:6px 8px;border-radius:4px;font-size:12px}
 #grad{width:180px;height:10px;margin:3px 0}
 #title{font-weight:600}
 #legend span{display:inline-block;width:10px;height:10px;border-radius:5px;margin:0 3px 0 8px}
</style>
<script src="__THREE__"></script><script src="__ORBIT__"></script>
</head><body>
<div id="bar">
 <span id="title">__TITLE__</span>
 <label>run <select id="bundle"></select></label>
 <button id="play">▶</button>
 <input id="time" type="range" min="0" value="0" style="width:200px"><span id="tlabel"></span>
 <label><input id="fixed" type="checkbox">fixed range (all times)</label>
 <label><input id="showMem" type="checkbox" checked>membrane</label>
 <label>open <input id="clip" type="range" min="0" max="100" value="45" style="width:90px"></label>
 <label><input id="showSlice" type="checkbox" checked>slice <select id="axis"><option>z</option><option>y</option><option>x</option></select></label>
 <label><input id="showPart" type="checkbox" checked>particles</label><span id="legend"></span>
</div>
<div id="view"><div id="cbar"><div id="cvar"></div><div id="grad"></div><div style="display:flex;justify-content:space-between"><span id="lo"></span><span id="hi"></span></div></div></div>
<script>
const DATA = __DATA__, VAR = __VAR__, ALL_RANGE = __RANGE__, CMAP = __CMAP__, SPC = __SPCOLORS__, CAMERA = __CAMERA__;
let RANGE = ALL_RANGE.slice();
const dec = (s, T) => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return new T(u.buffer); };
const stops = CMAP.map(h => new THREE.Color(h));
function cmap(v, out, i) {
  let x = (v - RANGE[0]) / (RANGE[1] - RANGE[0]); x = isFinite(x) ? Math.min(1, Math.max(0, x)) : 0;
  const f = x * (stops.length - 1), k = Math.min(stops.length - 2, Math.floor(f)), w = f - k;
  out[3*i] = stops[k].r + w*(stops[k+1].r - stops[k].r); out[3*i+1] = stops[k].g + w*(stops[k+1].g - stops[k].g); out[3*i+2] = stops[k].b + w*(stops[k+1].b - stops[k].b);
}
document.getElementById('cvar').textContent = VAR + ' (__UNITS__)';
document.getElementById('grad').style.background = 'linear-gradient(to right,' + CMAP.join(',') + ')';
function setRange() {  // this frame's range over every run (runs share their output times), or all times
  if (document.getElementById('fixed').checked) RANGE = ALL_RANGE.slice();
  else { RANGE = [Infinity, -Infinity]; for (const d of Object.values(DATA)) { const r = d.frame_ranges[Math.min(frame, d.frame_ranges.length - 1)]; RANGE[0] = Math.min(RANGE[0], r[0]); RANGE[1] = Math.max(RANGE[1], r[1]); }
         if (!(RANGE[1] > RANGE[0])) RANGE[1] = RANGE[0] + 1e-12; }
  document.getElementById('lo').textContent = RANGE[0].toPrecision(3); document.getElementById('hi').textContent = RANGE[1].toPrecision(3);
}
const view = document.getElementById('view');
const renderer = new THREE.WebGLRenderer({antialias: true}); renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(view.clientWidth, view.clientHeight); renderer.localClippingEnabled = true; renderer.setClearColor(0xfafafa);
view.appendChild(renderer.domElement);
const scene = new THREE.Scene(); scene.add(new THREE.AmbientLight(0xffffff, 0.75));
const light = new THREE.DirectionalLight(0xffffff, 0.45); scene.add(light);
const camera = new THREE.PerspectiveCamera(40, view.clientWidth / view.clientHeight, 0.01, 1e4);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
const clipPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);  // keeps y ≥ cut: opens the side facing the camera
function surfaceMesh(s, clipped) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(dec(s.points, Float32Array), 3));
  g.setIndex(new THREE.BufferAttribute(dec(s.cells, Uint32Array), 1));
  g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(3 * s.n).fill(0.8), 3));
  g.computeVertexNormals();
  const m = new THREE.MeshLambertMaterial({vertexColors: true, side: THREE.DoubleSide, clippingPlanes: clipped ? [clipPlane] : []});
  return new THREE.Mesh(g, m);
}
const runs = {};
for (const [label, d] of Object.entries(DATA)) {
  const group = new THREE.Group(), r = {d, group, slices: {}, parts: {}};
  r.mem = surfaceMesh(d.membrane, true); group.add(r.mem);
  r.memVals = d.membrane.values ? d.membrane.values.map(v => dec(v, Float32Array)) : null;
  for (const [ax, s] of Object.entries(d.slices)) { const m = surfaceMesh(s, false); m.visible = false; group.add(m); r.slices[ax] = {m, vals: s.values.map(v => dec(v, Float32Array))}; }
  Object.keys(d.particles).forEach((sp, k) => {
    const g = new THREE.BufferGeometry(); g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(0), 3));
    const p = new THREE.Points(g, new THREE.PointsMaterial({color: SPC[k % SPC.length], clippingPlanes: [clipPlane], size: 0.08 * Math.max(d.bounds[1]-d.bounds[0], d.bounds[3]-d.bounds[2]) / 9}));
    group.add(p); r.parts[sp] = {p, frames: d.particles[sp]};
  });
  group.visible = false; scene.add(group); runs[label] = r;
  const o = document.createElement('option'); o.textContent = label; document.getElementById('bundle').appendChild(o);
}
document.getElementById('legend').innerHTML = Object.keys(Object.values(DATA)[0].particles).map((s, k) => `<span style="background:${SPC[k % SPC.length]}"></span>${s}`).join('');
let cur = null, frame = 0, playing = null;
const el = id => document.getElementById(id);
function paint(mesh, vals) { const c = mesh.geometry.attributes.color; for (let i = 0; i < vals.length; i++) cmap(vals[i], c.array, i); c.needsUpdate = true; }
function update() {
  const r = runs[cur], d = r.d; frame = Math.min(frame, d.times.length - 1); setRange();
  el('time').max = d.times.length - 1; el('time').value = frame; el('tlabel').textContent = 't = ' + d.times[frame].toPrecision(3) + ' s';
  r.mem.visible = el('showMem').checked; if (r.memVals) paint(r.mem, r.memVals[frame]); else r.mem.material.color.set(0xcccccc);
  const b = d.bounds; clipPlane.constant = -(b[2] + (b[3] - b[2]) * el('clip').value / 100);
  for (const [ax, s] of Object.entries(r.slices)) { s.m.visible = el('showSlice').checked && ax === el('axis').value; if (s.m.visible) paint(s.m, s.vals[frame]); }
  for (const p of Object.values(r.parts)) { p.p.visible = el('showPart').checked; p.p.geometry.setAttribute('position', new THREE.BufferAttribute(dec(p.frames[frame], Float32Array), 3)); }
}
function select(label) {
  if (cur) runs[cur].group.visible = false; cur = label; runs[cur].group.visible = true;
  const b = runs[cur].d.bounds, c = new THREE.Vector3((b[0]+b[1])/2, (b[2]+b[3])/2, (b[4]+b[5])/2), s = Math.max(b[1]-b[0], b[3]-b[2], b[5]-b[4]);
  controls.target.copy(c);
  if (CAMERA === 'top') { camera.position.set(c.x, c.y - 0.25*s, c.z + 1.1*s); camera.up.set(0, 1, 0); }
  else { camera.position.set(c.x + 0.45*s, c.y - 1.6*s, c.z + 0.9*s); camera.up.set(0, 0, 1); }
  light.position.copy(camera.position); update();
}
el('bundle').onchange = e => select(e.target.value);
el('time').oninput = e => { frame = +e.target.value; update(); };
['fixed', 'showMem', 'clip', 'showSlice', 'axis', 'showPart'].forEach(id => el(id).oninput = update);
el('play').onclick = () => { if (playing) { clearInterval(playing); playing = null; el('play').textContent = '▶'; return; }
  el('play').textContent = '❚❚'; playing = setInterval(() => { frame = (frame + 1) % runs[cur].d.times.length; update(); }, 350); };
select(Object.keys(runs)[0]);
window.onresize = () => { renderer.setSize(view.clientWidth, view.clientHeight); camera.aspect = view.clientWidth / view.clientHeight; camera.updateProjectionMatrix(); };
(function loop() { requestAnimationFrame(loop); controls.update(); light.position.copy(camera.position); renderer.render(scene, camera); })();
</script></body></html>
"""
