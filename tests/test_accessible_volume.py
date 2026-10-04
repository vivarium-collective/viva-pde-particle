"""Phase 7e.5: accessible volumes, compartment-aware folding and the masked FV domain."""
from __future__ import annotations

import numpy as np
import pytest
from process_bigraph import allocate_core

from viva_pde_particle.geometry.accessible import effective_volumes, fold_map
from viva_pde_particle.geometry.inside import TriangleInside
from viva_pde_particle.grid import CartesianGrid


def _cube_triangles(lo, hi):
    """A closed axis-aligned box surface, 12 triangles."""
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]])
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (2, 3, 7), (2, 7, 6), (1, 2, 6), (1, 6, 5), (3, 0, 4), (3, 4, 7)]
    return v[np.array(faces)]


def test_triangle_inside_is_exact_on_lattice_points():
    """Lattice points lying in the planes of the surface's vertices and edges (the case VTK got wrong)."""
    inside = TriangleInside(_cube_triangles((1.0, 1.0, 1.0), (3.0, 3.0, 3.0)))
    g = np.arange(0.0, 4.01, 0.25)
    pts = np.stack(np.meshgrid(g, g, g, indexing="ij"), axis=-1).reshape(-1, 3)
    strict = ((pts > 1.0) & (pts < 3.0)).all(axis=1)
    on_face = ((pts >= 1.0) & (pts <= 3.0)).all(axis=1) & ~strict
    got = inside(pts)
    np.testing.assert_array_equal(got[~on_face], strict[~on_face])  # exact away from the surface itself
    rng = np.random.default_rng(0)
    mc = rng.uniform(0, 4, (200_000, 3))
    assert inside(mc).mean() * 64 == pytest.approx(8.0, rel=0.02)


def test_fold_map_sends_exterior_nodes_to_the_nearest_domain_node():
    g = CartesianGrid((0, 0, 0), (4, 4, 4), (5, 5, 5))
    mask = np.zeros(g.shape, bool)
    mask[1:4, 1:4, 1:4] = True
    fold = fold_map(mask, g)
    flat = np.arange(mask.size).reshape(mask.shape)
    assert (fold[mask.ravel()] == flat[mask]).all()
    assert fold[flat[2, 2, 0]] == flat[2, 2, 1]  # face neighbour
    assert fold[flat[0, 0, 0]] == flat[1, 1, 1]  # corner: the diagonal neighbour is the only in-mask one
    assert (fold >= 0).all()


def test_effective_volumes_per_correction():
    g = CartesianGrid((0, 0, 0), (4, 4, 4), (5, 5, 5))
    mask = np.zeros(g.shape, bool)
    mask[1:4, 1:4, 1:4] = True
    frac = np.zeros(g.shape)
    frac[mask] = 1.0
    frac[0, 2, 2] = 0.5  # an exterior node partly inside the smooth region
    fold = fold_map(mask, g)
    none = effective_volumes(g, mask, frac, fold, "none")
    acc = effective_volumes(g, mask, frac, fold, "adapters")
    assert none.sum() == pytest.approx(g.element_volumes[mask].sum()) and (none[~mask] == 0).all()
    assert acc.sum() == pytest.approx((frac * g.element_volumes).sum())  # every accessible bit, folded in
    assert acc[1, 2, 2] == pytest.approx(g.element_volumes[1, 2, 2] + 0.5 * g.element_volumes[0, 2, 2])
    with pytest.raises(ValueError):
        effective_volumes(g, mask, frac, fold, "some")


def test_masked_fv_conserves_with_custom_volume_fractions():
    from viva_pde_particle.processes import FVReactionDiffusion

    g = CartesianGrid((0, 0, 0), (4, 4, 4), (9, 9, 9))
    mask = np.zeros(g.shape, bool)
    mask[2:7, 2:7, 2:7] = True
    vf = np.where(mask, 0.6 + 0.4 * np.random.default_rng(1).random(g.shape), 1.0)
    proc = FVReactionDiffusion(config={"grid": g.to_config(), "dt": 0.05, "domain": {"mask": mask, "volume_fraction": vf},
                                       "pde": {"species": {"B": {"diffusion": 1.0}}, "terms": []}}, core=allocate_core())
    b = np.zeros(g.shape)
    b[4, 4, 4] = 10.0
    b[0, 0, 0] = 3.0  # outside the domain: frozen
    out = proc.update({"fields": {"B": b}, "external_conc": {}}, 1.0)["fields"]["B"]
    w = vf * g.full_volume * mask
    assert (out * w).sum() == pytest.approx((b * w).sum(), rel=1e-10)  # mass conserved with the custom volumes
    assert out[0, 0, 0] == 3.0 and (out[~mask & (np.arange(b.size).reshape(b.shape) != 0)] == 0).all()
    assert out[2, 2, 2] > 0  # diffused within the domain


def test_accessible_fractions_integrate_to_the_region_volume():
    pytest.importorskip("vcell_fenics")
    from viva_pde_particle.geometry import realize_fenics, sphere_in_box
    from viva_pde_particle.geometry.accessible import accessible_fractions

    g = CartesianGrid((0, 0, 0), (4, 4, 4), (17, 17, 17))
    real = realize_fenics(sphere_in_box((2.0, 2.0, 2.0), 1.5, (4.0, 4.0, 4.0)), 0.4)
    f = accessible_fractions(real, "cell", g)
    assert (f >= 0).all() and (f <= 1).all()
    assert (f * g.element_volumes).sum() == pytest.approx(real.volume("cell"), rel=0.01)
