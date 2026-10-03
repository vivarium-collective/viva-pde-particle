"""pyvcell smoothed VTK grid of native VCell results (viva_pde_particle.reference.vcell_vtk)."""
from __future__ import annotations

import numpy as np
import pytest

from viva_pde_particle.reference.vcell_native import hybrid_support_available
from viva_pde_particle.reference.vcell_vtk import hex_volumes

UNIT = np.array([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)], dtype=float)  # voxel order


def test_hex_volumes():
    assert hex_volumes(UNIT[None] * [2.0, 3.0, 0.5]) == pytest.approx([3.0])
    sheared = UNIT.copy()
    sheared[4:, 0] += 0.7  # shear the top face: volume unchanged
    assert hex_volumes(sheared[None]) == pytest.approx([1.0])


@pytest.mark.skipif(not hybrid_support_available(), reason="pyvcell without spatial-hybrid support")
def test_smoothed_domain_node_centred(tmp_path):
    pytest.importorskip("vtk")
    from viva_pde_particle.grid import CartesianGrid
    from viva_pde_particle.model import HybridModel, Reaction, Species
    from viva_pde_particle.reference.vcell_native import run_native
    from viva_pde_particle.reference.vcell_vtk import smoothed_domain

    g = CartesianGrid((0, 0, 0), (9.0, 9.0, 9.0), (19, 19, 19))
    m = HybridModel(g, [Species("A", 1.0, particle=True, initial=200), Species("B", 1.0, initial=0.0)],
                    [Reaction("convert", {"A": 1}, {"B": 1}, k=0.5)])
    run_native(m, t_end=0.02, dt=0.01, output_dt=0.01, seed=1, workdir=tmp_path,
               geometry={"kind": "sphere", "center": [4.5, 4.5, 4.5], "radius": 4.0})
    mesh_file = next(tmp_path.glob("*.mesh"))
    ball = 4 / 3 * np.pi * 4.0**3
    dom = smoothed_domain(mesh_file, "cell")
    # node-centred voxels: whole voxels of spacing 0.5, centred on their nodes
    assert dom.volumes.sum() == pytest.approx(len(dom.global_index) * 0.5**3, rel=1e-3)
    assert abs(dom.volumes.sum() / ball - 1) < 0.03
    k, rem = np.divmod(dom.global_index, 19 * 19)
    j, i = np.divmod(rem, 19)
    assert np.abs(dom.raw_corners.mean(axis=1) - np.stack([i, j, k], 1) * 0.5).max() < 1e-5
    # smoothing pulls the boundary towards the sphere
    r = lambda p: np.linalg.norm(p - 4.5, axis=1)  # noqa: E731
    assert np.sqrt(((r(dom.surface_points) - 4) ** 2).mean()) < np.sqrt(((r(dom.raw_surface_points) - 4) ** 2).mean())
    # pyvcell's own convention tiles the box with N elements: smaller by ((N-1)/N)^3
    as_is = smoothed_domain(mesh_file, "cell", node_centred=False)
    assert as_is.volumes.sum() / dom.volumes.sum() == pytest.approx((18 / 19) ** 3, rel=1e-3)
