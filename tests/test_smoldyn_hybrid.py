"""Phase 1: the Smoldyn hybrid extensions (virtualcell/Smoldyn, branch pyhybrid).

- ``HybridGrid`` geometry follows vcell-fvsolver's CartesianMesh (node-centred, nearest node).
- ``getMoleculePositions`` / ``getMoleculeHistogram`` are numpy views of the molecules.
- Rates written ``k*B;`` in a configuration file read field ``B`` from the grid at each
  molecule (1st order) or at each grid node (0th order, one Poisson draw per node).

Skipped unless Smoldyn was built with ``pixi run build-smoldyn`` (OPTION_VCELL).
"""
from __future__ import annotations

import math
import textwrap

import numpy as np
import pytest

smoldyn = pytest.importorskip("smoldyn")
_smoldyn = smoldyn._smoldyn
if not hasattr(_smoldyn, "HybridGrid"):
    pytest.skip("Smoldyn built without the hybrid extensions", allow_module_level=True)
from smoldyn._smoldyn import HybridGrid, MolecState  # noqa: E402


# ---------------------------------------------------------------- grid geometry

def test_grid_geometry_matches_vcell_cartesian_mesh():
    g = HybridGrid(origin=[0.0, 0.0], size=[10.0, 5.0], num=[11, 6])
    assert g.dim == 2
    assert g.num == [11, 6]
    assert g.spacing == pytest.approx([1.0, 1.0])  # L/(N-1)
    assert g.shape == [6, 11]  # (Ny, Nx): C order == VCell order
    assert g.numElements() == 66
    # node i at x0 + i*dx; linear index i + Nx*j
    assert g.center(0) == pytest.approx([0.0, 0.0])
    assert g.center(11 * 2 + 3) == pytest.approx([3.0, 2.0])
    # nearest node, (int)((x-x0)*(N-1)/L + 0.5), clamped
    assert g.index([3.4, 2.6]) == 3 + 11 * 3
    assert g.index([3.6, 2.4]) == 4 + 11 * 2
    assert g.index([-1.0, 99.0]) == 0 + 11 * 5


def test_single_node_axis_uses_domain_centre():
    g = HybridGrid(origin=[0.0, 0.0, 0.0], size=[4.0, 4.0, 2.0], num=[5, 5, 1])
    assert g.spacing == pytest.approx([1.0, 1.0, 2.0])  # L when N == 1
    assert g.center(0) == pytest.approx([0.0, 0.0, 1.0])


def test_set_and_get_field_roundtrip():
    g = HybridGrid(origin=[0.0, 0.0], size=[2.0, 1.0], num=[3, 2])
    values = np.arange(6, dtype=float).reshape(2, 3)  # (Ny, Nx)
    g.setField("B", values)
    np.testing.assert_array_equal(g.getField("B"), values)
    g.setField("B", values.ravel())  # flat VCell order is accepted too
    assert g.fieldNames() == ["B"]
    with pytest.raises(Exception, match="has 5 values, grid has 6"):
        g.setField("C", np.zeros(5))
    with pytest.raises(KeyError):
        g.getField("missing")


# ---------------------------------------------------------------- model helpers

def _write_model(tmp_path, body: str) -> str:
    path = tmp_path / "model.txt"
    path.write_text(textwrap.dedent(body))
    return str(path)


_BOX_2D = """\
    dim 2
    boundaries 0 0 10
    boundaries 1 0 10
    species A
    difc A 0
    time_start 0
    time_stop 1
    time_step 0.01
    random_seed 7
"""

# Grid nodes at 0.5, 1.5, ..., 9.5 (strictly inside the box), spacing 1.
def _grid():
    return HybridGrid(origin=[0.5, 0.5], size=[9.0, 9.0], num=[10, 10])


# ---------------------------------------------------------------- molecule access

def test_positions_and_histogram(tmp_path):
    model = _write_model(tmp_path, _BOX_2D + "mol 2000 A u u\n")
    grid = _grid()
    sim = _smoldyn.Simulation(model, "q", grid)
    pos = sim.getMoleculePositions("A")
    assert pos.shape == (2000, 2)
    assert ((pos >= 0) & (pos <= 10)).all()
    assert sim.getMoleculePositions().shape == (2000, 2)  # species defaults to "all"
    assert sim.getMoleculePositions("A", MolecState.soln).shape == (2000, 2)

    hist = sim.getMoleculeHistogram("A", grid)
    assert hist.shape == (10, 10)
    assert hist.sum() == 2000
    # same nearest-node binning computed in numpy from the positions
    idx = np.clip(np.floor((pos - 0.5) * 9 / 9 + 0.5).astype(int), 0, 9)
    expected = np.zeros((10, 10), dtype=int)
    np.add.at(expected, (idx[:, 1], idx[:, 0]), 1)
    np.testing.assert_array_equal(hist, expected)

    with pytest.raises(ValueError, match="unknown species"):
        sim.getMoleculePositions("Z")


# ---------------------------------------------------------------- field-dependent rates

def _left_right_counts(sim):
    x = sim.getMoleculePositions("A")[:, 0]
    return int((x < 5).sum()), int((x >= 5).sum())


def test_first_order_rate_reads_field_at_molecule(tmp_path):
    """A -> 0 at rate k*B with B = 0 (x < 5) and B = 1 (x >= 5): only the right half decays."""
    model = _write_model(tmp_path, _BOX_2D + "mol 4000 A u u\nreaction decay A -> 0 kdecay*B;\n"
                         .replace("kdecay", "1.0"))
    grid = _grid()
    sim = _smoldyn.Simulation(model, "q", grid)
    assert grid.requiredFields() == ["B"]

    b = np.zeros((10, 10))
    b[:, 5:] = 1.0  # node x-index >= 5  <=>  x >= 5
    grid.setField("B", b)

    left0, right0 = _left_right_counts(sim)
    sim.runUntil(1.0, 0.01, display=False)
    left1, right1 = _left_right_counts(sim)

    assert left1 == left0  # zero rate where B = 0
    expected = right0 * math.exp(-1.0)
    sd = math.sqrt(right0 * math.exp(-1.0) * (1 - math.exp(-1.0)))
    assert abs(right1 - expected) < 4 * sd, (right1, expected, sd)


def test_unset_field_means_zero_rate_and_updates_take_effect(tmp_path):
    model = _write_model(tmp_path, _BOX_2D + "mol 1000 A u u\nreaction decay A -> 0 2.0*B;\n")
    grid = _grid()
    sim = _smoldyn.Simulation(model, "q", grid)
    sim.runUntil(0.5, 0.01, display=False)  # B never set: rate 0
    assert sim.getMoleculeCount("A", MolecState.all) == 1000

    grid.setField("B", np.full((10, 10), 1.0))  # now rate 2.0 everywhere
    sim.runUntil(1.0, 0.01, display=False)
    n = sim.getMoleculeCount("A", MolecState.all)
    expected = 1000 * math.exp(-2.0 * 0.5)
    assert abs(n - expected) < 4 * math.sqrt(1000 * math.exp(-1) * (1 - math.exp(-1))), n


_COMPARTMENT_2D = """\
    start_surface walls
    action both all reflect
    panel rect +0 0 0 10
    panel rect -0 10 0 10
    panel rect +1 0 0 10
    panel rect -1 0 10 10
    end_surface
    start_compartment cell
    surface walls
    point 5 5
    end_compartment
"""


def test_zeroth_order_creation_per_grid_node(tmp_path):
    """0 -> A at rate k*B: one Poisson draw per grid node with mean k*B*dt*dx*dy (VCell)."""
    model = _write_model(tmp_path, _BOX_2D + _COMPARTMENT_2D +
                         "reaction_cmpt cell make 0 -> A 2.0*B;\n")
    grid = _grid()
    sim = _smoldyn.Simulation(model, "q", grid)
    b = np.zeros((10, 10))
    b[:, 5:] = 5.0  # only the right half produces
    grid.setField("B", b)
    sim.runUntil(1.0, 0.01, display=False)

    pos = sim.getMoleculePositions("A")
    expected = 50 * 2.0 * 5.0 * 1.0 * 1.0 * 1.0  # nodes * k * B * dx*dy * T = 500
    assert abs(len(pos) - expected) < 4 * math.sqrt(expected), len(pos)
    # created within the producing nodes' cells (x in [4.5, 10])
    assert (pos[:, 0] >= 4.5).all()


def test_zeroth_order_creation_matches_curved_compartment_area(tmp_path):
    """Field-dependent creation in a disk equals rate × the disk's area, not its voxelized area.

    Each grid cell draws Poisson(rate·dt·cell) placed in the cell, keeping only those inside
    the compartment. That thinning is exact only if every cell that overlaps the compartment
    draws, including those whose centre lies outside it (fork fix; VCell tests the centre only).
    """
    disk = """\
    start_surface membrane
    action both all reflect
    panel sph 5 5 3.3 60
    end_surface
    start_compartment cell
    surface membrane
    point 5 5
    end_compartment
"""
    model = _write_model(tmp_path, _BOX_2D + disk + "reaction_cmpt cell make 0 -> A 200.0*B;\n")
    grid = _grid()
    sim = _smoldyn.Simulation(model, "q", grid)
    grid.setField("B", np.full((10, 10), 1.0))
    sim.runUntil(1.0, 0.01, display=False)

    pos = sim.getMoleculePositions("A")
    expected = 200.0 * math.pi * 3.3**2  # k·B·area·T
    assert abs(len(pos) - expected) < 4 * math.sqrt(expected), (len(pos), expected)
    assert (np.hypot(pos[:, 0] - 5, pos[:, 1] - 5) <= 3.3 + 1e-9).all()
