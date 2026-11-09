"""PDE → well-mixed adapters: RegionMean / RegionTotal reduce a grid field over a region by FV control volume."""
from __future__ import annotations

import numpy as np
import pytest
from process_bigraph import Composite, allocate_core

from viva_pde_particle.core import build_core
from viva_pde_particle.grid import CartesianGrid
from viva_pde_particle.steps import RegionMean, RegionTotal
from viva_pde_particle.units import MOLECULES_PER_UM3_PER_UM


def _step(cls, grid, region, species=("A",)):
    return cls(config={"grid": grid.to_config(), "species": list(species), "region": region}, core=allocate_core())


def _disk(grid, center, radius):
    x, y = grid.node_coordinates()
    return (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2


def test_uniform_field_returns_its_value():
    g = CartesianGrid((0, 0, 0), (4, 3, 2), (5, 4, 3))
    x, y, z = g.node_coordinates()
    region = {"mask": (x + y + z) <= 3.5}  # irregular, touches faces, edges and a corner
    mean = _step(RegionMean, g, region)
    assert mean.convert({"A": np.full(g.shape, 4.0)}) == {"A": 4.0}  # power of two: exact
    assert mean.convert({"A": np.full(g.shape, 3.7)})["A"] == pytest.approx(3.7, rel=1e-15)
    total = _step(RegionTotal, g, region).convert({"A": np.full(g.shape, 3.7)})["A"]
    assert total == pytest.approx(3.7 * mean.volume * MOLECULES_PER_UM3_PER_UM, rel=1e-14)


def test_values_outside_the_region_do_not_matter():
    g = CartesianGrid((0, 0), (10, 8), (11, 9))
    rng = np.random.default_rng(0)
    mask = _disk(g, (4.0, 4.0), 2.5)
    region = {"mask": mask, "volume_fraction": rng.uniform(0.2, 1.0, g.shape)}
    field = rng.uniform(0.0, 5.0, g.shape)
    perturbed = field.copy()
    perturbed[~mask] += rng.uniform(1e3, 1e6, int((~mask).sum()))
    for cls in (RegionMean, RegionTotal):
        step = _step(cls, g, region)
        assert step.convert({"A": perturbed}) == step.convert({"A": field})


def test_mean_is_volume_weighted_with_a_volume_fraction():
    g = CartesianGrid((0, 0), (4, 2), (3, 3))  # spacing (2, 1): V_full = 2 µm³ (unit-thickness slab)
    mask = np.zeros(g.shape, bool)
    vf = np.zeros(g.shape)
    c = np.full(g.shape, 100.0)  # outside the region
    for (j, i), f, v in zip([(0, 0), (0, 1), (1, 1), (2, 2)], [0.25, 0.5, 1.0, 0.8], [1.0, 2.0, 3.0, 4.0], strict=True):
        mask[j, i], vf[j, i], c[j, i] = True, f, v
    region = {"mask": mask, "volume_fraction": vf}
    # Σ fᵢ·cᵢ = 0.25 + 1.0 + 3.0 + 3.2 = 7.45; Σ fᵢ = 2.55; V_full cancels in the mean, not in the total
    assert _step(RegionMean, g, region).convert({"A": c})["A"] == pytest.approx(7.45 / 2.55, rel=1e-15)
    assert _step(RegionTotal, g, region).convert({"A": c})["A"] == pytest.approx(
        2.0 * 7.45 * MOLECULES_PER_UM3_PER_UM, rel=1e-15)


def test_mean_without_a_volume_fraction_uses_the_grid_element_volumes():
    g = CartesianGrid((0, 0), (4, 2), (3, 3))  # element volumes: corners 0.5, edges 1, centre 2; Σ = 8
    c = np.zeros(g.shape)
    c[0, 0] = 8.0  # a corner node: the plain node average would be 8/9
    mean = _step(RegionMean, g, {"mask": np.ones(g.shape, bool)})
    assert mean.volume == 8.0
    assert mean.convert({"A": c})["A"] == pytest.approx(0.5, rel=1e-15)


def test_missing_species_gets_no_output():
    g = CartesianGrid((0, 0), (4, 2), (3, 3))
    step = _step(RegionMean, g, {"mask": np.ones(g.shape, bool)}, species=["A", "B"])
    assert set(step.update({"fields": {"A": np.ones(g.shape)}})["mean_conc"]) == {"A"}


@pytest.mark.parametrize("cls", [RegionMean, RegionTotal])
def test_empty_or_inconsistent_region_raises(cls):
    g = CartesianGrid((0, 0), (4, 2), (3, 3))
    with pytest.raises(ValueError, match="selects no grid node"):
        _step(cls, g, {"mask": np.zeros(g.shape, bool)})
    with pytest.raises(ValueError, match="region volume must be positive"):
        _step(cls, g, {"mask": np.ones(g.shape, bool), "volume_fraction": np.zeros(g.shape)})
    with pytest.raises(ValueError, match="region needs a mask"):
        _step(cls, g, {"volume_fraction": np.ones(g.shape)})
    with pytest.raises(ValueError, match=r"region mask has shape \(2, 2\)"):
        _step(cls, g, {"mask": np.ones((2, 2), bool)})
    with pytest.raises(ValueError, match=r"region volume_fraction has shape \(2, 2\)"):
        _step(cls, g, {"mask": np.ones(g.shape, bool), "volume_fraction": np.ones((2, 2))})
    with pytest.raises(ValueError, match="volume_fraction must be >= 0"):
        _step(cls, g, {"mask": np.ones(g.shape, bool), "volume_fraction": np.where(np.eye(3, dtype=bool), -1.0, 1.0)})


def test_shapes_are_checked_not_just_sizes():
    g = CartesianGrid((0, 0), (10, 8), (11, 9))  # array shape (9, 11)
    mask = np.zeros(g.shape, bool)
    mask[0, :3] = True
    with pytest.raises(ValueError, match=r"region mask has shape \(11, 9\)"):
        _step(RegionMean, g, {"mask": mask.reshape(11, 9)})  # same size, wrong layout
    step = _step(RegionMean, g, {"mask": mask})
    with pytest.raises(ValueError, match=r"field 'A' has shape \(11, 9\)"):
        step.convert({"A": np.ones((11, 9))})


def test_composite_domain_mean_times_volume_is_the_conserved_total():
    """FVReactionDiffusion (diffusion only, partial-volume subdomain) → RegionMean / RegionTotal in one Composite."""
    g = CartesianGrid((0, 0), (10, 8), (11, 9))
    rng = np.random.default_rng(1)
    domain = {"mask": _disk(g, (5.0, 4.0), 3.6), "volume_fraction": rng.uniform(0.3, 1.0, g.shape)}
    x, y = g.node_coordinates()
    u0 = np.where(domain["mask"], 10.0 * np.exp(-((x - 4.0) ** 2 + (y - 4.0) ** 2) / 2.0), 7.0)
    cell = _disk(g, (7.5, 4.0), 1.1) & domain["mask"]  # the voxels a well-mixed cell reads, away from the bump
    dt = 0.1

    doc = {
        "fields": {"TNF": u0},
        "pde": {"_type": "process", "address": "local:FVReactionDiffusion", "interval": dt,
                "config": {"grid": g.to_config(), "pde": {"species": {"TNF": {"diffusion": 1.0}}, "terms": []},
                           "dt": dt, "domain": domain},
                "inputs": {"fields": ["fields"], "external_conc": ["external_conc"]},
                "outputs": {"fields": ["fields"]}},
    }
    # the cell's region as JSON lists, the way a submitted document carries it
    cell_region = {"mask": cell.tolist(), "volume_fraction": domain["volume_fraction"].tolist()}
    adapters = [("domain_mean", "RegionMean", domain, "mean_conc", "tnf_domain_mean"),
                ("domain_total", "RegionTotal", domain, "molecules", "tnf_domain_molecules"),
                ("cell_mean", "RegionMean", cell_region, "mean_conc", "tnf_at_cell")]
    for node, name, region, port, store in adapters:
        doc[node] = {"_type": "step", "address": f"local:{name}",
                     "config": {"grid": g.to_config(), "species": ["TNF"], "region": region},
                     "inputs": {"fields": ["fields"]}, "outputs": {port: [store]}}
    sim = Composite({"state": doc}, core=build_core())  # output stores unseeded: the steps' initial_state fills them
    for store in ("tnf_domain_mean", "tnf_domain_molecules", "tnf_at_cell"):
        assert np.isnan(sim.state[store]["TNF"])  # nothing computed yet, and visibly so

    w = g.full_volume * domain["volume_fraction"]
    volume = float(w[domain["mask"]].sum())
    total0 = float((w * u0)[domain["mask"]].sum())  # µM·µm³, what the FV engine conserves
    naive0 = float(u0[domain["mask"]].mean())
    cell_means = []
    for _ in range(20):
        sim.run(dt)
        u = np.asarray(sim.state["fields"]["TNF"])
        assert sim.state["tnf_domain_mean"]["TNF"] * volume == pytest.approx(total0, rel=1e-12)
        assert sim.state["tnf_domain_molecules"]["TNF"] == pytest.approx(total0 * MOLECULES_PER_UM3_PER_UM, rel=1e-12)
        # the cell's mean is the current field's, not a stale one
        expected = float((w * u)[cell].sum() / w[cell].sum())
        assert sim.state["tnf_at_cell"]["TNF"] == pytest.approx(expected, rel=1e-12)
        cell_means.append(sim.state["tnf_at_cell"]["TNF"])
    assert cell_means[-1] > cell_means[0]  # the bump diffused toward the cell
    # the plain node average is not the conserved quantity on this partial-volume grid
    assert abs(float(u[domain["mask"]].mean()) - naive0) > 1e-3
