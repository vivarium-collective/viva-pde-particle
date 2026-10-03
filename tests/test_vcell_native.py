"""Native VCell hybrid reference (pyvcell -> libvcell -> pyvcell-fvsolver).

Validation runs everywhere; building and solving need a pyvcell with spatial-hybrid
support (the ``dev`` pixi environment until that pyvcell is released).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from viva_pde_particle.composites.examples import field_modulated_decay_model
from viva_pde_particle.grid import CartesianGrid
from viva_pde_particle.model import HybridModel, Species
from viva_pde_particle.reference.vcell_native import hybrid_support_available, validate_for_native

needs_hybrid = pytest.mark.skipif(not hybrid_support_available(), reason="pyvcell without spatial-hybrid support")


def test_validation_rejects_unsupported_models():
    with pytest.raises(ValueError, match="3D"):
        validate_for_native(field_modulated_decay_model())  # 2D
    slab2 = CartesianGrid((0, 0, 0), (10, 10, 1), (11, 11, 2))
    with pytest.raises(ValueError, match="3 nodes"):
        validate_for_native(HybridModel(slab2, [Species("A", 1, particle=True, initial=10), Species("B", 1)]))
    m = field_modulated_decay_model(thickness=1.0)
    validate_for_native(m)  # ok
    per_node = HybridModel(m.grid, [Species("A", 0, particle=True, initial=np.ones(m.grid.shape)), Species("B", 0)])
    validate_for_native(per_node)  # integer per-node counts are placed explicitly
    fractional = HybridModel(m.grid, [Species("A", 0, particle=True, initial=np.full(m.grid.shape, 0.5)),
                                      Species("B", 0)])
    with pytest.raises(ValueError, match="integers"):
        validate_for_native(fractional)


@needs_hybrid
def test_biomodel_is_a_hybrid_application():
    from pyvcell.vcml.models_app import HYBRID_SOLVER

    from viva_pde_particle.reference.vcell_native import to_biomodel

    bm = to_biomodel(field_modulated_decay_model(thickness=1.0), t_end=1.0, dt=0.01, output_dt=0.5, seed=3)
    app = bm.applications[0]
    assert app.stochastic and app.simulations[0].solver == HYBRID_SOLVER
    assert app.get_species_mapping("B").force_continuous and not app.get_species_mapping("A").force_continuous
    assert app.get_species_mapping("B").init_conc == "(x / 10.0)"
    assert app.simulations[0].mesh_size == (11, 11, 3)


@needs_hybrid
def test_native_field_modulated_decay_follows_local_field():
    from viva_pde_particle.reference.vcell_native import run_native

    model = field_modulated_decay_model(n_particles=4000, k=1.0, thickness=1.0)
    tr = run_native(model, t_end=1.0, dt=0.01, output_dt=1.0, seed=5)
    a, b = tr.particle_counts["A"], tr.fields["B"]
    assert a.shape == (2, 3, 11, 11)
    np.testing.assert_allclose(b[0, 0, 0], np.linspace(0, 1, 11), atol=1e-12)
    n0, n1 = a[0].sum(axis=(0, 1)), a[-1].sum(axis=(0, 1))
    p = np.exp(-b[0, 0, 0])
    sd = np.sqrt(np.maximum(n0 * p * (1 - p), 1e-12))
    assert n1[0] == n0[0]  # B = 0 column
    assert np.max(np.abs(n1 - n0 * p) / sd) < 4.5
    assert abs(n0.sum() - 4000) < 5 * math.sqrt(4000)  # Poisson initial total


def test_drop_noop_reactions_and_place_particles(tmp_path):
    from viva_pde_particle.reference.vcell_native import drop_noop_reactions, place_particles

    f = tmp_path / "model.smoldynInput"
    f.write_text("\n".join([
        "species C", "species O",
        "reaction open C -> O 1.0;",
        "reaction close O -> C 5.0;",
        "reaction influx O -> O 6022.14076;",
        "reaction_cmpt cell make 0 -> O 2.0*U;",
        "reaction dimer O + C -> C + O 3.0",
        "compartment_mol 7 C cell",
        "mol 3 O 1 2 3",
        "end_file",
    ]) + "\n")
    assert drop_noop_reactions(f) == ["influx", "dimer"]
    place_particles(f, {"C": np.array([[1.0, 2.0, 0.5], [3.0, 4.0, 0.5]])})
    lines = f.read_text().splitlines()
    assert "reaction open C -> O 1.0;" in lines and "reaction_cmpt cell make 0 -> O 2.0*U;" in lines
    assert not any("influx" in ln or "dimer" in ln for ln in lines)
    assert "compartment_mol 7 C cell" not in lines and "mol 3 O 1 2 3" in lines  # only C replaced
    assert lines[-3:] == ["mol 1 C 1.0 2.0 0.5", "mol 1 C 3.0 4.0 0.5", "end_file"]


def test_particle_positions_from_per_node_counts():
    from viva_pde_particle.reference.vcell_native import particle_positions

    g = CartesianGrid((0, 0, 0), (2.0, 2.0, 1.0), (3, 3, 3))
    counts = np.zeros(g.shape)
    counts[0, 1, 2] = 2  # z=0, y=1, x=2 -> node (2.0, 1.0, 0.0), twice
    m = HybridModel(g, [Species("C", 0, particle=True, initial=counts), Species("U", 1)])
    pts = particle_positions(m)["C"]
    assert pts.shape == (2, 3)
    np.testing.assert_allclose(pts[:, :2], [[2.0 - 2e-6, 1.0]] * 2, atol=1e-9)  # nudged inside x = 2
    assert np.all(pts[:, 2] > 0)


@needs_hybrid
def test_zero_reactant_reactions_become_general_kinetics():
    """VCell mass action drops Kf when there are no reactants; sources must be General kinetics."""
    from viva_pde_particle.benchmarks.calcium_sparks import TEST1, calcium_sparks_model
    from viva_pde_particle.reference.vcell_native import to_biomodel

    bm = to_biomodel(calcium_sparks_model(TEST1), t_end=0.1, dt=0.01, output_dt=0.1)
    leak = next(r for r in bm.model.reactions if r.name == "leak")
    assert leak.kinetics.kinetics_type == "GeneralKinetics"
    assert [(p.name, float(p.value)) for p in leak.kinetics.kinetics_parameters] == [("J", 0.1)]
    pump = next(r for r in bm.model.reactions if r.name == "pump")
    assert pump.kinetics.kinetics_type == "MassAction"
