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
    with pytest.raises(ValueError, match="per-node"):
        validate_for_native(per_node)


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
