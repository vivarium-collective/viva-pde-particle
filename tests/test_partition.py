"""HybridModel -> PDE/particle partition (port of VCell's ParticleMathMapping.combineHybrid)."""
from __future__ import annotations

import numpy as np
import pytest

from viva_pde_particle.grid import CartesianGrid
from viva_pde_particle.model import (
    HybridModel,
    Reaction,
    Species,
    initial_particle_counts,
    partition,
    write_smoldyn_config,
)
from viva_pde_particle.units import MOLECULES_PER_UM3_PER_UM as NA


def _model():
    g = CartesianGrid((0.0, 0.0), (10.0, 10.0), (11, 11))
    return HybridModel(
        grid=g,
        species=[
            Species("A", 1.0, particle=True, initial=100),
            Species("B", 2.0, initial=1.5),
            Species("C", 3.0),
        ],
        reactions=[
            Reaction("decay", {"A": 1}, {}, k=0.1),            # particle only
            Reaction("bind", {"A": 1, "B": 1}, {"C": 1}, k=0.2),  # mixed: field folded in
            Reaction("make", {"B": 1}, {"A": 1, "B": 1}, k=0.3),  # 0th order creation k*[B]
            Reaction("dimer", {"A": 2}, {}, k=0.4),            # 2nd order particle
            Reaction("cc", {"B": 1}, {"C": 1}, k=0.5),          # continuous only
        ],
    )


def test_pde_side_terms():
    pde = partition(_model()).pde
    assert set(pde["species"]) == {"B", "C"}
    assert pde["particle_species"] == ["A"]
    terms = {(t["reaction"], t["species"]): t["coeff"] for t in pde["terms"]}
    # bind consumes B and makes C; make leaves B unchanged (catalyst); cc converts B -> C
    assert terms == {("bind", "B"): -1, ("bind", "C"): 1, ("cc", "B"): -1, ("cc", "C"): 1}


def test_particle_side_reactions_and_units():
    rx = {r["name"]: r for r in partition(_model()).particles["reactions"]}
    assert set(rx) == {"decay", "bind", "make", "dimer"}  # cc is continuous-only
    assert rx["decay"] == {"name": "decay", "reactants": ["A"], "products": [], "order": 1,
                           "rate_constant": 0.1, "fields": []}
    assert rx["bind"]["reactants"] == ["A"] and rx["bind"]["products"] == []  # C stays on PDE side
    assert rx["bind"]["fields"] == ["B"] and rx["bind"]["rate_constant"] == pytest.approx(0.2)
    assert rx["make"]["order"] == 0 and rx["make"]["products"] == ["A"]
    assert rx["make"]["rate_constant"] == pytest.approx(0.3 * NA)  # µM/s -> molecules/µm³/s
    assert rx["dimer"]["reactants"] == ["A", "A"]
    assert rx["dimer"]["rate_constant"] == pytest.approx(0.4 / NA)  # 1/(µM s) -> µm³/s
    assert partition(_model()).particles["field_species"] == ["B"]


def test_smoldyn_config_text():
    m = _model()
    parts = partition(m)
    counts = initial_particle_counts(m, np.random.default_rng(0))
    assert counts["A"].sum() == 100
    text = write_smoldyn_config(parts.particles, m.grid, counts, time_step=0.02, seed=3)
    lines = text.splitlines()
    assert "dim 2" in lines and "boundaries 0 0.0 10.0 r" in lines
    assert "difc A 1.0" in lines and "time_step 0.02" in lines and "random_seed 3" in lines
    assert "reaction decay A -> 0 0.1" in lines
    assert "reaction bind A -> 0 0.2*B;" in lines
    assert any(ln.startswith("reaction_cmpt domain make 0 -> A ") and ln.endswith("*B;") for ln in lines)
    assert "start_compartment domain" in lines  # needed for grid-based 0th order creation
    assert sum(int(ln.split()[1]) for ln in lines if ln.startswith("mol ")) == 100
    assert lines[-1] == "end_file"


def test_unknown_species_and_too_many_particles_rejected():
    g = CartesianGrid((0.0,), (1.0,), (3,))
    with pytest.raises(ValueError, match="unknown species"):
        HybridModel(g, [Species("A", 1.0)], [Reaction("r", {"Z": 1}, {}, 1.0)])
    m = HybridModel(g, [Species("A", 1.0, particle=True)], [Reaction("r", {"A": 3}, {}, 1.0)])
    with pytest.raises(ValueError, match="at most 2"):
        partition(m)
