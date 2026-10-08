"""Phase 6: hybrid models through SBML Spatial (viva_pde_particle.model.sbml).

Needs libvcell built with VCell's SBML hybrid support (virtualcell/vcell#2175) and pyvcell's unit conversion (virtualcell/pyvcell#63);
skipped otherwise.
"""
import re

import numpy as np
import pytest

pytest.importorskip("libvcell")
pytest.importorskip("pyvcell")

from viva_pde_particle.composites.examples import two_way_exchange_model  # noqa: E402
from viva_pde_particle.model import Reaction  # noqa: E402
from viva_pde_particle.model.sbml import from_sbml, to_sbml  # noqa: E402
from viva_pde_particle.units import MOLECULES_PER_UM3_PER_UM  # noqa: E402


@pytest.fixture(scope="module")
def hybrid():
    """The two-way exchange slab with a starting particle count, a bimolecular and a zero-order reaction."""
    from pyvcell.vcml import utils

    if not hasattr(utils, "convert_vcml_units"):
        pytest.skip("pyvcell without unit conversion (load_sbml_str unit_system=)")
    m = two_way_exchange_model(1.0, 0.5, 0.2)
    m.species_by_name("A").initial = 2000
    m.reactions.append(Reaction("bind", {"A": 1, "B": 1}, {"B": 1}, k=0.3))
    m.reactions.append(Reaction("source", {}, {"B": 1}, k=0.05))
    try:
        sbml = to_sbml(m)
    except Exception as e:  # libvcell's VCell refuses spatial stochastic export
        pytest.skip(f"libvcell without SBML hybrid support: {e}")
    if "representation=" not in sbml:
        pytest.skip("libvcell without SBML hybrid support")
    return m, sbml


def annotations(sbml):
    return dict(re.findall(r'<species[^>]*\bid="(\w+)".*?vcell:representation="(\w+)"', sbml, re.DOTALL))


def test_representation_is_annotated(hybrid):
    _, sbml = hybrid
    assert annotations(sbml) == {"A": "particle", "B": "continuous"}


def test_round_trip(hybrid):
    m, sbml = hybrid
    r = from_sbml(sbml, num=m.grid.num)
    assert r.geometry is None and r.model.grid == m.grid
    for a, b in zip(m.species, r.model.species):
        assert (b.name, b.particle, b.diffusion) == (a.name, a.particle, a.diffusion)
        assert np.asarray(b.initial) == pytest.approx(np.asarray(a.initial))
    assert {(x.name, tuple(sorted(x.reactants.items())), tuple(sorted(x.products.items())), x.k)
            for x in r.model.reactions} == {
        (x.name, tuple(sorted(x.reactants.items())), tuple(sorted(x.products.items())), x.k) for x in m.reactions}


def test_absent_representation_is_continuous(hybrid):
    m, sbml = hybrid
    plain = re.sub(r"\s*<vcell:SpeciesContextSpecSettings[^>]*/>", "", sbml)
    r = from_sbml(plain, num=m.grid.num)
    assert r.model.particle_species == []
    # A is now a field: its 2000 molecules become their concentration over the 10×10×1 µm³ slab
    assert r.model.species_by_name("A").initial == pytest.approx(2000 / (MOLECULES_PER_UM3_PER_UM * 100.0))
