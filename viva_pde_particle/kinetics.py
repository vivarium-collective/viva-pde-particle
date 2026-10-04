"""Mass-action reaction terms, shared by every PDE engine (Phase 7f).

Kept outside the engines so that no engine imports another: FV, FEniCSx Q1 and the P1 mesh engine
all evaluate the partitioned model's PDE terms with :func:`reaction_rates`.
"""
from __future__ import annotations

import numpy as np


def reaction_rates(terms: list[dict], species: list[str], conc: dict[str, np.ndarray], shape) -> dict[str, np.ndarray]:
    """dC/dt from mass-action terms (µM/s) for each continuous species (shared by the PDE engines)."""
    rates = {s: np.zeros(shape) for s in species}
    for term in terms:
        r = np.full(shape, term["coeff"] * term["k"])
        for s, n in term["reactants"].items():
            r = r * conc[s] ** n
        rates[term["species"]] += r
    return rates
