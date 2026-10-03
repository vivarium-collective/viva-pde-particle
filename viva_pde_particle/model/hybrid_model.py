"""A minimal hybrid reaction-diffusion model and its PDE/particle partition.

One model describes all species and mass-action reactions. A set of species is
marked as particles, and :func:`partition` splits the model the way VCell's
``ParticleMathMapping.combineHybrid()`` does:

- **PDE side:** every continuous species keeps all its reaction terms. A particle
  species appearing in a term is read as its binned concentration (µM).
- **Particle side:** every reaction with a particle reactant or product becomes a
  Smoldyn reaction over the particle species only. Continuous reactants are folded
  into the rate (``k·[B]``), and continuous products are dropped (the PDE side adds
  them). Reactions among continuous species only are not sent to Smoldyn.

Neither side exchanges fluxes. Each applies its own half of a mixed reaction,
reading the other's state from the previous coupling step.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from viva_pde_particle.grid import CartesianGrid
from viva_pde_particle.units import smoldyn_rate_factor


@dataclass
class Species:
    name: str
    diffusion: float  # µm²/s
    particle: bool = False
    #: continuous: µM (scalar or array of grid shape); particle: total count (int,
    #: placed uniformly) or per-node counts (array of grid shape)
    initial: float | np.ndarray = 0.0


@dataclass
class Reaction:
    name: str
    reactants: dict[str, int]
    products: dict[str, int]
    k: float  # mass action, µM^(1-order)/s

    @property
    def order(self) -> int:
        return sum(self.reactants.values())


@dataclass
class HybridModel:
    grid: CartesianGrid
    species: list[Species]
    reactions: list[Reaction] = field(default_factory=list)

    def __post_init__(self):
        names = [s.name for s in self.species]
        if len(set(names)) != len(names):
            raise ValueError("duplicate species names")
        for r in self.reactions:
            unknown = (set(r.reactants) | set(r.products)) - set(names)
            if unknown:
                raise ValueError(f"reaction {r.name}: unknown species {sorted(unknown)}")

    @property
    def particle_species(self) -> list[str]:
        return [s.name for s in self.species if s.particle]

    @property
    def continuous_species(self) -> list[str]:
        return [s.name for s in self.species if not s.particle]

    def species_by_name(self, name: str) -> Species:
        return next(s for s in self.species if s.name == name)


@dataclass
class PartitionedModel:
    """Plain-data (config-serializable) PDE and particle halves of a HybridModel."""

    pde: dict
    particles: dict


def partition(model: HybridModel) -> PartitionedModel:
    particle = set(model.particle_species)
    continuous = model.continuous_species

    terms = []
    for r in model.reactions:
        net: dict[str, int] = {}
        for s, n in r.reactants.items():
            net[s] = net.get(s, 0) - n
        for s, n in r.products.items():
            net[s] = net.get(s, 0) + n
        for s, coeff in net.items():
            if s in continuous and coeff != 0:
                terms.append({"species": s, "coeff": coeff, "k": r.k, "reactants": dict(r.reactants), "reaction": r.name})

    pde = {
        "species": {
            s: {"diffusion": model.species_by_name(s).diffusion} for s in continuous
        },
        "particle_species": sorted(particle),
        "terms": terms,
    }

    reactions = []
    for r in model.reactions:
        p_rct = {s: n for s, n in r.reactants.items() if s in particle}
        p_prd = {s: n for s, n in r.products.items() if s in particle}
        if not p_rct and not p_prd:
            continue  # continuous-only: PDE side
        c_rct = {s: n for s, n in r.reactants.items() if s not in particle}
        order = sum(p_rct.values())
        reactants = [s for s, n in p_rct.items() for _ in range(n)]
        products = [s for s, n in p_prd.items() for _ in range(n)]
        if len(reactants) > 2 or len(products) > 2:
            raise ValueError(f"reaction {r.name}: Smoldyn supports at most 2 particle reactants/products")
        reactions.append({
            "name": r.name,
            "reactants": reactants,
            "products": products,
            "order": order,
            "rate_constant": r.k * smoldyn_rate_factor(order),
            "fields": [s for s, n in c_rct.items() for _ in range(n)],
        })

    particles = {
        "species": {
            s: {"diffusion": model.species_by_name(s).diffusion} for s in sorted(particle)
        },
        "field_species": sorted({f for rx in reactions for f in rx["fields"]}),
        "reactions": reactions,
    }
    return PartitionedModel(pde=pde, particles=particles)


def initial_fields(model: HybridModel) -> dict[str, np.ndarray]:
    """Initial continuous fields (µM), broadcast to the grid shape."""
    shape = model.grid.shape
    return {
        s.name: np.broadcast_to(np.asarray(s.initial, dtype=float), shape).copy()
        for s in model.species if not s.particle
    }


def initial_particle_counts(model: HybridModel, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Initial per-node molecule counts for each particle species.

    A scalar total is spread uniformly by volume (multinomial over element volumes,
    matching uniform random placement); an array is taken as per-node counts.
    """
    g = model.grid
    out = {}
    for s in model.species:
        if not s.particle:
            continue
        init = np.asarray(s.initial)
        if init.ndim == 0:
            p = (g.volume_fraction / g.volume_fraction.sum()).ravel()
            out[s.name] = rng.multinomial(int(init), p).reshape(g.shape).astype(float)
        else:
            if init.shape != g.shape:
                raise ValueError(f"species {s.name}: initial counts shape {init.shape} != grid {g.shape}")
            out[s.name] = init.astype(float)
    return out
