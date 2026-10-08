"""Hybrid models in SBML Spatial (Phase 6).

SBML Spatial carries the compartments, geometry, species, diffusion, initial conditions and reactions. Which species
are particles is one VCell-namespace annotation per species (virtualcell/vcell#2175):

    <vcell:SpeciesContextSpecSettings vcell:representation="particle"/>

with values ``continuous`` (the default when absent) and ``particle``. Plain SBML Spatial is therefore a deterministic
PDE model.

Both directions go through VCell, which owns the SBML Spatial mapping:

- :func:`to_sbml`: ``HybridModel`` → pyvcell BioModel (:func:`~viva_pde_particle.reference.vcell_native.to_biomodel`)
  → libvcell ``vcml_to_sbml``.
- :func:`from_sbml`: libvcell ``sbml_to_vcml`` → pyvcell BioModel in VCell units → ``HybridModel``, plus a vcell-fenics
  ``GeometryDescription`` when the geometry is not the plain box.

SBML carries its own units (VCell writes dm, µmol, l); :func:`from_sbml` converts to VCell's µm, µM and s with
``pyvcell.vcml.load_sbml_str(..., unit_system="vcell")``. Simulation settings (mesh, Δt, Smoldyn step multiplier, seed) are not part of the
model, so :func:`from_sbml` takes the mesh size, and located initial particles (per-node counts) cannot be written.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from viva_pde_particle.grid import CartesianGrid
from viva_pde_particle.model.hybrid_model import HybridModel, Reaction, Species
from viva_pde_particle.units import MOLECULES_PER_UM3_PER_UM

#: the application name written by :func:`to_sbml`
APPLICATION = "hybrid"

@dataclass
class SbmlHybrid:
    """A hybrid model read from SBML Spatial."""

    model: HybridModel
    #: None for the plain box (the species fill the grid); otherwise the geometry, in µm
    geometry: object | None
    #: the subvolume holding the species
    region: str


def to_sbml(model: HybridModel, *, geometry=None, region: str | None = None, domain_volume: float | None = None,
            round_trip_validation: bool = True) -> str:
    """SBML Spatial (Level 3) for a hybrid model, with each species' representation annotated.

    ``geometry``, ``region`` and ``domain_volume`` are as for
    :func:`~viva_pde_particle.reference.vcell_native.to_biomodel`. Particle species need scalar (total count)
    initial conditions; per-node counts cannot be written to SBML.
    """
    from pyvcell.vcml.utils import to_sbml_str

    from viva_pde_particle.reference.vcell_native import to_biomodel

    for s in model.species:
        if s.particle and np.asarray(s.initial).ndim != 0:
            raise ValueError(f"species {s.name}: per-node particle counts cannot be written to SBML")
    # the simulation is VCML-only; any valid settings will do
    bm = to_biomodel(model, t_end=1.0, dt=0.01, output_dt=0.1, geometry=geometry, region=region,
                     domain_volume=domain_volume, name=APPLICATION)
    return to_sbml_str(bm, APPLICATION, round_trip_validation=round_trip_validation)


def write_sbml(path: str | Path, model: HybridModel, **kwargs) -> Path:
    path = Path(path)
    path.write_text(to_sbml(model, **kwargs))
    return path


def from_sbml(sbml: str, *, num: tuple[int, int, int], domain_volume: float | None = None) -> SbmlHybrid:
    """A ``HybridModel`` from SBML Spatial.

    ``num`` is the mesh (node counts per axis); the grid spans the geometry's extent. Particle species start with
    ``initial concentration × region volume`` molecules, placed uniformly; ``domain_volume`` (µm³) is that region
    volume, needed only when particles start non-empty in a geometry other than the plain box.
    """
    from pyvcell.vcml.utils import load_sbml_str

    # VCell imports SBML in the SBML's own units; load it in VCell's (µm, µM, s), which the model here uses
    bm = load_sbml_str(sbml, unit_system="vcell")
    if len(bm.applications) != 1:
        raise ValueError(f"expected one application, found {len(bm.applications)}")
    app = bm.applications[0]
    geo = app.geometry
    if geo.dim != 3:
        raise NotImplementedError(f"{geo.dim}D geometry (hybrid models are 3D)")

    compartments = {s.compartment_name for s in bm.model.species}
    if len(compartments) != 1:
        raise NotImplementedError(f"species in several compartments {sorted(compartments)}; one is supported")
    (compartment,) = compartments
    region = next(m.geometry_class_name for m in app.compartment_mappings if m.compartment_name == compartment)

    grid = CartesianGrid(tuple(geo.origin), tuple(geo.extent), tuple(num))
    if len(geo.subvolumes) == 1:
        geometry = None
        volume = float(np.prod(geo.extent))
    else:
        from viva_pde_particle.geometry.vcml import from_vcml_geometry

        geometry = from_vcml_geometry(geo)
        volume = domain_volume

    species = []
    for sm in app.species_mappings:
        particle = bool(app.stochastic and not sm.force_continuous)
        diffusion = _number(sm.diff_coef, f"diffusion of {sm.species_name}")
        c0 = sm.init_conc
        if particle:
            c0 = _number(c0, f"initial concentration of particle species {sm.species_name}")
            if c0 and volume is None:
                raise ValueError(f"particle species {sm.species_name} starts non-empty: give domain_volume")
            initial = round(c0 * MOLECULES_PER_UM3_PER_UM * volume) if c0 else 0
            species.append(Species(sm.species_name, diffusion, particle=True, initial=initial))
        elif isinstance(c0, str):
            species.append(Species(sm.species_name, diffusion, initial=_evaluate_on_grid(c0, grid), initial_expression=c0))
        else:
            species.append(Species(sm.species_name, diffusion, initial=float(c0 or 0.0)))

    params = {p.name: _number(p.value, p.name) for p in bm.model.model_parameters}
    reactions = []
    for r in bm.model.reactions:
        reactions.extend(_mass_action(r, params))
    return SbmlHybrid(HybridModel(grid, species, reactions), geometry, region)


def read_sbml(path: str | Path, **kwargs) -> SbmlHybrid:
    return from_sbml(Path(path).read_text(), **kwargs)


# ---------------------------------------------------------------------------------------------------- helpers


def _number(value, what: str) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        raise NotImplementedError(f"{what} is an expression ({value!r}); only numbers are supported") from None


def _python(expr: str) -> str:
    import libvcell

    ok, msg, py = libvcell.vcell_infix_to_python_infix(expr)
    if not ok:
        raise ValueError(f"cannot translate VCell expression {expr!r}: {msg}")
    return py


def _evaluate_on_grid(expr: str, grid: CartesianGrid) -> np.ndarray:
    x, y, z = grid.node_coordinates()
    env = {"x": x, "y": y, "z": z, "np": np, "math": np}
    return np.broadcast_to(eval(compile(_python(expr), "<sbml>", "eval"), env), grid.shape).astype(float)


def _mass_action(reaction, params: dict[str, float]) -> list[Reaction]:
    """Mass-action ``Reaction``s from a VCell reaction, irreversible, in µM units.

    VCell's SBML import returns mass action as general kinetics ``J = kf·Π reactants − kr·Π products``. ``J`` is
    evaluated at random concentrations to recover kf and kr, and checked at further points, so anything that is not
    mass action is rejected rather than approximated.
    """
    kin = reaction.kinetics
    local = {p.name: p for p in kin.kinetics_parameters}
    reactants = {r.name: int(r.stoichiometry) for r in reaction.reactants}
    products = {p.name: int(p.stoichiometry) for p in reaction.products}
    if kin.kinetics_type == "MassAction":
        kf = _number(local["Kf"].value, f"{reaction.name} Kf")
        kr = _number(local["Kr"].value, f"{reaction.name} Kr")
    else:
        rate = next((p for p in kin.kinetics_parameters if p.role == "reaction rate"), None)
        if rate is None:
            raise NotImplementedError(f"reaction {reaction.name}: kinetics {kin.kinetics_type}")
        values = dict(params)
        for p in kin.kinetics_parameters:
            if p is not rate:
                values[p.name] = _number(p.value, f"{reaction.name} {p.name}")
        code = compile(_python(str(rate.value)), reaction.name, "eval")
        names = {n.id for n in ast.walk(ast.parse(_python(str(rate.value)))) if isinstance(n, ast.Name)}
        species = sorted(names - set(values))
        rng = np.random.default_rng(0)

        def J(s):
            return float(eval(code, {"math": np, "np": np}, {**values, **s}))

        def basis(s):
            return (np.prod([s.get(n, 0.0) ** k for n, k in reactants.items()]),
                    np.prod([s.get(n, 0.0) ** k for n, k in products.items()]))

        points = [{n: float(v) for n, v in zip(species, rng.uniform(0.5, 2.0, len(species)))} for _ in range(5)]
        A = np.array([[b[0], -b[1]] for b in map(basis, points)])
        j = np.array([J(s) for s in points])
        (kf, kr), *_ = np.linalg.lstsq(A, j, rcond=None)
        if not np.allclose(A @ [kf, kr], j, rtol=1e-9, atol=1e-12 * max(1.0, np.abs(j).max())):
            raise NotImplementedError(f"reaction {reaction.name}: rate {rate.value!r} is not mass action")
        # the fit is exact up to float roundoff: drop a vanishing term and round to 12 significant digits
        scale = max(abs(kf), abs(kr))
        kf, kr = (0.0 if abs(k) <= 1e-12 * scale else float(f"{k:.12g}") for k in (kf, kr))
    out = []
    if kf:
        out.append(Reaction(reaction.name, reactants, products, k=float(kf)))
    if kr:
        out.append(Reaction(f"{reaction.name}_reverse", products, reactants, k=float(kr)))
    return out
