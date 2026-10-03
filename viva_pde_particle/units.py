"""Unit conventions shared by the PDE and particle processes.

Units follow VCell: lengths in µm, time in s, volume concentrations in µM, and
particle counts as molecule numbers. Mass-action rate constants are µM-based,
``k`` in ``µM^(1-n)/s`` for a reaction of total order ``n``.

2D models are slabs of unit thickness (1 µm), as in VCell's 2D meshes. Volume and
area densities are then numerically equal, and so are the Smoldyn rate constants.
"""
from __future__ import annotations

#: molecules per µm³ in 1 µM (Avogadro's number x 1e-6 mol/L x 1e-15 L/µm³)
MOLECULES_PER_UM3_PER_UM = 602.214076


def uM_to_molecules_per_um3(conc_uM):
    return conc_uM * MOLECULES_PER_UM3_PER_UM


def molecules_per_um3_to_uM(density):
    return density / MOLECULES_PER_UM3_PER_UM


def smoldyn_rate_factor(particle_order: int) -> float:
    """Factor converting a µM-based rate (with field reactants folded in) to Smoldyn units.

    The rate is ``k·Π[C]`` with ``k`` mass action in µM and the continuous concentrations
    ``[C]`` in µM:

    - order 0 (creation): µM/s -> molecules/(µm³·s), factor ``602.214``;
    - order 1: 1/s, factor 1;
    - order 2: 1/(µM·s) -> µm³/s, factor ``1/602.214``.
    """
    if particle_order == 0:
        return MOLECULES_PER_UM3_PER_UM
    if particle_order == 1:
        return 1.0
    if particle_order == 2:
        return 1.0 / MOLECULES_PER_UM3_PER_UM
    raise ValueError(f"Smoldyn reactions have at most 2 reactants, got particle order {particle_order}")
