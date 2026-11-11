"""The port set of a component is fixed by its class, never by its config.

A static face lets a composite be checked against declared ports without instantiating anything
(the ``describe`` tooling reads ``inputs()``/``outputs()`` with the class standing in for ``self``).
"""
from __future__ import annotations

import pytest

from viva_pde_particle.processes import GridRecorder, MeshRecorder, SmoldynHybrid


def _face(cls):
    return cls.inputs(cls), cls.outputs(cls)


def test_smoldyn_hybrid_face_does_not_depend_on_config():
    inputs, outputs = _face(SmoldynHybrid)
    assert set(inputs) == {"fields"}
    assert set(outputs) == {"particle_counts", "particle_totals", "particle_positions"}


def test_recorders_have_one_fixed_face_each():
    assert _face(GridRecorder) == ({"fields": "map[array[float]]", "particle_positions": "map[array[float]]"}, {})
    assert _face(MeshRecorder) == ({"field_dofs": "map[array[float]]", "particle_positions": "map[array[float]]"}, {})


def test_recorder_refuses_the_wrong_layout(tmp_path):
    from process_bigraph import allocate_core

    with pytest.raises(ValueError, match="records a 'grid' layout"):
        GridRecorder(config={"out_dir": str(tmp_path / "x.fenics"), "base_dt": 0.1, "every": 1,
                             "layout": {"kind": "mesh"}, "species": ["A"]}, core=allocate_core())


def test_attach_recorder_picks_the_recorder_by_layout():
    from viva_pde_particle.composites.examples import two_way_exchange
    from viva_pde_particle.viz3d.record import attach_recorder

    doc = attach_recorder(two_way_exchange(core=None), "out", 0.1)
    assert doc["recorder"]["address"] == "local:GridRecorder"
    assert set(doc["recorder"]["inputs"]) == {"fields", "particle_positions"}
