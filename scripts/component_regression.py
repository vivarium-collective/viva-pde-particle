"""Bit-exact regression harness for the Phase 7 componentization refactor.

Records the outputs of six representative hybrid composites, then checks a later version of
the code against them array for array (np.array_equal, no tolerance):

    pixi run python scripts/component_regression.py refs.npz save    # on the commit before a change
    pixi run python scripts/component_regression.py refs.npz check   # after the change

References depend on the machine (Smoldyn RNG, BLAS), so they are not committed. Record and
check on the same machine. A refactor that only moves code between components must print
ALL IDENTICAL.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

from viva_pde_particle.composites.examples import two_way_exchange_model
from viva_pde_particle.composites.hybrid import (build_coupler_document, build_hybrid_document,
                                                 build_mesh_hybrid_document, run_document)
spec = importlib.util.spec_from_file_location("b2b", str(Path(__file__).resolve().parents[1] / "workspace/studies/sphere-cosim-vs-native/sims/run.py"))
b2b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b2b)

def flat(tr):
    out = {}
    for s, v in tr.fields.items(): out[f"fields_{s}"] = np.stack(v)
    for s, v in tr.particle_counts.items(): out[f"counts_{s}"] = np.stack(v)
    for s, v in tr.field_dofs.items(): out[f"dofs_{s}"] = np.stack(v)
    return out

cases = {
    "fv_fvsolver": lambda: run_document(build_hybrid_document(two_way_exchange_model(), 0.01, 2, "fvsolver", seed=3), 1.0, 0.01, 0.1),
    "fv_start": lambda: run_document(build_hybrid_document(two_way_exchange_model(), 0.01, 2, "start-of-interval", seed=3), 1.0, 0.01, 0.1),
    "q1": lambda: run_document(build_hybrid_document(two_way_exchange_model(), 0.01, 2, "fvsolver", seed=3, pde_engine="fenicsx"), 1.0, 0.01, 0.1),
    "coupler_strang": lambda: run_document(build_coupler_document(two_way_exchange_model(), 0.01, 4, "strang", seed=3), 1.0, 0.01, 0.2),
    "mesh_grid": lambda: run_document(build_mesh_hybrid_document(b2b.model("exchange"), b2b.MESH_SPHERE, 0.01, seed=3), 0.3, 0.01, 0.1),
    "mesh_positions": lambda: run_document(build_mesh_hybrid_document(b2b.model("conversion"), b2b.MESH_SPHERE, 0.01, seed=3, particle_transfer="positions"), 0.3, 0.01, 0.1),
}
if __name__ == "__main__":
    out = sys.argv[1]
    if sys.argv[2] == "save":
        arrays = {}
        for name, fn in cases.items():
            for k, v in flat(fn()).items(): arrays[f"{name}/{k}"] = v
        np.savez_compressed(out, **arrays); print("saved", len(arrays), "arrays")
    else:
        ref = np.load(out); bad = 0
        for name, fn in cases.items():
            got = flat(fn())
            for k, v in got.items():
                r = ref[f"{name}/{k}"]
                same = r.shape == v.shape and np.array_equal(r, v)
                diff = 0.0 if same or r.shape != v.shape else float(np.abs(r - v).max())
                if not same: bad += 1
                print(f"{'OK  ' if same else 'DIFF'} {name}/{k} {'' if same else diff}")
        print("ALL IDENTICAL" if bad == 0 else f"{bad} arrays differ")
