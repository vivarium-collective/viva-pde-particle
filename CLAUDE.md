@AGENTS.md

# viva-pde-particle

PDE/particle hybrid co-simulation with process-bigraph (finite volume or FEniCSx PDE process
plus a patched Smoldyn process), benchmarked against the hybrid solver embedded in
vcell-fvsolver.
- Plan and phase status: `docs/PLAN.md`.
- How the reference solver couples PDE and particles: `docs/fvsolver-hybrid-notes.md`.

## Environment (pixi, not uv)

    pixi install              # conda-forge engines + pypi ecosystem + this package (editable)
    pixi run build-smoldyn    # patch + build + install Smoldyn (OPTION_VCELL) from external/Smoldyn
    pixi run test             # pytest
    pixi run lint             # workspace lint
    pixi run check-engines    # what is importable
    pixi run serve            # vivarium-workbench dashboard

- **Always run Python through `pixi run`.** dolfinx, the Smoldyn build and pyvcell-fvsolver
  (Python 3.12, macOS 15+) exist only in the pixi env.
- **`pyproject.toml` stays light.** It is what the generated `workspace-ci.yml` installs with uv
  on Python 3.11. Engine tests must `pytest.importorskip` their engine.
- **Don't edit generated workflows by hand.** Files under `.github/workflows/` that carry the
  `viva-template-provenance` header are drift-guarded. Add new workflows as separate files.
- **`dev` environment:** `pixi run -e dev ...` uses `../pyvcell` editable (co-developing its
  spatial-hybrid support, virtualcell/pyvcell#62). The native VCell reference
  (`viva_pde_particle.reference.vcell_native`) and Study A2 part 3 need it. Build Smoldyn there
  too with `pixi run -e dev build-smoldyn`. CI uses `default`, where native tests skip.
- **`compose` environment:** `default` minus the `workbench` feature (vivarium-workbench). The compose-api image
  (`docker/compose.Dockerfile`) is built from it. Nothing under `viva_pde_particle/` may import `vivarium_workbench`
  at module level; `pyproject.toml` has it only as the `workbench` extra, which `dev` includes.
- **Native solves:** vcell-fvsolver segfaults on a second hybrid solve in one process. Always go
  through `run_native` / `run_native_ensemble`, which spawn a process per solve.

## Smoldyn submodule (virtualcell/Smoldyn fork)

- **Where it points:** `external/Smoldyn` tracks the `pyhybrid` branch of `virtualcell/Smoldyn`, a fork of
  ssandrews/Smoldyn. In the submodule, `origin` = upstream ssandrews (the diff base, `origin/master`) and
  `fork` = virtualcell (push here).
- **Hybrid extensions:**
  - `source/vcell/HybridGrid.h`
  - `source/vcell/GridValueProvider.{h,cpp}`
  - the `HybridGrid` / `getMoleculePositions` / `getMoleculeHistogram` bindings in `source/python/module.cpp`
  - the hybrid `Simulation(filepath, flags, grid)` constructor
- **After changing Smoldyn:** commit on `pyhybrid` and run `git -C external/Smoldyn push fork pyhybrid`,
  then commit the updated gitlink in this repo.
- **Rules for Smoldyn changes:**
  - every change must keep the vanilla `OPTION_VCELL=OFF` build compiling (`SMOLDYN_VCELL=OFF pixi run build-smoldyn`);
  - never commit the in-tree built `_smoldyn*.so`.
- **Ask first** before opening PRs to upstream ssandrews/Smoldyn.

## Coupling semantics to preserve

process-bigraph reads state at the start of each interval. fvsolver's Smoldyn step over
`[T, T+k·dt]` reads the field at `T+(k-1)·dt`. The `SmoldynHybrid` default `coupling: fvsolver`
mode reproduces this by running on the PDE clock (see `tests/test_scheduling_semantics.py`).
