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
- **Sibling checkouts:** `pixi run dev-link` installs `../pyvcell` etc. editable over the pins.
  Never commit local paths.

## Smoldyn submodule and patches

- **Where the fixes live:** `external/Smoldyn` is pinned to an upstream ssandrews/Smoldyn commit.
  Our changes live on the submodule's local `pyhybrid` branch and are exported as
  `patches/smoldyn/*.patch`. `scripts/build_smoldyn.sh` applies them idempotently.
- **After changing Smoldyn,** commit on `pyhybrid`, then re-export:
  `git -C external/Smoldyn format-patch --binary -o ../../patches/smoldyn origin/master..pyhybrid`
  (clear the old patches first). Keep the parent's gitlink at the upstream commit until a fork exists.
- **Patch rules:**
  - every change must keep the vanilla `OPTION_VCELL=OFF` build compiling (`SMOLDYN_VCELL=OFF pixi run build-smoldyn`);
  - never commit the in-tree built `_smoldyn*.so`.
- Creating the Smoldyn fork (proposed `virtualcell/Smoldyn`) needs the user's go-ahead.

## Coupling semantics to preserve

process-bigraph reads state at the start of each interval. fvsolver's Smoldyn step over
`[T, T+k·dt]` reads the field at `T+(k-1)·dt`. The `SmoldynHybrid` default `coupling: fvsolver`
mode reproduces this by running on the PDE clock (see `tests/test_scheduling_semantics.py`).
