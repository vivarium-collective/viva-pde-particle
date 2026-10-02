# fenics-smoldyn: research plan

PDE/particle hybrid co-simulation with process-bigraph, compared against the hybrid solver embedded in vcell-fvsolver.
See [fvsolver-hybrid-notes.md](fvsolver-hybrid-notes.md) for the detailed trace of the reference implementation.


## Context

VCell's finite volume solver (`virtualcell/vcell-fvsolver`, local `../vcell-fvsolver`) includes a
VCell-patched Smoldyn 2.38. Together they run PDE/particle hybrid simulations:

- A single VCell model is partitioned. Species flagged `forceContinuous` stay PDE fields, and all
  other species become particles. This happens in VCell Java,
  `ParticleMathMapping.combineHybrid()` (`../vcell/vcell-core/.../mapping/ParticleMathMapping.java:1082-1250`).
- Each step is coupled by explicit, lagged operator splitting (`VCell/src/SimTool.cpp:871-935`):
  1. The PDE takes one forward-Euler/explicit step.
  2. Every k-th PDE step (`SMOLDYN_STEP_MULTIPLIER`), Smoldyn takes one step of size k·dt.
  3. Particle counts are binned to concentration (`copyParticleCountsToConcentration`).
  4. All variables are copied from current to old.
- Both sides read only the **old** arrays (`SimulationExpression.cpp:398-470`). In effect this is Jacobi-style coupling.
- **PDE → particles:** `VCellValueProvider::getValue` (`bridgeVCellSmoldyn/VCellValueProvider.cpp:21-84`) evaluates
  rate expressions (`k*B;` in `.smoldynInput`) at a particle position, using a nearest-voxel lookup with no interpolation.
  - 1st-order reactions use the molecule's position.
  - 2nd-order reactions use the midpoint of the pair.
  - 0th-order reactions do a Poisson draw for each voxel.
- **Particles → PDE:** `VCellSmoldynOutput::computeHistogram` (`bridgeVCellSmoldyn/VCellSmoldynOutput.cpp:433-552`)
  bins each molecule to the nearest grid node (`dx = L/(N-1)`), then divides by the element volume or area
  (boundary voxels have reduced volume). The conversion to µM is written into the PDE expressions.
- **Mixed reactions are not exchanged as fluxes.** Each side applies its own half of each mixed reaction
  (Smoldyn: `A_p → …` with rate `k·[B]`; PDE: terms that read the binned `[A]`).

The fvsolver approach is fast but hard-coded: one mesh type, one splitting scheme, one Smoldyn version
(2.38, ~10 years old), and no step-level Python control (`pyvcell_fvsolver.solve()` only runs to completion).

**Goal:**
1. Reproduce this method as a process-bigraph co-simulation with two processes: a PDE process and a
   Smoldyn process.
2. Compare it with the embedded fvsolver on:
   - **accuracy:** ensemble statistics and convergence in the coupling interval
   - **performance:** wall time and overhead breakdown
   - **generalizability:** FEniCSx meshes, alternative splitting schemes, swapping engines
3. Package this as a viva workspace with investigations and studies that work with
   viva-superpowers and vivarium-workbench.

### Decisions already made
- **PDE engine:** both, in stages.
  - First, a numpy finite-volume process on the *same* Cartesian grid as fvsolver, so the comparison
    measures the coupling and not the discretization.
  - Then a FEniCSx (dolfinx) process, for generalizability.
- **Mixed (particle × field) reactions:** patch Smoldyn. Upstream Smoldyn (`../Smoldyn`, v2.73+) still has
  the VCell hooks:
  - `OPTION_VCELL` in `CMakeLists.txt:51`
  - `ValueProvider` / `ValueProviderFactory` / `AbstractMesh` in `source/Smoldyn/smoldyn.h:47-49,368-389,1071-1072`
  - `rxn->rateValueProvider` in `smolreact.c:2590-3164`
  - `smoldynhybrid.c`, `source/vcell/SimpleValueProvider.*`, `SimpleMesh.*`
  - the `#ifdef OPTION_VCELL` path in `source/python/module.cpp:89`

  The patch adds a grid-backed `ValueProvider` that is fed from Python, rather than inventing a new mechanism.
- **Code home:** a new package in this repo (`viva_fenics_smoldyn`). `viva-smoldyn` and `viva-fenics`
  serve as references and patterns. Useful generic changes go back to them later as PRs.

## Repo / environment

### Layout (viva-template workspace, nested investigations)
```
fenics-smoldyn/
  pixi.toml                    # single self-contained env (conda-forge + pypi)
  pyproject.toml               # hatchling; package viva_fenics_smoldyn
  workspace.yaml               # schema_version 2, package_path: viva_fenics_smoldyn, default_emitter: parquet
  AGENTS.md, CLAUDE.md, README.md, docs/PLAN.md, docs/fvsolver-hybrid-notes.md
  .pbg/schemas/                # from viva-template
  scripts/                     # lint-workspace.py, serve.sh (pixi-aware), build_smoldyn.sh
  external/Smoldyn/            # git submodule -> Smoldyn fork, branch `pyhybrid`
  viva_fenics_smoldyn/
    core.py                    # build_core(): register processes/steps/emitters
    units.py                   # µM <-> molecules/µm³ (602.214...), area/volume helpers
    grid.py                    # fvsolver-compatible node-centred Cartesian grid, element volumes, binning
    model/                     # HybridModel spec + partitioner (port of combineHybrid)
    processes/
      fv_reaction_diffusion.py # numpy explicit FV PDE process (Phase 2)
      smoldyn_hybrid.py        # patched-Smoldyn particle process (Phase 2)
      fenicsx_reaction_diffusion.py  # dolfinx PDE process (Phase 4)
      mesh_binning.py          # particle -> mesh-cell histogram for unstructured meshes (Phase 4)
    reference/
      fvsolver_inputs.py       # write hybrid .fvinput + .smoldynInput from HybridModel
      fvsolver_runner.py       # pyvcell_fvsolver.solve() + result reader (counts → conc)
    analysis/                  # ensemble stats, error metrics, timing
    composites/*.composite.yaml
  workspace/
    investigations/<inv>/investigation.yaml
    investigations/<inv>/studies/<study>/study.yaml (+ sims/run.py, viz/)
    references/papers.bib, notes/
  tests/
```

### Environment (pixi, osx-arm64 + linux-64)
- **Conda-forge packages:** `python 3.12.*`, `fenics-dolfinx 0.10.*` (matches vcell-fenics; viva-fenics is on 0.11,
  and we only borrow its patterns), `mpich`, `petsc4py`, `numpy`, `scipy`, `matplotlib`, `h5py`, `zarr`, `pyyaml`,
  plus build tools for Smoldyn: `cmake`, `cxx-compiler`, `pybind11`, `zlib`, `libtiff` as needed.
- **PyPI packages:**
  - `process-bigraph`, `bigraph-schema`, `viva-emitters`
  - `pyvcell[solver,native]`, which brings `pyvcell-fvsolver 0.10.7` (cp312 arm64 wheel exists) and
    `libvcell 0.0.18` (arm64 wheel exists)
  - `vivarium-workbench` and `viva-workspace` pulled from git `main`
  - `viva_fenics_smoldyn` itself as an editable install
- **Smoldyn:** built from the submodule by the `pixi run build-smoldyn` task, which runs
  `pip install ./external/Smoldyn` with `-DOPTION_VCELL=ON`. This also works around the PyPI problem:
  the current smoldyn 2.75 wheels are Windows-only, and the older macOS wheels are x86_64.
- **Local development against sibling checkouts in `../`:** a `pixi run dev-link` task runs
  `pip install -e ../pyvcell ../vivarium-workbench …`, so committed pins stay git/PyPI-based and the
  environment stays self-contained.
- **Workbench:** run inside this env (`pixi run serve`, i.e. `vivarium-workbench serve --workspace .`).
  The template CI moves from uv to `prefix-dev/setup-pixi`.

### Scaffold source
- Copy from GitHub `vivarium-collective/viva-template`. The local `../pbg-template` is stale and from before the rebrand.
- Use the `/viva-workspace` in-place mode with `--template-source` pointing to viva-template, or apply
  `template-init.sh` by hand.
- Then adapt it for pixi and for the nested layout (`layout:` keys like `../viva-smoldyn/workspace.yaml`).

## Technical design

### 1. Smoldyn patch (fork `external/Smoldyn`, branch `pyhybrid`)
- **`GridValueProvider` / `GridValueProviderFactory`** (`source/vcell/GridValueProvider.{h,cpp}`):
  - Rate strings in the VCell `…;` form are parsed as a constant times a product of named fields
    (a minimal parser: `k*B`, `k*B*C`, `B`). Anything more complex is an error in v1.
  - `getValue(t,x,y,z,rxn)` does the nearest-node lookup with the same indexing as fvsolver,
    `i = (int)((x-x0)/dx + 0.5)`. Membrane panel lookup is deferred.
- **`SimpleMesh`-based `AbstractMesh`** with the PDE grid geometry, so 0th-order per-voxel Poisson
  creation in `smoldynhybrid.c` works.
- **pybind11 additions** (`source/python/module.cpp`, gated by `OPTION_VCELL`):
  - `Simulation.setGrid(origin, spacing, shape)`
  - `Simulation.setField(name, np.ndarray)`: zero-copy view or memcpy into the provider's buffer; called once per
    coupling step
  - `Simulation.getMoleculePositions(species) -> ndarray[n,3]`, so positions are no longer read via `listmols` text
  - `Simulation.getMoleculeHistogram(species) -> ndarray[shape]`: nearest-node counts in C, mirroring
    `computeHistogram`
- **Model loading:** stays file-based through `.smoldynInput` (`simInitAndLoad`), or uses the Python builder API
  with expression rates. The particle process supports both.
- **Upstream:** keep the patch minimal and self-contained so it can be offered upstream (ssandrews/Smoldyn).
  The fork's GitHub location (proposed: `virtualcell/Smoldyn`) needs confirmation before it is created.

### 2. Processes and the coupling contract
Units on the wire:
- fields: µM, on the grid nodes
- particle histograms: counts per element

Each process owns its own unit conversion, using `units.py` and `grid.py`.

| Process | Inputs | Outputs | Interval |
|---|---|---|---|
| `FVReactionDiffusion` (numpy) | `particle_counts: map[array]` | `fields: map[array]` (overwrite) | `dt` |
| `SmoldynHybrid` | `fields: map[array]` | `particle_counts: map[array]`, `molecule_counts: map[int]`, optional positions | `k·dt` |
| `FenicsxReactionDiffusion` (Phase 4) | `particle_counts` (per mesh cell) | `fields` (DOF array + cell-avg view) | `dt` |

- **`FVReactionDiffusion`:**
  - Node-centred grid with `dx = L/(N-1)` and half/quarter boundary volumes (as in `CartesianMesh.cpp:804`).
  - Explicit forward Euler, zero-flux boundaries by default.
  - Reaction terms come from the `HybridModel`; particle species appear as read-only concentrations,
    `counts / (vol·602.214)`.
- **`SmoldynHybrid`:**
  - Persistent `smoldyn.Simulation`.
  - On `update(state, interval)`: `setField` for each input field, then `runUntil(t+interval)`, then return `getMoleculeHistogram`.
- **Scheduling semantics:**
  - Process-bigraph processes read the state at the start of their interval, and their updates are applied
    afterwards. This reproduces fvsolver's "everyone reads old" lagged coupling.
  - With PDE interval `dt` and Smoldyn interval `k·dt`, the PDE sees particle concentrations held for k steps,
    exactly as in fvsolver.
  - Phase 0 confirms this ordering on a trivial composite, and the confirmation is written up.
- **Alternative orchestration (Phase 5), not possible in fvsolver:**
  - A `HybridCoupler` Step-based orchestrator for Gauss-Seidel ordering or Strang splitting.
  - A sub-cycled PDE (implicit backward Euler with a larger dt).

### 3. Model spec and partitioner (`model/`)
- **`HybridModel`:** species with diffusion coefficients, compartment (a single volume compartment in v1),
  reactions (mass action), initial conditions (field arrays or particle counts) and a `particle_species` set.
- **`partition(model)`** ports `combineHybrid`:
  - The PDE side keeps every reaction term that involves a continuous species.
  - The Smoldyn side gets each reaction with its continuous reactants folded into the rate (`k*B;`).
    Continuous products are dropped, and continuous-only reactions are dropped.
- **Emitters** from the same `HybridModel`:
  1. composite state for process-bigraph
  2. `.fvinput` + `.smoldynInput` for the reference fvsolver (`reference/fvsolver_inputs.py`), using the
     keywords `VOLUME_PARTICLE`, `SMOLDYN_STEP_MULTIPLIER`, the `SMOLDYN_BEGIN/INPUT_FILE/END` block,
     `vcellWriteOutput` and the `highResVolumeSamples` map
- **Phase 3:** import hybrid VCML through `pyvcell` (`SpeciesMapping.force_continuous`, `ParticleProperties`).
  The cross-check uses `libvcell.vcml_to_finite_volume_input` to see whether VCell's own writer emits hybrid inputs.

### 4. Reference runner
- **Running:** `pyvcell_fvsolver.solve(fvinput, vcg, outdir)`.
- **Reading results:** read the `.sim`/`.zip`/`.hdf5` output with pyvcell `sim_results` (or a thin custom reader if
  the zarr path does not apply). Particle variables are stored as **raw counts per voxel**
  (`FVDataSet.cpp:413-418`).
- **Ensembles:** run N seeds (`random_seed` in `.smoldynInput`) in a process pool.

## Research program (investigations and studies)

### Investigation A: `cosim-vs-embedded-hybrid`
**Question:** can a process-bigraph co-simulation of {PDE, Smoldyn} reproduce the embedded fvsolver hybrid
solver to within its stochastic error, and at what cost?

| Study | What it tests | Pass criteria (behavior_tests) |
|---|---|---|
| `handoff-units-and-binning` | Pure diffusion: particle-only and field-only, then particle histogram → conc | Matches analytic; histogram conservation exact; boundary-volume handling matches fvsolver |
| `field-modulated-decay` | `A_p → ∅` with rate `k·[B](x)` and a static/diffusing B | Ensemble mean within 95% CI of analytic / ODE-PDE limit |
| `two-way-exchange` | `A_p → B_f` and 0th-order `∅ → A_p` at rate `k·[B]` | Total mass conservation; steady-state counts match |
| `bimolecular-hybrid` | `A_p + B_f → C_f` | Co-sim vs fvsolver: ensemble mean and variance per region (KS / CI overlap); high-copy limit vs deterministic PDE |
| `coupling-interval-convergence` | Sweep k and dt | First-order convergence in both; error-curve comparison |
| `performance-scaling` | Grid size × particle count × k | Wall time, per-component breakdown (PDE, Smoldyn, marshaling, scheduler overhead) |

### Investigation B: `hybrid-generalizability`
**Question:** what does the modular co-simulation enable that the embedded solver cannot?

| Study | Content |
|---|---|
| `fenicsx-same-problem` | FEniCSx PDE process on a matched mesh, compared with the numpy FV result (separates discretization from coupling) |
| `unstructured-geometry` | Non-box geometry: particle → cell binning via `dolfinx.geometry` bounding-box tree; Smoldyn surfaces generated from the mesh boundary |
| `splitting-schemes` | Lagged Jacobi (fvsolver) vs Gauss-Seidel vs Strang; accuracy vs cost |
| `engine-swap` | Same composite with the PDE engine swapped (FV ↔ FEniCSx) and, optionally, the particle engine swapped (Smoldyn ↔ simple python BD) |
| `membrane-particles` (stretch) | Membrane-bound particles and surface actions using panel-indexed binning |

**Common metrics:**
- region-wise ensemble mean and variance of particle counts
- field L2 / L∞ error against the reference ensemble mean
- mass balance
- wall time and peak memory

Results go to study-local parquet runs. Figures are produced with `/viva-viz`, and reports through `/viva-report`.

## Phases (execution order)

0. **Scaffold and spikes** (this session, after approval)
   - Commit `docs/PLAN.md` and `docs/fvsolver-hybrid-notes.md`. The notes hold the coupling trace above,
     with file:line references.
   - Scaffold from viva-template: `workspace.yaml`, the package with an empty `build_core`, AGENTS/CLAUDE.md, `.gitignore`.
   - Write `pixi.toml` and get `pixi install` working. Check that `pyvcell_fvsolver`, `libvcell`, `dolfinx` and
     `process_bigraph` import.
   - Add the Smoldyn submodule (upstream for now) and run the `build-smoldyn` spike with `OPTION_VCELL=ON`.
     This is the main risk: the VCell build path may pull in VCell-only sources or zlib quirks.
   - Run a scheduling spike: two dummy processes with intervals dt and k·dt confirm the "read old" semantics.
   - Create empty Investigations A and B with `investigation.yaml`, and run `scripts/lint-workspace.py` until it reports OK.
1. **Smoldyn patch:** `GridValueProvider`, mesh, and the pybind accessors, with C++ and python unit tests.
2. **Core co-sim:** `grid.py`, `units.py`, `HybridModel` + partitioner, `FVReactionDiffusion`, `SmoldynHybrid`, composites.
   This phase also runs Studies A1–A2.
3. **Reference path:** fvsolver input writer and runner. A hand-built hybrid case is validated against VCell
   desktop if needed, and VCML import goes through pyvcell. This phase runs Studies A3–A6.
4. **FEniCSx process and mesh binning:** Studies B1–B2.
5. **Orchestration variants:** Studies B3–B5. Write-up.

## Risks / open items
- **`OPTION_VCELL` in upstream Smoldyn** may not build cleanly through the python path. Fallback: compile only
  the needed hybrid sources and the provider into the python module, without the full VCell option.
- **Hybrid input generation:** if `libvcell` does not emit hybrid fvinput, our own writer is the path, and it
  needs one VCell-desktop-generated golden case to validate against. Check `../vcell` or a VCML from the VCell
  database for an existing hybrid model.
- **Smoldyn version gap:** the reference uses Smoldyn 2.38 and the co-sim uses 2.7x. Pure-Smoldyn baselines
  (no fields) are run in both to measure this.
- **Smoldyn's own Python API is awkward for stepping** (segfault if `molpos` is read before the first run, per viva-smoldyn).
  The new numpy accessors avoid the text output path entirely.
- **Outward-facing steps that need explicit confirmation when they come up:** creating the GitHub repo for
  fenics-smoldyn, creating the Smoldyn fork, and any PRs to upstream repos.

## Verification
- `pixi install && pixi run build-smoldyn && pixi run test`. Tests import every engine, check unit/binning
  round-trips, and run the scheduling-semantics test.
- `pixi run python scripts/lint-workspace.py` reports `workspace lint: OK`.
- `pixi run serve` starts the workbench, and it lists both investigations.
- From Phase 2 on, each study's `behavior_tests` run through `/viva-study run-baseline` or `canonical_runs`
  scripts, and the report card shows pass/fail against the criteria above.
