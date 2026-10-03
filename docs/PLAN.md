# viva-pde-particle: research plan

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
- **Code home:** a new package in this repo (`viva_pde_particle`). `viva-smoldyn` and `viva-fenics`
  serve as references and patterns. Useful generic changes go back to them later as PRs.

## Repo / environment

### Layout (viva-template workspace, nested investigations)
```
viva-pde-particle/
  pixi.toml                    # single self-contained env (conda-forge + pypi)
  pyproject.toml               # hatchling; package viva_pde_particle
  workspace.yaml               # schema_version 2, package_path: viva_pde_particle, default_emitter: parquet
  AGENTS.md, CLAUDE.md, README.md, docs/PLAN.md, docs/fvsolver-hybrid-notes.md
  .pbg/schemas/                # from viva-template
  scripts/                     # lint-workspace.py, serve.sh (pixi-aware), build_smoldyn.sh
  external/Smoldyn/            # git submodule -> Smoldyn fork, branch `pyhybrid`
  viva_pde_particle/
    core.py                    # build_core(): register processes/steps/emitters
    units.py                   # µM <-> molecules/µm³ (602.214...), area/volume helpers
    grid.py                    # fvsolver-compatible node-centred Cartesian grid, element volumes, binning
    model/                     # HybridModel spec + partitioner (port of combineHybrid)
    processes/
      fv_reaction_diffusion.py # numpy explicit FV PDE process (Phase 2)
      smoldyn_hybrid.py        # patched-Smoldyn particle process (Phase 2)
      fenicsx_reaction_diffusion.py  # dolfinx PDE process (Phase 4)
      mesh_binning.py          # particle -> mesh-cell histogram for unstructured meshes (Phase 4)
    reference/                 # Phase 3: drive the embedded hybrid solver via pyvcell / libvcell / pyvcell-fvsolver
    analysis/                  # ensemble stats, error metrics, timing
    composites/*.composite.yaml
  workspace/
    investigations/<inv>/investigation.yaml
    studies/<study>/study.yaml (+ sims/run.py, results/metrics.json, viz/); `investigation:` back-reference
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
  - `viva_pde_particle` itself as an editable install
- **Smoldyn:** built from the submodule by the `pixi run build-smoldyn` task (`scripts/build_smoldyn.sh`).
  The task applies `patches/smoldyn/*.patch`, configures CMake with `-DOPTION_VCELL=ON` and graphics off,
  then pip-installs the staged `build/.../py` package. This also works around the PyPI problem:
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

### 1. Smoldyn extensions (fork `virtualcell/Smoldyn`, branch `pyhybrid`) — implemented in Phase 1
- **`HybridGrid`** (`source/vcell/HybridGrid.h`, header-only, all builds):
  - a node-centred Cartesian grid with VCell `CartesianMesh` geometry:
    - `dx = L/(N-1)`, or `L` when `N = 1`;
    - nodes at `x0 + i·dx`, or the domain centre when `N = 1`;
    - nearest node `(int)((x-x0)·(N-1)/L + 0.5)`, clamped;
  - holds named scalar fields in VCell order (x fastest), i.e. numpy shape `(Nz, Ny, Nx)` in C order.
- **`GridValueProvider` / `GridValueProviderFactory`** (`source/vcell/GridValueProvider.{h,cpp}`, `OPTION_VCELL`):
  - **Parsing:** a rate written `k*B;` in the configuration file is parsed as a product of numbers and field names.
    Names set by `define` are substituted by Smoldyn first. Any other expression is a load error.
  - **Evaluation:** the rate is evaluated at the nearest grid node. An unset field evaluates to 0.
  - **Membrane/surface variants** use the same volume lookup. The panel-indexed membrane lookup is deferred.
- **`GridMesh`** (an `AbstractMesh` over `HybridGrid`) drives Smoldyn's 0th-order creation: one
  Poisson(`rate·dt·dx·dy·dz`) draw per node whose centre is in the compartment, placed uniformly within that
  node's cell. This matches vcell-fvsolver, including its full-volume treatment of boundary nodes.
- **Python** (`_smoldyn`):
  - `HybridGrid(origin, size, num)` with `setField(name, array)`, `getField`, `index`, `center`, `shape`,
    `spacing` and `requiredFields()`. The last is the set of field names the loaded rates reference.
  - `Simulation(filepath, flags, grid)` (`OPTION_VCELL`) loads a configuration file with the grid-backed
    factory and mesh.
  - `Simulation.getMoleculePositions(species="all", state=all)` returns an `(n, dim)` array.
  - `Simulation.getMoleculeHistogram(species, grid, state=all)` returns nearest-node counts shaped like the grid.
    It mirrors `VCellSmoldynOutput::computeHistogram`.
  - The accessors exist in all builds.
- **Model loading is file-based** (`.smoldynInput`-style text). The builder API cannot attach a factory, because
  rates are parsed while the file is read.
- **Upstream:** the changes are kept self-contained so they can be offered to ssandrews/Smoldyn. The vanilla
  `OPTION_VCELL=OFF` build is verified to still compile.

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
  - Semi-implicit, matching vcell-fvsolver's `FV_SOLVER` (`SparseVolumeEqnBuilder` + PCG), the PDE solver
    used in hybrid runs. Diffusion is backward Euler, and reactions are forward Euler from the start-of-step
    state: `s_i·uⁿ⁺¹ + D·dt·K·uⁿ⁺¹ = s_i·(uⁿ + R·dt)`, with zero-flux boundaries. (Phase 2 finding: "forward
    Euler" applies to the reactions and the coupling, not to diffusion.)
  - Reaction terms come from the `HybridModel`; particle species appear as read-only concentrations,
    `counts / (vol·602.214)`.
- **`SmoldynHybrid`:**
  - Persistent `smoldyn.Simulation`.
  - On `update(state, interval)`: `setField` for each input field, then `runUntil(t+interval)`, then return `getMoleculeHistogram`.
- **Scheduling semantics** (confirmed in Phase 0 by `tests/test_scheduling_semantics.py`):
  - Process-bigraph processes read the state at the start of their interval, and their updates are applied
    afterwards. This reproduces fvsolver's "everyone reads old" lagged coupling.
  - With PDE interval `dt` and Smoldyn interval `k·dt`, the PDE sees particle concentrations held for k steps,
    exactly as in fvsolver.
  - **Difference for k > 1:** fvsolver runs the Smoldyn step after the k-th PDE iterate (`SimTool.cpp:889-893`),
    so a Smoldyn step over `[T, T+k·dt]` sees the field at `T+(k-1)·dt`. A process-bigraph process with
    `interval = k·dt` sees the field at `T`.
  - So `SmoldynHybrid` gets a `coupling` option:
    - `fvsolver` (default): run on the PDE's `dt` clock and step Smoldyn by `k·dt` on every k-th call, which
      replicates fvsolver exactly.
    - `start-of-interval`: use `interval = k·dt`, the plain process-bigraph behavior. This is a variant
      worth comparing in its own right.
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
- **Outputs from the same `HybridModel`:**
  1. the composite state for process-bigraph (`composites/hybrid.py`), with Smoldyn config text from
     `model/smoldyn_config.py`;
  2. the same model as a VCell hybrid application for the reference solver (Phase 3, see §4).

### 4. Reference runner (Phase 3): teach the VCell Python stack, no VCell desktop
The reference inputs come from VCell's own math generation, not from a hand-written fvinput/smoldynInput writer
and not from VCell desktop. This may require extending:
- **pyvcell:** build a spatial application with a particle-species selection, i.e. a hybrid/stochastic
  application with `SpeciesMapping.force_continuous` for the field species. The VCML model classes can already
  represent this. Today `add_application` hardcodes `stochastic=False`, and `simulate` runs only the FV path.
- **libvcell:** VCML → solver inputs through VCell's `ParticleMathMapping.combineHybrid()` and its writers,
  emitting the combined `.fvinput` + `.smoldynInput`. Check what `vcml_to_finite_volume_input` already does for
  hybrid applications.
- **pyvcell-fvsolver:** run the hybrid inputs. The solver already embeds Smoldyn 2.38. Results must expose the
  particle variables, which are stored as raw counts per voxel (`FVDataSet.cpp:413-418`).

Then:
- **Running:** `pyvcell_fvsolver.solve(fvinput, vcg, outdir)`, driven through pyvcell.
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

0. **Scaffold and spikes**: done 2026-10-03; see "Phase 0 results" below.
   - Commit `docs/PLAN.md` and `docs/fvsolver-hybrid-notes.md`. The notes hold the coupling trace above,
     with file:line references.
   - Scaffold from viva-template: `workspace.yaml`, the package with an empty `build_core`, AGENTS/CLAUDE.md, `.gitignore`.
   - Write `pixi.toml` and get `pixi install` working. Check that `pyvcell_fvsolver`, `libvcell`, `dolfinx` and
     `process_bigraph` import.
   - Add the Smoldyn submodule (upstream for now) and run the `build-smoldyn` spike with `OPTION_VCELL=ON`.
     This is the main risk: the VCell build path may pull in VCell-only sources or zlib quirks.
   - Run a scheduling spike: two dummy processes with intervals dt and k·dt confirm the "read old" semantics.
   - Create empty Investigations A and B with `investigation.yaml`, and run `scripts/lint-workspace.py` until it reports OK.
1. **Smoldyn extensions:** done 2026-10-03; see "Phase 1 results" below.
2. **Core co-sim.**
   - **2a, done 2026-10-03:** `grid.py`, `units.py`, `HybridModel` + partitioner, `FVReactionDiffusion`,
     `SmoldynHybrid` and the composite builder. See "Phase 2a results" below.
   - **2b, done 2026-10-03:** Studies A1–A2 as workbench studies. See "Phase 2b results" below.
3. **Reference path:** teach pyvcell (and, as needed, libvcell and pyvcell-fvsolver) to build, generate and run
   hybrid VCell applications headlessly (see §4). This phase runs Studies A3–A6.
4. **FEniCSx process and mesh binning:** Studies B1–B2.
5. **Orchestration variants:** Studies B3–B5. Write-up.
6. **Standards-based model description (follow-on; options to be discussed when we get there).**
   - **Goal:** describe the hybrid model with SBML plus the Spatial package instead of VCML.
   - **How VCell does it today:** a single model is augmented with the choice of which species are particles.
     VCell's math generation (`ParticleMathMapping.combineHybrid()`, see the notes) then produces both the
     fvsolver input and the Smoldyn input and combines them.
   - **Open design question:** how to express the particle/continuous partition in a standards-based way.
     Candidates to evaluate:
     - a sidecar list of particle species (with per-species particle properties) next to an SBML Spatial document;
     - SBML annotations on species;
     - a SED-ML or simulation-level setting;
     - reusing VCell's own SBML Spatial import/export (`libvcell`/`pyvcell` converters) to reach the existing math
       generation.
   - **Prerequisite:** the `HybridModel` + `partition()` layer from Phase 2, which is the target this format
     would load into.

## Phase 0 results (2026-10-03)

- **Scaffold:** rendered from `vivarium-collective/viva-template` (workspace `pde-particle`, package
  `viva_pde_particle`). The generated `workspace-ci.yml` is drift-guarded, so it stays as generated: it
  installs the light pure-Python package with uv on Python 3.11, and engine tests skip there.
- **Environment:** `pixi.toml` provides Python 3.12, dolfinx 0.10.0, `pyvcell-fvsolver` 0.10.7 (reports
  "with smoldyn version 2.38"), `libvcell` 0.0.18, `pyvcell` 0.4.1, process-bigraph 1.8.5 and
  vivarium-workbench from git main.
  - **Needs macOS 15+:** `[system-requirements] macos = "15.0"`, because the `pyvcell-fvsolver` wheels are
    tagged `macosx_15_0`.
  - **Ecosystem drift:** `viva-marketplace` is now the `viva-catalog` distribution, and pixi does not read
    transitive `[tool.uv.sources]`. So `pixi.toml` mirrors vivarium-workbench main's git sources.
  - `pixi run check-engines` reports what is importable.
- **Smoldyn `OPTION_VCELL` build:** upstream (`ssandrews/Smoldyn` `e21d6dd`) did not compile with
  `OPTION_VCELL`. The fixes are committed on the submodule's local `pyhybrid` branch and exported to
  `patches/smoldyn/0001-*.patch`:
  - compile the sources as C++, as VCell's vendored Smoldyn does
  - `extern "C"` guards on libSteve headers
  - use the system zlib and drop the stale Windows `zlib.h`/`zconf.h`
  - fix a swallowed brace in `smolcomparts.c`
  - update a stale `surfsetrate` call
  - `calloc` casts, a missing `<sstream>`, and missing includes in `module.cpp`

  The vanilla `OPTION_VCELL=OFF` build still compiles. The parent repo pins the submodule at the upstream
  commit, and `build_smoldyn.sh` applies the patch, so the repo builds anywhere until a fork exists.
  `tests/test_smoldyn_build.py` checks first-order decay against the analytic solution.
- **Scheduling:** process-bigraph reads state at the start of each interval (Jacobi). This matches fvsolver
  except for the k > 1 field-read difference above, which is handled by the `coupling` option.
- **Workbench:** `pixi run lint` reports `workspace lint: OK`. `pixi run serve` starts the dashboard on the
  pixi interpreter, and `/api/workspace` lists both investigations. The planned studies are recorded in each
  investigation's `at_a_glance`, and each study's `study.yaml` is created when its phase starts (lint
  rejects `studies:` entries without one).

## Phase 1 results (2026-10-03)

- **Fork:** `virtualcell/Smoldyn` (a fork of ssandrews/Smoldyn), branch `pyhybrid`. The submodule now tracks it,
  and the Phase 0 patch files are retired.
- **Implemented** the `HybridGrid`, `GridValueProvider`/`Factory`, `GridMesh` and numpy accessors described in
  Technical design §1.
- **Two pre-existing `OPTION_VCELL` bugs fixed** along the way:
  - **Compartments crashed.** Compartment setup always used VCell's voxel-map path, which dereferences a NULL
    `volumeSamplesPtr` unless the model has `highResVolumeSamples`. It now falls back to geometric compartments,
    as `posincompart` already did.
  - **Out-of-bounds reads in 1D/2D models.** The hybrid rate evaluation read `pos[2]` past the end of the
    position array. Positions are now padded to 3D.
- **Tests:** `tests/test_smoldyn_hybrid.py` covers:
  - grid geometry against the VCell formulas;
  - field round-trips;
  - positions and histogram against a numpy re-binning;
  - 1st order: `A → ∅` at rate `k·[B]` with a half-domain field, where the zero-field half is exactly unchanged
    and the other half matches `exp(-k t)` within 4σ;
  - an unset field acting as rate 0, and field updates taking effect between `runUntil` calls;
  - 0th order: `∅ → A` at rate `k·[B]`, where the count matches `nodes·k·B·dx·dy·T` and molecules appear only
    in the producing cells.

## Phase 2a results (2026-10-03)

**Package `viva_pde_particle`:**
- `units.py`: µM ↔ molecules/µm³ (602.214), and Smoldyn rate-unit factors by particle order.
- `grid.py`: `CartesianGrid` with VCell conventions:
  - spacing, node coordinates and nearest-node binning;
  - element volume fractions (½, ¼, ⅛ at boundaries);
  - counts ↔ µM;
  - the volume-scaled zero-flux diffusion operator.
- `model/`: `HybridModel` (`Species`, `Reaction`) and `partition()`, a port of `combineHybrid`. The particle
  side folds continuous reactants into the rate as `k·[B]` and converts it to Smoldyn units: ×602.214 for 0th
  order, ×1 for 1st, ÷602.214 for 2nd. `write_smoldyn_config()` emits the configuration, adding a
  domain-covering compartment when grid-based 0th-order creation is needed.
- **`processes/FVReactionDiffusion`:** the semi-implicit `FV_SOLVER` step, using an `splu` factorization per
  species. Particle species enter the reactions as binned µM.
- **`processes/SmoldynHybrid`:** the patched Smoldyn with a `HybridGrid`.
  - `coupling: fvsolver` (default): runs on the PDE clock and takes one `k·dt` step every k-th call.
  - `coupling: start-of-interval`: interval `k·dt`.
  - Exact step counts come from `breaktime = t + (n-½)·dt`.
- `composites/hybrid.py`: `build_hybrid_document()` and `run_hybrid()`, which records the trajectory.

**Implementation notes:**
- **Overwrite types for outputs:** process-bigraph's plain `map[array[float]]` output is additive, so writers
  declare `map[overwrite[array[float]]]`.

**Tests** (29 passed, 1 skipped):
- the cosine eigenmodes of the FV operator decay at exactly the backward-Euler rate;
- diffusion conserves mass, and the explicit reaction term matches;
- particle binning respects boundary volumes;
- the partition gives the expected terms, units and config text;
- end to end:
  - field-modulated decay leaves zero-field regions exactly unchanged and decays the rest as `exp(-k[B]t)`;
  - `A_p → B_f` conserves A+B molecules within 2%;
  - with `fvsolver` coupling and k = 4, particle counts change only every 4th PDE step;
  - both coupling modes agree for slow dynamics.

## Phase 2b results (2026-10-03)

**Study layout:**
- Studies live flat in `workspace/studies/<slug>/` with an `investigation:` back-reference, as in viva-fenics.
- The template's lint scans only the flat layout, while the workbench resolves both.

**Each study holds:**
- `study.yaml` (schema v4), whose baseline is a `@composite_generator` composite (`viva_pde_particle.composites.examples.*`);
- `sims/run.py`, the canonical run, which records events in `.pbg/runs.jsonl`;
- `results/metrics.json`;
- `viz/*.html` (plotly).

**Grading:** `viva_pde_particle/evaluators.py` registers each metric as a derived scalar, so the workbench's
`study_evaluator.evaluate_test` grades `behavior_tests` natively. All 8 tests pass through it.

**A1 `handoff-units-and-binning`:**
- FV vs analytic diffusion: 6.0e-4 relative error.
- Binned Smoldyn variance vs 2Dt: 0.96%.
- Histogram molecule mismatch: 0.
- Node-class counts vs element-volume shares: max |z| = 1.15. Corner and edge µM are within 1.3% of interior.

**A2 `field-modulated-decay`:**
- Static-field column survival vs `exp(-k·B·t)`: max |z| = 1.76.
- Decays where B = 0: none.
- Diffusing A and B, ensemble-mean A vs the continuum FV solution: 2.9% relative L2 (at the sampling-noise level).
- Total A vs continuum: 0.14%.

**Support code:**
- `viva_pde_particle/analysis.py`:
  - `run_ensemble`;
  - `continuum_model` + `run_continuum` (the deterministic limit through the FV process alone);
  - `histogram_variance`;
  - `write_metrics`.

**Next:**
- Phase 3, the reference path through pyvcell/libvcell/pyvcell-fvsolver. A3–A6 compare against it.
- The coupling-only checks of A3 (`two-way-exchange`) can run on the co-simulation before that.

## Risks / open items
- ~~**`OPTION_VCELL` in upstream Smoldyn** may not build cleanly through the python path.~~ Resolved in
  Phase 0: it needed 9 small build fixes (`patches/smoldyn/0001-*.patch`), and it builds and runs natively
  on arm64.
- **Hybrid input generation:** VCell's math generation (`ParticleMathMapping`) is the source of truth. If
  `libvcell` does not yet expose the hybrid path, extending it (and pyvcell) is Phase 3 work in those repos. We
  will not write our own fvinput/smoldynInput generator.
- **Smoldyn version gap:** the reference uses Smoldyn 2.38 and the co-sim uses 2.7x. Pure-Smoldyn baselines
  (no fields) are run in both to measure this.
- **Smoldyn's own Python API is awkward for stepping** (segfault if `molpos` is read before the first run, per viva-smoldyn).
  The new numpy accessors avoid the text output path entirely.
- **Outward-facing steps that need explicit confirmation when they come up:** PRs to upstream repos (e.g.
  offering the Smoldyn changes to ssandrews/Smoldyn). The GitHub repo and the `virtualcell/Smoldyn` fork were
  created at the user's request on 2026-10-02 and 2026-10-03.

## Verification
- `pixi install && pixi run build-smoldyn && pixi run test`. Tests import every engine, check unit/binning
  round-trips, and run the scheduling-semantics test.
- `pixi run python scripts/lint-workspace.py` reports `workspace lint: OK`.
- `pixi run serve` starts the workbench, and it lists both investigations.
- From Phase 2 on, each study's `behavior_tests` run through `/viva-study run-baseline` or `canonical_runs`
  scripts, and the report card shows pass/fail against the criteria above.
