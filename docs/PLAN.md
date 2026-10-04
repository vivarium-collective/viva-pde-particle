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
  - A `HybridCoupler` orchestrator (implemented as one Process owning both engines) for Gauss-Seidel ordering or Strang splitting.
    Replaced in Phase 7d by the engine-agnostic `SplittingCoordinator`.
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
**Question:** what does decomposing the hybrid solver into interchangeable processes add (alternative PDE
discretizations, splitting orders, engines), and at what cost relative to VCell's own discretization of the same
problems?

| Study | Content |
|---|---|
| `fenicsx-same-problem` | FEniCSx PDE process on a matched mesh, compared with the numpy FV result (separates discretization from coupling) |
| `unstructured-geometry` | Curved domain with a body-fitted P1 mesh, compared with native VCell's Cartesian/staircase discretization of the same analytic geometry |
| `splitting-schemes` | Lagged Jacobi (fvsolver) vs Gauss-Seidel vs Strang; accuracy vs cost |
| `engine-swap` | Same composite with the PDE engine swapped (FV ↔ FEniCSx) and, optionally, the particle engine swapped (Smoldyn ↔ simple python BD) |
| `membrane-particles` (stretch) | Membrane-bound particles and surface actions using panel-indexed binning |

**Common metrics:**
- region-wise ensemble mean and variance of particle counts
- field L2 / L∞ error against the reference ensemble mean
- mass balance
- wall time and peak memory

Results go to study-local parquet runs. Figures are produced with `/viva-viz`, and reports through `/viva-report`.

### Investigation C: `vcell-hybrid-paper-benchmarks`
The validation suite of the VCell hybrid methods paper, run against both VCell's native hybrid solver (through
the Phase 3 pyvcell path) and the co-simulation, with its figures regenerated:
- Schaff, Gao, Li, Novak & Slepchenko (2016), *Numerical approach to spatial deterministic-stochastic models
  arising in cell biology*, PLoS Comput Biol 12(12): e1005236, doi:10.1371/journal.pcbi.1005236.

**Question:** do the native solver and the co-simulation both reproduce the paper's results, and where do they
differ?

**Model mapping (calcium sparks):** stochastic two-state channels are immobile particle species placed at fixed
nodes (`Ch_closed`, `Ch_open`, D = 0), coupled to a diffusing Ca²⁺ field `U`:
- opening `Ch_closed → Ch_open` at `k_on` (Test 1) or `k_on·U/U₀` (field-dependent, Test 2+);
- closing at `k_off`;
- open channels as a field source `J` (`Ch_open → Ch_open + U`);
- removal by pumps (`V_p`).

| Study | Paper | Model / regime | Reference | Paper's metric |
|---|---|---|---|---|
| `separable-calcium-sparks` | Test 1, Fig 1 | 24 channels, field-independent gating; D = 1 µm²/s; quasi-2D 10.1×2.1×0.5 µm³, Δx = 0.1 µm; Δt 0.002–0.02 s; T = 5 s | Analytic expectation (separable case) | Error ε(N, Δt), ∝ N^-1/2 |
| `coupled-sparks-fast-diffusion` | Test 2, Figs 2–3 | Field-dependent opening `k_on·U/U₀`, k_on = 0.1 s⁻¹, D = 1000 µm²/s, Δt = 0.2 ms | Non-spatial Gibson–Bruck SSA (we implement) | p(U,t), P(n,t) at t = 1, 2, 3 s and steady state; L2 ≈ 1% of max |
| `single-channel-fokker-planck` | Tests 3–5, Figs 4–6 | Dimensionless single channel, fast and finite diffusion | Direct functional Fokker–Planck solution (to implement) | L2 of p(ρ), 1–3% of max |
| `gated-binding-3d` | Test 6, Fig 7 | M (inert↔reactive) particles + ligand field L in 10³ µm³, h = 0.2 and 0.034 µm | t^-3/2 asymptotics | Relaxation function; documents the point-particle limitation |
| `cell-polarization` (stretch) | Test 7, Fig 8 | Membrane receptors in a sphere, R = 4 µm | Qualitative | Needs membranes (Investigation B `membrane-particles`) |

Each study runs ensembles through both solvers. The figures are regenerated with the native and co-simulation
results overlaid, and the paper's numbers are recorded as `pass_if` provenance. Parameter details are in the
paper's Methods and S1 text; they are to be transcribed into each `study.yaml` when the study is built.

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
   hybrid VCell applications headlessly (see §4).
   - **First target:** the calcium-spark model of Investigation C (`separable-calcium-sparks`, then
     `coupled-sparks-fast-diffusion`), exercising the native solver on a published, well-characterized case.
   - **Then** Studies A3–A6.
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
7. **Componentization: reusable components with general engines (started 2026-10-03; design in
   [DESIGN.md §13](DESIGN.md#13-steps-and-the-component-architecture-phase-7)).**
   - **Goal:** a set of process-bigraph components (Processes and Steps) that assemble into an accurate hybrid
     co-simulation. The core PDE solver and the core particle solver should be general, and know nothing about each
     other.
   - **Where we are:** the engines can be swapped behind one fixed coupling contract (one `CartesianGrid`, µM and
     counts, configurations generated by the partitioner), and B1 showed the PDE swap works. But the coupling was
     buried in the engines:
     - counts → µM inside the FV and Q1 engines;
     - Pᵀ, the P1 point load and P·u inside the mesh engine;
     - fvsolver timing inside `SmoldynHybrid`;
     - `HybridCoupler` hard-codes FV + Smoldyn.
     Every accuracy problem the studies found (vcell-fvsolver#25 and #26, B2c, B2d) was in exactly these adapter and
     geometry pieces.
   - **Sub-phases.** Each must reproduce the previous outputs bit-for-bit on the regression harness
     (`scripts/component_regression.py`, six composites), or justify the difference:
     - **7a:** particle → PDE adapters as Steps; PDE engines take a generic `external_conc`. Done; see "Phase 7a
       results".
     - **7b:** PDE → particle adapter Step (`MeshToGridField`, P·u); the mesh engine publishes only DOFs. Done; see
       "Phase 7b results".
     - **7c:** take the PDE clock out of the particle engine. Replace `coupling="fvsolver"` with a scheduling or
       field-snapshot construct that keeps fvsolver timing exact; the Investigation A studies must reproduce. Done;
       see "Phase 7c results".
     - **7d:** a generic splitting coordinator. It sequences any child processes and adapters (Jacobi, Gauss–Seidel,
       Strang) in place of `HybridCoupler`'s hard-coded classes; B3 must reproduce. Done; see "Phase 7d results".
     - **7e:** a geometry compiler. One spec yields the PDE mesh, the Smoldyn membrane, the volume samples and the
       accessible-volume fractions, so the domains cannot diverge (B2d). It also covers compartment-aware and
       accessible-volume binning options for the adapters (the vcell-fvsolver#25 and B2c lessons). In progress; see
       "Phase 7e plan".
     - **7f:** packaging. A written component contract (units, array layouts, timing), catalog and workbench
       registration, and example composites that mix components; for example the FV engine with
       positions-based binning on a mesh-derived grid. Done; see "Phase 7f results".
   - **Acceptance:** B1, B2b, B2d and B3 reproduce with the componentized composites, and no engine imports or
     names another engine's concepts.

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

## Phase 3a results (2026-10-03): headless native reference

**Pipeline** (`viva_pde_particle/reference/vcell_native.py`), with no VCell desktop and no hand-written inputs:
- `HybridModel` → pyvcell BioModel: a stochastic spatial application, `force_continuous` fields, and the
  `"Finite Volume Standalone, Regular Grid"` solver;
- → libvcell, VCell's `ParticleMathMapping.combineHybrid()`, which writes `.fvinput` + `.smoldynInput`;
- → pyvcell-fvsolver, i.e. vcell-fvsolver with embedded Smoldyn 2.38;
- → pyvcell's `PdeDataSet` reader: fields in µM, particles as counts per voxel.

**Upstream:**
- **pyvcell** needed hybrid authoring and result support: [virtualcell/pyvcell#62](https://github.com/virtualcell/pyvcell/pull/62),
  open for review. It adds:
  - `add_hybrid_sim`, `SmoldynSimulationOptions` and `Simulation.time_step`;
  - `map_species(force_continuous=)`;
  - particle channels in `Result`.
- **libvcell** needed no change.
- **Local use:** the `dev` pixi environment installs `../pyvcell` editable. The `default` environment (CI) keeps
  the PyPI pyvcell, and native tests skip there.

**Findings and decisions:**
- **3D only:** VCell's spatial stochastic/hybrid math requires 3D geometry, with at least 3 mesh nodes per axis.
  2D problems run as quasi-2D slabs (`Nz = 3`), with the co-simulation on the identical grid.
  (The paper's 2-node z-axis would be rejected by current VCell.)
- **Particle initial conditions** in a concentration-mode application are Poisson per node, so the initial total
  is Poisson(N). The co-simulation matches with `particle_init="poisson"`. Exact placement needs a
  math-level model (Phase 3b).
- **vcell-fvsolver can't run two hybrid solves in one process:** the second segfaults inside `solve()`,
  presumably from embedded-Smoldyn or hybrid globals. `run_native` therefore spawns a fresh process per solve,
  and `run_native_ensemble` runs seeds in parallel (8 seeds in ~2 s). The solver bug is worth fixing upstream.
- **zarr conflict:** pyvcell's zarr-based `Result` pins zarr 2, and the workbench needs zarr 3, so our runner reads
  the `.sim` output directly.

**First co-sim vs native comparison** (Study A2, part 3):
- native survival vs `exp(-k·B·t)`: |z| = 1.68;
- co-sim vs native, static-field survival: |z| = 1.47;
- co-sim vs native, diffusing x-profiles: |z| = 2.58;
- total survival differs by 0.25%;
- fields are identical.

**Next:**
- Phase 3b: math-level models with located particles, for the paper's calcium sparks (Investigation C).
- Studies A3–A6 on both solvers.

## Phase 3b results (2026-10-03): paper benchmark C1 on both solvers

**Per-node particle placement in the native path:**
- A BioModel cannot express per-node particle counts (VCell seeds particles from a concentration as Poisson draws).
- Species with per-node counts therefore get zero initial concentration, and their molecules are written as
  `mol 1 X x y z` into the VCell-generated `.smoldynInput`.
- The math stays VCell's.
- **Long-term fix:** located initial counts in VCell, or MathModel support in libvcell (it only accepts BioModels
  today).

**VCell finding, no-op particle jump processes:**
- **The problem:** a particle that catalyses a continuous source (channel influx `O → O + U`) becomes a
  destroy/create jump process `O → O` at rate k (~6000/s). This happens even when the particle is written as a
  modifier.
- **Why it matters:** Smoldyn splits competing first-order reactions in proportion to their rates, so this no-op
  suppresses channel closing. The open fraction rises from 1/6 to ~0.9.
- **What we do:** `drop_noop_reactions` removes such processes from the `.smoldynInput`. The source stays in the
  PDE, so this is exact.
- **Proper fix:** `ParticleMathMapping.combineHybrid` should drop jump processes whose actions cancel. Worth an
  upstream VCell issue/PR.

**Study C1 `separable-calcium-sparks`** (Test 1 / Fig 1B; 128 trials × Δt ∈ {0.02, 0.01, 0.005} × both solvers):
- ε vs N log-log slope: −0.46 co-sim, −0.43 native (theory −1/2).
- ε at N = 128: 1.4% co-sim, 1.6% native.
- ε is flat in Δt, i.e. statistically dominated, as the paper reports.
- Co-sim vs native difference: 0.75× the combined sampling error.
- All behavior tests pass.

**Deviations (provisional):**
- Δz = 0.25, because VCell needs ≥ 3 z-nodes (the paper used 0.5).
- Channel coordinates use defaults approved by the author: columns at x = 2, 4, 6, 8; rows at y = 0.3–1.8.

**Next:** C2 and A3; see the Phase 3c results.

## Phase 3c results (2026-10-03): C2, A3, and three more VCell/Smoldyn findings

**C2 `coupled-sparks-fast-diffusion`** (Test 2 / Fig 2):
- The reference is an exact event-driven sampler of the well-mixed piecewise-deterministic process
  (`benchmarks/pdmp.py`, 20,000 realizations). It validates against the analytic open probability on Test 1.
- With 500 trials per solver:
  - mean |z| vs reference: 1.57 (co-sim), 0.64 (native);
  - P(n, t) L2: 0.05 for both, at the sampling floor;
  - co-sim vs native KS p ≥ 0.035.
- **Open observation:** both solvers share a marginal KS p ≈ 7·10⁻⁴ against the well-mixed reference, likely
  residual spatial structure at D = 1000. A follow-up should use larger ensembles or D → ∞.

**A3 `two-way-exchange`** (A_p ⇌ B_f with field-dependent creation):
- Co-sim conserves A + B to 1.2%, reaches k₂/k₁ to 0.3%, and creates 0.989× the exact expectation.
- Native overproduces creation by 1.813×, which is exactly the predicted boundary-volume factor 1.815×, so it does
  not conserve mass.

**Findings:**
1. **VCell zero-reactant mass action** keeps only −Kr·Π(products) and drops Kf. A constant source (`0 → U`)
   silently vanished from the native PDE and biased C2 (|z| 11.5 before the fix). The converter now writes
   zero-reactant reactions as General kinetics. This is by design in VCell, but it's a trap for converters.
2. **Smoldyn grid 0th-order creation** used the full cell volume at boundary nodes, overproducing by
   Σfull/Σactual. This is in VCell's vendored Smoldyn: [virtualcell/vcell-fvsolver#24](https://github.com/virtualcell/vcell-fvsolver/issues/24).
   Fixed in our fork.
3. **The same path skipped boundary nodes** with geometric compartments: centres on the walls fail `posincompart`.
   Fixed in our fork by nudging the test point inward.
4. **Residual start-up artifact:** Smoldyn skips 0th-order creation for about the first 2 steps (~1% over 2 s).
   Not yet fixed.

**Upstream issues filed:**
- [virtualcell/vcell#2158](https://github.com/virtualcell/vcell/issues/2158): no-op particle jump processes.
- [virtualcell/vcell-fvsolver#23](https://github.com/virtualcell/vcell-fvsolver/issues/23): second hybrid solve
  segfaults.
- [virtualcell/vcell-fvsolver#24](https://github.com/virtualcell/vcell-fvsolver/issues/24): boundary
  overproduction.

**Infrastructure:** `run_ensemble(workers=…)` runs co-sim ensembles in spawned processes.

**Next:**
- A4 `bimolecular-hybrid`;
- A5 coupling-interval convergence;
- A6 performance;
- Investigation B (FEniCSx, Phase 4).

## Phase 3d results (2026-10-03): record-time fix and A4

**Harness bug, now fixed:**
- **Cause:** process-bigraph accumulates each process's time as a float sum of its interval, and
  25 × 0.01 = 0.25000000000000006 > 0.25. The update ending at a record time was deferred, so every
  co-simulation state recorded at t was really at t − dt (99 updates by t = 1.0).
- **Fix:** `run_hybrid` advances half a step past each record time. A regression test is in
  `tests/test_hybrid_cosim.py`.
- **Effect:** all studies were re-run.
  - A1 binned variance vs 2Dt: 0.96% → 0.13%.
  - A2 total vs continuum: 0.14% → 0.05%.
  - A4's apparent solver disagreement disappeared (C |z| 71 → 0.85).
- **Lesson:** process-bigraph composites need care with time arithmetic when results are sampled at exact
  times. Worth an upstream note: an integer-tick or rational-time scheduler would avoid it.

**A4 `bimolecular-hybrid`** (A_p + B_f → C_f, B substantially depleted):
- co-sim vs native: survival |z| = 1.79, C |z| = 0.85, profiles |z| = 2.10;
- both within 0.3% of the continuum survival;
- B + C conserved to round-off.

## Phase 3e results (2026-10-03): A5, A6; Investigation A complete

**A5 `coupling-interval-convergence`** (A_p + B_f → C_f, k ∈ {1…32}, 64 seeds):
- Both fvsolver-semantics solvers are first order in k·Δt: fitted order 1.23 (co-sim) and 1.07 (native).
- Their errors match: ratio 1.05 at k = 16.
- `start-of-interval` coupling roughly halves the error at large k.
- Native crashes at k = 32 (Smoldyn step = output interval).

**A6 `performance-scaling`:**
- The co-sim is faster than the native solve in every configuration: 0.25–0.93×.
  - Grid sweep 11→81 (20k particles): 0.14→0.90 s, against native 0.48→0.97 s.
  - 200k particles: 2.6 s, against native 3.8 s.
- process-bigraph overhead is ≤ 3 ms per step (3–42%, worst for tiny problems).

**Investigation A verdict:** supported. The co-simulation reproduces the embedded solver within stochastic error,
with the same coupling error and at lower cost. Where they differ, the embedded solver has defects (see the
investigation's executive summary).

**Next:** Investigation B (Phase 4: FEniCSx PDE process, unstructured meshes, splitting schemes), and the remaining
C studies (Fokker–Planck, gated binding).

## Phase 4a results (2026-10-03): FEniCSx engine and B1

**`FenicsxReactionDiffusion`** (`processes/fenicsx_reaction_diffusion.py`):
- Q1 Lagrange elements on a structured quad/hex mesh whose vertices are the grid nodes, built with dolfinx 0.10.
  Q1 rather than P1 tetrahedra because P1-tet lumped masses depend on the diagonal orientation.
- The Q1 lumped mass equals the FV dual-cell volumes exactly, so particle binning and units are shared with FV.
- Same ports and time scheme as FV; mass is `lumped` or `consistent`.
- Selected by `build_hybrid_document(pde_engine="fenicsx")`.
- Reaction terms are shared with FV through `processes.fv_reaction_diffusion.reaction_rates`.

**B1 `fenicsx-same-problem`:**
- **Discretization:** all three engines are second order on the cosine mode.
- **Coupled:** with the same seeds, the engine swap is invisible (paired |z| ≤ 1.5; survival differences ≤ 3·10⁻⁴),
  and all engines are within 0.27% of the continuum.
- **Analysis note:** comparisons that share seeds must use `analysis.paired_z` (common random numbers). The
  two-sample z drastically understates their sensitivity.

**Next (Phase 4b):** unstructured geometry B2. That needs particle → mesh-cell binning, and field → particle values on
non-grid meshes, either a mesh-based ValueProvider or interpolation onto a HybridGrid. Then splitting schemes B3.

## Phase 4b results (2026-10-03): unstructured geometry (B2)

**Correction (2026-10-03):** an earlier version of this section, and of the B2 study, framed the unstructured mesh as
something VCell cannot do. That's wrong. VCell handles analytic, image-segmented and CSG geometries. Its Cartesian
volume grid is coupled to a membrane grid that is geometrically a staircase surface but numerically uses local
tangent-plane projections and Voronoi neighbours, and it converges well. pyvcell and VCell can also turn FV output
into smoothed VTK unstructured grids for analysis. B2 demonstrates an alternative, body-fitted discretization under
the same coupling. The meaningful test is a comparison with native VCell on the same analytic sphere (B2b).

**`viva_pde_particle/mesh.py`:**
- gmsh ball meshes (gmsh/python-gmsh added to pixi; used as a meshing tool).
- `MeshGridTransfer`: one sparse P1 interpolation matrix P from a background grid to mesh DOFs. Fields go
  mesh → grid as P·u; particle counts go grid → mesh as Pᵀ·h. Rows sum to 1, so the handoff conserves molecules.

**`FenicsxMeshReactionDiffusion`:**
- P1 on the unstructured mesh, lumped mass.
- The authoritative state is `field_dofs`. It outputs grid-sampled `fields`, so `SmoldynHybrid` is unchanged apart
  from the geometry.
- Smoldyn config gains `geometry={"kind": "sphere"}` (reflecting `panel sph`, inside compartment) and explicit
  initial positions.
- Built by `build_mesh_hybrid_document`, run by `run_document`.

**B2 `unstructured-geometry`** (ball R = 4 µm):
- Conversion: decay |z| ≤ 1.8; A + B conserved to 0.4%; B vs continuum FEM 1.0% L2.
- Exchange: mass balance 1.0%; steady ratio 0.33%.
- The mesh volume is −1.4% of the ball.

**Next:**
- B3 splitting schemes: a `HybridCoupler` Process for Gauss–Seidel and Strang ordering.
- Remaining C studies.

## Phase 4c results (2026-10-03): B2b, the same ball under native VCell and the co-simulation

`to_biomodel(geometry={"kind": "sphere"})` builds VCell analytic geometry: a `cell` sphere in an `ec` background,
with a membrane.

**Results** (box [0,9]³, Δ = 0.25, R = 4, 8 seeds):
- **Volume:** VCell's representation of the ball is more accurate (−0.47%) than the co-sim mesh at h = 0.8 (−1.4%).
- **Conversion:** the solvers agree (|z| = 0.73; radial B within 3.5%).
- **Exchange: the native solver loses molecules at the curved membrane.** Signed balance at 5 s: native −6.8%,
  co-sim −1.2% (+0.51% after the B2d fixes). Native A is 6.1% below the co-sim (8.1% after B2d). An earlier
  version said "grows"; that misread an unsigned metric.
  The loss is consistent with two mechanisms:
  - **Exterior binning:** 1.2% of native A is binned to nodes outside the voxelized cell, where its PDE source is
    dropped.
  - **Creation deficit:** native creation runs at 0.960 of exact (co-sim 0.974 before B2d). B2d found the cause:
    Smoldyn's hybrid `zeroreact` creates only in cells whose centre is inside the compartment.
  The rough budget matches. These are coupling details at curved membranes, not VCell geometry/PDE limitations.

### Study B2c: near-membrane fields (`near-membrane-fields`)
Ball conversion with A uniform, so E[B] is uniform. Native fields are mapped onto pyvcell's smoothed unstructured grid
by the new `reference/vcell_vtk.py`. The co-sim uses P1 DOFs. 32 seeds per solver.
- **Exporter convention:** pyvcell and VCell's Java vis exporter draw element i as `[i·L/N, (i+1)·L/N]`, but the fvsolver
  mesh is node-centred (`i·L/(N−1)`). On 37³ the exported domain is 8.4% too small. `smoothed_domain(node_centred=True)`
  corrects this on the analysis side, giving −0.50% volume error. Smoothing cuts the surface RMS radial error from 0.108
  to 0.065 µm.
- **Interior:** both solvers match E[B] (native +0.5%, co-sim −0.4%; +0.25% after B2d).
- **Outer shell (r > R − 2Δ), opposite-sign artifacts:**
  - Native is 2.3% low (z −20): B2b's exterior-binning loss, localized.
  - Co-sim is 1.3% high (z 10) with Smoldyn confined by the exact sphere. B2d traced this to the sphere/mesh domain
    mismatch, not to Pᵀ; with the mesh membrane it is +0.12% (z 1.2).
  - Smoothing does not change the native bias; the artifact is in the particle→field binning, not the geometry/PDE.
  - **Code:** `VCellSmoldynOutput::computeHistogram` (vcell-fvsolver `bridgeVCellSmoldyn/VCellSmoldynOutput.cpp:476–548`)
    bins to the nearest node. Its same-compartment neighbour correction is present but commented out, and is the
    likely native fix.

### Study B2d: co-sim boundary transfer (`cosim-boundary-transfer`)
Four variants (`particle_transfer` grid/positions × `membrane` sphere/mesh), 32 seeds each:

| transfer / membrane | shell bias (z) | exchange balance | wall s per sim s |
|---|---|---|---|
| grid / sphere | +1.27% (10.3) | +2.9% | 0.71 |
| positions / sphere | +1.24% (10.0) | +2.9% | 2.98 |
| grid / mesh | +0.12% (1.2) | +0.69% | 1.07 |
| positions / mesh | +0.07% (0.7) | +1.0% | 3.56 |

- **The cause is the membrane, not the transfer.** The fix is to confine Smoldyn by the mesh's boundary triangles,
  now the default (`build_mesh_hybrid_document(membrane="mesh")`). B2, B2b and B2c were re-run with it:
  - B2 conversion balance 0.11%, B vs continuum 0.53% L2;
  - B2b co-sim exchange +0.51%;
  - B2c co-sim shell +0.12%.
- **Credit:** VCell's Smoldyn input already confines particles with `panel tri` from its geometry (9,516 triangles,
  261.8 µm³), and the co-sim now follows that practice.
  - **Difference:** native's smoothed triangulation and voxel PDE domain differ by 1.8% in volume. The co-sim's are
    identical.
  - **Native creation:** 0.982 × 0.976 (centre-only creation) = 0.958, matching the measured 0.960.
- **Smoldyn fork fix (pyhybrid a773562).** Field-dependent 0th-order creation in a compartment only drew in cells whose
  centre was inside. That gave a −2.4% deficit for this ball, the B2b "creation deficit", which had been cancelling
  the sphere's +1.4% extra volume. Every cell now draws, with exact thinning.
  `test_zeroth_order_creation_matches_curved_compartment_area` covers it.
- **Upstream:** vcell-fvsolver's vendored Smoldyn has the same centre test, plausibly the native 0.960. Filed as
  virtualcell/vcell-fvsolver#26. The commented-out same-compartment binning correction is
  virtualcell/vcell-fvsolver#25.
- **Triangulated-membrane cost (N molecules × M triangles):**
  - Smoldyn sized its virtual boxes from the initial count, so exchange, which starts empty, got 1 box.
  - Fix: always write `boxsize` (10×), plus VCell's `highResVolumeSamples` compartment map, now parsed under
    `OPTION_VCELL` (fork e86d552), for a further 1.8×.
  - Mesh-membrane exchange: 4.55 → 0.25 s per simulated s (sphere 0.21), with identical results.
  - VCell's Smoldyn writer sets no `boxsize` either: virtualcell/vcell#2159.
- **New infrastructure:** positions transfer, the voxel→tet `PointLocator`, and the mesh membrane (PR #21).
  `docs/DESIGN.md` explains all approaches.

## Phase 5 results (2026-10-03): splitting schemes (B3)

**`HybridCoupler`** (`processes/hybrid_coupler.py`; replaced by `SplittingCoordinator` in Phase 7d):
- One process composes the FV and Smoldyn engines and orders their substeps per coupling interval:
  `jacobi`, `gs_particles_first`, `gs_pde_first` or `strang`.
- `jacobi` reproduces the two-process `coupling="fvsolver"` composite exactly (regression test).
- Built by `build_coupler_document`.

**B3 `splitting-schemes`** (k ∈ {2…32}, 64 seeds):
- Jacobi and the two Gauss–Seidel orderings are first order (1.0–1.23).
- Strang keeps the error near the sampling floor up to τ = 0.16 s: 25× smaller than Jacobi there.
- So the particle side can step ~32× less often at fvsolver-level accuracy. vcell-fvsolver's hybrid loop
  (`SimTool.cpp`) implements one fixed order, so trying Strang there would mean changing the solver; in the
  co-simulation it is a coupler option. It's a candidate default for particle-heavy models.

## Phase 7a results (2026-10-03): particle → PDE adapters as Steps

- **New Steps** (`viva_pde_particle/steps/transfer.py`):
  - `GridCountsToConcentration`: grid histogram → µM on the same grid (FV, Q1);
  - `GridCountsToMeshConcentration`: Pᵀ·counts / (M_L·602.214);
  - `PositionsToMeshConcentration`: the exact P1 point load.
  Each runs when its input store changes, after that time point's process updates and before the next process
  intervals, so coupling timing is unchanged.
- **Engines:**
  - `FVReactionDiffusion`, `FenicsxReactionDiffusion` and `FenicsxMeshReactionDiffusion` now read `external_conc`
    for species they use but do not evolve (inferred from the terms, or `pde.external_species`). They have no
    particle ports or particle logic.
  - The mesh engine's `particle_transfer` option is gone; the builder now picks the Step.
  - The mesh objects (mesh, P, lumped mass, locator) are shared through per-process caches in `mesh.py`.
- **Equivalence:** all 14 arrays are bit-identical to the pre-refactor code (`np.array_equal`) on six composites:
  - FV in both coupling modes;
  - FEniCSx Q1;
  - `HybridCoupler` with Strang splitting;
  - the mesh engine with grid transfer and with positions transfer.
- **Cost (single-process timing, wall s per simulated s):**

  | case | before | after |
  |---|---|---|
  | FV | 0.044 | 0.046 |
  | mesh, grid transfer | 1.17–1.38 | 0.37 |
  | mesh, positions transfer | 1.77–1.80 | 0.96–1.02 |

  The likely reason: the PDE engine no longer receives the grid histogram or the positions array through its ports
  every PDE step. Not profiled.
- **Tests:** `tests/test_transfer_steps.py`. The existing engine tests were updated to compose Step + engine.

## Phase 7b results (2026-10-03): PDE → particle adapter as a Step

- **New Step:** `MeshToGridField` (`steps/transfer.py`) samples P1 DOF fields onto the particle engine's lookup grid
  (P·u; nearest DOF for nodes outside the mesh). FV and Q1 already produce grid fields and need no adapter.
- **Mesh engine:** `FenicsxMeshReactionDiffusion` now publishes only `field_dofs` and takes no `grid` config. It
  builds from `mesh.mesh_space(spec)`, a cached mesh with its P1 space and lumped mass, and knows neither the grid
  nor the particles.
- **Shared caches:** `mesh_transfer` (for the Steps) is built on the same cached space, so the DOF numbering is shared
  across components. `mesh_locator` now needs only the mesh spec.
- **Equivalence and cost:** bit-identical to the pre-refactor code on the six regression composites. Timing is the
  same as 7a (mesh with grid transfer 0.37, with positions 0.95 wall s per simulated s).

## Phase 7c results (2026-10-03): the particle engine without the PDE clock

- **`SmoldynHybrid`:** a plain engine. Each update reads the fields at its start and advances Smoldyn by the update's
  interval. The `coupling` config, the call counter and the PDE clock are gone.
- **`Stepper`** (`processes/stepper.py`): a generic Process that ticks every `dt` and calls any child process on tick
  `phase` of every `every`, with interval `every·dt`. With `every = k` and `phase = k−1`:
  - the child reads its inputs at T+(k−1)·dt;
  - its outputs are first seen at T+k·dt.
  That is vcell-fvsolver's hybrid timing, which plain process-bigraph scheduling (read at start, publish at end)
  cannot express.
- **Builders:** `composites.hybrid.particle_node` builds the particle node for either coupling.
  - `start-of-interval`: `SmoldynHybrid` with interval k·dt.
  - `fvsolver`: `Stepper(SmoldynHybrid)` with interval dt.
- **`HybridCoupler`:** no longer passes a coupling mode.
- **Verification:**
  - bit-identical on the six regression composites, including fvsolver mode with k = 2;
  - `tests/test_stepper.py` checks the read and publish times for k = 1, 2, 3 with toy clock processes;
  - timing within noise of 7b.

## Phase 7d results (2026-10-04): generic splitting coordinator

- **`SplittingCoordinator`** (`processes/splitting.py`) replaces `HybridCoupler`, which instantiated FV + Smoldyn
  directly.
  - It holds a PDE engine, a particle engine and adapter Steps as nodes (address, config, port → store wiring), just
    as in a composite document.
  - Over each interval τ = k·dt it runs the engines in the scheme's order on an internal copy of the stores, and runs
    the adapters whose inputs changed after each run.
  - The schedules are explicit (`splitting.schedule`); Jacobi holds the particle update back until the end of the
    interval.
- **Builders:** `composites.hybrid.to_splitting_document(doc, scheme, dt, k)` converts any start-of-interval
  two-process document, FV, Q1 or mesh. `build_coupler_document` wraps it and gains `pde_engine` and `pde_options`.
- **New capability:** splitting with the unstructured-mesh engine and both mesh adapters (Strang test,
  `tests/test_splitting.py`).
- **Equivalence:** the regression harness now covers all four schemes (nine composites, 20 arrays). The result is
  bit-identical to the code before 7d, and therefore to the original pre-Phase-7 code.
- **Cleanup:** the FV and Q1 engines lost their `particle_species` alias, which only `HybridCoupler` used.

## Phase 7e plan (2026-10-04): geometry compilers

**Decisions (from discussion):**
1. **Geometry format:** the internal format is vcell-fenics' `GeometryDescription` (VCell-style analytic, CSG, image
   or compartmental subvolumes, surface classes and named faces). It is new and used only by vcell-fenics, so it sits
   behind adapters. VCML `Geometry` (pyvcell) and SBML-Spatial adapters come later; SBML-Spatial ties in with Phase 6.
2. **Meshing:** Netgen, through vcell-fenics' `realize()`. vcell-fenics is public and is a pinned git dependency.
3. **Smoldyn membrane:** always the **smooth** surface, with an **accessible-volume correction**. A staircase
   membrane is not faithful to VCell, which corrects the staircase with implicit surfaces (membrane areas and
   normals) but does not remesh it watertight. The correction is switchable:
   - `"adapters+volumes"` (default): the particle→field adapter divides by the accessible volume Aᵢ, and the FV
     engine uses volume fractions Aᵢ/Vᵢ at membrane voxels. Both come from the same smooth surface, so the result is
     conservative and unbiased.
   - `"adapters"` (VCell-faithful PDE volumes): only the adapter divides by Aᵢ. The FV engine keeps vcell-fvsolver's
     full-voxel volumes. This overcounts particle sources by Vᵢ/Aᵢ (about +1.9% for the B2b ball).
   - For the body-fitted mesh, the mesh boundary is the smooth surface, so Aᵢ equals the lumped mass and the
     correction is the identity.

**Architecture.** Each PDE geometry compiler turns the `GeometryDescription` into a `RealizedGeometry`:
- region names and a vectorized `locate(points) → region`;
- each region's smooth boundary as outward-oriented triangles;
- the measure behind each PDE value;
- the bounding box.

Everything on the particle side is then derived from that `RealizedGeometry`, never from the description
independently. This is the lesson of B2b–B2d, where independently realized domains disagreed at the membrane. The
derived pieces:
- the Smoldyn membrane, compartments and interior points;
- the `highResVolumeSamples` map and `boxsize`;
- the adapters' accessible volumes and node region labels.

The two compilers:
- **FEniCS (Netgen):** `vcell_fenics.backend.realize`, giving a tet mesh per subvolume plus the membrane surface mesh.
- **VCell-FV:** VCell's own `CartesianMesh` via libvcell (`.vcg` region map), with VCell's smooth membrane
  triangulation and volume samples (from the `.smoldynInput` that libvcell writes). The co-sim FV then runs on
  exactly the geometry native VCell uses.

**Sub-steps:**
- **7e.1 (done):** dependencies and plan.
  - vcell-fenics is pinned to its main branch at the merge of virtualcell/vcell-fenics#211, which makes its backend imports lazy, so `realize()` needs
    none of the solver-only packages such as `scifem`.
  - Added from conda-forge: Netgen, scikit-image, pydantic and VTK.
  - Netgen realization of the B2b ball: at h = 0.8, 0.6 s and −2.1% volume; at h = 0.5, −1.1%.
- **7e.2 (done):** `viva_pde_particle/geometry`.
  - `sphere_in_box`, and a dict carrier for process configs.
  - The `RealizedGeometry` interface: `locate`, outward boundary triangles, interior points and volumes.
  - `FenicsRealization` / `realize_fenics` (Netgen, cached).
  - `build_mesh` accepts `{"kind": "geometry", "description", "region", "h"}`, and
    `build_mesh_hybrid_document(..., geometry=...)` builds the composite from a description. The Smoldyn membrane is
    the region mesh's boundary, with several interior points. The gmsh sphere path is kept, bit-identical, for the
    pre-7e studies.
  - A hybrid on the Netgen ball conserves mass (1 s: conversion +0.64%, exchange +0.45%) and confines its particles
    to the PDE domain (`tests/test_geometry.py`).
- **7e.3 (done):** `geometry/smoldyn.py`, the Smoldyn geometry derived from a `RealizedGeometry`.
  - `smoldyn_geometry(realized, region, grid, volume_samples)` gives the membrane (the region's boundary triangles),
    several compartment points, and the `highResVolumeSamples` map rasterized with the realization's own `inside`.
    `RealizedGeometry.uniform_points` gives the initial positions.
  - The gmsh ball is wrapped as a one-region `MeshRealization`, so both mesh paths share the derivation.
    Bit-identical on the regression harness.
  - **Non-convex check** (a dumbbell: two overlapping balls joined by a neck, realized by Netgen). Field-driven
    creation fills the compartment at 1.009× rate·volume, evenly between the lobes. With a single compartment point
    it falls to 0.882× and is lopsided, because Smoldyn's line-of-sight compartment test can't see the whole region
    (`tests/test_smoldyn_geometry.py`).
- **7e.4 (done):** the VCell-FV compiler (`geometry/vcell_fv.py`) and the VCML adapter (`geometry/vcml.py`).
  - **`to_vcml_geometry`** turns a description into a pyvcell `Geometry`. Analytic subvolumes; `geom.x[i]` becomes
    `x/y/z` and `**` becomes `^`.
    - Coverage rule: the last subvolume is written as `1.0`. VCell needs every point owned, and complementary
      strict predicates leave interface nodes unowned (libvcell then fails with `null`). Under VCell's priority rule
      this changes only those gap points.
    - `from_vcml_geometry` is the reverse, via vcell-fenics' importer. `to_biomodel` now accepts a description.
  - **`VCellFVRealization`** asks libvcell to write a geometry-only hybrid input (no solve) and reads:
    - the `.vcg`: VCell's node → region map and staircase volumes (`node_regions`, `node_mask`, `pde_volume`);
    - the `.smoldynInput`: VCell's smooth membrane triangulation and compartment points.
    `inside`/`locate` follow the smooth surface (VTK enclosed-points test).
  - **The B2b ball reproduces native exactly:** staircase `cell` 266.734 µm³ over 17,071 nodes (nodes strictly
    inside r = 4); smooth membrane of 9,516 triangles enclosing 261.818 µm³; VCell's 26 compartment points. Smoldyn
    geometry derives from it unchanged. The tests need the `dev` env (spatial-hybrid pyvcell).
- **7e.5 (done):** the co-simulation on VCell's own geometry, with the accessible-volume correction.
  - **Pieces:**
    - `geometry/accessible.py`: accessible fractions, the fold map, effective volumes;
    - adapters: `GridCountsToConcentration(staircase={fold, volumes})` and the new `ExtendGridField` Step;
    - an FV `domain` option (mask and volume fractions);
    - `build_vcell_geometry_hybrid_document(..., correction=)`.
  - **Study B2e (`vcell-geometry-cosim`, 32 seeds):**
    - `none` reproduces native VCell with the vcell-fvsolver#25 correction: shell −1.36%, exchange −2.7%.
    - `adapters` is unbiased but overcounts sources (conversion +1.30%).
    - **`adapters+volumes`** is unbiased (shell +0.02%, z 0.15) and conservative (conversion +0.12%, exchange
      +0.96%, the coupling-lag level).
  - **VTK finding:** `vtkSelectEnclosedPoints` misclassifies lattice-aligned points against VCell's surface,
    depending on process state. The VCell realization now uses `geometry/inside.py` (vertical-ray parity). The B2c
    forward model was re-checked: unchanged.
- **7e.6 (done):** a non-sphere geometry end to end: Study B2f (`dumbbell-three-ways`).
  - One non-convex description (two balls joined by a thin neck) drove the native VCell solver (via VCML), the
    co-sim on VCell's geometry and the co-sim on a Netgen mesh.
  - VCell's smooth membrane is −12.4% of the analytic volume.
  - Native VCell loses 25.9% of A + B in exchange. The co-sim with `none` loses 18.2%, which is the volume mismatch
    alone.
  - `adapters+volumes` gives −0.07%, with no near-membrane bias. The Netgen path gives −0.12%.
  - The native runners now pass `geometry`, `region` and `domain_volume` through.
  - Remaining for later: CSG and image subvolumes in the VCML adapter, and the SBML-Spatial adapter (Phase 6).

## Phase 7f results (2026-10-04): the component contract and assembly; Phase 7 complete

- **Component contract:** written in DESIGN.md §13. It covers units, array layouts, timing, and the ports and config
  of every Process and Step.
- **`kinetics.py`:** holds `reaction_rates`, shared by the FV, Q1 and P1 engines. No engine imports another any
  more; the FEniCSx engines used to import it from the FV module.
- **Workbench generators:**
  - `ball_netgen_hybrid`: a `GeometryDescription` ball, Netgen mesh, both transfer Steps;
  - `ball_vcell_geometry_hybrid`: VCell's own geometry with the accessible-volume correction;
  - `exchange_splitting`: any PDE engine under any splitting scheme. This includes the FEniCSx Q1 engine under
    Strang splitting, which the original code could not express (`tests/test_phase7_generators.py`).
- **Phase 7 acceptance:**
  - The componentized composites reproduce B1 (Q1), B2/B2b/B2d (mesh) and B3 (all four schemes) bit-for-bit on the
    regression harness (`scripts/component_regression.py`).
  - No engine imports or names another engine's concepts.
  - Every coupling concern is a separate component: transfers, scheduling, splitting, geometry.

## Phase 8 (2026-10-04): 3D storage, static 3D figures, interactive spatial viewing

**Why:** runs could not be viewed in 3D. Study scripts kept fields, positions and geometry in memory and wrote only
metrics and 1D/2D plotly pages.

**Decisions (user):**
- **Storage:** both.
  - vcell-fenics results bundles (ADR 010: VTU meshes + zarr point arrays), which VCell's field viewer already reads.
  - The workbench's xarray/zarr emitter for run history.
- **Interactive:** both.
  - VCell's vtk.wasm field viewer, fed by a Python server.
  - A self-contained HTML viewer embedded in the workbench.
- **Static:** PyVista off-screen PNG and GIF.

**Steps:**
1. **8a (done):** bundles and the recorder.
   - `viz3d/bundle.py` writes vcell-fenics schema-1 bundles from numpy (no dolfinx).
     - Grid fields go on the node lattice cut into 6 tets per hex, keeping hexes that touch the PDE domain.
     - P1 mesh fields use the engine's own cells.
     - Membranes are triangle domains.
     - Particle positions use an extension key outside the vcell-fenics manifest.
   - `viz3d/record.py`:
     - `BundleRecorder` writes the bundle rows; membrane values are sampled from the extended field.
     - `attach_recorder(doc, out_dir, output_dt, membranes=)` works for the grid, mesh, VCell-geometry and splitting
       documents.
     - `write_native_bundle` handles native VCell trajectories.
   - `processes/recorder.py`: `SpatialRecorder` ticks on the shortest process interval, so its clock is the PDE's
     float for float. It has no outputs, so recording leaves the run bit-identical (tested). `run_document` records the
     final state through `close`.
   - **Emitter switch:** `workspace.yaml` now uses the xarray emitter (the workbench default) for scalar run history.
     The workbench's xarray wiring emits scalar stores but not field arrays (it needs per-variable coordinates), so
     spatial data lives in bundles.
2. **8b and 8c (done):**
   - `viz3d/static.py`:
     - `render_png`: panels per bundle on a shared colour range. The mid-plane slice with the membrane outline, and the
       field on a cut-away membrane with the particles.
     - `render_gif`: one run over time.
   - `viz3d/html.py` `bundle_html`: a self-contained three.js page.
     - Layers: the membrane coloured by the field with a clipping plane, x/y/z mid-plane slices, and particles.
     - Controls: a time slider and play button, a run selector, and a colour bar.
     - Data: base64 float32, subsampled; about 0.2 MB for a ball.
   - pyvista and imageio are in pixi only.
3. **8d (done):** `viz3d/viewer_server.py` (`pixi run view3d run.fenics ...`).
   - A stdlib loopback server that serves `../vcell/webapp-viewer` and implements `FieldViewerServer`'s JSON
     contract for bundles: `/health`, `/info`, `/grid`, `/field`, `/stats` and `/timeseries`, ported from
     `FenicsBundleViews.java`.
   - It also serves a `/particles` extension.
   - VCell's own viewer renders our bundles. Checked with Playwright (Chromium 1.63, the viewer tests' pin):
     "rendered B @ t = 1 on cell ✓".
   - Membrane fields are named `<field>_<membrane>`, VCell's convention, because the viewer selects variables by
     name.
   - Still to do: the vcell PR that draws particles in the viewer (opened only).
4. **8e:** study figures (B2f first).

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
