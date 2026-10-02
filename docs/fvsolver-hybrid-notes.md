# How vcell-fvsolver's embedded hybrid PDE/Smoldyn solver works

These are reference notes for reproducing the method as a process-bigraph co-simulation (see [PLAN.md](PLAN.md)).

**Sources:**
- `virtualcell/vcell-fvsolver`: local checkout `../vcell-fvsolver`, branch `SmoldynHashingSpeedup`, traced 2026-10-02.
- `virtualcell/vcell` (Java side): `../vcell`.
- Upstream Smoldyn: `../Smoldyn`, ssandrews/Smoldyn `v2.73-5`.

Line numbers are for those checkouts and will drift.

## Key finding

The two sides **never exchange reaction fluxes**:
- Each side applies only its own half of every mixed (particle × field) reaction.
- Each side reads the other side's state from the previous coupling step.
- The partitioning is done ahead of time by VCell Java.

## Code map

| Role | Location |
|---|---|
| Driver and main loop | `VCell/src/SimTool.cpp`: `start1()` 785–946, loop 871–935, `copyParticleCountsToConcentration()` 950–978 |
| Hybrid glue | `bridgeVCellSmoldyn/vcellhybrid.cpp`: `smoldynInit` 24–101, `smoldynOneStep` 103–106 (`simulatetimestep` + `computeHistogram`), `smoldynEnd` 108–113; `setHybrid()` called in `FiniteVolume.cpp:47`, `SolverMain.cpp:62` |
| PDE → Smoldyn values | `bridgeVCellSmoldyn/VCellValueProvider.cpp:21-84` (also the rate-expression factory) |
| Mesh adapter | `bridgeVCellSmoldyn/VCellMesh.cpp` |
| Smoldyn → PDE binning | `bridgeVCellSmoldyn/VCellSmoldynOutput.cpp`: `parseInput` 117–311, `computeHistogram` 433–552 |
| Particle variables | `VCell/include/VCELL/ParticleVariable.h`, `VolumeParticleVariable.cpp`, `MembraneParticleVariable.cpp` (value arrays + `moleculeCounts`) |
| Smoldyn hybrid hooks (`OPTION_VCELL`) | `smoldyn-2.38/source/Smoldyn/smoldynhybrid.c` (rate evaluation, random placement within a voxel); `smolreact.c` 0th order 2432–2491, 1st 2534–2556, 2nd 2618–2650; `smolsurface.c` `surfUpdateRate` 2241–2275; rate parsing `smolsim.c:1256-1320`, `smolsurface.c:3046` |
| Partitioning (Java) | `vcell-core/.../mapping/ParticleMathMapping.java` `combineHybrid()` 1082–1250; rate unit conversion 820–900 |
| Input writers (Java) | `FiniteVolumeFileWriter.java:287-314,1347,1512`; `SmoldynFileWriter.java:579-602,1135-1138` |

`VCell/src/ParticleContext.cpp` and the particle blocks in `SerialScheduler.cpp` are older particle code that is
commented out. They are **not** part of this coupling.

## Time stepping (`SimTool.cpp:871-935`)

**Setup:**
1. `initSimulation` runs, then `smoldynInit`.
2. Smoldyn's start-up commands run `vcellWriteOutput`, which bins the particles.
3. Counts are converted to concentration.
4. Every variable's `update()` copies current → old (803–811).

**Each PDE step:**
1. `simulation->iterate()` takes one PDE step (`SerialScheduler`).
2. Every k-th step (k = `SMOLDYN_STEP_MULTIPLIER`), the solver calls `smoldynOneStep` and then `copyParticleCountsToConcentration`.
3. `simulation->update()` copies current → old for all variables, particle variables included (`Variable.cpp:84`).

**Both sides read old values only:**
- Every symbol-table value points at `getOld()` (`SimulationExpression.cpp:398-470`).
- This covers the particle variables and the Smoldyn rate expressions.
- So the scheme is **explicit, lagged (Jacobi-style) operator splitting**:
  - Smoldyn sees the PDE values from before the current PDE step.
  - The PDE sees particle concentrations from the last Smoldyn step, held fixed for k PDE steps.
- The fully implicit Sundials PDE solver is refused for hybrid runs (`SimulationExpression.cpp:254`).
- The Smoldyn `time_step` is PDE `dt × k` (`SmoldynFileWriter.java:1135-1138`).

## PDE → particles (`VCellValueProvider::getValue`)

**Lookup:**
- **Volume:** the value is taken from the voxel containing the point (`CartesianMesh::getVolumeIndex`). There is **no interpolation**.
- **Membrane:** the element index comes from the panel name `tri_a_b_<memIndex>`.

**Where each reaction order samples it:**
- **0th order:** loops over every voxel whose centre is in the compartment. Count ~ Poisson(rate·dt·dx·dy·dz), placed
  uniformly at random inside that voxel. On membranes, it uses each triangle's centre and area.
- **1st order:** the rate is evaluated at the molecule's position, then converted to a probability (`rxnsetrate`).
- **2nd order:** the rate is evaluated at the **midpoint** of the two reactants.
- **Surface actions:** the rate is evaluated at the molecule's position, then `surfupdateparams` is called.

## Particles → PDE (`computeHistogram`)

**Binning:**
- **Volume molecules** go to the nearest grid node: `i = (int)((x-x0)/dx + 0.5)` with `dx = extent/(Nx-1)`. This
  matches the PDE mesh's node-centred convention (`CartesianMesh.cpp:804`).
- **Membrane molecules** go by the membrane index in the panel name.

**Units:**
- **Volume:** concentration = count / `getVolumeOfElement_cu(j)`, giving molecules/µm³. Boundary elements have reduced
  (half/quarter) volumes.
- **Membrane:** concentration = count / element area, giving molecules/µm².
- **Inconsistency:** 0th-order creation uses the *full* voxel volume even at the boundary.
- The conversion to VCell units (µM, molecules/µm²) is **not** in the C++ code. It is a factor written into the PDE expressions.

## Model partitioning (`combineHybrid`)

- **Which species become particles:** species with `isForceContinuous()` stay continuous (PDE/ODE). All others become particles.
- **Building the math:**
  1. The full deterministic math is generated first.
  2. For each particle species, the concentration variable is replaced by a function: unit factor × particle variable
     (molecules/µm³ → µM).
  3. That species' own PDE is removed.
- **PDE side:** keeps all reaction terms for continuous species. Where a particle species takes part, the term reads
  its binned concentration.
- **Smoldyn side:** gets one reaction per original reaction.
  - Continuous reactants are removed from the reactant list, and the rate is multiplied by their concentration.
    Example: `A_p + B_f → …` becomes `A_p → …` with rate `k·[B]`.
  - Actions on continuous species are dropped.
  - Reactions involving only continuous species are dropped.

## Input format keywords

**`.fvinput`** (`FVSolver.cpp`):
- `VOLUME_PARTICLE <var> <feature>` and `MEMBRANE_PARTICLE <var> <membrane>` (388–397)
- `SMOLDYN_STEP_MULTIPLIER k` (1103)
- A `SMOLDYN_BEGIN` / `SMOLDYN_INPUT_FILE <path>` / `SMOLDYN_END` block (1461, 1505–1530)

**`.smoldynInput`, VCell-specific parts:**
- **Field-dependent rates:** a rate written as an expression ending in `;` (e.g. `k*B;`) is bound to PDE fields.
  A plain number stays constant. This applies to `reaction`, `reaction_cmpt`, `reaction_surface`, `reaction_rate`
  and surface `rate`.
- **Compartment map:** `start_highResVolumeSamples … end_highResVolumeSamples` is a compressed hex voxel map used to
  decide which compartment a point is in.
- **Membrane panels** must be named `tri_<i>_<j>_<memIndex>`.
- **`vcellWriteOutput`** is written as
  `cmd N n vcellWriteOutput begin … dimension / sampleSize Nx Ny Nz / numMembraneElements / variable <name> volume|membrane <domain> … end`.
  The `variable` lines must follow the same order as the species declarations.
- **Other commands:** `vcellPrintProgress`, `vcellDataProcess`.

## Build and Python access

**Build:**
- `OPTION_TARGET_PYTHON_BINDING` (default ON; scikit-build only) forces `OPTION_VCELL` ON. Native builds need
  `-DOPTION_TARGET_PYTHON_BINDING=OFF`.
- Smoldyn is always linked: the `vcell` library links `smoldyn_static` and `vcellsmoldynbridge`
  (`VCell/CMakeLists.txt:220`), and the bridge adds `-DVCELL_HYBRID -DVCELL`.

**Python:**
- **`pyvcell_fvsolver`** (PyPI 0.10.7, with cp312 arm64 wheels) exposes only `version()` and
  `solve(fvinput, vcg, outdir)`, which runs to completion. There is no stepping and no access to field values.
- **`libvcell`** only converts files (VCML/SBML → fvinput). Whether it emits hybrid inputs is not yet verified.
- **pyvcell** can *represent* particle math (`ParticleProperties`, `ParticleJumpProcess`,
  `SpeciesMapping.force_continuous`), but cannot run it: `add_application` hardcodes `stochastic=False`.

**Test cases:** the repo has **no hybrid test cases**. A golden hybrid case has to be generated from VCML via VCell or `libvcell`.

## Output

- The usual FV outputs (`.sim` per save in `.zip`, `.log`, `.mesh`, `.meshmetrics`, `.hdf5`).
- **Particle variables are written as raw molecule counts per voxel or membrane element, not concentrations**
  (`FVDataSet.cpp:413-418`).
- In hybrid mode `VCellSmoldynOutput::write()` returns straight after binning (313–317), so Smoldyn writes nothing of its
  own and no particle positions are saved.

## Upstream Smoldyn (2.7x) for the co-simulation

**Python API** (`source/python/module.cpp`):
- **Stepping:** `runTimeStep`, `runUntil`, `setSimTimes`, `updateSim`.
- **Adding molecules:** `addSolutionMolecules(species, n, lowpos, highpos)` (one box per call).
- **Removing molecules:** only through commands (`killmol*`, `fixmolcount*`).
- **Reading positions:** through text output (`listmols`/`molpos` + `getOutputData`).
- **Rates:** `setReactionRate` takes a single scalar per reaction.

**Field-dependent rates are not available from Python:**
- The VCell hooks still exist upstream behind `OPTION_VCELL` (default OFF):
  - `ValueProvider` / `ValueProviderFactory` / `AbstractMesh` in `smoldyn.h`
  - `rxn->rateValueProvider` in `smolreact.c`
  - `smoldynhybrid.c`
  - `source/vcell/SimpleValueProvider.*` and `SimpleMesh.*`
  - an `#ifdef OPTION_VCELL` load path in `module.cpp`
- The plan is to add a grid-backed `ValueProvider` fed from numpy, plus numpy accessors for positions and histograms
  (see PLAN.md §Technical design 1).

**Packaging:** the PyPI `smoldyn` 2.75 wheels are Windows-only, and older macOS wheels are x86_64. Build from source instead.
