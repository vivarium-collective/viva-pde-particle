# Running ensembles on compose-api

compose-api (https://compose.cam.uchc.edu) runs process-bigraph documents as SLURM jobs on the UCHC
cluster. Each job runs an Apptainer image with this command:

    singularity run --compat <image> run /experiment/<id>.omex -o /experiment/output -n <interval>

(The runner also accepts the form without `run`, `<doc> -o <dir> -n <t>`, which Viva Core uses.)

The service zips the output directory for download.

## The image

`docker/compose.Dockerfile` builds `ghcr.io/vivarium-collective/viva-pde-particle-compose`.

**Contents:**
- the locked pixi `compose` environment: `default` without the vivarium-workbench dashboard (the `workbench`
  feature), so process-bigraph's package discovery does not import a web app on every job;
- the patched Smoldyn (OPTION_VCELL), built from `external/Smoldyn`;
- this package.

Its entrypoint is `viva_pde_particle.compose_runner`. It honours the job command above with our own core.
pbest's runtime does the same, but pins process-bigraph 1.0.5 and bigraph-schema 1.0.14, which are older than
this workspace's.

**Build:** `.github/workflows/compose-image.yml` builds it for linux/amd64, then tests it in two steps:
- `smoke`: a small co-simulation ensemble, run in Docker;
- the exact job contract under `apptainer run --compat` with `/experiment` bound.

It pushes `sha-<short>` and `latest` from `main`.

    docker run --rm ghcr.io/vivarium-collective/viva-pde-particle-compose smoke
    apptainer run --compat --bind $PWD/exp:/experiment image.sif example /experiment/experiment.omex
    apptainer run --compat --bind $PWD/exp:/experiment image.sif run /experiment/experiment.omex -o /experiment/output -n 1

**Not included:** native VCell. pyvcell 0.4.2 (2026-10-08) has the spatial-hybrid support (pyvcell#62, #63), but the
`compose` environment carries neither pyvcell nor libvcell nor pyvcell-fvsolver, so the image runs the co-simulation
only.

## Ensembles as documents

A stochastic study is thousands of short trials. An `viva_pde_particle.ensemble.Ensemble` subclass runs a
*trial function* over a block of seeds in one Process, using the CPUs the job was given (the affinity
mask). A study is therefore a few dozen documents rather than thousands.

```python
from viva_pde_particle.ensemble import collect, ensemble_documents, write_omex

docs = ensemble_documents("viva_pde_particle.benchmarks.fokker_planck:trial_rho",
                          {"test": "test3", "dtau": 0.001, "sample_times": [30, 35, 40, 45, 50]},
                          seeds=4000, block=100)
for i, d in enumerate(docs):
    write_omex(d, f"block_{i:03d}.omex")      # submit each to POST /simulation/run
# ... after downloading each job's results.zip:
seeds, values = collect(glob.glob("results/*/results_*.pber"))
```

**Trial functions:** `fn(seed, **params) -> list[float]`, each run by its own `Ensemble` subclass, for example
`FokkerPlanckEnsemble` for `benchmarks.fokker_planck.trial_rho`. The document names the class by its address,
`local:viva_pde_particle.ensemble.FokkerPlanckEnsemble`, so a registry check of addresses (compose-api's) covers
which trial runs, and nothing in a document's config is imported. Callers still name a trial `"module:function"`;
`ensemble.ENSEMBLES` maps each name to its class, and a name with no class is refused. A new trial function needs a
new subclass in `ensemble.py`. Documents use fully qualified `local:` addresses, the emitter's included
(`local:process_bigraph.emitter.RAMEmitter`), which `ensemble.build_runner_core` registers; it still registers the
short `ram-emitter` for older documents.

**Block size:** a job has a 30-minute limit and 2 CPUs. At 3–7 s per Test 3 trial, 100–200 seeds per
block fit.

## What compose-api still needs

A submission can't yet name its image: every job runs one shared container whose package list is fixed
in compose-api. The plan is a compose-api change that accepts a prebuilt image from an allowlist and pulls
it (`singularity pull docker://…`) instead of building the shared container.

## The studies' simulations on compose-api

`scripts/compose_submit_studies.py` submits one document per study baseline (and per variant the image can run):
seed 1, the study's model, time step and end time, with a spatial recorder so the run leaves a web-viewable
results bundle (docs/web-bundle.md). Every run is tagged `pde-particle`, with the study slug and its investigation
slug, and carries a one-line description. Tags and the description are given at submission and cannot be edited.

    compose-api simulations list --tag pde-particle                         # all of them
    compose-api simulations list --tag pde-particle --tag dumbbell-three-ways
    pixi run python scripts/compose_submit_studies.py status                # refresh the record

The record is `workspace/reports/compose-submissions.json` (tracked): study, document name, simulation id, tags,
interval, status, and the ids of earlier attempts. The documents themselves are under
`workspace/studies/<slug>/runs/compose/` (gitignored). The script's docstring has the build/submit steps and the
environments each needs: the grid documents build in `default`, the VCell-realization documents in `dev`
(libvcell realizes the geometry; running them needs nothing of VCell), and the unstructured-mesh documents **in
the compose image itself**.

**Why mesh documents are built in the image.** A mesh document carries a mesh *spec* (`kind: sphere` or
`kind: geometry`, plus `h`), and every process re-meshes it at run time. gmsh and Netgen on Linux produce a
different mesh than on macOS (653 vs 643 P1 nodes for the h = 0.8 ball), so a document built on a Mac arrives
with initial `field_dofs`, boundary triangles and volume samples that do not fit the image's mesh, and fails in
its first transfer step. Building in the image (`docker run --platform linux/amd64 ...`, see the docstring) sidesteps
this; carrying the realized mesh (points and cells) in the document would remove it.

**Not submitted:** the native VCell runs (no pyvcell in the image) and study C3's trial ensembles, which ran earlier
as `EnsembleRun` blocks (`workspace/studies/single-channel-fokker-planck/runs/compose/`) before tags existed.
