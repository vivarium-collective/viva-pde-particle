# Running ensembles on compose-api

compose-api (https://compose.cam.uchc.edu) runs process-bigraph documents as SLURM jobs on the UCHC
cluster. Each job runs an Apptainer image with this command:

    singularity run --compat <image> run /experiment/<id>.omex -o /experiment/output -n <interval>

(The runner also accepts the form without `run`, `<doc> -o <dir> -n <t>`, which Viva Core uses.)

The service zips the output directory for download.

## The image

`docker/compose.Dockerfile` builds `ghcr.io/vivarium-collective/viva-pde-particle-compose`.

**Contents:**
- the locked pixi `default` environment;
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

**Not included yet:** native VCell. It needs pyvcell with spatial-hybrid support (pyvcell#62, not yet
released), so the image runs the co-simulation only.

## Ensembles as documents

A stochastic study is thousands of short trials. `viva_pde_particle.ensemble.EnsembleRunner` runs a
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

**Trial functions:** an importable `fn(seed, **params) -> list[float]`, for example
`benchmarks.fokker_planck.trial_rho`. Documents use fully qualified `local:` addresses, which
`ensemble.build_runner_core` registers.

**Block size:** a job has a 30-minute limit and 2 CPUs. At 3–7 s per Test 3 trial, 100–200 seeds per
block fit.

## What compose-api still needs

A submission can't yet name its image: every job runs one shared container whose package list is fixed
in compose-api. The plan is a compose-api change that accepts a prebuilt image from an allowlist and pulls
it (`singularity pull docker://…`) instead of building the shared container.
