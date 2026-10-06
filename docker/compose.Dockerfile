# Container image for running viva-pde-particle documents on compose-api (SLURM + Apptainer).
#
# compose-api's job runs:  singularity run --compat <image> run /experiment/<id>.omex -o /experiment/output -n <t>
# so the entrypoint is viva_pde_particle.compose_runner (see that module): it runs a process-bigraph document with this
# workspace's core and writes results_*.pber like pbest does.
#
# Contents: the pixi `default` environment (exactly as locked: --frozen, since the `dev` env's path deps are not in the
# build context; conda-forge dolfinx, netgen, ... + the pypi deps), the patched
# Smoldyn (OPTION_VCELL) built from external/Smoldyn, and this package. linux/amd64 only (the cluster's architecture
# and pixi.lock's linux platform).
#
#   docker build -f docker/compose.Dockerfile -t viva-pde-particle-compose .
#   docker run --rm viva-pde-particle-compose smoke
ARG PIXI_VERSION=0.68.1

FROM ghcr.io/prefix-dev/pixi:${PIXI_VERSION}-bookworm AS build
# binutils: CMake looks for `ar` on the system (the conda compilers do not put a plain `ar` on PATH)
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates binutils \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . .
RUN pixi install --frozen -e default \
    && pixi run --frozen -e default build-smoldyn \
    && pixi shell-hook --frozen -e default > /app/docker/shell-hook.sh \
    && rm -rf build ~/.cache/rattler ~/.cache/pip

FROM debian:bookworm-slim AS runtime
COPY --from=build /app /app
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /app
ENTRYPOINT ["/bin/bash", "/app/docker/compose-entrypoint.sh"]
CMD ["--help"]
