#!/bin/bash
# Activate the pixi environment (conda activation scripts: dolfinx, PETSc, ...) and run the compose runner.
# Apptainer --compat gives a read-only image and an isolated, writable /tmp: caches go there.
set -e
source /app/docker/shell-hook.sh
export TMPDIR="${TMPDIR:-/tmp}"
export MPLCONFIGDIR="$TMPDIR/matplotlib" XDG_CACHE_HOME="$TMPDIR/cache"
exec python -m viva_pde_particle.compose_runner "$@"
