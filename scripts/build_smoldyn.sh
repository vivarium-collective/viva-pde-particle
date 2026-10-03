#!/usr/bin/env bash
# Build and install the Smoldyn python module from external/Smoldyn into the
# active (pixi) environment.
#
#   pixi run build-smoldyn                 # OPTION_VCELL=ON (hybrid ValueProvider hooks)
#   SMOLDYN_VCELL=OFF pixi run build-smoldyn   # vanilla upstream build, for A/B comparisons
#
# OPTION_VCELL compiles smoldynhybrid.c + source/vcell/{SimpleMesh,SimpleValueProvider}
# into libsmoldyn so reactions can carry position-dependent rates (rateValueProvider).
# Graphics are disabled: we run headless and never want the OpenGL/GLUT dependency.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/external/Smoldyn"
VCELL="${SMOLDYN_VCELL:-ON}"
BUILD="$ROOT/build/smoldyn-vcell-$VCELL"
PY="$(command -v python)"

[ -f "$SRC/CMakeLists.txt" ] || { echo "external/Smoldyn missing: git submodule update --init external/Smoldyn" >&2; exit 1; }

# Apply our Smoldyn patches (patches/smoldyn/*.patch, exported from the submodule's
# `pyhybrid` branch) when the submodule is checked out at plain upstream. Idempotent:
# a patch that is already present (e.g. on the pyhybrid branch) is skipped.
shopt -s nullglob
for patch in "$ROOT"/patches/smoldyn/*.patch; do
  if git -C "$SRC" apply --check --reverse "$patch" 2>/dev/null; then
    echo "already applied: $(basename "$patch")"
  elif git -C "$SRC" apply --check "$patch" 2>/dev/null; then
    git -C "$SRC" apply "$patch"
    echo "applied: $(basename "$patch")"
  else
    echo "ERROR: $(basename "$patch") neither applies nor is already applied to external/Smoldyn" >&2
    exit 1
  fi
done
shopt -u nullglob

cmake -S "$SRC" -B "$BUILD" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DOPTION_VCELL="$VCELL" \
  -DOPTION_PYTHON=ON \
  -DOPTION_TARGET_SMOLDYN=ON \
  -DOPTION_TARGET_LIBSMOLDYN=ON \
  -DOPTION_USE_OPENGL=OFF \
  -DOPTION_USE_LIBTIFF=OFF \
  -DOPTION_NSV=ON \
  -DOPTION_EXAMPLES=OFF \
  -DOPTION_DOCS=OFF \
  -DPython3_EXECUTABLE="$PY"

cmake --build "$BUILD" --parallel

# source/python/CMakeLists.txt stages a setup.py + package tree under <build>/py.
"$PY" -m pip install --no-deps --no-build-isolation --force-reinstall "$BUILD/py"

"$PY" -c "import smoldyn, smoldyn._smoldyn as m; print('smoldyn', smoldyn.__version__, 'from', m.__file__)"
