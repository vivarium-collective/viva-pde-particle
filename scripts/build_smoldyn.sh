#!/usr/bin/env bash
# Build and install the Smoldyn python module from external/Smoldyn (the
# virtualcell/Smoldyn fork, branch pyhybrid) into the active (pixi) environment.
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
# one build tree per pixi environment (the CMake cache pins the Python interpreter)
ENV_NAME="$(basename "${CONDA_PREFIX:-default}")"
BUILD="$ROOT/build/smoldyn-vcell-$VCELL-$ENV_NAME"
PY="$(command -v python)"

[ -f "$SRC/CMakeLists.txt" ] || { echo "external/Smoldyn missing: git submodule update --init external/Smoldyn" >&2; exit 1; }

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
