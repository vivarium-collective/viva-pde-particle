"""Report which simulation engines are importable in the current environment.

Exit status is non-zero if any *required* engine is missing. Smoldyn is reported
separately because it is built by ``pixi run build-smoldyn``, not installed by
``pixi install``.
"""
from __future__ import annotations

import importlib
import sys

REQUIRED = {
    "process_bigraph": lambda m: getattr(m, "__version__", "?"),
    "bigraph_schema": lambda m: getattr(m, "__version__", "?"),
    "dolfinx": lambda m: m.__version__,
    "pyvcell_fvsolver": lambda m: m.version(),
    "libvcell": lambda m: getattr(m, "__version__", "?"),
    "pyvcell": lambda m: getattr(m, "__version__", "?"),
    "vivarium_workbench": lambda m: getattr(m, "__version__", "?"),
    "viva_pde_particle": lambda m: "editable",
}
OPTIONAL = {
    "smoldyn": lambda m: m.__version__,
}


def probe(name, version_of):
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # noqa: BLE001 - report any import failure
        return False, f"{type(exc).__name__}: {exc}"
    try:
        return True, str(version_of(mod))
    except Exception as exc:  # noqa: BLE001
        return True, f"imported (version unknown: {exc})"


def main() -> int:
    missing = []
    for label, table in (("required", REQUIRED), ("optional", OPTIONAL)):
        for name, version_of in table.items():
            ok, info = probe(name, version_of)
            print(f"[{'ok' if ok else 'MISSING':7}] {label:8} {name:20} {info}")
            if not ok and label == "required":
                missing.append(name)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
