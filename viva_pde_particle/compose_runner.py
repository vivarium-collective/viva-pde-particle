"""The container entrypoint for compose-api: ``[run] <experiment.omex|.pbg> -o <output dir> -n <interval>``.

compose-api's SLURM job runs ``singularity run --compat <image> run /experiment/<id>.<suffix> -o
/experiment/output -n <interval>`` and zips the output directory. pbest's ``main.py`` normally answers
that command, but it pins process-bigraph 1.0.5 and bigraph-schema 1.0.14, which are older than this
workspace's. This module honours the same contract with our own core (:func:`viva_pde_particle.ensemble.
build_runner_core`):

- reads the first ``.pbg`` in the archive, or the ``.pbg`` given directly;
- builds a ``Composite`` and runs it for the interval;
- writes ``results_<date>.pber``: ``gather_emitter_results`` as JSON, as pbest does.

    python -m viva_pde_particle.compose_runner run experiment.omex -o output -n 1
    python -m viva_pde_particle.compose_runner example experiment.omex   # a small real ensemble document
    python -m viva_pde_particle.compose_runner smoke                     # example + run + check, in a temp dir
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def load_document(path: Path, workdir: Path) -> dict:
    if path.suffix == ".omex":
        with zipfile.ZipFile(path) as zf:
            zf.extractall(workdir)
        pbgs = sorted(workdir.rglob("*.pbg"))
        if not pbgs:
            raise FileNotFoundError(f"no .pbg in {path}")
        doc = json.loads(pbgs[0].read_text())
    else:
        doc = json.loads(path.read_text())
    return doc if "state" in doc else {"state": doc}


def run(path: Path, output: Path, interval: float) -> Path:
    """Run the document and write its results; on compose-api, inside a ``task`` span that announces the results
    file as a dataset (:func:`_announce`)."""
    from process_bigraph import events

    emitter = events.get_emitter()  # configured from PBG_* when compose-api's job passes them; silent otherwise
    with emitter.span("task", document=path.name, interval=interval):
        out = _run(path, output, interval)
        _announce(emitter, out)
    emitter.flush()
    return out


def _run(path: Path, output: Path, interval: float) -> Path:
    from process_bigraph import Composite, gather_emitter_results

    from viva_pde_particle.ensemble import build_runner_core

    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        doc = load_document(path, Path(tmp))
        sim = Composite(doc, core=build_runner_core())
        sim.run(interval)
        results = gather_emitter_results(sim)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d#%H-%M-%S")
    out = output / f"results_{stamp}.pber"
    out.write_text(json.dumps({"/".join(map(str, k)) if isinstance(k, tuple) else str(k): _jsonable(v)
                               for k, v in results.items()}))
    return out


def _announce(emitter, out: Path) -> None:
    """``artifact.written`` for the results file: compose-api registers it as a dataset of kind ``results`` with the
    seeds it holds (compose-api docs/plan-observability.md O5). The job script also records the file, by size and
    checksum alone; this event adds what it is."""
    from viva_pde_particle.ensemble import collect

    attributes: dict = {"format": "pber"}
    try:
        seeds, _ = collect([out])
        if seeds:
            attributes.update(seeds=len(seeds), first_seed=min(seeds), last_seed=max(seeds))
    except Exception:  # not an ensemble's results: a plain document run
        pass
    data = out.read_bytes()
    emitter.event("artifact.written", component="viva_pde_particle", uri=str(out.resolve()), kind="results",
                  name=f"ensemble results ({attributes['seeds']} seeds)" if "seeds" in attributes else out.name,
                  bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), attributes=attributes)


EXAMPLE_TRIAL = "viva_pde_particle.benchmarks.fokker_planck:trial_rho"
EXAMPLE_PARAMS = {"test": "test4", "tau": 0.2, "dtau": 1e-3}  # a short co-simulation trial (PDE + Smoldyn)


def write_example(path: Path) -> Path:
    from viva_pde_particle.ensemble import ensemble_document, write_omex

    return write_omex(ensemble_document(EXAMPLE_TRIAL, EXAMPLE_PARAMS, 1, 4), path)


def smoke() -> int:
    """Write the example, run it as compose-api would, and check that four trials came back."""
    from viva_pde_particle.ensemble import collect

    with tempfile.TemporaryDirectory() as tmp:
        omex = write_example(Path(tmp) / "experiment.omex")
        out = run(omex, Path(tmp) / "output", 1.0)
        seeds, values = collect([out])
    ok = seeds == [1, 2, 3, 4] and all(len(v) == 2 for v in values) and any(v[0] > 0 for v in values)
    print(f"smoke {'OK' if ok else 'FAILED'}: seeds {seeds}, values {values}")
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="compose_runner")
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run a process-bigraph document (.omex or .pbg)")
    r.add_argument("input", type=Path)
    r.add_argument("-o", "--output", type=Path, default=Path("output"))
    r.add_argument("-n", "--interval", type=float, default=1.0)
    e = sub.add_parser("example", help="write a small ensemble document (.omex) to run")
    e.add_argument("path", type=Path)
    sub.add_parser("smoke", help="run the example in a temporary directory and check its results")
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in ("run", "example", "smoke", "-h", "--help"):
        argv = ["run", *argv]  # `<doc> -o <dir> -n <t>` without the subcommand (Viva Core's sbatch calls it so)
    args = parser.parse_args(argv)
    if args.command == "example":
        print(f"wrote {write_example(args.path)}")
        return 0
    if args.command == "smoke":
        return smoke()
    out = run(args.input, args.output, args.interval)
    print(f"results written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
