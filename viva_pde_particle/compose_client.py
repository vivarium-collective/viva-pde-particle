"""Run an ensemble on compose-api (compose.cam.uchc.edu: SLURM + Apptainer on UCHC's mantis cluster).

The ensemble is split into blocks of seeds. Each block is one :func:`~viva_pde_particle.ensemble.ensemble_document`,
submitted as an ``.omex`` to ``POST /simulation/run?simulator=viva-pde-particle``. compose-api runs it in the
prebuilt image (:mod:`viva_pde_particle.compose_runner`) as one SLURM job: 2 CPUs, 8 GB, 30 minutes.

Everything about a run lives in one directory, so an interrupted run resumes where it stopped:

- ``manifest.json``: the trial, its parameters and each block's simulation id, status and attempts;
- ``blocks/<first>-<last>.omex``: the documents submitted;
- ``results/<first>-<last>.pber``: each finished block's emitter results.

Failures:
- a block that **failed** is resubmitted, up to ``max_attempts`` times;
- a block that **timed out** or ran **out of memory** is split in two, since resubmitting it won't help.

    python -m viva_pde_particle.compose_client run DIR --trial module:fn --params '{"test": "test4"}' \\
        --seeds 12500 --block 600
    python -m viva_pde_particle.compose_client status DIR
    python -m viva_pde_particle.compose_client submit experiment.omex      # one document, prints its id
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from viva_pde_particle.ensemble import collect, ensemble_document, write_omex

DEFAULT_URL = "https://compose.cam.uchc.edu"
SIMULATOR = "viva-pde-particle"
JOB_CPUS = 2
JOB_WALL_S = 30 * 60
DONE = "completed"
TERMINAL = {"completed", "failed", "cancelled", "timeout", "out_of_memory"}
SPLIT = {"timeout", "out_of_memory"}


def block_size(seconds_per_trial: float, cpus: int = JOB_CPUS, wall_s: float = JOB_WALL_S, fill: float = 0.5) -> int:
    """Seeds per block so a job uses ``fill`` of its wall time on ``cpus`` CPUs."""
    return max(1, int(fill * wall_s * cpus / seconds_per_trial))


class ComposeClient:
    """The compose-api endpoints an ensemble needs: submit, status, results."""

    def __init__(self, url: str = DEFAULT_URL, simulator: str = SIMULATOR, timeout: float = 300.0):
        self.url = url.rstrip("/")
        self.simulator = simulator
        self.timeout = timeout

    def _request(self, path: str, query: dict, data: bytes | None = None, headers: dict | None = None):
        req = urllib.request.Request(f"{self.url}{path}?{urllib.parse.urlencode(query)}", data=data,
                                     headers=headers or {}, method="POST" if data is not None else "GET")
        return urllib.request.urlopen(req, timeout=self.timeout)

    def submit(self, omex: Path, interval: float = 1.0) -> int:
        """Submit one document; its simulation id."""
        boundary = uuid.uuid4().hex
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"uploaded_file\"; "
                f"filename=\"{Path(omex).name}\"\r\nContent-Type: application/zip\r\n\r\n").encode()
        body += Path(omex).read_bytes() + f"\r\n--{boundary}--\r\n".encode()
        with self._request("/simulation/run", {"simulator": self.simulator, "interval_time": interval}, body,
                           {"Content-Type": f"multipart/form-data; boundary={boundary}"}) as r:
            return int(json.load(r)["simulation_database_id"])

    def status(self, sim_id: int) -> dict:
        """The job record, ``status`` lower-cased. Until the SLURM job exists (while the image is fetched, say)
        compose-api answers 404, which reads as ``{"status": "submitting"}``. A submission that fails before
        reaching SLURM stays 404 forever: :class:`EnsembleRun` gives up on it after ``submit_timeout_s``."""
        try:
            with self._request("/results/simulation/status", {"simulation_id": sim_id}) as r:
                rec = json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return {"status": "submitting"}
            raise
        rec["status"] = str(rec.get("status", "")).lower()
        return rec

    def results(self, sim_id: int) -> dict[str, bytes]:
        """The files in a finished simulation's results archive, by name."""
        with self._request("/results/simulation/results/file", {"simulation_id": sim_id}) as r:
            data = r.read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            return {Path(n).name: zf.read(n) for n in zf.namelist() if not n.endswith("/")}


@dataclass
class Block:
    first: int
    last: int
    sim_id: int | None = None
    status: str = "new"  # new | submitting | pending | running | … | completed | failed | split
    attempts: int = 0
    submitted: float | None = None  # epoch seconds of the latest submission
    slurm_job: int | None = None
    seconds: float | None = None
    history: list = field(default_factory=list)  # (sim_id, final status) of earlier attempts

    @property
    def name(self) -> str:
        return f"{self.first}-{self.last}"


class EnsembleRun:
    """One ensemble on compose-api, kept in ``directory`` (see the module docstring)."""

    def __init__(self, directory, trial: str | None = None, params: dict | None = None, seeds: int | None = None,
                 block: int | None = None, workers: int = 0, client: ComposeClient | None = None,
                 max_attempts: int = 3, max_in_flight: int = 40, submit_timeout_s: float = 1800.0, log=print,
                 clock=time.time, max_outage_s: float = 3600.0):
        self.dir = Path(directory)
        self.client = client or ComposeClient()
        self.max_attempts = max_attempts
        self.max_in_flight = max_in_flight
        self.submit_timeout_s = submit_timeout_s
        self.max_outage_s = max_outage_s
        self.log = log
        self.clock = clock
        manifest = self.dir / "manifest.json"
        if manifest.exists():
            m = json.loads(manifest.read_text())
            wanted = {"trial": trial, "params": params, "seeds": seeds, "block": block}
            clash = {k: (m[k], v) for k, v in wanted.items() if v is not None and m[k] != v}
            if clash:
                raise ValueError(f"{manifest} is a different ensemble: {clash}")
            self.trial, self.params, self.seeds, self.block = m["trial"], m["params"], m["seeds"], m["block"]
            self.workers = m.get("workers", 0)
            self.blocks = [Block(**b) for b in m["blocks"]]
        else:
            if None in (trial, seeds, block):
                raise ValueError(f"no manifest in {self.dir}: give trial, seeds and block")
            self.trial, self.params, self.seeds, self.block, self.workers = trial, params or {}, seeds, block, workers
            self.blocks = [Block(s, min(s + block - 1, seeds)) for s in range(1, seeds + 1, block)]
            self.save()

    # ------------------------------------------------------------ state

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        m = {"url": self.client.url, "simulator": self.client.simulator, "trial": self.trial, "params": self.params,
             "seeds": self.seeds, "block": self.block, "workers": self.workers,
             "blocks": [asdict(b) for b in self.blocks]}
        tmp = self.dir / "manifest.json.tmp"
        tmp.write_text(json.dumps(m, indent=1))
        tmp.replace(self.dir / "manifest.json")

    def _live(self) -> list[Block]:
        return [b for b in self.blocks if b.status != "split"]

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for b in self._live():
            out[b.status] = out.get(b.status, 0) + 1
        return out

    def finished(self) -> bool:
        return all(b.status == DONE or (b.status in TERMINAL and b.attempts >= self.max_attempts)
                   for b in self._live())

    # ------------------------------------------------------------ steps

    def _submit(self, b: Block) -> None:
        omex = self.dir / "blocks" / f"{b.name}.omex"
        omex.parent.mkdir(parents=True, exist_ok=True)
        write_omex(ensemble_document(self.trial, self.params, b.first, b.last, self.workers), omex)
        if b.sim_id is not None:
            b.history.append([b.sim_id, b.status])
        b.sim_id = self.client.submit(omex)
        b.status, b.attempts, b.slurm_job, b.seconds = "submitting", b.attempts + 1, None, None
        b.submitted = self.clock()
        self.log(f"submitted seeds {b.name} as simulation {b.sim_id} (attempt {b.attempts})")

    def submit(self) -> int:
        """Submit blocks that are new or failed (attempts left), keeping at most ``max_in_flight`` in flight."""
        n = 0
        in_flight = sum(b.status not in TERMINAL and b.status != "new" for b in self._live())
        for b in list(self._live()):
            if in_flight >= self.max_in_flight:
                break
            if b.status == "new" or (b.status in TERMINAL - {DONE} - SPLIT and b.attempts < self.max_attempts):
                self._submit(b)
                self.save()
                in_flight += 1
                n += 1
        return n

    def poll(self) -> None:
        """Refresh the blocks in flight; fetch finished results; split blocks that ran out of time or memory."""
        for b in list(self._live()):
            if b.status in TERMINAL or b.status == "new":
                continue
            rec = self.client.status(b.sim_id)
            b.status, b.slurm_job = rec["status"], rec.get("slurmjobid", b.slurm_job)
            if b.status == "submitting" and self.clock() - (b.submitted or 0) > self.submit_timeout_s:
                b.status = "failed"
                self.log(f"seeds {b.name}: simulation {b.sim_id} never reached SLURM in {self.submit_timeout_s:.0f} s")
            elif b.status == DONE:
                self._fetch(b, rec)
            elif b.status in SPLIT:
                self._split(b)
            elif b.status in TERMINAL:
                self.log(f"seeds {b.name}: simulation {b.sim_id} {b.status}: {rec.get('error_message')}")
            self.save()

    def _fetch(self, b: Block, rec: dict) -> None:
        files = {n: d for n, d in self.client.results(b.sim_id).items() if n.endswith(".pber")}
        if not files:
            b.status = "failed"
            self.log(f"seeds {b.name}: simulation {b.sim_id} completed with no results file")
            return
        out = self.dir / "results" / f"{b.name}.pber"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(next(iter(files.values())))
        seeds, _ = collect([out])
        if seeds != list(range(b.first, b.last + 1)):
            out.unlink()
            b.status = "failed"
            self.log(f"seeds {b.name}: simulation {b.sim_id} returned seeds {seeds[:3]}…, not the block")
            return
        b.seconds = _elapsed(rec)
        self.log(f"seeds {b.name}: done" + (f" in {b.seconds:.0f} s" if b.seconds else ""))

    def _split(self, b: Block) -> None:
        if b.first == b.last:
            self.log(f"seed {b.first}: simulation {b.sim_id} {b.status} on a single seed; not splitting")
            b.attempts = self.max_attempts
            return
        mid = (b.first + b.last) // 2
        i = self.blocks.index(b)
        self.blocks[i + 1:i + 1] = [Block(b.first, mid), Block(mid + 1, b.last)]
        self.log(f"seeds {b.name}: simulation {b.sim_id} {b.status}; split into {b.first}-{mid} and {mid + 1}-{b.last}")
        b.status = "split"

    def run(self, poll_s: float = 30.0, sleep=time.sleep) -> tuple[list[int], list[list[float]]]:
        """Submit, poll and resubmit until every block finished; the seeds and values (see :meth:`collect`)."""
        last, outage = None, None
        while True:
            try:
                self.submit()
                self.poll()
            except OSError as e:  # urllib's URLError and HTTPError, socket timeouts: the service or the network
                outage = outage if outage is not None else self.clock()
                if self.clock() - outage > self.max_outage_s:
                    raise
                self.log(f"compose-api unreachable ({e}); retrying in {poll_s:.0f} s")
                sleep(poll_s)
                continue
            if outage is not None:
                self.log(f"compose-api reachable again after {self.clock() - outage:.0f} s")
                outage = None
            c = self.counts()
            if c != last:
                self.log(f"blocks: {c}")
                last = c
            if self.finished():
                break
            sleep(poll_s)
        return self.collect()

    def collect(self, partial: bool = False) -> tuple[list[int], list[list[float]]]:
        """Seeds and values in seed order, from ``results/``. Raises if seeds are missing, unless ``partial``."""
        seeds, values = collect(sorted((self.dir / "results").glob("*.pber")))
        missing = sorted(set(range(1, self.seeds + 1)) - set(seeds))
        if missing and not partial:
            raise RuntimeError(f"{len(missing)} of {self.seeds} seeds have no results (first: {missing[:5]}); "
                               f"see {self.dir / 'manifest.json'}")
        return seeds, values

    def summary(self) -> str:
        done = [b for b in self._live() if b.status == DONE]
        secs = [b.seconds / (b.last - b.first + 1) for b in done if b.seconds]
        rate = f", {sum(secs) / len(secs):.2f} s per trial per job" if secs else ""
        return (f"{self.dir}: {self.trial} {self.params}, seeds 1-{self.seeds} in {len(self._live())} blocks: "
                f"{self.counts()}{rate}")


def _elapsed(rec: dict) -> float | None:
    from datetime import datetime

    try:
        return (datetime.fromisoformat(rec["end_time"]) - datetime.fromisoformat(rec["start_time"])).total_seconds()
    except (KeyError, TypeError, ValueError):
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="compose_client")
    parser.add_argument("--url", default=DEFAULT_URL)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="run (or resume) an ensemble kept in DIR")
    r.add_argument("dir", type=Path)
    r.add_argument("--trial")
    r.add_argument("--params", type=json.loads, default=None)
    r.add_argument("--seeds", type=int)
    r.add_argument("--block", type=int)
    r.add_argument("--max-in-flight", type=int, default=40)
    r.add_argument("--poll", type=float, default=30.0)
    s = sub.add_parser("status", help="refresh and show an ensemble's blocks")
    s.add_argument("dir", type=Path)
    one = sub.add_parser("submit", help="submit one document (.omex)")
    one.add_argument("omex", type=Path)
    args = parser.parse_args(argv)
    client = ComposeClient(args.url)
    if args.command == "submit":
        print(client.submit(args.omex))
        return 0
    if args.command == "status":
        run = EnsembleRun(args.dir, client=client)
        run.poll()
        print(run.summary())
        return 0
    run = EnsembleRun(args.dir, args.trial, args.params, args.seeds, args.block, client=client,
                      max_in_flight=args.max_in_flight)
    try:
        seeds, _ = run.run(args.poll)
    except RuntimeError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"{run.summary()}\n{len(seeds)} seeds collected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
