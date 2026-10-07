"""Ensembles on compose-api (EnsembleRun), against a fake compose-api that runs each block locally."""
from __future__ import annotations

import io
import json
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from viva_pde_particle import compose_runner
from viva_pde_particle.compose_client import ComposeClient, EnsembleRun, block_size
from viva_pde_particle.ensemble import collect

TRIAL = "viva_pde_particle.ensemble:example_trial"


class FakeCompose:
    """Runs a submitted block at its first status call. ``outcomes`` scripts the statuses of given submissions
    (by submission order, 0-based); every other submission completes."""

    url = "fake://compose"
    simulator = "viva-pde-particle"

    def __init__(self, tmp_path, outcomes=None):
        self.tmp = tmp_path
        self.outcomes = outcomes or {}
        self.jobs: dict[int, dict] = {}

    def submit(self, omex, interval=1.0):
        sim_id = 100 + len(self.jobs)
        self.jobs[sim_id] = {"omex": omex.read_bytes(), "n": len(self.jobs), "polls": 0}
        return sim_id

    def status(self, sim_id):
        job = self.jobs[sim_id]
        job["polls"] += 1
        if job["polls"] == 1:
            return {"status": "submitting"}
        status = self.outcomes.get(job["n"], "completed")
        if status == "completed" and "out" not in job:
            src = self.tmp / f"in{sim_id}.omex"
            src.write_bytes(job["omex"])
            compose_runner.main(["run", str(src), "-o", str(self.tmp / f"out{sim_id}"), "-n", "1"])
            job["out"] = next((self.tmp / f"out{sim_id}").glob("*.pber")).read_bytes()
        return {"status": status, "slurmjobid": sim_id + 5000, "error_message": None,
                "start_time": "2026-10-07 00:00:00", "end_time": "2026-10-07 00:00:10"}

    def results(self, sim_id):
        return {"results_x.pber": self.jobs[sim_id]["out"]}


def _run(tmp_path, client, **kw):
    kw = {"trial": TRIAL, "params": {"scale": 2.0}, "seeds": 10, "block": 4, "workers": 1, **kw}
    return EnsembleRun(tmp_path / "ens", client=client, log=lambda *_: None, **kw)


def test_block_size_fills_half_the_job():
    assert block_size(1.0) == 1800  # 30 min × 2 CPUs × ½
    assert block_size(10_000.0) == 1


def test_runs_every_block_and_collects_in_seed_order(tmp_path):
    fake = FakeCompose(tmp_path)
    run = _run(tmp_path, fake)
    seeds, values = run.run(poll_s=0, sleep=lambda s: None)
    assert seeds == list(range(1, 11))
    assert values == [[2.0 * s, float(s * s)] for s in seeds]
    assert [b.name for b in run.blocks] == ["1-4", "5-8", "9-10"]
    assert all(b.status == "completed" and b.seconds == 10.0 for b in run.blocks)


def test_failed_blocks_are_resubmitted_and_timeouts_split(tmp_path):
    fake = FakeCompose(tmp_path, outcomes={0: "failed", 1: "timeout"})  # block 1-4 fails once; 5-8 times out
    run = _run(tmp_path, fake)
    seeds, _ = run.run(poll_s=0, sleep=lambda s: None)
    assert seeds == list(range(1, 11))
    by_name = {b.name: b for b in run.blocks}
    assert by_name["1-4"].attempts == 2 and by_name["1-4"].history == [[100, "failed"]]
    assert by_name["5-8"].status == "split"
    assert by_name["5-6"].status == by_name["7-8"].status == "completed"


def test_gives_up_after_max_attempts(tmp_path):
    fake = FakeCompose(tmp_path, outcomes={n: "failed" for n in range(20)})
    run = _run(tmp_path, fake, seeds=4, block=4, max_attempts=2)
    with pytest.raises(RuntimeError, match="4 of 4 seeds have no results"):
        run.run(poll_s=0, sleep=lambda s: None)
    assert run.blocks[0].attempts == 2 and len(fake.jobs) == 2


def test_submission_that_never_reaches_slurm_times_out(tmp_path):
    class Stuck(FakeCompose):
        def status(self, sim_id):
            return {"status": "submitting"}

    now = [0.0]
    run = _run(tmp_path, Stuck(tmp_path), seeds=4, block=4, max_attempts=1, submit_timeout_s=60, clock=lambda: now[0])
    run.submit()
    run.poll()
    assert run.blocks[0].status == "submitting"
    now[0] = 61.0
    run.poll()
    assert run.blocks[0].status == "failed" and run.finished()


def test_resumes_from_the_manifest_without_resubmitting(tmp_path):
    fake = FakeCompose(tmp_path)
    first = _run(tmp_path, fake, max_in_flight=2)
    first.submit()
    assert len(fake.jobs) == 2
    again = EnsembleRun(tmp_path / "ens", client=fake, log=lambda *_: None)  # parameters come from the manifest
    assert (again.trial, again.seeds, again.block) == (TRIAL, 10, 4)
    assert [b.sim_id for b in again.blocks] == [100, 101, None]
    seeds, _ = again.run(poll_s=0, sleep=lambda s: None)
    assert seeds == list(range(1, 11)) and len(fake.jobs) == 3
    with pytest.raises(ValueError, match="different ensemble"):
        EnsembleRun(tmp_path / "ens", trial=TRIAL, seeds=20, block=4, client=fake)


def test_wrong_seeds_in_results_fail_the_block(tmp_path):
    class Wrong(FakeCompose):
        def results(self, sim_id):
            return {"results_x.pber": json.dumps({"e": [{"seeds": [99], "values": [[0.0]]}]}).encode()}

    run = _run(tmp_path, Wrong(tmp_path), seeds=4, block=4, max_attempts=1)
    run.submit()
    run.poll()
    run.poll()
    assert run.blocks[0].status == "failed" and not list((tmp_path / "ens" / "results").glob("*"))


def test_http_client_against_a_stub_server(tmp_path):
    """The requests ComposeClient makes: a multipart upload, then status (404 = submitting), then a results zip."""
    seen = {}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("results_1.pber", json.dumps({"e": [{"seeds": [1], "values": [[1.0]]}]}))

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            seen["post"] = (self.path, self.headers["Content-Type"],
                            self.rfile.read(int(self.headers["Content-Length"])))
            self._send(200, json.dumps({"simulation_database_id": 4192}).encode())

        def do_GET(self):
            if self.path.startswith("/results/simulation/status"):
                if "4192" in self.path:
                    self._send(200, json.dumps({"status": "COMPLETED", "slurmjobid": 1}).encode())
                else:
                    self._send(404, b'{"detail": "not found"}')
            else:
                self._send(200, buf.getvalue(), "application/zip")

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        client = ComposeClient(f"http://127.0.0.1:{server.server_port}")
        omex = tmp_path / "experiment.omex"
        omex.write_bytes(b"PK-archive-bytes")
        assert client.submit(omex) == 4192
        path, ctype, body = seen["post"]
        assert path == "/simulation/run?simulator=viva-pde-particle&interval_time=1.0"
        assert ctype.startswith("multipart/form-data; boundary=")
        assert b'name="uploaded_file"; filename="experiment.omex"' in body and b"PK-archive-bytes" in body
        assert client.status(4192)["status"] == "completed"
        assert client.status(1) == {"status": "submitting"}
        files = client.results(4192)
        (tmp_path / "r.pber").write_bytes(files["results_1.pber"])
        assert collect([tmp_path / "r.pber"]) == ([1], [[1.0]])
    finally:
        server.shutdown()


def test_network_errors_are_retried_then_raised(tmp_path):
    class Flaky(FakeCompose):
        down = 2

        def status(self, sim_id):
            if self.down:
                self.down -= 1
                raise TimeoutError("timed out")
            return super().status(sim_id)

    seeds, _ = _run(tmp_path, Flaky(tmp_path)).run(poll_s=0, sleep=lambda s: None)
    assert seeds == list(range(1, 11))

    class Down(FakeCompose):
        def status(self, sim_id):
            raise OSError("unreachable")

    now = iter(range(0, 10_000, 100))
    run = _run(tmp_path / "down", Down(tmp_path), clock=lambda: next(now), max_outage_s=500)
    with pytest.raises(OSError):
        run.run(poll_s=0, sleep=lambda s: None)
