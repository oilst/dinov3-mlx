"""Tests T1 to T8 of specs/06-speed/plan.html: the GPU lock (F14).

The tests use a temporary lock folder, a test pattern with a unique token and child processes
without DINOV3_GPU_LOCK. So they never wait for a real job, and the lock of the session does not
let them pass at once.
"""

import itertools
import json
import os
import re
import subprocess
import sys
import time
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from dinov3_mlx import gpu_lock
from dinov3_mlx.gpu_lock import (
    HEAVY_JOB_PATTERN,
    LOCK_ENV,
    GpuLockTimeoutError,
    heavy_job,
    other_heavy_jobs,
)

# One waiter: python -c WAITER <lock dir> <pattern> <poll s> <hold s> <label> [token].
WAITER = """
import json, sys, time
from pathlib import Path
from dinov3_mlx.gpu_lock import heavy_job

folder, pattern, poll, hold, label = Path(sys.argv[1]), sys.argv[2], *sys.argv[3:6]
asked = time.time()
with heavy_job(
    label,
    poll_s=float(poll),
    lock_path=folder / "gpu.lock",
    registry=folder / "gpu-lock.d",
    pattern=pattern,
    log_path=folder / "gpu-lock.log",
):
    enter = time.time()
    time.sleep(float(hold))
    leave = time.time()
print(json.dumps({"label": label, "asked": asked, "enter": enter, "leave": leave}))
"""
# A parent whose command line matches the test pattern, as uv run does: it runs one waiter.
PARENT = """
import subprocess, sys
done = subprocess.run([sys.executable, "-c", sys.argv[1], *sys.argv[2:-1]], capture_output=True,
                      text=True)
sys.stdout.write(done.stdout)
sys.stderr.write(done.stderr)
"""


@pytest.fixture
def token() -> str:
    """A unique word for the command lines of the test processes."""
    return f"gpulocktest{uuid.uuid4().hex[:12]}"


def child_env() -> dict[str, str]:
    """The environment of a child process without the lock of the session."""
    return {k: v for k, v in os.environ.items() if k != LOCK_ENV}


def start_waiter(
    folder: Path, pattern: str, hold_s: float, label: str, *extra: str, poll_s: float = 0.05
) -> subprocess.Popen:
    """Start one waiter in a child process."""
    args = [str(folder), pattern, str(poll_s), str(hold_s), label, *extra]
    return subprocess.Popen(
        [sys.executable, "-c", WAITER, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=child_env(),
    )


def result_of(process: subprocess.Popen) -> dict:
    """Wait for a waiter and return its JSON line."""
    stdout, stderr = process.communicate(timeout=30)
    assert process.returncode == 0, stderr
    return json.loads(stdout.splitlines()[-1])


def wait_for_hold(folder: Path, label: str, timeout_s: float = 10.0) -> None:
    """Wait until the lock log has the hold line of one label."""
    log = folder / "gpu-lock.log"
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if log.exists() and any(
            line.split("\t")[2:4] == ["hold", label] for line in log.read_text().splitlines()
        ):
            return
        time.sleep(0.02)
    raise AssertionError(f"no hold line of {label}")


def test_heavy_job_pattern_matches_the_job_names() -> None:
    """T1."""
    jobs = [
        "/repo/.venv/bin/python3 /repo/.venv/bin/pytest -k 7B",
        "python scripts/bench_speed.py matrix",
        "python scripts/bench_mlx_port.py --priority P2",
        "python scripts/benchmark.py --dry-run",
        "python scripts/cross_check.py",
        "python scripts/make_reference.py --streamed",
        "python scripts/profile_block.py",
        "python -m dinov3_mlx.gpu_lock -- true",
    ]
    for job in jobs:
        assert re.search(HEAVY_JOB_PATTERN, job), job
    for other in ["/repo/.venv/bin/ruff check", "python -c pass"]:
        assert not re.search(HEAVY_JOB_PATTERN, other), other


def test_heavy_job_waits_for_the_lock_holder(tmp_path: Path, token: str) -> None:
    """T2."""
    a = start_waiter(tmp_path, token + "x", 1.0, "A")
    time.sleep(0.2)
    b = start_waiter(tmp_path, token + "x", 0.0, "B")
    first, second = result_of(a), result_of(b)
    assert second["enter"] >= first["leave"]


def test_heavy_job_waits_for_a_matching_process(tmp_path: Path, token: str) -> None:
    """T3: the waiter has the token in its own command line, so it must leave itself out."""
    # start_new_session: the job of another agent runs in its own process group.
    command = [sys.executable, "-c", "import time; time.sleep(1.0)", token]
    other = subprocess.Popen(command, start_new_session=True)
    started = time.time()
    try:
        waiter = start_waiter(tmp_path, token, 0.0, "W", token)
        result = result_of(waiter)
    finally:
        other.wait(timeout=10)
    assert result["enter"] >= started + 0.9
    assert other.returncode == 0


def test_other_heavy_jobs_leaves_out_own_process_and_ancestors(tmp_path: Path) -> None:
    """T3: the own process and its parent (for example uv run) never count."""
    table = gpu_lock._processes()
    for pid in (os.getpid(), os.getppid()):
        command = table[pid][1]
        assert command not in other_heavy_jobs(re.escape(command), tmp_path)


def test_other_heavy_jobs_leaves_out_the_group_of_a_registered_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Finding V16 of specs/06-speed/verify.html: the forked subshell of a pipeline keeps the
    command line of its shell and shares the process group of the registered job, so it does not
    count. A process of another group still counts."""
    shell = "zsh -c uv run python -m dinov3_mlx.gpu_lock -- python scripts/bench_x.py | grep x"
    table = {
        900: (1, 900, "zsh"),
        901: (900, 901, shell),  # the shell of the job: the leader of the job's process group
        902: (901, 901, "uv run python -m dinov3_mlx.gpu_lock -- python scripts/bench_x.py"),
        903: (901, 901, shell),  # the forked subshell of the pipeline: a sibling of 902
        950: (1, 950, "python scripts/bench_y.py"),  # a heavy job of another shell
    }
    monkeypatch.setattr(gpu_lock, "_process_groups", lambda: table)
    (tmp_path / "902").touch()
    found = other_heavy_jobs(r"scripts/bench_", tmp_path)
    assert found == ["python scripts/bench_y.py"]


def test_child_of_the_lock_holder_does_not_wait(tmp_path: Path, token: str) -> None:
    """T4: LOCK_ENV names an ancestor of the child (this test process)."""
    holder = start_waiter(tmp_path, token + "x", 2.0, "A")
    wait_for_hold(tmp_path, "A")
    env = child_env() | {LOCK_ENV: str(os.getpid())}
    args = [str(tmp_path), token + "x", "0.05", "0", "C"]
    done = subprocess.run(
        [sys.executable, "-c", WAITER, *args],
        check=False,
        capture_output=True,
        text=True,
        env=env,
        timeout=30,
    )
    holder_result = result_of(holder)
    child = json.loads(done.stdout.splitlines()[-1])
    assert child["enter"] - child["asked"] < 0.1
    assert child["enter"] < holder_result["leave"]


def test_two_waiters_enter_one_after_the_other(tmp_path: Path, token: str) -> None:
    """T5: B and C run under a parent that matches the test pattern, as uv run does. Each parent
    has its own session, as two jobs of two shells (finding V14 of specs/06-speed/verify.html):
    then other_heavy_jobs sees the other parent, and the wait loop before the flock runs."""
    holder = start_waiter(tmp_path, token, 1.0, "A")
    time.sleep(0.2)
    parents = [
        subprocess.Popen(
            [
                sys.executable,
                "-c",
                PARENT,
                WAITER,
                str(tmp_path),
                token,
                "0.05",
                "0.3",
                label,
                token,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=child_env(),
            start_new_session=True,
        )
        for label in ("B", "C")
    ]
    start = time.time()
    results = [result_of(holder)] + [result_of(parent) for parent in parents]
    assert all(result["enter"] - start < 5 for result in results[1:])
    holds = sorted((r["enter"], r["leave"]) for r in results)
    for (_, leave), (enter, _) in itertools.pairwise(holds):
        assert enter >= leave
    assert list((tmp_path / "gpu-lock.d").iterdir()) == []


def test_heavy_job_writes_the_lock_log(
    tmp_path: Path, token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T6."""
    monkeypatch.delenv(LOCK_ENV, raising=False)
    holder = start_waiter(tmp_path, token + "x", 0.5, "A")
    wait_for_hold(tmp_path, "A")
    log = tmp_path / "gpu-lock.log"
    with heavy_job(
        "T6",
        poll_s=0.05,
        lock_path=tmp_path / "gpu.lock",
        registry=tmp_path / "gpu-lock.d",
        pattern=token + "x",
        log_path=log,
    ):
        assert os.environ[LOCK_ENV] == str(os.getpid())
    result_of(holder)
    mine = [line.split("\t") for line in log.read_text().splitlines() if "\tT6" in line]
    events = [fields[2] for fields in mine]
    assert events[0] == "wait" and events[-2:] == ["hold", "end"]
    assert "wait for" in events
    assert all(fields[1] == f"pid {os.getpid()}" and fields[3] == "T6" for fields in mine)
    assert LOCK_ENV not in os.environ


def test_heavy_job_timeout_ends_a_wait(
    tmp_path: Path, token: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T7 (R54)."""
    monkeypatch.delenv(LOCK_ENV, raising=False)
    holder = start_waiter(tmp_path, token + "x", 3.0, "A")
    wait_for_hold(tmp_path, "A")
    start = time.monotonic()
    with (
        pytest.raises(GpuLockTimeoutError),
        heavy_job(
            "T7",
            poll_s=20.0,
            lock_path=tmp_path / "gpu.lock",
            registry=tmp_path / "gpu-lock.d",
            pattern=token + "x",
            log_path=tmp_path / "gpu-lock.log",
            timeout_s=0.3,
        ),
    ):
        pass
    assert 0.3 <= time.monotonic() - start <= 1.0
    assert not (tmp_path / "gpu-lock.d" / str(os.getpid())).exists()
    command = [sys.executable, "-m", "dinov3_mlx.gpu_lock", "--lock-dir", str(tmp_path)]
    done = subprocess.run(
        [*command, "--timeout", "0.3", "--", "true"],
        check=False,
        env=child_env(),
        capture_output=True,
    )
    assert done.returncode != 0
    result_of(holder)


def test_gpu_lock_command_runs_the_command(tmp_path: Path) -> None:
    """The command line returns the exit code of its command.

    The temporary lock folder and a pattern that matches no process keep the test free of the
    jobs and the lock of other sessions (V1 of specs/06-speed/verify.html).
    """
    nothing = f"[n]o-heavy-job-{uuid.uuid4().hex}"
    command = [sys.executable, "-m", "dinov3_mlx.gpu_lock", "--lock-dir", str(tmp_path)]
    done = subprocess.run(
        [
            *command,
            *("--pattern", nothing, "--poll", "0.05", "--timeout", "30"),
            *("--", sys.executable, "-c", "raise SystemExit(3)"),
        ],
        check=False,
        env=child_env(),
        capture_output=True,
        timeout=60,
    )
    assert done.returncode == 3, done.stderr
    assert (tmp_path / gpu_lock.LOG_PATH.name).exists()
    assert gpu_lock.LOG_PATH.name == "gpu-lock.log"


def test_gpu_lock_command_keeps_the_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the hidden options, the command line uses the real pattern and lock paths."""
    seen: dict[str, object] = {}

    @contextmanager
    def recording(label: str, **kwargs: object) -> Iterator[None]:
        seen.update(kwargs, label=label)
        yield

    monkeypatch.setattr(gpu_lock, "heavy_job", recording)
    assert gpu_lock.main(["--", sys.executable, "-c", "raise SystemExit(0)"]) == 0
    assert seen["pattern"] == HEAVY_JOB_PATTERN
    assert {"lock_path", "registry", "log_path"}.isdisjoint(seen)


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """A recording heavy_job in each entry point and in conftest.py."""
    import bench_mlx_port
    import bench_speed
    import benchmark
    import conftest
    import cross_check
    import make_reference
    import profile_block

    found: list[str] = []

    @contextmanager
    def recording(label: str, *args: object, **kwargs: object) -> Iterator[None]:
        found.append("enter")
        yield
        found.append("exit")

    for module in (
        bench_speed,
        bench_mlx_port,
        benchmark,
        cross_check,
        make_reference,
        profile_block,
        conftest,
    ):
        monkeypatch.setattr(module, "heavy_job", recording)
    return found


def test_drivers_take_the_lock_before_the_first_worker(
    tmp_path: Path, events: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """T8: main(argv) of each driver enters the GPU lock before the first worker or model load
    and exits after the last; a dry run enters no lock. The conftest hooks hold the lock from
    pytest_sessionstart to pytest_sessionfinish."""
    import bench_mlx_port
    import bench_speed
    import benchmark
    import conftest
    import cross_check
    import make_reference
    import profile_block

    from dinov3_mlx import bench

    def work(*args: object, **kwargs: object) -> Counter:
        events.append("worker")
        return Counter()

    monkeypatch.setattr(bench, "_parameter_count", lambda model_size: 21_596_544)
    monkeypatch.setattr(bench_speed, "run_specs", work)
    monkeypatch.setattr(bench_mlx_port, "run_plan", work)
    monkeypatch.setattr(benchmark, "run_configs", work)
    monkeypatch.setattr(benchmark, "write_deferred", lambda *args: 0)
    monkeypatch.setattr(cross_check, "run", lambda *args: work() and 0)
    monkeypatch.setattr(make_reference, "run_all", work)
    monkeypatch.setattr(make_reference, "run_streamed", work)
    monkeypatch.setattr(make_reference, "write_inputs", work)
    monkeypatch.setattr(profile_block, "run_all", work)
    monkeypatch.setattr(make_reference, "INPUTS_FILE", tmp_path / "inputs.safetensors")
    results = str(tmp_path / "lines.jsonl")
    runs = {
        "bench_speed": lambda: bench_speed.main(["sweep", "--results", results]),
        "bench_mlx_port": lambda: bench_mlx_port.main(["--rounds", "1", "--results", results]),
        "benchmark": lambda: benchmark.main(["--model-size", "ViT-S/16", "--results", results]),
        "cross_check": lambda: cross_check.main(["--model-size", "ViT-S/16"]),
        "make_reference": lambda: make_reference.main(["ViT-S/16"]),
        "profile_block": lambda: profile_block.main(["--results", results]),
    }
    for name, run in runs.items():
        events.clear()
        try:
            run()
        except SystemExit as stop:
            assert not stop.code, name
        assert events[0] == "enter" and events[-1] == "exit", (name, events)
        assert "worker" in events, name
    dry_runs = [
        lambda: bench_speed.main(["sweep", "--dry-run", "--results", results]),
        lambda: bench_mlx_port.main(["--dry-run"]),
        lambda: benchmark.main(["--model-size", "ViT-S/16", "--dry-run", "--results", results]),
    ]
    for run in dry_runs:
        events.clear()
        run()
        assert "enter" not in events

    events.clear()
    session = SimpleNamespace(config=SimpleNamespace(args=["-k", "x"]))
    conftest.pytest_sessionstart(session)
    assert events == ["enter"]
    conftest.pytest_sessionfinish(session, 0)
    assert events == ["enter", "exit"]
