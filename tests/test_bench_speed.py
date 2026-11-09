"""Tests T9, T11 and T21 to T24 of specs/06-speed/plan.html: the driver scripts/bench_speed.py."""

import json
import logging
import os
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import pytest

from dinov3_mlx import bench
from dinov3_mlx.bench import DTYPE_MODE_NAMES

REPO_ROOT = Path(__file__).parents[1]
# The parameter counts of the safetensors headers, with the mask token.
COUNTS = {
    "ViT-S/16": 21_596_544,
    "ViT-S+/16": 28_700_000,
    "ViT-B/16": 85_660_416,
    "ViT-L/16": 303_100_000,
    "ViT-H+/16": 840_592_640,
    "ViT-7B/16": 6_716_035_072,
}


@pytest.fixture
def counts(monkeypatch: pytest.MonkeyPatch) -> None:
    """The needs read no safetensors header."""
    monkeypatch.setattr(bench, "_parameter_count", COUNTS.__getitem__)


def test_matrix_plan_pairs_each_configuration(counts: None) -> None:
    """T21 (Table 11 of the plan)."""
    import bench_speed

    specs = bench_speed.matrix_plan(quantized=True)
    assert len(specs) == 576
    groups: dict[int, list[dict]] = {}
    for spec in specs:
        groups.setdefault(spec["group"], []).append(spec)
    assert len(groups) == 54
    sizes = {bench_speed.spec_config(group[0]).model_size: len(group) for group in groups.values()}
    assert sizes == {
        "ViT-S/16": 10,
        "ViT-S+/16": 10,
        "ViT-B/16": 10,
        "ViT-L/16": 10,
        "ViT-H+/16": 14,
        "ViT-7B/16": 10,
    }
    for group, members in groups.items():
        configs = [bench_speed.spec_config(spec) for spec in members]
        assert all(config.compile for config in configs)
        torch_modes = [
            c.dtype_mode for s, c in zip(members, configs) if s["backend"] == "torch-mps"
        ]
        giant = configs[0].model_size == "ViT-7B/16"
        assert sorted(torch_modes) == sorted(
            ["fp16-cast", "bf16-cast"] if giant else DTYPE_MODE_NAMES
        )
        assert all(
            (c.bits is not None) == (s["backend"] == "mlx" and c.bits is not None)
            for s, c in zip(members, configs)
        )
        assert [s["position"] for s in members] == list(range(1, len(members) + 1))
        base = bench_speed.group_workers(configs[0].model_size, quantized=True)
        first = (members[0]["backend"], configs[0].dtype_mode, configs[0].bits)
        assert first == base[group % len(base)]  # the rotation by the group number
        for spec in members:
            env = bench_speed.worker_env(spec, {"PATH": "/usr/bin", "MLX_ENABLE_TF32": "1"})
            assert env["MLX_ENABLE_TF32"] == "0"
            watermark = env.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO")
            assert watermark == ("0.8" if spec["backend"] == "torch-mps" else None)
    giant_torch = [
        s for s in specs if s["backend"] == "torch-mps" and "7B" in s["config"]["model_size"]
    ]
    assert len(giant_torch) == 18 and {s["config"]["run"] for s in giant_torch} == {"B4"}
    bits = Counter(s["config"]["bits"] for s in specs)
    assert bits == {None: 504, 8: 36, 4: 36}
    order = [bench_speed.model_order(s["config"]["model_size"]) for s in specs]
    assert order == sorted(order)  # ViT-7B/16 last
    assert len(bench_speed.matrix_plan(["ViT-S/16"])) == 90


def test_matrix_plan_follow_ups_gate_and_reruns(counts: None) -> None:
    """T21: the gate J13, the reruns of skipped groups (J15) and the follow-up groups (J16)."""
    import bench_speed

    gate = bench_speed.gate_plan()
    assert [(s["backend"], s["config"]["compile"]) for s in gate] == [
        ("torch-mps", False),
        ("torch-mps", True),
        ("mlx", True),
    ] * 2
    specs = bench_speed.matrix_plan(["ViT-S/16"])
    skipped = dict(
        specs[3]["config"], backend=specs[3]["backend"], status="skipped", phase="matrix"
    )
    rerun = bench_speed.only_skipped(specs, [skipped])
    assert {s["group"] for s in rerun} == {specs[3]["group"]} and len(rerun) == 10
    done = [
        dict(s["config"], backend=s["backend"], status="done", phase="matrix")
        for s in specs
        if s["group"] == 0
    ]
    assert {s["group"] for s in bench_speed.skip_done(specs, done)} == set(range(1, 9))
    line = dict(specs[0]["config"], backend="mlx", status="done", valid=True, phase="matrix")
    line |= {"dtype_mode": "fp32", "frame": {"throughput_ips": 10.0}}
    follow = bench_speed.follow_up_plan([("ViT-S/16", 224, 1, "fp32")], [line])
    assert [(s["backend"], s["config"]["compile"], s["config"]["dtype_mode"]) for s in follow] == [
        ("torch-mps", False, "fp32"),
        ("torch-mps", True, "fp32"),
        ("mlx", True, "fp32"),
    ]
    assert {s["phase"] for s in follow} == {"compile-check"}


def test_pairs_plan_rotates_five_rounds(counts: None) -> None:
    """T22: the paired runs and the change plan."""
    import bench_speed

    specs = bench_speed.pairs_plan(["P1", "P2"], 5, "1")
    assert len(specs) == 100
    for priority in ("P1", "P2"):
        mine = [s for s in specs if s["priority"] == priority]
        assert len(mine) == 50
        assert {s["run_id"] for s in mine} == {"1"}
        for r in range(1, 6):
            in_round = [s for s in mine if s["round"] == r]
            modes = [s["config"]["dtype_mode"] for s in in_round[::2]]
            assert modes == DTYPE_MODE_NAMES[r - 1 :] + DTYPE_MODE_NAMES[: r - 1]
            first = "torch-mps" if r % 2 else "mlx"
            for a, b in zip(in_round[::2], in_round[1::2], strict=True):
                assert a["config"] == b["config"] and a["config"]["compile"]
                assert (a["backend"], b["backend"]) == (
                    first,
                    {"mlx": "torch-mps"}.get(first, "mlx"),
                )

    changes = bench_speed.change_plan(
        ["compile", "rope", "fp16-output", "autocast", "quantized"], ["P1", "P2"], 5
    )
    by_change = Counter(s["change"] for s in changes)
    assert by_change == {"C1": 10, "C2": 10, "C4": 8, "C3": 40, "C5": 30}
    for spec in changes:
        if spec["change"] in ("C1", "C2", "C4"):
            assert spec["kind"] == "change" and len(spec["forms"]) == 2 and spec["rounds"] == 5
        else:
            assert spec["kind"] == "bench" and "form" in spec
    c3 = [s for s in changes if s["change"] == "C3" and s["priority"] == "P2"]
    bf16 = [s["form"] for s in c3 if s["form"].startswith("bf16")]
    assert bf16[:4] == ["bf16-cast", "bf16-autocast", "bf16-autocast", "bf16-cast"]
    c5 = [s["form"] for s in changes if s["change"] == "C5"][:6]
    assert c5 == ["unquantized", "8-bit", "4-bit", "8-bit", "4-bit", "unquantized"]
    assert {s["priority"] for s in changes if s["change"] == "C5"} == {"P2"}
    assert len(bench_speed.sweep_plan()) == 22
    assert Counter(s["kind"] for s in bench_speed.learn_plan()) == {"bench": 20, "token-range": 2}
    assert len(bench_speed.profile_plan(["P1"])) == 2


DRY_RUN = """
import sys
import bench_speed
from dinov3_mlx import bench

bench._parameter_count = {counts}.__getitem__
for argv in {commands}:
    bench_speed.main(argv)
print(sorted(m for m in ("torch", "mlx.core") if m in sys.modules))
"""


def test_bench_speed_dry_run_imports_no_backend() -> None:
    """T23: the dry run of each subcommand prints its workers and loads no backend."""
    commands = [
        [
            "change",
            "--change",
            "compile",
            "rope",
            "autocast",
            "quantized",
            "fp16-output",
            "--dry-run",
        ],
        ["matrix", "--quantized", "--dry-run"],
        ["pairs", "--run", "1", "--dry-run"],
        ["sweep", "--dry-run"],
        ["learn", "--dry-run"],
        ["profile", "--dry-run"],
        ["preprocess", "--dry-run"],
    ]
    code = DRY_RUN.format(counts=repr(COUNTS), commands=repr(commands))
    env = {k: v for k, v in os.environ.items() if k != "MLX_DISABLE_COMPILE"}
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT / "scripts",
        env=env,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines[-1] == "[]"
    totals = [int(line.split()[0]) for line in lines if line.endswith(" workers")]
    assert totals == [98, 576, 100, 22, 22, 2, 1]
    code = "import sys, dinov3_mlx.mlx_bench; print('mlx.core' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.stdout.strip() == "False", result.stderr


def test_bench_speed_refuses_mlx_disable_compile(tmp_path: Path) -> None:
    """T24 (R42)."""
    env = os.environ | {"MLX_DISABLE_COMPILE": "1"}
    result = subprocess.run(
        [sys.executable, "bench_speed.py", "sweep", "--results", str(tmp_path / "x.jsonl")],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT / "scripts",
        env=env,
        timeout=60,
    )
    assert result.returncode == 1
    assert "MLX_DISABLE_COMPILE" in result.stderr
    assert not (tmp_path / "x.jsonl").exists()
    assert not (tmp_path / "x-batches").exists()


def test_driver_records_other_heavy_jobs(
    tmp_path: Path, counts: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T9: a process that matches the test pattern runs during worker 1, outside the lock. The
    line of worker 1 has its command line in other_heavy_jobs_seen; the line of worker 2 has an
    empty list."""
    import bench_speed

    token = f"heavyjobtest{uuid.uuid4().hex[:10]}"
    monkeypatch.setattr(logging.getLogger("bench_speed"), "handlers", [])
    other: list[subprocess.Popen] = []

    def runner(spec: dict, bench_dir: Path) -> dict:
        if not other:  # worker 1: the other job starts during it
            command = [sys.executable, "-c", "import time; time.sleep(30)", token]
            other.append(subprocess.Popen(command, start_new_session=True))
            time.sleep(0.3)
        return {"status": "done", "reason": ""}

    read_state = bench_speed.read_machine_state
    calls: list[int] = []

    def read_machine_state() -> object:
        calls.append(1)
        if len(calls) == 2:  # before worker 2: the other job has ended
            other[0].kill()
            other[0].wait()
        return read_state()

    monkeypatch.setattr(bench_speed, "read_machine_state", read_machine_state)
    specs = bench_speed.sweep_plan()[:2]
    results = tmp_path / "lines.jsonl"
    try:
        bench_speed.run_specs(specs, results, tmp_path, runner, pattern=token, registry=tmp_path)
    finally:
        for process in other:
            process.kill()
    lines = [json.loads(text) for text in results.read_text().splitlines()]
    assert len(lines) == 2
    assert any(token in command for command in lines[0]["other_heavy_jobs_seen"])
    assert lines[1]["other_heavy_jobs_seen"] == []
    assert lines[0]["backend"] == "mlx" and lines[0]["phase"] == "sweep"


# A worker of bench_speed.py that runs the real load up to its memory check and stops before the
# weights.
WORKER_UNTIL_THE_CHECK = """
from typing import NoReturn

import bench_speed
from dinov3_mlx import mlx_weights, reference
from dinov3_mlx.hub import WeightsNotFoundError


def stop(model_size: str) -> NoReturn:
    raise WeightsNotFoundError("stop after the memory check")


mlx_weights.required_mlx_load_bytes = lambda *args, **kwargs: 1
mlx_weights.weights_dir = stop
reference.required_load_bytes = lambda *args, **kwargs: 1
reference.weights_dir = stop
bench_speed.main()
"""


@pytest.mark.parametrize("backend", ["torch-mps", "mlx"])
def test_bench_speed_worker_memory_check_reaches_the_driver_log(
    backend: str,
    tmp_path: Path,
    counts: None,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """T11: the memory check of each worker entry of bench_speed.py reaches the driver log
    through run_worker_subprocess."""
    import bench_speed

    spec = next(s for s in bench_speed.pairs_plan(["P1"], 1, "1") if s["backend"] == backend)
    payload = json.dumps(spec | {"bench_dir": str(tmp_path)})
    worker = subprocess.run(
        [sys.executable, "-c", WORKER_UNTIL_THE_CHECK, "--worker", payload],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT / "scripts",
        env=bench_speed.worker_env(spec, dict(os.environ)),
        timeout=120,
    )
    label = "ViT-S/16 mlx fp" if backend == "mlx" else "ViT-S/16 torch-mps fp"
    assert "memory check: " + label in worker.stderr, worker.stderr
    monkeypatch.setattr(subprocess, "run", lambda command, **kwargs: worker)
    monkeypatch.setattr(logging.getLogger("dinov3_mlx.memory"), "handlers", [])
    with caplog.at_level(logging.INFO):
        result = bench_speed.run_worker(spec, tmp_path)
    assert result["status"] == "error"
    records = [r for r in caplog.records if r.name == "dinov3_mlx.memory"]
    assert len(records) == 1 and label in records[0].getMessage()
