"""Tests T21 to T25 of specs/04-mlx-port/plan.html: TF32, imports, the moved readers, the TF32
probe and the dry run of the Step 4 benchmark. Also the fields of its mlx worker."""

import dataclasses
import json
import logging
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import pytest
from small_model import small_hf_model, small_mlx_model

from dinov3_apple import mlx_weights, preprocess, reference, stored
from dinov3_apple.bench import BenchConfig, TimeBudget
from dinov3_apple.mlx_model import MlxBackbone, tf32_enabled

pytestmark = pytest.mark.mlx

REPO_ROOT = Path(__file__).parents[1]

MOVED_FROM_REFERENCE = [
    "Configuration",
    "configurations",
    "read_inputs",
    "read_reference",
    "reference_path",
    "model_slug",
    "REPO_ROOT",
    "IMAGES_DIR",
    "REFERENCE_DIR",
    "INPUTS_FILE",
    "SQUARE_SIZES",
    "HIDDEN_STATES_CONFIG",
]
CONFIG_IDS = [
    "iss-kibo-lab/224x224",
    "iss-kibo-lab/448x448",
    "iss-kibo-lab/512x512",
    "iss-kibo-lab/224x400",
    "forest/224x224",
    "forest/448x448",
    "forest/512x512",
    "forest/224x304",
    "iss-conference/224x224",
    "iss-conference/448x448",
    "iss-conference/512x512",
    "iss-conference/304x224",
]

# One fp32 matmul on the GPU against NumPy float64 (D7 of explore.html).
TF32_PROBE = """
import json, os
import numpy as np
import mlx.core as mx

mx.set_memory_limit(10**9)
rng = np.random.default_rng(0)
a = rng.standard_normal((512, 384)).astype(np.float32)
b = rng.standard_normal((384, 1536)).astype(np.float32)
out = mx.matmul(mx.array(a), mx.array(b), stream=mx.gpu)
mx.eval(out)
expected = a.astype(np.float64) @ b.astype(np.float64)
error = np.abs(np.array(out, dtype=np.float64) - expected).max() / np.abs(expected).max()
print(json.dumps({"MLX_ENABLE_TF32": os.environ["MLX_ENABLE_TF32"], "relative": float(error)}))
"""


def test_conftest_turns_tf32_off() -> None:
    assert os.environ["MLX_ENABLE_TF32"] == "0"
    assert tf32_enabled() is False


def test_tf32_enabled_reads_the_environment_as_mlx_does() -> None:
    assert tf32_enabled({}) is True
    assert tf32_enabled({"MLX_ENABLE_TF32": "1"}) is True
    assert tf32_enabled({"MLX_ENABLE_TF32": "0"}) is False
    assert tf32_enabled({"MLX_ENABLE_TF32": "no"}) is False


def test_mlx_modules_do_not_import_torch() -> None:
    code = (
        "import sys\n"
        "import dinov3_apple.stored, dinov3_apple.mlx_model, dinov3_apple.mlx_weights\n"
        "print(sorted(name for name in ('torch', 'transformers') if name in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=os.environ | {"MLX_ENABLE_TF32": "0"},
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"


def test_stored_readers_are_reexported() -> None:
    for name in MOVED_FROM_REFERENCE:
        assert getattr(reference, name) is getattr(stored, name), name
        assert name in reference.__all__, name
    assert "TEST_IMAGES" in reference.__all__
    assert preprocess.own_aspect_size is stored.own_aspect_size
    assert "own_aspect_size" in preprocess.__all__
    assert [config.id for config in stored.configurations()] == CONFIG_IDS
    if stored.INPUTS_FILE.exists():
        assert set(stored.read_inputs()) == set(CONFIG_IDS)


def test_mlx_tf32_probe(report: Callable[..., None]) -> None:
    errors = {}
    for value in ("0", "1"):
        result = subprocess.run(
            [sys.executable, "-c", TF32_PROBE],
            check=False,
            capture_output=True,
            text=True,
            env=os.environ | {"MLX_ENABLE_TF32": value},
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        line = json.loads(result.stdout.strip().splitlines()[-1])
        assert line["MLX_ENABLE_TF32"] == value
        errors[value] = line["relative"]
    report("test_mlx_tf32_probe", tf32_off=errors["0"], tf32_on=errors["1"])
    # TF32 off: 7.8e-7 with mlx 0.32.4. If MLX ignored MLX_ENABLE_TF32=0, the error would be the
    # TF32 value, about 7e-4. The TF32-on value is a report only (F7 of verify.html).
    assert 0 <= errors["0"] <= 1e-5


def _other(backend: str) -> str:
    """Return the other worker backend of the paired benchmark."""
    return "mlx" if backend == "torch-mps" else "torch-mps"


def test_bench_mlx_port_dry_run() -> None:
    from bench_mlx_port import worker_env, worker_plan

    plan = worker_plan(3)
    assert len(plan) == 12
    assert [entry[0] for entry in plan] == [1] * 4 + [2] * 4 + [3] * 4
    for round_ in (1, 2, 3):
        entries = [entry for entry in plan if entry[0] == round_]
        assert [entry[1] for entry in entries] == [8, 8, 1, 1]
        first = entries[0][2]
        assert [entry[2] for entry in entries] == [first, _other(first), first, _other(first)]
    assert [entry[2] for entry in plan[::4]] == ["torch-mps", "mlx", "torch-mps"]
    for _, _, backend, tf32 in plan:
        env = worker_env(backend, tf32, {"PATH": "/usr/bin", "MLX_ENABLE_TF32": "1"})
        if backend == "mlx":
            assert tf32 is False
            assert env["MLX_ENABLE_TF32"] == "0"
            assert "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in env
        else:
            assert tf32 is None
            assert env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.8"
            assert env["PYTORCH_MPS_LOW_WATERMARK_RATIO"] == "0.6"
    # The dry run prints the plan and the memory needs. It loads no model and imports no backend.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, bench_mlx_port as b; b.dry_run(b.worker_plan(3)); "
                "print(sorted(m for m in ('torch', 'mlx.core') if m in sys.modules))"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT / "scripts",
    )
    if "WeightsNotFoundError" in result.stderr:
        pytest.skip("the ViT-S/16 weights are not on this Mac")
    assert result.returncode == 0, result.stderr
    lines = result.stdout.strip().splitlines()
    assert lines[-1] == "[]"
    assert sum("ViT-S/16" in line for line in lines) == 12


def test_mlx_worker_records_its_memory_limit_and_model_slug(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker records the value that its load gave to mx.set_memory_limit (Table 10).

    The load gives the small model of tests/small_model.py, so the test needs no gated weights.
    ViT-S+/16 has the width 384 of the small model and a model slug other than vits16.
    """
    import bench_mlx_port

    if not stored.INPUTS_FILE.exists():
        pytest.skip(f"{stored.INPUTS_FILE.name} is missing. Run scripts/make_reference.py.")
    limit = 3 * 10**9  # not the need of a load, not the limit of the test session
    model = small_mlx_model(small_hf_model(seed=0))

    def load(model_size: str, dtype_mode: str = "fp32", tf32: bool = False) -> MlxBackbone:
        mx.set_memory_limit(limit)
        return model

    monkeypatch.setattr(mlx_weights, "load_mlx_backbone", load)
    short = TimeBudget(min_batches=2, min_seconds=0.0)
    config = BenchConfig("T", "ViT-S+/16", "fp32", 224, 1)
    session_limit = mx.get_memory_limit()
    try:
        worker = bench_mlx_port.run_mlx_worker(config, short, short, tmp_path)
    finally:
        mx.set_memory_limit(session_limit)
    assert worker["status"] == "done"
    assert worker["mlx_memory_limit_bytes"] == limit
    assert worker["tf32"] is False
    assert worker["timed_batches"] >= 2
    assert [path.name for path in tmp_path.glob("*.json")] == ["vits16plus_fp32_224_b1_r0.json"]


def test_bench_mlx_port_files_follow_the_results_path(tmp_path: Path) -> None:
    """With --results, the per-batch files and driver.log go next to the results file (F5 of
    specs/04-mlx-port/verify.html). The default results file keeps the folder of the build."""
    import bench_mlx_port

    assert bench_mlx_port.bench_dir_of(bench_mlx_port.RESULTS) == bench_mlx_port.BENCH_DIR
    results = tmp_path / "bench.jsonl"
    assert bench_mlx_port.bench_dir_of(results) == tmp_path / "bench-batches"
    default_log = bench_mlx_port.BENCH_DIR / "driver.log"
    before = default_log.read_bytes() if default_log.exists() else None
    # 0 rounds: the driver opens its log and starts no worker, so no model load.
    result = subprocess.run(
        [sys.executable, "bench_mlx_port.py", "--rounds", "0", "--results", str(results)],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT / "scripts",
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "run of 0 workers" in (tmp_path / "bench-batches" / "driver.log").read_text()
    assert (default_log.read_bytes() if default_log.exists() else None) == before


# A worker that runs the real load_mlx_backbone up to its memory check and stops before the
# weights: the need is 1 byte, and weights_dir raises.
WORKER_UNTIL_THE_CHECK = """
from typing import NoReturn

import bench_mlx_port
from dinov3_apple import mlx_weights
from dinov3_apple.hub import WeightsNotFoundError


def stop(model_size: str) -> NoReturn:
    raise WeightsNotFoundError("stop after the memory check")


mlx_weights.required_mlx_load_bytes = lambda *args: 1
mlx_weights.weights_dir = stop
bench_mlx_port.main()
"""


def test_mlx_worker_memory_check_reaches_the_driver_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The memory check of the load in an mlx worker reaches the driver log, with the time of the
    worker (AC11; F6 of specs/04-mlx-port/verify.html). Also after a worker error."""
    import bench_mlx_port

    config = bench_mlx_port.bench_config(1, 1)
    spec = json.dumps(dataclasses.asdict(config) | {"backend": "mlx", "bench_dir": str(tmp_path)})
    start = time.time()
    worker = subprocess.run(
        [sys.executable, "-c", WORKER_UNTIL_THE_CHECK, "--worker", spec],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT / "scripts",
        env=bench_mlx_port.worker_env("mlx", False, os.environ),
        timeout=120,
    )
    end = time.time()
    assert "memory check: ViT-S/16 mlx fp32" in worker.stderr, worker.stderr

    # The driver side: run_worker_subprocess gets the output of this worker.
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        commands.append(command)
        return worker

    monkeypatch.setattr(subprocess, "run", run)
    # The record must not reach memory-checks.log of the test session.
    monkeypatch.setattr(logging.getLogger("dinov3_apple.memory"), "handlers", [])
    with caplog.at_level(logging.INFO):
        result = bench_mlx_port.run_worker_subprocess(config, "mlx", False, tmp_path)
    assert json.loads(commands[0][-1])["bench_dir"] == str(tmp_path)
    assert result["status"] == "error"
    assert "stop after the memory check" in result["reason"]
    records = [record for record in caplog.records if record.name == "dinov3_apple.memory"]
    assert len(records) == 1
    assert re.fullmatch(
        r"memory check: ViT-S/16 mlx fp32: needs 0\.0 GB, \d+\.\d GB available, ok",
        records[0].getMessage(),
    )
    assert start <= records[0].created <= end
    line = logging.Formatter(bench_mlx_port.LOG_FORMAT).format(records[0])
    assert re.fullmatch(
        r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} dinov3_apple\.memory memory check: .+ ok", line
    )
