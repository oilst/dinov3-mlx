"""Tests T21 to T25 of specs/04-mlx-port/plan.html: TF32, imports, the moved readers, the TF32
probe and the dry run of the Step 4 benchmark. Also the fields of its mlx worker. Tests T28 to T30
of specs/05-all-sizes/plan.html: the model sizes under test, the plan of the benchmark at P2 and
the host copy of bf16 output."""

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
import numpy as np
import pytest
from fakes import fail_load
from small_model import small_hf_model, small_mlx_model, small_pixels

from dinov3_apple import mlx_weights, preprocess, reference, stored
from dinov3_apple.bench import DTYPE_MODE_NAMES, BenchConfig, TimeBudget
from dinov3_apple.hub import MODEL_IDS
from dinov3_apple.mlx_model import MlxBackbone, run_mlx, tf32_enabled
from dinov3_apple.outputs import NUM_PREFIX_TOKENS

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
    """T54: no torch import in the mlx modules and in the Step 6 modules."""
    code = (
        "import sys\n"
        "import dinov3_apple.stored, dinov3_apple.mlx_model, dinov3_apple.mlx_weights\n"
        "import dinov3_apple.memory, dinov3_apple.mlx_preprocess, dinov3_apple.mlx_bench\n"
        "import dinov3_apple.speedup, dinov3_apple.gpu_lock\n"
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


def _fake_load(monkeypatch: pytest.MonkeyPatch, limit: int, name: str = "vits16") -> MlxBackbone:
    """Replace load_mlx_backbone with a load of a small model that sets the MLX limit to limit."""
    model = small_mlx_model(small_hf_model(seed=0, name=name))

    def load(model_size: str, dtype_mode: str = "fp32", tf32: bool = False, **kwargs: object):
        mx.set_memory_limit(limit)
        return model

    monkeypatch.setattr(mlx_weights, "load_mlx_backbone", load)
    return model


def test_mlx_worker_raises_the_limit_to_the_bench_need(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T29 (D8): the worker loads with the load need as the MLX limit, then raises the limit to
    the benchmark need of mlx. The batch file name has the model slug, bits, the compile flag and
    the run id. The small model has the width 384 of ViT-S+/16 and needs no gated weights."""
    from dinov3_apple import bench, mlx_bench

    if not stored.INPUTS_FILE.exists():
        pytest.skip(f"{stored.INPUTS_FILE.name} is missing. Run scripts/make_reference.py.")
    limit = 3 * 10**9  # not the need of a load, not the limit of the test session
    _fake_load(monkeypatch, limit)
    monkeypatch.setattr(bench, "_parameter_count", lambda model_size: 21_596_544)
    short = TimeBudget(min_batches=2, min_seconds=0.0)
    session_limit = mx.get_memory_limit()
    names = []
    try:
        for bits, run_id in ((None, None), (8, "1"), (4, "1"), (4, "2")):
            config = BenchConfig("S6", "ViT-S+/16", "fp32", 224, 1, compile=True, bits=bits)
            worker = mlx_bench.run_mlx_worker(config, short, short, tmp_path, run_id=run_id)
            names.append(worker["batch_file"])
            assert worker["mlx_load_limit_bytes"] == limit
            assert worker["mlx_memory_limit_bytes"] == bench.required_bench_bytes(
                config, backend="mlx"
            )
    finally:
        mx.set_memory_limit(session_limit)
    assert names == [
        "vits16plus_fp32_224_b1_compile_r0.json",
        "vits16plus_fp32_224_b1_compile_8bit_r0_run1.json",
        "vits16plus_fp32_224_b1_compile_4bit_r0_run1.json",
        "vits16plus_fp32_224_b1_compile_4bit_r0_run2.json",
    ]
    assert sorted(path.name for path in tmp_path.glob("*.json")) == sorted(names)
    assert worker["load_report_peak_bytes"] is None
    assert worker["tf32"] is False


def test_mlx_worker_records_versions_and_compile_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T30 (D19, R38, R42): the compile trace, the architecture, the versions, the value of
    MLX_DISABLE_COMPILE, the MLX cache and the MLX peak of the timed batches."""
    from dinov3_apple import bench, mlx_bench

    if not stored.INPUTS_FILE.exists():
        pytest.skip(f"{stored.INPUTS_FILE.name} is missing. Run scripts/make_reference.py.")
    _fake_load(monkeypatch, 3 * 10**9, "vits16plus")
    monkeypatch.setattr(bench, "_parameter_count", lambda model_size: 21_596_544)
    monkeypatch.delenv("MLX_DISABLE_COMPILE", raising=False)
    short = TimeBudget(min_batches=2, min_seconds=0.0)
    config = BenchConfig("S6", "ViT-S+/16", "bf16-cast", 224, 1, compile=True)
    session_limit = mx.get_memory_limit()
    try:
        worker = mlx_bench.run_mlx_worker(config, short, short, tmp_path)
        monkeypatch.setenv("MLX_DISABLE_COMPILE", "1")
        disabled = mlx_bench.run_mlx_worker(config, short, short, tmp_path)
    finally:
        mx.set_memory_limit(session_limit)
    assert 0 < worker["compile_trace_s"] < worker["first_call_s"]
    assert worker["mlx_architecture"] == mx.device_info()["architecture"]
    assert worker["versions"]["mlx-metal"] == "0.32.4"
    assert worker["mlx_disable_compile"] is None
    assert disabled["mlx_disable_compile"] == "1"
    assert worker["mlx_cache_bytes"] >= 0
    assert worker["mlx_timed_peak_bytes"] > 0
    assert worker["output_transfer"] == "fp16-patches"
    assert set(bench.MLX_FIELDS) <= set(worker)
    assert "load_peak_bytes" not in worker


def test_bench_mlx_port_files_follow_the_results_path(tmp_path: Path) -> None:
    """With --results, the per-batch files and driver.log go next to the results file (F5 of
    specs/04-mlx-port/verify.html). The default results file keeps the folder of the build."""
    import bench_mlx_port

    assert bench_mlx_port.bench_dir_of(bench_mlx_port.RESULTS) == bench_mlx_port.BENCH_DIR
    # The default results file of each priority has its own folder (Table 2 of the Step 5 plan).
    assert bench_mlx_port.RESULTS.name == "bench-mlx-port.jsonl"
    assert bench_mlx_port.BENCH_DIR.name == "mlx-port"
    assert bench_mlx_port.RESULTS_P2.name == "bench-all-sizes.jsonl"
    assert bench_mlx_port.BENCH_DIR_P2.name == "all-sizes"
    assert bench_mlx_port.bench_dir_of(bench_mlx_port.RESULTS_P2) == bench_mlx_port.BENCH_DIR_P2
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

    # The driver side: run_worker gets the output of this worker.
    commands: list[list[str]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        commands.append(command)
        return worker

    monkeypatch.setattr(subprocess, "run", run)
    # The record must not reach memory-checks.log of the test session.
    monkeypatch.setattr(logging.getLogger("dinov3_apple.memory"), "handlers", [])
    with caplog.at_level(logging.INFO):
        result = bench_mlx_port.run_worker(config, "mlx", False, tmp_path)
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


def test_mlx_model_sizes_under_test() -> None:
    """T28: mlx runs ViT-S/16, ViT-S+/16 and ViT-B/16 by default, all six with DINOV3_SLOW=1."""
    code = "import json, under_test; print(json.dumps(under_test.MLX_MODEL_SIZES_UNDER_TEST))"
    sizes = {}
    for slow in ("0", "1"):
        env = {k: v for k, v in os.environ.items() if k != "DINOV3_SLOW"}
        if slow == "1":
            env["DINOV3_SLOW"] = "1"
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT / "tests",
            env=env,
        )
        assert result.returncode == 0, result.stderr
        sizes[slow] = json.loads(result.stdout.strip().splitlines()[-1])
    assert sizes["0"] == ["ViT-S/16", "ViT-S+/16", "ViT-B/16"]
    assert sizes["1"] == list(MODEL_IDS)


def test_bench_p2_worker_plan(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """T29: 3 rounds x 5 dtype modes x 2 backends at P2; the order of the backends alternates from
    round to round. The P1 plan keeps its 12 workers. The dry run loads no model."""
    import bench_mlx_port

    assert len(bench_mlx_port.worker_plan(3)) == 12
    plan = bench_mlx_port.worker_plan(3, "P2")
    assert len(plan) == 30
    assert [entry[0] for entry in plan] == [1] * 10 + [2] * 10 + [3] * 10
    for round_ in (1, 2, 3):
        entries = [entry for entry in plan if entry[0] == round_]
        assert [entry[1] for entry in entries] == [m for m in DTYPE_MODE_NAMES for _ in range(2)]
        first = "torch-mps" if round_ % 2 else "mlx"
        assert [entry[2] for entry in entries] == [first, _other(first)] * 5
    for _, _, backend, tf32 in plan:
        env = bench_mlx_port.worker_env(backend, tf32, {"PATH": "/usr/bin", "MLX_ENABLE_TF32": "1"})
        if backend == "mlx":
            assert tf32 is False
            assert env["MLX_ENABLE_TF32"] == "0"
            assert "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in env
        else:
            assert tf32 is None
            assert env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.8"
            assert env["PYTORCH_MPS_LOW_WATERMARK_RATIO"] == "0.6"
    config = bench_mlx_port.bench_config(2, "bf16-autocast", "P2")
    assert (config.run, config.model_size, config.dtype_mode) == (
        "step5",
        "ViT-H+/16",
        "bf16-autocast",
    )
    assert (config.side, config.batch_size, config.repeat) == (512, 8, 1)
    # The needs read the headers of ViT-H+/16. The test gives the count, so it reads no file.
    monkeypatch.setattr("dinov3_apple.bench._parameter_count", lambda model_size: 840_592_640)
    monkeypatch.setattr("dinov3_apple.mlx_weights.load_mlx_backbone", fail_load)
    bench_mlx_port.dry_run(plan, "P2")
    lines = capsys.readouterr().out.strip().splitlines()
    workers = [line for line in lines if "ViT-H+/16" in line]
    assert len(workers) == 30
    # The needs of each backend (D8 of specs/06-speed/plan.html).
    mlx_needs = {"fp32": 8.88, "cast": 6.44, "autocast": 6.86}
    for line in workers:
        kind = "fp32" if " fp32 " in line else ("cast" if "-cast" in line else "autocast")
        if line.split()[2] == "mlx":
            assert f"need {mlx_needs[kind]:4.2f} GB" in line, line
        else:
            assert f"need {7.12 if kind == 'cast' else 10.23:4.2f} GB" in line, line


def test_time_batch_mlx_gives_fp16_patches() -> None:
    """T56 (D13): in bf16-cast the frame embeddings are fp32 and the patch embeddings fp16; the
    values equal run_mlx cast to fp16. In fp32 both stay fp32."""
    from dinov3_apple import mlx_bench
    from dinov3_apple.mlx_model import compile_backbone

    model = small_mlx_model(small_hf_model(seed=0), "bf16-cast")
    batch = small_pixels((2, 3, 224, 224))
    times, frame, patches = mlx_bench.time_batch_mlx(compile_backbone(model), batch, "bf16-cast")
    assert len(times) == 3
    assert frame.dtype == np.float32
    assert patches.dtype == np.float16
    assert mlx_bench.output_transfer(frame, patches) == "fp16-patches"
    tokens = run_mlx(model, batch, "bf16-cast")
    assert np.array_equal(frame, tokens[:, 0])
    assert np.array_equal(patches, tokens[:, NUM_PREFIX_TOKENS:].astype(np.float16))
    fp32 = small_mlx_model(small_hf_model(seed=0), "fp32")
    _, frame32, patches32 = mlx_bench.time_batch_mlx(fp32, batch, "fp32")
    assert frame32.dtype == patches32.dtype == np.float32
    assert mlx_bench.output_transfer(frame32, patches32) == "native"
    # The lines of Step 5 hold bf16 bits as uint16: host_to_fp32 still reads them.
    bits = np.array([0x3F80], np.uint16)
    assert mlx_bench.host_to_fp32(bits)[0] == 1.0
    assert mlx_bench.output_transfer(bits, bits) == "view-uint16"


@pytest.mark.parametrize(
    ("change", "forms"), [("C1", ["E", "A"]), ("C2", ["A", "R"]), ("C4", ["R-fp32", "R-fp16"])]
)
def test_change_worker_forms(
    change: str, forms: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """R56 and T22: the forms of one change run in one process and give the same tokens; each
    round gives one ratio. A small model, so the test needs no gated weights."""
    from dinov3_apple import mlx_bench

    if not stored.INPUTS_FILE.exists():
        pytest.skip(f"{stored.INPUTS_FILE.name} is missing. Run scripts/make_reference.py.")
    _fake_load(monkeypatch, mx.get_memory_limit(), "vits16plus")
    config = BenchConfig("S6", "ViT-S+/16", "bf16-cast", 224, 2, compile=True)
    short = TimeBudget(min_batches=2, min_seconds=0.0)
    result = mlx_bench.run_change_worker(config, change, forms, 2, short, short)
    assert result["frame_equal"]
    assert result["patches_equal"]
    assert len(result["ratios"]["frame"]) == 2
    assert all(len(result["throughput"][form]["patch"]) == 2 for form in forms)


def test_profile_worker_parts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Table 12 of specs/06-speed/plan.html: the profile worker gives the host share, the 11
    parts of block 1, the kernel counts and the throughput at batch sizes 8 and 32."""
    from dinov3_apple import mlx_bench
    from dinov3_apple.flops import PART_NAMES

    if not stored.INPUTS_FILE.exists():
        pytest.skip(f"{stored.INPUTS_FILE.name} is missing. Run scripts/make_reference.py.")
    _fake_load(monkeypatch, mx.get_memory_limit(), "vits16plus")
    config = BenchConfig("S6", "ViT-S+/16", "fp16-cast", 224, 8, compile=True)
    result = mlx_bench.run_profile_worker(config, tmp_path)
    assert list(result["parts_s"]) == PART_NAMES
    assert 0 < result["host"]["host_share"]
    assert result["kernels"]["eager"] > result["kernels"]["compiled"] > 0
    assert set(result["batch_sizes"]) == {"8", "32"}
