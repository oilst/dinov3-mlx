"""Tests T21 to T25 of specs/04-mlx-port/plan.html: TF32, imports, the moved readers, the TF32
probe and the dry run of the Step 4 benchmark. Also the fields of its mlx worker."""

import json
import os
import subprocess
import sys
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
    assert all(0 <= error < 1 for error in errors.values())


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
