"""Test T33 of specs/05-all-sizes/plan.html: the dry run of scripts/cross_check.py, and the
run-rule guard, the comparison and the worker command line of the script."""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fakes import VIT7B16_PARAMETER_COUNT, fail_if_called, fail_load

REPO_ROOT = Path(__file__).parents[1]

# A driver run of scripts/cross_check.py with its folder in a temporary place. The memory read and
# the header read fail if the driver calls them before the run-rule guard. The test puts tests/ on
# PYTHONPATH for the import of fakes.
REFUSED_DRIVER = """
import sys
from pathlib import Path

import cross_check
from fakes import fail_if_called

cross_check.CROSS_DIR = Path(sys.argv[1])
cross_check.available_memory_bytes = fail_if_called("available_memory_bytes")
cross_check.parameter_count = fail_if_called("parameter_count")
cross_check.main(sys.argv[2:])
"""


def test_cross_check_dry_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """T33: two workers in order (torch first), 20.1 GB each, with their environments. The dry
    run loads no model. The test gives the parameter count, so it reads no weights file."""
    import cross_check

    monkeypatch.setattr(cross_check, "parameter_count", lambda model_size: VIT7B16_PARAMETER_COUNT)
    monkeypatch.setattr("dinov3_mlx.reference.load_torch_backbone", fail_load)
    monkeypatch.setattr("dinov3_mlx.mlx_weights.load_mlx_backbone", fail_load)
    cross_check.main(
        [
            "--model-size",
            "ViT-7B/16",
            "--dtype-mode",
            "bf16-cast",
            "--torch-device",
            "mps",
            "--dry-run",
        ]
    )
    lines = capsys.readouterr().out.strip().splitlines()
    workers = [line for line in lines if line.startswith("worker")]
    assert len(workers) == 2
    assert workers[0].startswith("worker 1: torch-mps ViT-7B/16 bf16-cast")
    assert workers[1].startswith("worker 2: mlx ViT-7B/16 bf16-cast")
    for line in workers:
        assert "need 20.1 GB" in line
    assert "PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8" in workers[0]
    assert "PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6" in workers[0]
    assert "MLX_ENABLE_TF32" not in workers[0]
    assert "MLX_ENABLE_TF32=0" in workers[1]
    assert "PYTORCH_MPS" not in workers[1]
    assert "configurations: 12" in lines


def test_cross_check_refuses_7b_autocast_on_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    """The torch worker of ViT-7B/16 in an autocast mode has fp32 weights: the run rule stops the
    dry run before any worker, before the memory read and before parameter_count (F10 of
    specs/05-all-sizes/verify.html)."""
    import cross_check

    monkeypatch.setattr(cross_check, "parameter_count", fail_if_called("parameter_count"))
    monkeypatch.setattr(
        cross_check, "available_memory_bytes", fail_if_called("available_memory_bytes")
    )
    with pytest.raises(SystemExit, match="CLAUDE.md run rules"):
        cross_check.main(
            ["--model-size", "ViT-7B/16", "--dtype-mode", "bf16-autocast", "--dry-run"]
        )


def test_cross_check_refusal_reaches_the_driver_log(tmp_path: Path) -> None:
    """A refused run writes the refusal to driver.log, before the memory read and before
    parameter_count (F10). The driver runs in a subprocess, so its log handlers stay out of the
    test session."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            REFUSED_DRIVER,
            str(tmp_path),
            "--model-size",
            "ViT-7B/16",
            "--dtype-mode",
            "bf16-autocast",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT / "scripts",
        env=os.environ | {"PYTHONPATH": str(REPO_ROOT / "tests")},
        timeout=120,
    )
    assert result.returncode == 1, result.stderr
    assert "AssertionError" not in result.stderr
    assert result.stderr.rstrip().endswith("(CLAUDE.md run rules)")
    log = (tmp_path / "driver.log").read_text()
    assert re.search(r"cross_check refused: ViT-7B/16 torch-mps bf16-autocast: .+ run rules", log)


def test_cross_check_compare_tokens() -> None:
    """The comparison of one configuration: a parity for finite tokens, the status non-finite
    with the finite flag of each backend otherwise."""
    import cross_check

    rng = np.random.default_rng(0)
    torch_tokens = rng.standard_normal((5 + 196, 8)).astype(np.float32)
    status, parity, values = cross_check.compare_tokens(torch_tokens, torch_tokens.copy())
    assert status == "measured"
    assert parity["frame"] == 1.0
    assert values == {"torch_finite": True, "mlx_finite": True, "max_abs": 0.0}
    broken = torch_tokens.copy()
    broken[3, 2] = np.inf
    status, parity, values = cross_check.compare_tokens(torch_tokens, broken)
    assert (status, parity) == ("non-finite", None)
    assert values == {"torch_finite": True, "mlx_finite": False}


def test_cross_check_worker_command_parses(monkeypatch: pytest.MonkeyPatch) -> None:
    """The driver starts each worker with --worker and --spec only. The parser accepts this
    command line without --model-size and gives the spec to worker_main."""
    import cross_check

    commands: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout='{"status": "done"}\n', stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    spec = {"model_size": "ViT-7B/16", "dtype_mode": "bf16-cast", "torch_device": "mps"}
    assert cross_check.run_worker("torch-mps", spec) == {"status": "done"}
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        cross_check, "worker_main", lambda backend, spec_json: calls.append((backend, spec_json))
    )
    cross_check.main(commands[0][2:])
    assert calls == [("torch", json.dumps(spec))]
