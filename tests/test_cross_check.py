"""Test T33 of specs/05-all-sizes/plan.html: the dry run of scripts/cross_check.py, and the
run-rule guard, the comparison and the worker command line of the script."""

import json
import subprocess

import numpy as np
import pytest

VIT7B16_PARAMETER_COUNT = 6_716_035_072


def _fail_load(*args: object, **kwargs: object) -> None:
    raise AssertionError("the dry run loaded a model")


def test_cross_check_dry_run(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """T33: two workers in order (torch first), 20.1 GB each, with their environments. The dry
    run loads no model. The test gives the parameter count, so it reads no weights file."""
    import cross_check

    monkeypatch.setattr(cross_check, "parameter_count", lambda model_size: VIT7B16_PARAMETER_COUNT)
    monkeypatch.setattr("dinov3_apple.reference.load_torch_backbone", _fail_load)
    monkeypatch.setattr("dinov3_apple.mlx_weights.load_mlx_backbone", _fail_load)
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
    cross-check before any worker and before parameter_count."""
    import cross_check

    monkeypatch.setattr(cross_check, "parameter_count", _fail_load)
    with pytest.raises(SystemExit, match="CLAUDE.md run rules"):
        cross_check.main(
            ["--model-size", "ViT-7B/16", "--dtype-mode", "bf16-autocast", "--dry-run"]
        )


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

    monkeypatch.setattr(cross_check.subprocess, "run", fake_run)
    spec = {"model_size": "ViT-7B/16", "dtype_mode": "bf16-cast", "torch_device": "mps"}
    assert cross_check.run_worker_subprocess("torch-mps", spec) == {"status": "done"}
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        cross_check, "worker_main", lambda backend, spec_json: calls.append((backend, spec_json))
    )
    cross_check.main(commands[0][2:])
    assert calls == [("torch", json.dumps(spec))]
