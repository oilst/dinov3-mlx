"""Tests T11 to T15 of specs/05-all-sizes/plan.html: the run-rule guards of both loaders, of
scripts/make_reference.py and of the backbone slot, and the rule for non-finite outputs (M2)."""

import json
import logging
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from parity_checks import check_finite, run_rule_skip

from dinov3_apple.hub import MODEL_IDS
from dinov3_apple.memory import InsufficientMemoryError, RunRuleError
from dinov3_apple.parity import FP16_OVERFLOW, THRESHOLDS
from dinov3_apple.reference import LOAD_RATIO, load_torch_backbone

VIT7B16_PARAMETER_COUNT = 6_716_035_072
# The torch dtype modes with fp32 weights: the run rule refuses ViT-7B/16 in each of them.
FP32_WEIGHT_MODES = ["fp32", "fp16-autocast", "bf16-autocast"]


def _fail(name: str) -> Callable[..., None]:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"{name} was called")

    return fail


@pytest.mark.parametrize("dtype_mode", [*FP32_WEIGHT_MODES, "bf16-cast"])
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_load_torch_backbone_refuses_7b_fp32_weights(
    device: str, dtype_mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T11: ViT-7B/16 with fp32 weights stops before the memory check; bf16-cast passes the guard
    and stops at the memory check."""
    # The lines of a monkeypatched memory check must not reach memory-checks.log of the session.
    monkeypatch.setattr(logging.getLogger("dinov3_apple.memory"), "handlers", [])
    monkeypatch.setattr("dinov3_apple.reference.parameter_count", _fail("parameter_count"))
    monkeypatch.setattr("dinov3_apple.memory.available_memory_bytes", _fail("memory check"))
    monkeypatch.setattr(
        "dinov3_apple.reference.DINOv3ViTModel.from_pretrained", _fail("from_pretrained")
    )
    if dtype_mode in FP32_WEIGHT_MODES:
        with pytest.raises(RunRuleError, match="ViT-7B/16"):
            load_torch_backbone("ViT-7B/16", device, dtype_mode)
        return
    monkeypatch.setattr(
        "dinov3_apple.reference.parameter_count", lambda model_size: VIT7B16_PARAMETER_COUNT
    )
    monkeypatch.setattr("dinov3_apple.memory.available_memory_bytes", lambda: 0)
    with pytest.raises(InsufficientMemoryError, match="needs 20.1 GB"):
        load_torch_backbone("ViT-7B/16", device, dtype_mode)


def test_make_reference_refuses_7b(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """T12: make_reference.py records a skip with the run-rule reason and starts no subprocess."""
    import make_reference

    runs_log = tmp_path / "runs.jsonl"
    monkeypatch.setattr(make_reference, "RUNS_LOG", runs_log)
    # load_ratio binds its default RUNS_LOG when Python defines it, so the test replaces it.
    monkeypatch.setattr(make_reference, "load_ratio", lambda: LOAD_RATIO)
    monkeypatch.setattr(make_reference, "parameter_count", _fail("parameter_count"))
    monkeypatch.setattr(make_reference, "check_free_memory", _fail("memory check"))
    monkeypatch.setattr(subprocess, "run", _fail("subprocess.run"))
    make_reference.run_all(["ViT-7B/16"])
    lines = [json.loads(text) for text in runs_log.read_text().splitlines()]
    assert len(lines) == 1
    assert lines[0]["model_size"] == "ViT-7B/16"
    assert lines[0]["status"] == "skipped"
    assert "CLAUDE.md run rules" in lines[0]["reason"]
    assert "fp32_weight_bytes" not in lines[0]


def test_run_rule_skip() -> None:
    """T13: 60 triples. mlx skips only (ViT-7B/16, fp32); torch-mps skips ViT-7B/16 in the three
    dtype modes with fp32 weights."""
    skips = {
        (model_size, dtype_mode, backend)
        for model_size in MODEL_IDS
        for dtype_mode in THRESHOLDS
        for backend in ("mlx", "torch-mps")
        if run_rule_skip(model_size, dtype_mode, backend) is not None
    }
    assert skips == {("ViT-7B/16", "fp32", "mlx")} | {
        ("ViT-7B/16", dtype_mode, "torch-mps") for dtype_mode in FP32_WEIGHT_MODES
    }
    assert len(MODEL_IDS) * len(THRESHOLDS) * 2 - len(skips) == 56
    assert "CLAUDE.md run rules" in run_rule_skip("ViT-7B/16", "fp32", "mlx")


def test_check_finite_marks_only_listed_pairs() -> None:
    """T14: a non-finite output xfails only for a pair of EXPECTED_FP16_OVERFLOW."""
    notes: list[tuple[str, str]] = []

    def note_of(*pair: str) -> Callable[[], None]:
        return lambda: notes.append(pair)

    finite = np.ones((5, 4), np.float32)
    check_finite(finite, "ViT-L/16", "fp16-cast", note_of("ViT-L/16", "fp16-cast"))
    assert notes == []
    nan = finite.copy()
    nan[2, 1] = np.nan
    with pytest.raises(pytest.xfail.Exception, match="fp16 overflow") as xfail:
        check_finite(nan, "ViT-L/16", "fp16-cast", note_of("ViT-L/16", "fp16-cast"))
    assert str(xfail.value) == FP16_OVERFLOW
    assert notes == [("ViT-L/16", "fp16-cast")]
    for pair in (("ViT-S/16", "fp16-cast"), ("ViT-7B/16", "fp16-cast")):
        notes.clear()
        with pytest.raises(pytest.fail.Exception, match="not an expected failure"):
            check_finite(nan, *pair, note_of(*pair))
        assert notes == [pair]


# The pair of T15 as parameters: the sort key of tests/conftest.py puts the test with the other
# users of the backbone slot, after the tests without a model size.
@pytest.mark.mlx
@pytest.mark.parametrize(
    ("model_size", "dtype_mode"), [pytest.param("ViT-7B/16", "fp32", id="mlx-ViT-7B/16-fp32")]
)
def test_backbone_slot_skips_a_run_rule_error(
    model_size: str,
    dtype_mode: str,
    backbone: Callable[[str, str, str], Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """T15: the backbone slot skips ViT-7B/16 in fp32 on mlx with the run-rule reason, before any
    file read and without a memory-check line."""
    monkeypatch.setattr("dinov3_apple.mlx_weights.parameter_count", _fail("parameter_count"))
    monkeypatch.setattr("dinov3_apple.mlx_weights.weights_dir", _fail("weights_dir"))
    caplog.set_level(logging.INFO, logger="dinov3_apple.memory")
    with pytest.raises(pytest.skip.Exception, match="CLAUDE.md run rules"):
        backbone(model_size, "mlx", dtype_mode)
    assert [r for r in caplog.records if r.name == "dinov3_apple.memory"] == []
