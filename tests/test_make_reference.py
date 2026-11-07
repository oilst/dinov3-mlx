import json
import logging
import subprocess
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from fakes import fail_if_called
from make_reference import load_ratio
from safetensors import safe_open
from small_model import save_small_model, small_hf_model, small_pixels
from transformers import DINOv3ViTModel

from dinov3_apple.reference import (
    HIDDEN_STATES_CONFIG,
    LOAD_RATIO,
    REQUIRED_METADATA,
    configurations,
    run_torch,
    run_torch_hidden_states,
)


def _write(path: Path, lines: list[dict]) -> Path:
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def test_load_ratio_without_runs_is_the_default(tmp_path: Path) -> None:
    assert load_ratio(tmp_path / "runs.jsonl") == LOAD_RATIO


def test_load_ratio_keeps_the_largest_measured_ratio_of_large_models(tmp_path: Path) -> None:
    runs = _write(
        tmp_path / "runs.jsonl",
        [
            {"status": "done", "measured_ratio": 2.04, "fp32_weight_bytes": 86_000_000},
            {"status": "done", "measured_ratio": 1.21, "fp32_weight_bytes": 1_212_000_000},
            {"status": "done", "measured_ratio": 1.11, "fp32_weight_bytes": 3_362_000_000},
            {"status": "skipped", "reason": "memory check"},
        ],
    )
    assert load_ratio(runs) == 1.21


def test_load_ratio_never_goes_below_the_default(tmp_path: Path) -> None:
    runs = _write(
        tmp_path / "runs.jsonl",
        [{"status": "done", "measured_ratio": 1.05, "fp32_weight_bytes": 3_362_000_000}],
    )
    assert load_ratio(runs) == LOAD_RATIO


# The tests of the options --streamed and --check (path D4-Y, Table 7 of
# specs/05-all-sizes/plan.html).
VIT7B16_STREAMED_NEED = 4_820_818_739
CHILD_RUN = {
    "run_time_s": 300.0,
    "load_time_s": 40.0,
    "peak_memory_bytes": 2_500_000_000,
    "rss_before_load_bytes": 400_000_000,
    "largest_block_bytes": 671_285_248,
    "outer_bytes": 12_730_368,
}


def _fake_children(outputs: dict[str, dict], commands: list[list[str]]) -> Callable[..., object]:
    """Return a stand-in of subprocess.run that prints the JSON line of each child."""

    def run(command: list[str], **kwargs: object) -> SimpleNamespace:
        commands.append(command)
        child = outputs[command[command.index("--child") + 1]]
        return SimpleNamespace(returncode=0, stdout=f"progress\n{json.dumps(child)}\n")

    return run


def test_load_ratio_ignores_streamed_runs(tmp_path: Path) -> None:
    """A streamed run holds one block at a time, so its peak says nothing about the load of the
    whole model: its lines never feed load_ratio."""
    runs = _write(
        tmp_path / "runs.jsonl",
        [
            {"status": "done", "measured_ratio": 1.21, "fp32_weight_bytes": 1_212_000_000},
            {
                "status": "done",
                "streamed": True,
                "measured_ratio": 3.5,
                "fp32_weight_bytes": 26_864_140_288,
            },
            {"model_size": "ViT-7B/16", "status": "done", "streamed": True, **CHILD_RUN},
            {"model_size": "ViT-H+/16", "status": "checked", "streamed": True, "check": True},
        ],
    )
    assert load_ratio(runs) == 1.21


def test_run_streamed_records_need_and_peak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--streamed ViT-7B/16: the run-rule reading of the streamed path, then the memory check
    with the streamed need (4.82 GB), then one subprocess. The line has "streamed": true, the need
    and the peak, and no load ratio. load_ratio, parameter_count and the full-model need are never
    used."""
    import make_reference

    runs_log = tmp_path / "runs.jsonl"
    monkeypatch.setattr(make_reference, "RUNS_LOG", runs_log)
    monkeypatch.setattr(make_reference, "load_ratio", fail_if_called("load_ratio"))
    monkeypatch.setattr(make_reference, "parameter_count", fail_if_called("parameter_count"))
    monkeypatch.setattr(
        make_reference, "required_load_bytes", fail_if_called("required_load_bytes")
    )
    monkeypatch.setattr(make_reference, "required_streamed_bytes", lambda m: VIT7B16_STREAMED_NEED)
    checks: list[tuple[int, str]] = []

    def check(required_bytes: int, label: str) -> int:
        checks.append((required_bytes, label))
        return 30 * 10**9

    monkeypatch.setattr(make_reference, "check_free_memory", check)
    commands: list[list[str]] = []
    child = CHILD_RUN | {"file": "reference-vit7b16.safetensors", "file_bytes": 250_000_000}
    monkeypatch.setattr(subprocess, "run", _fake_children({"ViT-7B/16": child}, commands))
    make_reference.run_streamed(["ViT-7B/16"], check=False)

    label = "ViT-7B/16 torch-cpu fp32, streamed: one block at a time (make_reference.py)"
    assert checks == [(VIT7B16_STREAMED_NEED, label)]
    assert [command[2:] for command in commands] == [["--child", "ViT-7B/16", "--streamed"]]
    lines = [json.loads(text) for text in runs_log.read_text().splitlines()]
    assert len(lines) == 1
    line = lines[0]
    assert line["model_size"] == "ViT-7B/16"
    assert line["status"] == "done"
    assert line["streamed"] is True
    assert "one block" in line["run_rule"]
    assert line["need_bytes"] == VIT7B16_STREAMED_NEED
    assert line["available_before_bytes"] == 30 * 10**9
    assert line["peak_memory_bytes"] == 2_500_000_000
    assert {"measured_ratio", "ratio", "fp32_weight_bytes"}.isdisjoint(line)
    assert load_ratio(runs_log) == LOAD_RATIO


def test_run_streamed_check_records_and_stops_above_the_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--streamed --check: each model size with a reference file in its own subprocess, one after
    the other. A model size without a file gets a skip line. A largest difference above 1e-6
    stops the script after the last model size."""
    import make_reference

    runs_log = tmp_path / "runs.jsonl"
    monkeypatch.setattr(make_reference, "RUNS_LOG", runs_log)
    files = {"ViT-S/16": tmp_path / "s.safetensors", "ViT-B/16": tmp_path / "b.safetensors"}
    for file in files.values():
        file.write_bytes(b"stored")
    monkeypatch.setattr(
        make_reference, "reference_path", lambda m: files.get(m, tmp_path / "missing.safetensors")
    )
    monkeypatch.setattr(make_reference, "required_streamed_bytes", lambda m: 4 * 10**9)
    monkeypatch.setattr(make_reference, "check_free_memory", lambda need, label: 30 * 10**9)
    differences = {
        "ViT-S/16": {"configurations": 12, "tokens_max_abs": 0.0, "hidden_states_max_abs": 0.0},
        "ViT-B/16": {"configurations": 12, "tokens_max_abs": 0.0, "hidden_states_max_abs": 2e-6},
    }
    outputs = {model_size: CHILD_RUN | values for model_size, values in differences.items()}
    commands: list[list[str]] = []
    monkeypatch.setattr(subprocess, "run", _fake_children(outputs, commands))
    with pytest.raises(SystemExit, match="ViT-B/16"):
        make_reference.run_streamed(["ViT-S/16", "ViT-B/16", "ViT-L/16"], check=True)

    assert [command[2:] for command in commands] == [
        ["--child", "ViT-S/16", "--streamed", "--check"],
        ["--child", "ViT-B/16", "--streamed", "--check"],
    ]
    lines = [json.loads(text) for text in runs_log.read_text().splitlines()]
    assert [(line["model_size"], line["status"]) for line in lines] == [
        ("ViT-S/16", "checked"),
        ("ViT-B/16", "checked"),
        ("ViT-L/16", "skipped"),
    ]
    assert [line.get("passed") for line in lines] == [True, False, None]
    assert all(line["streamed"] and line["check"] for line in lines)
    assert lines[0]["largest_difference"] == 0.0
    assert lines[1]["largest_difference"] == 2e-6
    assert lines[1]["target"] == 1e-6
    assert "missing" in lines[2]["reason"]
    assert all(file.read_bytes() == b"stored" for file in files.values())


def test_streamed_differences() -> None:
    """The largest absolute difference of the tokens of each configuration and of the hidden
    states. Another set of configurations or another shape stops the check."""
    from make_reference import streamed_differences

    hidden_key = f"{HIDDEN_STATES_CONFIG}/hidden_states"
    tokens = {
        "a/224x224": np.zeros((201, 4), np.float32),
        "b/448x448": np.ones((789, 4), np.float32),
    }
    hidden = np.zeros((3, 201, 4), np.float32)
    stored = {name: values.copy() for name, values in tokens.items()} | {hidden_key: hidden.copy()}
    stored["b/448x448"][7, 2] += 0.25
    stored[hidden_key][2, 0, 1] -= 0.5
    assert streamed_differences(stored, tokens, hidden) == {
        "configurations": 2,
        "tokens_max_abs": 0.25,
        "hidden_states_max_abs": 0.5,
    }
    with pytest.raises(ValueError, match="configurations"):
        streamed_differences(stored, {"a/224x224": tokens["a/224x224"]}, hidden)
    with pytest.raises(ValueError, match="shape"):
        streamed_differences(stored, tokens, hidden[:2])


@pytest.fixture
def small_weights(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """A small SwiGLU model as the weights of ViT-S+/16, random pixels for the 12
    configurations, and its full-model reference output (tokens and hidden states)."""
    import make_reference

    hf_model = small_hf_model(seed=6, name="vits16plus")
    folder = tmp_path / "weights"
    save_small_model(hf_model, folder, files=2)
    pixels = {
        config.id: small_pixels((1, 3, config.height, config.width), seed=i)
        for i, config in enumerate(configurations())
    }
    full = DINOv3ViTModel.from_pretrained(folder, dtype=torch.float32).eval()
    stored = {config: run_torch(full, values, "fp32")[0] for config, values in pixels.items()}
    hidden = run_torch_hidden_states(full, pixels[HIDDEN_STATES_CONFIG])
    stored[f"{HIDDEN_STATES_CONFIG}/hidden_states"] = hidden
    out = tmp_path / "reference-small.safetensors"
    monkeypatch.setattr(make_reference, "weights_dir", lambda m: folder)
    monkeypatch.setattr("dinov3_apple.reference.weights_dir", lambda m: folder)
    monkeypatch.setattr(make_reference, "read_inputs", lambda: pixels)
    monkeypatch.setattr(make_reference, "reference_path", lambda m: out)
    monkeypatch.setattr(make_reference, "load_torch_backbone", fail_if_called("load_torch"))
    monkeypatch.setattr(DINOv3ViTModel, "from_pretrained", fail_if_called("from_pretrained"))
    monkeypatch.setattr(logging.getLogger("dinov3_apple.memory"), "handlers", [])
    monkeypatch.setattr("dinov3_apple.memory.available_memory_bytes", lambda: 10**12)
    return {"stored": stored, "out": out}


def test_streamed_child_writes_the_reference_file(
    small_weights: dict, capsys: pytest.CaptureFixture
) -> None:
    """The child of --streamed writes the reference file with the header fields of the others plus
    "streamed": true. Its tokens and hidden states equal the full model bit for bit."""
    import make_reference

    make_reference.run_streamed_child("ViT-S+/16", check=False)
    child = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    out = small_weights["out"]
    assert child["file"] == out.name
    assert child["file_bytes"] == out.stat().st_size
    assert child["peak_memory_bytes"] >= child["rss_before_load_bytes"] > 0
    with safe_open(out, "np") as tensors:
        metadata = json.loads(tensors.metadata()["reference"])
        data = {key.removesuffix("/tokens"): tensors.get_tensor(key) for key in tensors.keys()}  # noqa: SIM118
    assert set(REQUIRED_METADATA) <= set(metadata)
    assert metadata["streamed"] is True
    assert metadata["attn_implementation"] == "sdpa"
    assert metadata["num_blocks"] == 2
    assert metadata["model_size"] == "ViT-S+/16"
    assert sorted(metadata["weight_sha256"]) == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ]
    stored = small_weights["stored"]
    assert sorted(data) == sorted(stored)
    for key, values in stored.items():
        assert np.abs(data[key] - values).max() == 0.0, key


def test_streamed_child_check_writes_no_file(
    small_weights: dict, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    """The child of --streamed --check compares with the stored file and writes no file."""
    import make_reference

    monkeypatch.setattr(make_reference, "read_reference", lambda m: (small_weights["stored"], {}))
    monkeypatch.setattr(make_reference, "save_file", fail_if_called("save_file"))
    make_reference.run_streamed_child("ViT-S+/16", check=True)
    child = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert child["configurations"] == 12
    assert child["tokens_max_abs"] == 0.0
    assert child["hidden_states_max_abs"] == 0.0
    assert not small_weights["out"].exists()
