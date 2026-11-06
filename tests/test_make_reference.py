import json
from pathlib import Path

from make_reference import load_ratio

from dinov3_apple.reference import LOAD_RATIO


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
