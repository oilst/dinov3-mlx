import sys
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from dinov3_apple import bench
from dinov3_apple.bench import (
    TIMED,
    WARM_UP,
    BenchConfig,
    TimeBudget,
    add_rounds,
    baseline_rows,
    bench_configurations,
    check_validity,
    keep_going,
    make_batch,
    reference_tokens,
    required_bench_bytes,
    result_line,
    summarize,
)
from dinov3_apple.hub import WeightsNotFoundError
from dinov3_apple.parity import TEST_IMAGES


def _batches(budget: TimeBudget, seconds_per_batch: float) -> tuple[int, float]:
    batches, elapsed = 0, 0.0
    while keep_going(budget, batches, elapsed):
        batches += 1
        elapsed += seconds_per_batch
    return batches, elapsed


def test_keep_going_small_batches_reach_30_and_10_s() -> None:
    batches, elapsed = _batches(TIMED, 0.003)
    assert batches == 3334
    assert elapsed >= 10.0


def test_keep_going_stops_at_60_s_after_10_batches() -> None:
    assert _batches(TIMED, 2.5) == (24, 60.0)


def test_keep_going_slow_batches_run_10() -> None:
    assert _batches(TIMED, 40.0) == (10, 400.0)


def test_warm_up_needs_3_batches_and_2_s() -> None:
    assert _batches(WARM_UP, 0.003)[0] == 667
    assert _batches(WARM_UP, 5.0)[0] == 3


def test_summarize_throughput_median_p90() -> None:
    times = [0.1, 0.2, 0.3, 0.4, 1.0]
    summary = summarize(times, batch_size=8)
    assert summary["throughput_ips"] == pytest.approx(8 * 5 / 2.0)
    assert summary["median_s"] == pytest.approx(np.median(times))
    assert summary["p90_s"] == pytest.approx(np.percentile(times, 90))


def test_bench_configurations_counts() -> None:
    main = bench_configurations()
    assert len(main) == 225
    assert len(set(main)) == 225
    assert {config.run for config in main} == {"B1"}
    giant = bench_configurations(model_sizes=["ViT-7B/16"])
    assert len(giant) == 18
    assert {config.dtype_mode for config in giant} == {"fp16-cast", "bf16-cast"}
    assert {config.run for config in giant} == {"B4"}
    priority = bench_configurations(priority=True)
    assert len(priority) == 10
    assert {(c.model_size, c.side, c.batch_size) for c in priority} == {
        ("ViT-S/16", 224, 8),
        ("ViT-H+/16", 512, 8),
    }
    repeats = bench_configurations(priority=True, repeats=2)
    assert len(repeats) == 20
    assert {c.repeat for c in repeats} == {1, 2}
    assert {c.run for c in repeats} == {"B2"}
    compiled = bench_configurations(priority=True, compile=True)
    assert len(compiled) == 10
    assert all(c.compile and c.run == "B3" for c in compiled)


def _need(model_size: str, dtype_mode: str, side: int, batch_size: int) -> int:
    try:
        return required_bench_bytes(BenchConfig("B1", model_size, dtype_mode, side, batch_size))
    except WeightsNotFoundError as error:
        pytest.skip(str(error))


def test_required_bench_bytes_grows_with_batch_and_side() -> None:
    from dinov3_apple.reference import DTYPE_MODES, required_load_bytes

    load = required_load_bytes("ViT-S/16", DTYPE_MODES["fp32"].weights)
    small = _need("ViT-S/16", "fp32", 224, 1)
    assert small >= load
    assert _need("ViT-S/16", "fp32", 224, 8) > small
    assert _need("ViT-S/16", "fp32", 512, 1) > small
    assert _need("ViT-S/16", "fp16-cast", 224, 1) < small


def test_required_bench_bytes_examples() -> None:
    assert 15e9 <= _need("ViT-H+/16", "fp32", 512, 32) <= 19e9
    assert 26e9 <= _need("ViT-7B/16", "fp16-cast", 512, 32) <= 30e9


def test_bench_module_does_not_import_torch() -> None:
    code = "import sys, dinov3_apple.bench, dinov3_apple.flops; print('torch' in sys.modules)"
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", code], check=False, capture_output=True, text=True
    )
    assert result.stdout.strip() == "False", result.stderr


def test_make_batch_cycles_test_images() -> None:
    inputs = {
        f"{image}/{side}x{side}": np.full((1, 3, side, side), i, np.float32)
        for i, image in enumerate(TEST_IMAGES)
        for side in (224, 448)
    }
    batch = make_batch(inputs, 448, 8)
    assert batch.shape == (8, 3, 448, 448)
    assert batch.dtype == np.float32
    assert [int(image[0, 0, 0]) for image in batch] == [0, 1, 2, 0, 1, 2, 0, 1]


def _tokens(n: int = 16, d: int = 8, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((1 + 4 + n, d)).astype(np.float32)


def test_check_validity_flags_nan() -> None:
    reference = _tokens()
    output = reference.copy()
    output[7, 3] = np.nan
    validity = check_validity(output[0], output[5:], reference, "fp16-cast")
    assert not validity.valid
    assert not validity.finite
    assert "non-finite" in validity.note


def test_check_validity_passes_reference() -> None:
    reference = _tokens()
    output = reference + 1e-6
    validity = check_validity(output[0], output[5:], reference, "fp32")
    assert validity.valid
    assert validity.finite
    assert validity.parity.frame > 0.9999


def test_check_validity_fails_parity() -> None:
    reference = _tokens()
    output = _tokens(seed=1)
    validity = check_validity(output[0], output[5:], reference, "bf16-cast")
    assert validity.finite
    assert not validity.valid
    assert "parity" in validity.note


def test_check_validity_without_reference_is_finite_only() -> None:
    output = _tokens()
    validity = check_validity(output[0], output[5:], None, "fp16-cast")
    assert validity.valid
    assert validity.parity is None
    assert "finite only" in validity.note


def test_reference_tokens_read_image_0_or_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "reference.safetensors"
    tokens = np.arange(12, dtype=np.float32).reshape(3, 4)
    save_file(
        {
            f"{TEST_IMAGES[0]}/224x224/tokens": tokens,
            f"{TEST_IMAGES[0]}/448x448/tokens": tokens + 1,
            f"{TEST_IMAGES[1]}/224x224/tokens": tokens + 2,
        },
        path,
    )
    monkeypatch.setattr(bench, "reference_path", lambda model_size: path)
    assert np.array_equal(reference_tokens("ViT-S/16", 224), tokens)
    assert np.array_equal(reference_tokens("ViT-S/16", 448), tokens + 1)
    monkeypatch.setattr(bench, "reference_path", lambda model_size: tmp_path / "missing")
    assert reference_tokens("ViT-S/16", 224) is None


def _line(dtype_mode: str, throughput: float, **fields: object) -> dict:
    line = {
        "model_size": "ViT-S/16",
        "dtype_mode": dtype_mode,
        "side": 224,
        "batch_size": 8,
        "compile": False,
        "repeat": 0,
        "status": "done",
        "reason": "",
        "valid": True,
        "frame": {"throughput_ips": throughput},
        "patch": {"throughput_ips": throughput / 2},
    }
    return line | fields


def test_baseline_rows_pick_fastest_valid() -> None:
    lines = [
        _line("fp32", 100.0),
        _line("fp16-cast", 300.0),
        _line("fp16-autocast", 200.0),
        _line("bf16-cast", 150.0),
        _line("bf16-autocast", 250.0),
    ]
    rows = baseline_rows(lines, "frame")
    assert rows[("ViT-S/16", 224, 8, "fp32")].line["dtype_mode"] == "fp32"
    assert rows[("ViT-S/16", 224, 8, "fp16")].line["dtype_mode"] == "fp16-cast"
    assert rows[("ViT-S/16", 224, 8, "bf16")].line["dtype_mode"] == "bf16-autocast"


def test_baseline_rows_skip_invalid_and_include_compile() -> None:
    # The user decided on 2026-11-06 that the compile rows join the baseline.
    lines = [
        _line("fp16-cast", 500.0, valid=False, reason="non-finite output"),
        _line("fp16-autocast", 400.0, compile=True),
        _line("fp16-autocast", 200.0),
        # One fast repeat does not win: the median of [200, 900] is 200 (finding F1).
        _line("fp16-autocast", 900.0, repeat=1),
    ]
    row = baseline_rows(lines, "patch")[("ViT-S/16", 224, 8, "fp16")]
    assert row.line["dtype_mode"] == "fp16-autocast"
    assert row.line["compile"]


def test_baseline_rows_empty_with_reason() -> None:
    lines = [
        _line("fp16-cast", 300.0, valid=False, validity_note="non-finite output"),
        _line("fp16-autocast", 0.0, status="skipped", reason="needs 9.9 GB, 7.0 GB available"),
    ]
    row = baseline_rows(lines, "frame")[("ViT-S/16", 224, 8, "fp16")]
    assert row.line is None
    assert "fp16-cast" in row.reason and "non-finite" in row.reason
    assert "fp16-autocast" in row.reason and "skipped" in row.reason


def test_baseline_rows_use_latest_line() -> None:
    skipped = _line("fp32", 0.0, status="skipped", reason="does not fit")
    rows = baseline_rows([skipped, _line("fp32", 120.0)], "frame")
    assert rows[("ViT-S/16", 224, 8, "fp32")].line["frame"]["throughput_ips"] == 120.0


def test_result_line_without_worker_has_a_time() -> None:
    from dinov3_apple.machine import read_machine_state

    config = BenchConfig("B1", "ViT-S/16", "fp32", 224, 1)
    line = result_line(
        config, {"status": "skipped", "reason": "does not fit"}, 1, 0, read_machine_state()
    )
    assert line["time"] is not None
    assert line["start_time"] is None


def test_baseline_rows_use_median_of_runs() -> None:
    # Finding F1 of verify.html: one run can be off by up to 2x on this Mac.
    lines = [
        _line("fp32", 600.0),
        _line("fp32", 320.0, repeat=1),
        _line("fp32", 330.0, repeat=2),
    ]
    row = baseline_rows(lines, "frame")[("ViT-S/16", 224, 8, "fp32")]
    assert row.line["frame"]["throughput_ips"] == 330.0
    assert row.runs == 3


def test_baseline_rows_median_skips_invalid_runs() -> None:
    lines = [
        _line("fp16-cast", 300.0),
        _line("fp16-cast", 900.0, repeat=1, valid=False),
        _line("fp16-cast", 0.0, repeat=2, status="skipped", reason="does not fit"),
    ]
    row = baseline_rows(lines, "frame")[("ViT-S/16", 224, 8, "fp16")]
    assert row.line["frame"]["throughput_ips"] == 300.0
    assert row.runs == 1


def test_add_rounds_rotates_and_numbers_repeats() -> None:
    configs = [
        BenchConfig("B1", "ViT-S/16", "fp32", 224, 1),
        BenchConfig("B1", "ViT-S/16", "fp32", 224, 8),
    ]
    # 224 px batch size 8 has a main run and repeat 1 already; batch size 1 has no line.
    lines = [_line("fp32", 100.0), _line("fp32", 100.0, repeat=1)]
    rounds = add_rounds(configs, lines, rounds=2)
    assert [(c.batch_size, c.repeat) for c in rounds] == [(1, 0), (8, 2), (1, 1), (8, 3)]
    assert add_rounds(configs, lines, rounds=0) == configs
