"""Tests T25 to T28 of specs/06-speed/plan.html: the speedup of each configuration, the pass rule
(D21) and the compile check."""

import pytest

from dinov3_apple import speedup
from dinov3_apple.speedup import (
    compile_check,
    pass_verdict,
    single_verdict,
    speedup_rows,
    spread_between_runs,
)


def line(
    backend: str,
    mode: str,
    frame: float,
    patch: float | None = None,
    valid: bool = True,
    **fields: object,
) -> dict:
    """One result line made by hand: ViT-S/16, 224 px, batch size 8 unless fields say else."""
    base = {
        "run": "S6",
        "backend": backend,
        "model_size": "ViT-S/16",
        "dtype_mode": mode,
        "side": 224,
        "batch_size": 8,
        "compile": True,
        "bits": None,
        "repeat": 0,
        "status": "done",
        "reason": "",
        "valid": valid,
        "validity_note": "parity passes" if valid else "parity below the threshold: frame",
        "parity": {"frame": 0.999, "patch_mean": 0.998, "patch_p01": 0.99, "patch_min": 0.9},
        "frame": {"throughput_ips": frame, "median_s": 0.01, "p90_s": 0.011},
        "patch": {"throughput_ips": patch or frame, "median_s": 0.01, "p90_s": 0.011},
        "phase": "matrix",
        "group": 0,
    }
    return base | fields


def test_speedup_rows_follow_adr_0004_on_both_sides() -> None:
    """T25: the mlx side takes the fastest valid unquantized mode of the dtype; the torch-mps side
    the fastest valid row, or the faster row of a follow-up group; lines with the same key on two
    backends both stay."""
    lines = [
        line("mlx", "bf16-cast", 1000.0),
        line("mlx", "bf16-autocast", 900.0),
        line("mlx", "bf16-autocast", 5000.0, bits=8),  # quantized: never in the choice
        line("mlx", "fp16-cast", 3000.0, valid=False),  # not valid: never in the choice
        line("mlx", "fp16-autocast", 800.0),
        line("torch-mps", "bf16-cast", 1000.0),  # the same key as the mlx bf16-cast line
        line("torch-mps", "bf16-autocast", 500.0),
        line("torch-mps", "fp16-cast", 700.0),
        line("torch-mps", "fp16-cast", 900.0, compile=False, phase="compile-check"),
    ]
    rows = {(r.dtype, r.output, r.bits): r for r in speedup_rows(lines)}
    bf16 = rows[("bf16", "frame", None)]
    assert bf16.mlx_mode == "bf16-cast"
    assert bf16.torch_mode == "bf16-cast"
    assert bf16.ratio == pytest.approx(1.0)
    fp16 = rows[("fp16", "frame", None)]
    assert fp16.mlx_mode == "fp16-autocast"
    assert fp16.torch_mode == "fp16-cast"
    assert not fp16.torch_compile  # the faster eager row of the follow-up group
    assert fp16.ratio == pytest.approx(800.0 / 900.0)
    quantized = rows[("bf16", "frame", 8)]
    assert quantized.ratio == pytest.approx(5.0)
    assert quantized.unquantized_ratio == pytest.approx(5000.0 / 900.0)


def test_row_pairs_both_sides_of_one_group() -> None:
    """Finding V10 of specs/06-speed/verify.html: the ratio takes both sides from the group of the
    fastest torch-mps row, not the fastest row of each side from two groups. A group without an
    mlx row of the dtype takes the fastest mlx row of the cell."""
    lines = [
        line("mlx", "bf16-cast", 1000.0),
        line("torch-mps", "bf16-cast", 900.0),
        line("mlx", "bf16-cast", 800.0, phase="compile-check", group=7),
        line("torch-mps", "bf16-cast", 950.0, phase="compile-check", group=7),
        line("mlx", "fp16-cast", 700.0),
        line("torch-mps", "fp16-cast", 600.0),
        line("torch-mps", "fp16-cast", 650.0, compile=False, phase="compile-check", group=7),
    ]
    rows = {(r.dtype, r.output, r.bits): r for r in speedup_rows(lines)}
    assert rows[("bf16", "frame", None)].ratio == pytest.approx(800.0 / 950.0)
    assert rows[("fp16", "frame", None)].ratio == pytest.approx(700.0 / 650.0)


def test_pass_rule() -> None:
    """T26 (D21 option b)."""
    assert pass_verdict([1.05, 1.04], 1.04) == "above 1"
    assert pass_verdict([1.05, 0.98], 1.01) == "not above 1, inside the noise"
    # Finding V5 of specs/06-speed/verify.html: the pooled median within 1 ± NOISE gets the mark.
    assert pass_verdict([1.013, 1.018], 1.013) == "above 1, inside the noise"
    assert pass_verdict([1.05, 0.90], 0.95) == "not above 1"
    assert single_verdict(1.04) == "above 1"
    assert single_verdict(1.02) == "above 1, inside the noise"
    assert single_verdict(0.99) == "not above 1, inside the noise"
    assert single_verdict(0.90) == "not above 1"
    assert spread_between_runs([1.0, 1.2]) == pytest.approx(0.2)
    assert speedup._lower_median([0.98, 0.99, 1.01, 1.02]) == 0.99
    ratios = [0.95, 0.96, 0.97, 0.98, 0.99, 1.01, 1.02, 1.03, 1.04, 1.05]
    assert speedup._lower_median(ratios) == 0.99
    assert pass_verdict([1.01, 1.01], speedup._lower_median(ratios)) == (
        "not above 1, inside the noise"
    )


def test_pair_rows_take_both_runs() -> None:
    """T26: the paired runs at P1: the mode on each side over all rounds of both runs, one ratio
    for each round, the run medians, the pooled median and the verdict."""
    lines = []
    for run_id, factor in (("1", 1.1), ("2", 0.95)):
        for round_ in range(1, 3):
            for mode in ("bf16-cast", "bf16-autocast"):
                torch_ips = 1000.0 + round_
                mlx_ips = torch_ips * factor * (1.0 if mode == "bf16-cast" else 0.9)
                common = {"run_id": run_id, "round": round_, "priority": "P1"}
                lines.append(line("mlx", mode, mlx_ips, **common))
                lines.append(line("torch-mps", mode, torch_ips, **common))
    rows = {(r.priority, r.dtype, r.output): r for r in speedup.pair_rows(lines)}
    row = rows[("P1", "bf16", "frame")]
    assert row.mlx_mode == "bf16-cast"
    assert row.run_medians == pytest.approx([1.1, 0.95])
    assert row.pooled_median == pytest.approx(0.95)
    assert row.ratios == 4
    assert row.verdict == "not above 1"
    assert row.spread == pytest.approx(1.1 / 0.95 - 1)


def test_speedup_table_covers_every_configuration() -> None:
    """T27: 324 cells; the 18 ViT-7B/16 fp32 cells empty with the reason of ADR 0008; a skip line
    gives an empty cell with its reason; a 4-bit row below THRESHOLDS[mode] shows "not valid"
    with its parity; an eager fallback carries its mark."""
    lines = []
    for model_size in speedup.MODEL_SIZES:
        for side in (224, 448, 512):
            for batch_size in (1, 8, 32):
                cell = {"model_size": model_size, "side": side, "batch_size": batch_size}
                for mode in ("fp32", "fp16-cast", "bf16-cast"):
                    if model_size == "ViT-7B/16" and mode == "fp32":
                        continue
                    lines.append(line("mlx", mode, 120.0, **cell))
                    lines.append(line("torch-mps", mode, 100.0, **cell))
    p2 = {"model_size": "ViT-H+/16", "side": 512, "batch_size": 8}
    lines.append(line("mlx", "bf16-autocast", 50.0, bits=4, valid=False, **p2))
    lines.append(
        line("mlx", "fp16-cast", 0.0, status="skipped", reason="needs 9 GB", model_size="ViT-B/16")
    )
    # The compiled worker of one cell ends with an error; the eager worker carries the mark.
    lines.append(line("torch-mps", "fp32", 0.0, status="error", model_size="ViT-L/16"))
    lines.append(
        line(
            "torch-mps",
            "fp32",
            90.0,
            compile=False,
            mark="eager torch: torch.compile failed",
            model_size="ViT-L/16",
        )
    )
    cells = speedup.matrix_cells(lines)
    assert len(cells) == 324
    empty_7b = [c for c in cells if c.model_size == "ViT-7B/16" and c.dtype == "fp32"]
    assert len(empty_7b) == 18
    assert all("ADR 0008" in c.reason and c.ratio is None for c in empty_7b)
    skipped = next(
        c
        for c in cells
        if (c.model_size, c.side, c.batch_size, c.dtype) == ("ViT-B/16", 224, 8, "fp16")
    )
    assert skipped.ratio is None and "needs 9 GB" in skipped.reason
    eager = next(
        c
        for c in cells
        if (c.model_size, c.side, c.batch_size, c.dtype, c.output)
        == ("ViT-L/16", 224, 8, "fp32", "frame")
    )
    assert eager.mark == "eager torch: torch.compile failed"
    assert eager.ratio == pytest.approx(120.0 / 90.0)
    assert not eager.torch_compile
    quantized = [r for r in speedup_rows(lines) if r.bits == 4]
    assert quantized and all(r.verdict == "not valid" for r in quantized)
    assert "frame 0.999" in quantized[0].reason


def test_compile_check_lists_slow_compile_rows() -> None:
    """T28: a compiled torch-mps row below the Step 3 eager median of its mode is listed."""
    step3 = [
        line("torch-mps", "fp32", 100.0, compile=False, run="B1"),
        line("torch-mps", "fp32", 110.0, compile=False, run="B2", repeat=1),
        line("torch-mps", "fp32", 120.0, compile=False, run="B2", repeat=2),
        line("torch-mps", "bf16-cast", 100.0, compile=False, run="B1"),
    ]
    matrix = [
        line("torch-mps", "fp32", 105.0),  # below the median 110
        line("torch-mps", "bf16-cast", 150.0),
        line("mlx", "fp32", 10.0),
    ]
    assert compile_check(matrix, step3) == [("ViT-S/16", 224, 8, "fp32")]


def test_speedup_table_script_writes_all_sections(tmp_path) -> None:
    """J20: scripts/speedup_table.py writes the sections of Table 13 from the result files, and
    a second run gives the same text (VJ3)."""
    import speedup_table

    matrix = [
        line("mlx", "fp32", 120.0),
        line("torch-mps", "fp32", 100.0),
        line("mlx", "bf16-autocast", 90.0, bits=8, model_size="ViT-H+/16"),
    ]
    pairs = [
        line("mlx", "bf16-cast", 110.0, run_id="1", round=1, priority="P1"),
        line("torch-mps", "bf16-cast", 100.0, run_id="1", round=1, priority="P1"),
    ]
    files = {name: [] for name in speedup_table.FILES}
    files |= {"matrix": matrix, "pairs": pairs}
    text = speedup_table.write_table(files)
    for title in ("## 1 P1 and P2", "## 2 Matrix", "## 3 Quantized", "## 4 Change", "## 5 Best"):
        assert title in text
    assert "## 6 Autocast / cast" in text and "## 7 Compile check" in text
    rows = [row for row in text.split("## 2 Matrix")[1].split("Table 2.")[0].splitlines()]
    assert sum(row.startswith("| ViT") for row in rows) == 324
    assert "| P1 | bf16 | frame | bf16-cast | bf16-cast (compile) | 1.100 |" in text
    assert speedup_table.write_table(files) == text
