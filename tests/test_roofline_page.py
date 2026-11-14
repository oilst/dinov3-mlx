"""Tests of scripts/roofline_page.py: the points, the roof and the page from the result files."""

from pathlib import Path

import pytest
import roofline_page
from roofline_page import (
    MATMUL_PARTS,
    POINTS,
    attainable,
    block_points,
    concat_seconds,
    flag_cost,
    page,
    read_lines,
    tile_gain,
)

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "results/speed-profile-p2.jsonl"
ROOFS = ROOT / "results/roofs.jsonl"
JOINED = ROOT / "results/speed-joined.jsonl"
KERNELS = ROOT / "results/kernels-c1.jsonl"
RESULT_FILES = (PROFILE, ROOFS, JOINED, KERNELS)


def profile_line(mode: str = "fp16-cast") -> dict:
    """Return a profile line where each part and the block take 1 ms."""
    parts = dict.fromkeys(roofline_page.PART_NAMES, 1e-3)
    return {"dtype_mode": mode, "parts_s": parts, "block_s": 1e-3}


def test_block_points_use_flops_and_estimates() -> None:
    """Each part and the block get one point; only the element-wise parts have estimated FLOPs."""
    points = {p["part"]: p for p in block_points(profile_line())}
    assert list(points) == POINTS
    estimated = {n for n, p in points.items() if p["estimated"]}
    assert estimated == set(roofline_page.ELEMENTWISE_PER_VALUE)
    # qkv_proj of ViT-H+/16 at 512 px, batch size 8: 3 · 2 · B T D².
    assert points["qkv_proj"]["flops"] == 6 * 8 * 1029 * 1280**2
    assert points["qkv_proj"]["tflops"] == pytest.approx(points["qkv_proj"]["flops"] / 1e-3 / 1e12)
    assert points["block"]["flops"] == sum(points[n]["flops"] for n in POINTS[:-1])
    assert all(points[n]["intensity"] > 100 for n in MATMUL_PARTS)


def test_attainable_is_the_lower_roof() -> None:
    """Below the ridge point the memory roof limits, above it the matmul roof."""
    roofs = {"fp16": 50.0, "bf16": 40.0, "memory_gbs": 500.0}
    assert attainable(roofs, "fp16-cast", 10) == pytest.approx(5.0)
    assert attainable(roofs, "fp16-cast", 1000) == 50.0
    assert attainable(roofs, "bf16-cast", 1000) == 40.0


def c7_line(priority: str, r: list[float], jff: list[float]) -> dict:
    """Return a C7 line with the frame throughput of R and R-Jff in each round."""
    return {
        "change": "C7",
        "status": "done",
        "priority": priority,
        "batch_size": 8,
        "throughput": {"R": {"frame": r}, "R-Jff": {"frame": jff}},
    }


def test_flag_cost_is_the_r_jff_difference_at_p1() -> None:
    """The cost is the batch time of R-Jff minus that of R at P1; the P2 lines do not count."""
    lines = [c7_line("P1", [800.0, 800.0], [400.0, 400.0]), c7_line("P2", [1.0], [0.5])]
    cost, batch_time = flag_cost(lines)
    assert cost == pytest.approx(8 / 400 - 8 / 800)
    assert batch_time == pytest.approx(8 / 800)


def test_concat_seconds_counts_both_joins() -> None:
    """The concat reads and writes the q, k, v weights (3 D²) and the gate, up weights (2 D F)."""
    d, f = 1280, 5120
    assert concat_seconds({"memory_gbs": 500.0}) == pytest.approx(
        2 * (3 * d * d + 2 * d * f) * 2 / 500e9
    )


def test_tile_gain_uses_the_ratio_to_the_wheel() -> None:
    """T1 shrinks qkv_proj and o_proj, T3 shrinks sdpa; T2 does not count."""
    by = {("fp16-cast", n): {"seconds": 1e-3} for n in ("qkv_proj", "o_proj", "sdpa")}
    best = [
        {"target": "T1", "ratio_to_wheel": 1.25},
        {"target": "T2", "ratio_to_wheel": 3.0},
        {"target": "T3", "ratio_to_wheel": 2.0},
    ]
    assert tile_gain(by, best) == pytest.approx(2 * 1e-3 * 0.2 + 1e-3 * 0.5)


@pytest.mark.skipif(not all(f.is_file() for f in RESULT_FILES), reason="no result files")
def test_page_from_the_result_files() -> None:
    """The page has two marks for each point in Figure 1 and for each matmul point in Figure 2,
    and the two sections of Step 6e with Tables 6 and 7."""
    profiles = [p for p in read_lines(PROFILE) if p["dtype_mode"] in roofline_page.MODES]
    text = page(profiles, read_lines(ROOFS)[-1], read_lines(JOINED), read_lines(KERNELS))
    assert text.count('class="pt ') == 2 * len(POINTS) + 2 * (len(MATMUL_PARTS) + 1)
    assert "Table 5." in text and "nan" not in text
    assert 'id="joined"' in text and 'id="tiles"' in text
    assert "Table 6." in text and "Table 7." in text and "<h2>12 Sources</h2>" in text
    assert "set_joined_projections" in text and "_joined_bias" in text
