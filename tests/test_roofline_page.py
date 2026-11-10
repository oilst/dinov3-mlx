"""Tests of scripts/roofline_page.py: the points, the roof and the page from the result files."""

from pathlib import Path

import pytest
import roofline_page
from roofline_page import MATMUL_PARTS, POINTS, attainable, block_points, page, read_lines

ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "results/speed-profile-p2.jsonl"
ROOFS = ROOT / "results/roofs.jsonl"


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


@pytest.mark.skipif(not (PROFILE.is_file() and ROOFS.is_file()), reason="no result files")
def test_page_from_the_result_files() -> None:
    """The page has two marks for each point in Figure 1 and for each matmul point in Figure 2."""
    profiles = [p for p in read_lines(PROFILE) if p["dtype_mode"] in roofline_page.MODES]
    text = page(profiles, read_lines(ROOFS)[-1])
    assert text.count('class="pt ') == 2 * len(POINTS) + 2 * (len(MATMUL_PARTS) + 1)
    assert "Table 5." in text and "nan" not in text
