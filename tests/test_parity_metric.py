import numpy as np
import pytest

from dinov3_mlx.outputs import NUM_PREFIX_TOKENS
from dinov3_mlx.parity import THRESHOLDS, measure_parity


def test_measure_parity_worked_example() -> None:
    # Table 10 of docs/plan.html.
    reference = np.tile(np.array([1.0, 2.0, 2.0], dtype=np.float32), (NUM_PREFIX_TOKENS + 4, 1))
    output = np.tile(np.array([1.0, 2.0, 2.1], dtype=np.float32), (NUM_PREFIX_TOKENS + 4, 1))
    parity = measure_parity(reference, output)
    assert round(parity.frame, 4) == 0.9997
    assert parity.frame < THRESHOLDS["fp32"].frame
    assert parity.frame >= THRESHOLDS["fp16-cast"].frame
    assert parity.frame >= THRESHOLDS["bf16-cast"].frame


def test_measure_parity_of_equal_vectors_is_one() -> None:
    rng = np.random.default_rng(0)
    tokens = rng.standard_normal((NUM_PREFIX_TOKENS + 196, 384)).astype(np.float32)
    parity = measure_parity(tokens, tokens.copy())
    # fp64 rounding leaves at most a few units of 1e-16.
    for value in (parity.frame, parity.patch_mean, parity.patch_p01, parity.patch_min):
        assert value == pytest.approx(1.0, abs=1e-12)


def test_measure_parity_patch_percentile_and_minimum() -> None:
    # 101 patches with cosines 0.00, 0.01, ..., 1.00. The 1st percentile is the second value.
    cosines = np.linspace(0.0, 1.0, 101)
    angles = np.arccos(cosines)
    reference = np.zeros((NUM_PREFIX_TOKENS + 101, 2))
    reference[:, 0] = 1.0
    output = reference.copy()
    output[NUM_PREFIX_TOKENS:] = np.stack([np.cos(angles), np.sin(angles)], axis=1)
    parity = measure_parity(reference, output)
    assert parity.frame == pytest.approx(1.0, abs=1e-12)
    assert parity.patch_mean == pytest.approx(0.5)
    assert parity.patch_p01 == pytest.approx(0.01)
    assert parity.patch_min == pytest.approx(0.0, abs=1e-12)


def test_thresholds_of_table_6() -> None:
    # bf16 got its own thresholds on 2026-11-06 (user decision after the bf16 probe).
    assert THRESHOLDS["fp32"].frame == 0.9999
    assert THRESHOLDS["fp32"].patch_mean == 0.9999
    assert THRESHOLDS["fp32"].patch_p01 == 0.9999
    for mode in ("fp16-cast", "fp16-autocast"):
        assert THRESHOLDS[mode].frame == 0.999
        assert THRESHOLDS[mode].patch_mean == 0.999
        assert THRESHOLDS[mode].patch_p01 == 0.99
    for mode in ("bf16-cast", "bf16-autocast"):
        assert THRESHOLDS[mode].frame == 0.998
        assert THRESHOLDS[mode].patch_mean == 0.995
        assert THRESHOLDS[mode].patch_p01 == 0.96
