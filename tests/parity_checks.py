"""Shared checks of the parity tests of the backends torch-mps and mlx."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np
import pytest

from dinov3_apple.parity import THRESHOLDS, Parity
from dinov3_apple.stored import configurations

CONFIG_IDS = [config.id for config in configurations()]
# The batch of 8: the three images at 224 x 224 px, repeated.
BATCH_CONFIGS = [CONFIG_IDS[i % 3 * 4] for i in range(8)]


@contextmanager
def record_skip(
    record: Callable[..., None], *key: str, backend: str = "torch-mps"
) -> Iterator[None]:
    """Record a skip, for example a missing reference file or a failed memory check."""
    try:
        yield
    except pytest.skip.Exception as skip:
        record(*key, None, "skipped", str(skip), backend=backend)
        raise


def assert_thresholds(parity: Parity, dtype_mode: str) -> None:
    """Assert the frame, patch mean and patch 1st percentile thresholds of one dtype mode."""
    threshold = THRESHOLDS[dtype_mode]
    assert parity.frame >= threshold.frame
    assert parity.patch_mean >= threshold.patch_mean
    assert parity.patch_p01 >= threshold.patch_p01


def min_token_cosine(reference: np.ndarray, output: np.ndarray) -> float:
    """Return the smallest cosine similarity of two token sequences, token by token, in fp64.

    Args:
        reference: Tokens with shape (T, D).
        output: Tokens with shape (T, D).

    Returns:
        The minimum over the T tokens. The cosine does not depend on the scale of a token.
    """
    ref, out = reference.astype(np.float64), output.astype(np.float64)
    cos = (ref * out).sum(-1) / (np.linalg.norm(ref, axis=-1) * np.linalg.norm(out, axis=-1))
    return float(np.minimum(cos, 1.0).min())
