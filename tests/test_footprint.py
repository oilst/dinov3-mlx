import os

import pytest
import torch

from dinov3_mlx.footprint import read_footprint

MB = 10**6


def test_read_footprint_lifetime_maximum_is_at_least_current() -> None:
    footprint = read_footprint()
    assert footprint.current_bytes > 0
    assert footprint.lifetime_max_bytes >= footprint.current_bytes
    assert read_footprint(os.getpid()).current_bytes > 0


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no torch-mps")
def test_read_footprint_counts_mps_tensor() -> None:
    before = read_footprint()
    tensor = torch.ones(256 * MB // 4, dtype=torch.float32, device="mps")
    torch.mps.synchronize()
    after = read_footprint()
    del tensor
    torch.mps.empty_cache()
    assert after.current_bytes - before.current_bytes >= 230 * MB
    assert after.lifetime_max_bytes >= after.current_bytes
