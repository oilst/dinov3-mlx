import os

import pytest

from dinov3_apple.memory import InsufficientMemoryError, available_memory_bytes, check_free_memory


def test_available_memory_is_positive_and_below_total() -> None:
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    assert 0 < available_memory_bytes() < total


def test_check_free_memory_raises_when_memory_is_too_small() -> None:
    with pytest.raises(InsufficientMemoryError):
        check_free_memory(required_bytes=10**15, label="too large")


def test_check_free_memory_passes_when_memory_is_enough() -> None:
    assert check_free_memory(required_bytes=1, label="one byte") > 1
