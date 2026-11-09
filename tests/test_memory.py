import logging
import os

import pytest

from dinov3_mlx.memory import InsufficientMemoryError, available_memory_bytes, check_free_memory


@pytest.fixture(autouse=True)
def _no_memory_log(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the lines of these memory checks out of memory-checks.log of the test session.

    The backbone slot can attach its file handler before these tests run (F12 of
    specs/05-all-sizes/verify.html).
    """
    monkeypatch.setattr(logging.getLogger("dinov3_mlx.memory"), "handlers", [])


def test_available_memory_is_positive_and_below_total() -> None:
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    assert 0 < available_memory_bytes() < total


def test_check_free_memory_raises_when_memory_is_too_small() -> None:
    with pytest.raises(InsufficientMemoryError):
        check_free_memory(required_bytes=10**15, label="too large")


def test_check_free_memory_passes_when_memory_is_enough() -> None:
    assert check_free_memory(required_bytes=1, label="one byte") > 1


VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                    10.
Pages active:                                1000.
Pages inactive:                               100.
Pages speculative:                              5.
Pages purgeable:                                3.
File-backed pages:                             40.
Anonymous pages:                             2000.
"""


def test_available_memory_counts_only_pages_without_compression() -> None:
    # free + speculative + purgeable + inactive pages up to the file-backed pages.
    assert available_memory_bytes(VM_STAT) == (10 + 5 + 3 + 40) * 16384


def test_available_memory_counts_all_inactive_pages_when_file_backed() -> None:
    text = VM_STAT.replace(
        "File-backed pages:                             40.",
        "File-backed pages:                            500.",
    )
    assert available_memory_bytes(text) == (10 + 5 + 3 + 100) * 16384
