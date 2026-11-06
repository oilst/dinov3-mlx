"""Free memory on macOS and the memory check before each model load.

This Mac crashed from out-of-memory before (CLAUDE.md). Every model load calls check_free_memory
first.
"""

import logging
import re
import subprocess

logger = logging.getLogger(__name__)

GB = 1e9


class InsufficientMemoryError(MemoryError):
    """A model load needs more memory than is available."""


def available_memory_bytes(vm_stat_text: str | None = None) -> int:
    """Return the memory that a new allocation can get without compression or swap, from vm_stat.

    The sum counts the free, speculative and purgeable pages, plus the inactive pages up to the
    number of file-backed pages. macOS can drop file-backed pages at once. The other inactive pages
    hold app memory that macOS must compress or swap before it can reuse them, so they do not count.

    Args:
        vm_stat_text: The output of vm_stat. None runs vm_stat.

    Returns:
        The available memory in bytes.
    """
    if vm_stat_text is None:
        vm_stat_text = subprocess.run(
            ["vm_stat"], capture_output=True, text=True, check=True
        ).stdout
    page_size = int(re.search(r"page size of (\d+) bytes", vm_stat_text).group(1))
    pages = {
        name: int(count)
        for name, count in re.findall(
            r"^(Pages free|Pages inactive|Pages speculative|Pages purgeable|File-backed pages):"
            r"\s+(\d+)\.",
            vm_stat_text,
            flags=re.MULTILINE,
        )
    }
    droppable_inactive = min(pages["Pages inactive"], pages["File-backed pages"])
    free = pages["Pages free"] + pages["Pages speculative"] + pages["Pages purgeable"]
    return (free + droppable_inactive) * page_size


def check_free_memory(required_bytes: int, label: str) -> int:
    """Log the memory check of one model load and stop the load if it does not fit.

    Args:
        required_bytes: The memory that the load and the run need.
        label: The name of the load in the log line, for example "ViT-S/16 torch-cpu fp32".

    Returns:
        The available memory in bytes.

    Raises:
        InsufficientMemoryError: If the available memory is less than required_bytes.
    """
    available = available_memory_bytes()
    fits = available >= required_bytes
    message = (
        f"memory check: {label}: needs {required_bytes / GB:.1f} GB, "
        f"{available / GB:.1f} GB available, {'ok' if fits else 'does not fit'}"
    )
    logger.info(message)
    if not fits:
        raise InsufficientMemoryError(message)
    return available
