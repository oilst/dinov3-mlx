"""Free memory on macOS, the run rules and the memory check before each model load.

This Mac crashed from out-of-memory before (CLAUDE.md). Every model load calls check_run_rule and
then check_free_memory first. No torch and no mlx import.
"""

import logging
import re
import subprocess
from typing import TextIO

logger = logging.getLogger(__name__)

GB = 1e9

# The first word of a memory-check line that a worker subprocess sends to its driver.
FORWARD_PREFIX = "memory-check-record"

# The memory check of a model load: LOAD_RATIO x the weight bytes + LOAD_RESERVE_BYTES.
LOAD_RATIO = 1.2
LOAD_RESERVE_BYTES = 4 * 10**9

# CLAUDE.md, run rules: ViT-7B/16 runs in fp16, bf16 or with quantized weights. In fp32 its
# weights alone need about 27 GB.
NO_FP32_WEIGHTS = frozenset({"ViT-7B/16"})


class InsufficientMemoryError(MemoryError):
    """A model load needs more memory than is available."""


class RunRuleError(RuntimeError):
    """A load breaks a run rule of CLAUDE.md. The load stops before the memory check."""


def check_run_rule(model_size: str, bytes_per_weight: int, label: str) -> None:
    """Stop a load that breaks a run rule, before the memory check and before any file read.

    Args:
        model_size: A key of MODEL_IDS.
        bytes_per_weight: The bytes of each loaded weight: 4 for fp32 weights, 2 for fp16 or bf16.
        label: The name of the load, for example "ViT-7B/16 torch-cpu fp32".

    Raises:
        RunRuleError: For a model size in NO_FP32_WEIGHTS with 4 bytes for each weight.
    """
    if model_size in NO_FP32_WEIGHTS and bytes_per_weight >= 4:
        raise RunRuleError(
            f"{label}: {model_size} runs with fp16, bf16 or quantized weights only "
            "(CLAUDE.md run rules)"
        )


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


def record_load_peak(label: str, peak_bytes: int, cache_bytes: int) -> None:
    """Log the load peak of one mlx load on the logger of the memory checks.

    The line goes where the memory-check lines go: memory-checks.log of the tests, or the driver
    log of a benchmark through send_checks_to and replay_checks.

    Args:
        label: The name of the load, for example "ViT-S/16 mlx bf16-cast".
        peak_bytes: The load peak in bytes.
        cache_bytes: The MLX cache memory after the load.
    """
    logger.info(
        f"load peak: {label}: {peak_bytes / 1e6:.1f} MB, cache {cache_bytes} bytes after the load"
    )


def load_bytes(weight_bytes: int, ratio: float = LOAD_RATIO) -> int:
    """Return the memory that the memory check asks for before a model load.

    Args:
        weight_bytes: The bytes of the loaded weights.
        ratio: The ratio of the load peak to the weight bytes.

    Returns:
        ratio x weight_bytes + LOAD_RESERVE_BYTES.
    """
    return round(ratio * weight_bytes) + LOAD_RESERVE_BYTES


def send_checks_to(stream: TextIO) -> logging.Handler:
    """Send the memory-check lines of this process to stream, for replay_checks in the driver.

    A benchmark worker calls this once, before its model load. Its stderr then carries one line
    for each memory check: FORWARD_PREFIX, the time of the check, the logger name, the message.

    Args:
        stream: The stream of the lines, for example sys.stderr.

    Returns:
        The new handler of the memory-check logger.
    """
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter(f"{FORWARD_PREFIX} %(created).6f %(name)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return handler


def replay_checks(text: str) -> int:
    """Log the memory-check lines of a worker again in this process, with the time of the worker.

    The records go to the handlers of this process, for example to the driver log, in the format
    of these handlers. The other lines of text stay out.

    Args:
        text: The stderr of a worker that called send_checks_to.

    Returns:
        The number of replayed lines.
    """
    count = 0
    for line in text.splitlines():
        prefix, _, rest = line.partition(" ")
        fields = rest.split(" ", 2)
        if prefix != FORWARD_PREFIX or len(fields) != 3:
            continue
        created, name, message = float(fields[0]), fields[1], fields[2]
        record = logging.makeLogRecord(
            {
                "name": name,
                "msg": message,
                "levelno": logging.INFO,
                "levelname": logging.getLevelName(logging.INFO),
                "created": created,
                "msecs": (created % 1) * 1000,
            }
        )
        logging.getLogger(name).handle(record)
        count += 1
    return count
