"""Shared test setup. The MPS watermarks must be set before torch is imported (CLAUDE.md)."""

import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")

import gc
import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import torch

from dinov3_apple.hub import MODEL_IDS, WeightsNotFoundError
from dinov3_apple.memory import InsufficientMemoryError
from dinov3_apple.parity import Parity
from dinov3_apple.reference import (
    INPUTS_FILE,
    load_torch_backbone,
    read_inputs,
    read_reference,
    reference_path,
)

PARITY_LOG = Path(__file__).parent / "fixtures" / "generated" / "parity" / "torch-mps.jsonl"
MEMORY_LOG = PARITY_LOG.with_name("memory-checks.log")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Group the tests that need a model by model size and dtype mode.

    The backbone fixture keeps one model in memory. This order loads each model once.
    """
    model_order = list(MODEL_IDS)

    def key(item: pytest.Item) -> tuple[int, str]:
        callspec = getattr(item, "callspec", None)
        params = callspec.params if callspec else {}
        if "model_size" not in params:
            return (-1, "")
        return (model_order.index(params["model_size"]), params.get("dtype_mode", ""))

    items.sort(key=key)


class _BackboneSlot:
    """Holds at most one backbone. A new load frees the previous one first."""

    def __init__(self) -> None:
        self.key: tuple[str, str, str] | None = None
        self.model: torch.nn.Module | None = None
        self.handler = logging.FileHandler(MEMORY_LOG)
        self.handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logging.getLogger("dinov3_apple.memory").addHandler(self.handler)
        logging.getLogger("dinov3_apple.memory").setLevel(logging.INFO)

    def free(self) -> None:
        self.model, self.key = None, None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def get(self, model_size: str, device: str, dtype_mode: str) -> torch.nn.Module:
        key = (model_size, device, dtype_mode)
        if self.key != key:
            self.free()
            try:
                self.model = load_torch_backbone(model_size, device, dtype_mode)
            except (WeightsNotFoundError, InsufficientMemoryError) as error:
                pytest.skip(str(error))
            self.key = key
        return self.model


@pytest.fixture(scope="session")
def backbone() -> Iterator[Callable[[str, str, str], torch.nn.Module]]:
    """Return a loader that keeps one model in memory and checks the free memory before a load."""
    MEMORY_LOG.parent.mkdir(parents=True, exist_ok=True)
    slot = _BackboneSlot()
    yield slot.get
    slot.free()
    logging.getLogger("dinov3_apple.memory").removeHandler(slot.handler)


@pytest.fixture(scope="session")
def inputs() -> dict[str, np.ndarray]:
    """Return the stored pixels, keyed by configuration id."""
    if not INPUTS_FILE.exists():
        pytest.skip(f"{INPUTS_FILE} is missing. Run scripts/make_reference.py.")
    return read_inputs()


@pytest.fixture(scope="session")
def reference() -> Callable[[str], dict[str, np.ndarray]]:
    """Return a reader of the reference tokens of one model size. It keeps one file in memory."""
    cache: dict[str, dict[str, np.ndarray]] = {}

    def read(model_size: str) -> dict[str, np.ndarray]:
        if model_size not in cache:
            if not reference_path(model_size).exists():
                pytest.skip(f"{reference_path(model_size).name} is missing.")
            cache.clear()
            cache[model_size] = read_reference(model_size)[0]
        return cache[model_size]

    return read


@pytest.fixture(scope="session")
def record() -> Callable[..., None]:
    """Return a writer that appends one JSON line per measured configuration to PARITY_LOG."""
    PARITY_LOG.parent.mkdir(parents=True, exist_ok=True)

    def write(
        kind: str,
        model_size: str,
        dtype_mode: str,
        config: str,
        parity: Parity | None,
        status: str,
        reason: str = "",
    ) -> None:
        line = {
            "kind": kind,
            "backend": "torch-mps",
            "model_size": model_size,
            "dtype_mode": dtype_mode,
            "config": config,
            "status": status,
            "reason": reason,
            "parity": asdict(parity) if parity else None,
            "torch": torch.__version__,
            "time": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        with PARITY_LOG.open("a") as file:
            file.write(json.dumps(line) + "\n")

    return write
