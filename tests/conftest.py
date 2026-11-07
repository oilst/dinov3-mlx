"""Shared test setup. The MPS watermarks must be set before torch is imported (CLAUDE.md).

The fp32 dtype mode of mlx runs with TF32 off (D7 of specs/04-mlx-port/explore.html). MLX reads
MLX_ENABLE_TF32 once, at its first use, so the value goes into the environment before the first
import of mlx.
"""

import os

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.6")
os.environ["MLX_ENABLE_TF32"] = "0"

import gc
import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlx.core as mx
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

PARITY_DIR = Path(__file__).parent / "fixtures" / "generated" / "parity"
MEMORY_LOG = PARITY_DIR / "memory-checks.log"
# The measured values of the mlx tests without a model size, next to their targets.
MLX_REPORT = Path(__file__).parent / "fixtures" / "generated" / "mlx" / "report.jsonl"

# The package whose version goes into the lines of each backend.
BACKEND_PACKAGES = {"torch-mps": ("torch", torch.__version__), "mlx": ("mlx", mx.__version__)}

# The MLX memory limit of the test session. The tests without the gated weights need less than
# 100 MB. load_mlx_backbone sets the limit to the need of its load (Section 3.7 of the plan).
MLX_SESSION_LIMIT_BYTES = 4 * 10**9
mx.set_memory_limit(MLX_SESSION_LIMIT_BYTES)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Group the tests that need a model by model size, backend and dtype mode.

    The backbone fixture keeps one model in memory. This order loads each model once. The mlx
    tests without a model size come first, so the TF32 probe (two small subprocesses) runs
    before the first model load of the session.
    """
    model_order = list(MODEL_IDS)

    def key(item: pytest.Item) -> tuple[int, str, str]:
        callspec = getattr(item, "callspec", None)
        params = callspec.params if callspec else {}
        mlx = item.get_closest_marker("mlx") is not None
        if "model_size" not in params:
            return (-2 if mlx else -1, "", "")
        backend = "mlx" if mlx else "torch"
        return (model_order.index(params["model_size"]), backend, params.get("dtype_mode", ""))

    items.sort(key=key)


class _BackboneSlot:
    """Holds at most one backbone, torch or mlx. A new load frees the previous one first."""

    def __init__(self) -> None:
        self.key: tuple[str, str, str] | None = None
        self.model: Any = None
        self.handler = logging.FileHandler(MEMORY_LOG)
        self.handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logging.getLogger("dinov3_apple.memory").addHandler(self.handler)
        logging.getLogger("dinov3_apple.memory").setLevel(logging.INFO)

    def free(self) -> None:
        self.model, self.key = None, None
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        mx.clear_cache()

    def get(self, model_size: str, device: str, dtype_mode: str) -> Any:
        key = (model_size, device, dtype_mode)
        if self.key != key:
            self.free()
            try:
                if device == "mlx":
                    from dinov3_apple.mlx_weights import load_mlx_backbone

                    if os.environ.get("DINOV3_MLX_DEVICE") == "cpu":
                        mx.set_default_device(mx.cpu)
                    self.model = load_mlx_backbone(model_size, dtype_mode)
                else:
                    self.model = load_torch_backbone(model_size, device, dtype_mode)
            except (WeightsNotFoundError, InsufficientMemoryError) as error:
                pytest.skip(str(error))
            self.key = key
        return self.model


@pytest.fixture(scope="session")
def backbone() -> Iterator[Callable[[str, str, str], Any]]:
    """Return a loader that keeps one model in memory and checks the free memory before a load.

    The device is "cpu" or "mps" for the Hugging Face backbone on torch, or "mlx" for the MLX
    backbone. With DINOV3_MLX_DEVICE=cpu the MLX backbone runs on the CPU stream (a diagnostic).
    """
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
    """Return a writer that appends one JSON line per measured configuration.

    The lines of a backend go to tests/fixtures/generated/parity/{backend}.jsonl, with the version
    of the backend package ("torch" or "mlx").
    """
    PARITY_DIR.mkdir(parents=True, exist_ok=True)

    def write(
        kind: str,
        model_size: str,
        dtype_mode: str,
        config: str,
        parity: Parity | None,
        status: str,
        reason: str = "",
        backend: str = "torch-mps",
        values: dict | None = None,
    ) -> None:
        package, package_version = BACKEND_PACKAGES[backend]
        line = {
            "kind": kind,
            "backend": backend,
            "model_size": model_size,
            "dtype_mode": dtype_mode,
            "config": config,
            "status": status,
            "reason": reason,
            "parity": asdict(parity) if parity else None,
            "values": values,
            package: package_version,
            "time": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        with (PARITY_DIR / f"{backend}.jsonl").open("a") as file:
            file.write(json.dumps(line) + "\n")

    return write


@pytest.fixture(scope="session")
def report() -> Callable[..., None]:
    """Return a writer that appends the measured values of one mlx test to MLX_REPORT."""
    MLX_REPORT.parent.mkdir(parents=True, exist_ok=True)

    def write(test: str, **values: Any) -> None:
        line = {"test": test, **values, "mlx": mx.__version__}
        line["time"] = datetime.now(UTC).isoformat(timespec="seconds")
        with MLX_REPORT.open("a") as file:
            file.write(json.dumps(line) + "\n")

    return write
