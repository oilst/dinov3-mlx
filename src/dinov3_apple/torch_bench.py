"""The worker of one benchmark configuration on torch-mps (Sections 3.3 to 3.6 of
specs/03-baseline/plan.html).

The worker runs in its own subprocess, so the lifetime maximum of its footprint belongs to one
configuration. The driver sets the MPS watermarks in the environment before torch starts.
"""

import json
import time
from collections.abc import Callable
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open
from transformers import DINOv3ViTModel

from dinov3_apple.bench import (
    TIMED,
    WARM_UP,
    BenchConfig,
    TimeBudget,
    check_validity,
    keep_going,
    make_batch,
    summarize,
)
from dinov3_apple.footprint import read_footprint
from dinov3_apple.hub import MODEL_IDS
from dinov3_apple.machine import read_versions
from dinov3_apple.outputs import NUM_PREFIX_TOKENS
from dinov3_apple.parity import TEST_IMAGES
from dinov3_apple.reference import (
    DTYPE_MODES,
    REPO_ROOT,
    DtypeMode,
    load_torch_backbone,
    read_inputs,
    reference_path,
)

BENCH_DIR = REPO_ROOT / "tests" / "fixtures" / "generated" / "bench"


def _autocast(mode: DtypeMode) -> torch.autocast | nullcontext:
    if mode.autocast is None:
        return nullcontext()
    return torch.autocast(device_type="mps", dtype=mode.autocast)


@torch.inference_mode()
def time_batch(
    model: DINOv3ViTModel, batch: np.ndarray, mode: DtypeMode
) -> tuple[tuple[float, float, float], torch.Tensor, torch.Tensor]:
    """Run one batch host to host and time it (Table 4 of the plan).

    Args:
        model: A backbone on torch-mps.
        batch: fp32 pixels with shape (B, 3, H, W) in NumPy.
        mode: The dtype mode of the backbone.

    Returns:
        The device time, the frame time and the patch time in seconds, then the frame embeddings
        (B, D) and the patch embeddings (B, N, D) on the host, in the dtype of the backend output.
        They stay torch tensors because NumPy has no bf16; .numpy() of them costs no copy.
    """
    t0 = time.perf_counter()
    pixels = torch.from_numpy(batch).to("mps", mode.weights)
    with _autocast(mode):
        tokens = model(pixel_values=pixels).last_hidden_state
    torch.mps.synchronize()
    t1 = time.perf_counter()
    frame = tokens[:, 0].cpu()
    t2 = time.perf_counter()
    patches = tokens[:, NUM_PREFIX_TOKENS:].cpu()
    t3 = time.perf_counter()
    return (t1 - t0, t2 - t0, (t1 - t0) + (t3 - t2)), frame, patches


def _reference_tokens(model_size: str, side: int) -> np.ndarray | None:
    path = reference_path(model_size)
    if not path.exists():
        return None
    with safe_open(path, "np") as tensors:
        return tensors.get_tensor(f"{TEST_IMAGES[0]}/{side}x{side}/tokens")


def _phase(
    model: DINOv3ViTModel,
    batch: np.ndarray,
    mode: DtypeMode,
    budget: TimeBudget,
    on_batch: Callable[[], object] = lambda: None,
) -> list[tuple[float, float, float]]:
    times: list[tuple[float, float, float]] = []
    elapsed = 0.0
    while keep_going(budget, len(times), elapsed):
        batch_times, _, _ = time_batch(model, batch, mode)
        times.append(batch_times)
        elapsed += batch_times[2]
        on_batch()
    return times


def run_worker(
    config: BenchConfig,
    warm_up: TimeBudget = WARM_UP,
    timed: TimeBudget = TIMED,
    bench_dir: Path = BENCH_DIR,
) -> dict:
    """Load the backbone, check its first output, warm up, time the batches and return the result.

    The elapsed time of a phase is the sum of the patch times, the longest of the three times.

    Args:
        config: The configuration.
        warm_up: The time budget of the warm-up batches.
        timed: The time budget of the timed batches.
        bench_dir: The folder for the per-batch times.

    Returns:
        The worker fields of LINE_FIELDS, with status "done".

    Raises:
        ValueError: For ViT-7B/16 with fp32 weights (CLAUDE.md run rules).
        InsufficientMemoryError: If the memory check of the load fails.
    """
    mode = DTYPE_MODES[config.dtype_mode]
    if config.model_size == "ViT-7B/16" and mode.weights == torch.float32:
        raise ValueError("ViT-7B/16 runs on torch-mps with fp16 or bf16 weights only")
    start_time = datetime.now(UTC).isoformat(timespec="seconds")
    base = read_footprint()

    start = time.perf_counter()
    model = load_torch_backbone(config.model_size, "mps", config.dtype_mode)
    torch.mps.synchronize()
    load_time_s = time.perf_counter() - start
    attn_implementation = model.config._attn_implementation
    if config.compile:
        model = torch.compile(model)
    load_peak = read_footprint().lifetime_max_bytes

    batch = make_batch(read_inputs(), config.side, config.batch_size)
    (_, _, first_call_s), frame, patches = time_batch(model, batch, mode)
    output_dtype = str(frame.dtype).removeprefix("torch.")
    validity = check_validity(
        frame[0].float().numpy(),
        patches[0].float().numpy(),
        _reference_tokens(config.model_size, config.side),
        config.dtype_mode,
    )
    del frame, patches

    warm = _phase(model, batch, mode, warm_up)
    driver_max = [torch.mps.driver_allocated_memory()]
    timed_times = _phase(
        model, batch, mode, timed, lambda: driver_max.append(torch.mps.driver_allocated_memory())
    )
    end = read_footprint()

    device, frame_t, patch_t = (list(column) for column in zip(*timed_times, strict=True))
    bench_dir.mkdir(parents=True, exist_ok=True)
    slug = MODEL_IDS[config.model_size].split("/")[1]
    name = f"{slug}_{config.dtype_mode}_{config.side}_b{config.batch_size}"
    name += f"{'_compile' if config.compile else ''}_r{config.repeat}.json"
    (bench_dir / name).write_text(
        json.dumps({"config": config.key, "device": device, "frame": frame_t, "patch": patch_t})
    )
    above_load = end.lifetime_max_bytes > load_peak
    return {
        "status": "done",
        "reason": "",
        "valid": validity.valid,
        "finite": validity.finite,
        "parity": None if validity.parity is None else validity.parity.__dict__,
        "validity_note": validity.note,
        "load_time_s": load_time_s,
        "first_call_s": first_call_s,
        "warm_up_batches": len(warm),
        "warm_up_s": sum(t[2] for t in warm),
        "timed_batches": len(timed_times),
        "timed_s": sum(patch_t),
        "frame": summarize(frame_t, config.batch_size),
        "patch": summarize(patch_t, config.batch_size),
        "device": summarize(device, config.batch_size),
        "output_dtype": output_dtype,
        "footprint_base_bytes": base.current_bytes,
        "load_peak_bytes": load_peak,
        "peak_bytes": end.lifetime_max_bytes,
        "run_peak_bytes": end.lifetime_max_bytes if above_load else None,
        "run_peak_note": "" if above_load else f"below the load peak; end {end.current_bytes}",
        "mps_driver_max_bytes": max(driver_max),
        "versions": read_versions(),
        "attn_implementation": attn_implementation,
        "start_time": start_time,
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
    }
