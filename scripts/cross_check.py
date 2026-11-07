"""Cross-check of mlx against torch in the same dtype mode, without the reference output.

D4 Option C of specs/05-all-sizes/explore.html (path D4-N) and the bf16 diagnostic of risk R23.
Report only: no threshold applies. The parity of two backends does not measure the parity against
the fp32 reference output.

A torch worker runs first, in its own subprocess. It loads the Hugging Face backbone on torch-mps
or torch-cpu, runs each configuration at batch size 1 and stores the tokens in
tests/fixtures/generated/cross-check/. Then a separate mlx worker loads the MLX backbone, runs the
same configurations and compares its tokens with the stored torch tokens. It appends one line of
kind "cross-check" for each configuration to tests/fixtures/generated/parity/mlx.jsonl.

Before the workers and before any read of the free memory or of a header, the driver checks the
run rules of CLAUDE.md for both workers. A refused run writes the refusal to driver.log. Before
each worker, the driver checks the free memory. Each worker checks the run rule and the free
memory again at its load and sends its memory-check and load-peak lines to the driver, which
writes them to driver.log. The torch worker runs with the MPS watermarks 0.8 and 0.6. The mlx
worker runs with MLX_ENABLE_TF32=0 and imports no torch.

Usage:
    uv run python scripts/cross_check.py --model-size ViT-7B/16 --dtype-mode bf16-cast \
        --torch-device mps --dry-run
    uv run python scripts/cross_check.py --model-size ViT-7B/16 --dtype-mode bf16-cast \
        --torch-device mps
    uv run python scripts/cross_check.py --model-size ViT-H+/16 --dtype-mode bf16-cast \
        --torch-device cpu --configs forest/224x224
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from bench_mlx_port import worker_env
from safetensors import safe_open

from dinov3_apple.bench import DTYPE_MODE_NAMES, bytes_per_weight, run_worker_subprocess
from dinov3_apple.footprint import read_footprint
from dinov3_apple.gpu_lock import heavy_job
from dinov3_apple.hub import MODEL_IDS, parameter_count
from dinov3_apple.memory import (
    GB,
    RunRuleError,
    available_memory_bytes,
    check_run_rule,
    load_bytes,
    send_checks_to,
)
from dinov3_apple.parity import measure_parity
from dinov3_apple.stored import configurations, model_slug, read_inputs

REPO_ROOT = Path(__file__).parents[1]
CROSS_DIR = REPO_ROOT / "tests" / "fixtures" / "generated" / "cross-check"
PARITY_LOG = REPO_ROOT / "tests" / "fixtures" / "generated" / "parity" / "mlx.jsonl"
LOG_FORMAT = "%(asctime)s %(name)s %(message)s"
WORKER_TIMEOUT_S = 1800
STDERR_LINES = 20

logger = logging.getLogger("cross_check")


def tokens_path(model_size: str, dtype_mode: str, torch_device: str) -> Path:
    """Return the file of the stored torch tokens of one cross-check."""
    return CROSS_DIR / f"{model_slug(model_size)}-{dtype_mode}-torch-{torch_device}.safetensors"


def worker_environment(backend: str, base: dict[str, str]) -> dict[str, str]:
    """Return the environment of one worker: the MPS watermarks for torch (also on torch-cpu),
    MLX_ENABLE_TF32=0 for mlx (worker_env of scripts/bench_mlx_port.py)."""
    return worker_env("mlx" if backend == "mlx" else "torch-mps", False, base)


def worker_plan(
    model_size: str, dtype_mode: str, torch_device: str
) -> list[tuple[str, int, dict[str, str]]]:
    """Return (backend, need, the variables that the worker environment sets) of the two
    workers, torch first.

    The run rules come first, before parameter_count reads a header.

    Raises:
        RunRuleError: If one of the two loads breaks a run rule.
    """
    sizes = {
        backend: bytes_per_weight(backend, dtype_mode)
        for backend in (f"torch-{torch_device}", "mlx")
    }
    for backend, size in sizes.items():
        check_run_rule(model_size, size, f"{model_size} {backend} {dtype_mode}")
    count = parameter_count(model_size)
    return [
        (backend, load_bytes(count * size), worker_environment(backend, {}))
        for backend, size in sizes.items()
    ]


def dry_run(model_size: str, dtype_mode: str, torch_device: str, configs: list[str]) -> None:
    """Print the two workers, their needs and their environments. Load no model.

    The run-rule guard of worker_plan comes before the read of the free memory.
    """
    plan = worker_plan(model_size, dtype_mode, torch_device)
    available = available_memory_bytes()
    print(f"available memory: {available / GB:.1f} GB")
    for index, (backend, need, env) in enumerate(plan, start=1):
        fits = "fits" if need <= available else "does not fit"
        variables = " ".join(f"{key}={value}" for key, value in sorted(env.items()))
        print(
            f"worker {index}: {backend} {model_size} {dtype_mode}: need {need / GB:.1f} GB, "
            f"{fits}; env {variables}"
        )
    print(f"configurations: {len(configs)}")


def compare_tokens(
    torch_tokens: np.ndarray, mlx_tokens: np.ndarray
) -> tuple[str, dict | None, dict[str, float | bool]]:
    """Compare the tokens of one configuration on the two backends.

    Returns:
        The status ("measured" or "non-finite"), the parity of mlx against torch (None for a
        non-finite output) and the values of the line.
    """
    finite = {
        "torch_finite": bool(np.isfinite(torch_tokens).all()),
        "mlx_finite": bool(np.isfinite(mlx_tokens).all()),
    }
    if not all(finite.values()):
        return "non-finite", None, finite
    parity = measure_parity(torch_tokens, mlx_tokens)
    largest = float(np.abs(mlx_tokens.astype(np.float64) - torch_tokens).max())
    return "measured", asdict(parity), finite | {"max_abs": largest}


def run_torch_worker(spec: dict) -> dict:
    """Load the torch backbone, run the configurations and store the tokens."""
    import torch
    from safetensors.numpy import save_file

    from dinov3_apple.reference import load_torch_backbone, run_torch

    inputs = read_inputs()
    start = time.perf_counter()
    model = load_torch_backbone(spec["model_size"], spec["torch_device"], spec["dtype_mode"])
    tokens = {
        config: np.ascontiguousarray(run_torch(model, inputs[config], spec["dtype_mode"])[0])
        for config in spec["configs"]
    }
    run_time_s = time.perf_counter() - start
    path = tokens_path(spec["model_size"], spec["dtype_mode"], spec["torch_device"])
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tokens, path, metadata={"torch": torch.__version__})
    return {
        "status": "done",
        "file": path.name,
        "run_time_s": round(run_time_s, 2),
        "peak_bytes": read_footprint().lifetime_max_bytes,
    }


def run_mlx_worker(spec: dict) -> dict:
    """Load the MLX backbone, run the configurations, compare them with the stored torch tokens
    and append one cross-check line for each configuration."""
    import mlx.core as mx

    from dinov3_apple.mlx_model import compile_backbone, run_mlx
    from dinov3_apple.mlx_weights import load_mlx_backbone

    model_size, dtype_mode = spec["model_size"], spec["dtype_mode"]
    path = tokens_path(model_size, dtype_mode, spec["torch_device"])
    with safe_open(path, "np") as stored:
        torch_version = stored.metadata().get("torch")
        torch_tokens = {config: stored.get_tensor(config) for config in spec["configs"]}
    inputs = read_inputs()
    start = time.perf_counter()
    model = load_mlx_backbone(model_size, dtype_mode, tf32=False)
    forward = compile_backbone(model)  # the compiled forward pass (D4 of specs/06-speed/plan.html)
    PARITY_LOG.parent.mkdir(parents=True, exist_ok=True)
    statuses = []
    for config in spec["configs"]:
        tokens = run_mlx(model, inputs[config], dtype_mode, forward)[0]
        status, parity, values = compare_tokens(torch_tokens[config], tokens)
        line = {
            "kind": "cross-check",
            "backend": "mlx",
            "model_size": model_size,
            "dtype_mode": dtype_mode,
            "config": config,
            "status": status,
            "reason": "",
            "parity": parity,
            "values": values | {"torch_device": spec["torch_device"], "torch": torch_version},
            "mlx": mx.__version__,
            "time": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        with PARITY_LOG.open("a") as file:
            file.write(json.dumps(line) + "\n")
        statuses.append(status)
    return {
        "status": "done",
        "run_time_s": round(time.perf_counter() - start, 2),
        "lines": len(statuses),
        "non_finite": statuses.count("non-finite"),
        "load_report": asdict(model.load_report),
        "peak_bytes": read_footprint().lifetime_max_bytes,
        "torch_imported": "torch" in sys.modules,
    }


def worker_main(backend: str, spec_json: str) -> None:
    """Run one worker in this process and print the result as one JSON line."""
    from dinov3_apple.memory import InsufficientMemoryError

    send_checks_to(sys.stderr)
    spec = json.loads(spec_json)
    try:
        result = run_mlx_worker(spec) if backend == "mlx" else run_torch_worker(spec)
    except (InsufficientMemoryError, RunRuleError) as error:
        result = {"status": "skipped", "reason": f"load check in the worker: {error}"}
    print(json.dumps(result))


def run_worker(backend: str, spec: dict) -> dict:
    """Run one worker in a new subprocess and return its result. The memory-check and load-peak
    lines of the worker go to the log of the driver (run_worker_subprocess)."""
    worker = "mlx" if backend == "mlx" else "torch"
    command = [sys.executable, __file__, "--worker", worker, "--spec", json.dumps(spec)]
    env = worker_environment(backend, dict(os.environ))
    return run_worker_subprocess(command, env, WORKER_TIMEOUT_S, STDERR_LINES)


def run(model_size: str, dtype_mode: str, torch_device: str, configs: list[str]) -> int:
    """Run the torch worker, then the mlx worker, one after the other.

    Returns:
        0 if both workers are done, else 1.
    """
    spec = {
        "model_size": model_size,
        "dtype_mode": dtype_mode,
        "torch_device": torch_device,
        "configs": configs,
    }
    for backend, need, _ in worker_plan(model_size, dtype_mode, torch_device):
        available = available_memory_bytes()
        fits = available >= need
        logger.info(
            f"memory check: {backend} {model_size} {dtype_mode} (cross_check.py): needs "
            f"{need / GB:.1f} GB, {available / GB:.1f} GB available, "
            f"{'ok' if fits else 'does not fit'}"
        )
        if not fits:
            logger.warning(f"{backend}: skipped. The cross-check stops.")
            return 1
        start = time.perf_counter()
        result = run_worker(backend, spec)
        logger.info(
            f"{backend}: {result['status']} in {time.perf_counter() - start:.1f} s: "
            f"{json.dumps({k: v for k, v in result.items() if k != 'status'})}"
        )
        if result["status"] != "done":
            return 1
    return 0


def main(argv: list[str] | None = None) -> None:
    """Parse the arguments and run the cross-check."""
    config_ids = [config.id for config in configurations()]
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    # Required for the driver only: a worker gets its model size from --spec.
    parser.add_argument("--model-size", choices=list(MODEL_IDS))
    parser.add_argument("--dtype-mode", choices=DTYPE_MODE_NAMES, default="bf16-cast")
    parser.add_argument("--torch-device", choices=["mps", "cpu"], default="mps")
    parser.add_argument("--configs", nargs="+", choices=config_ids, default=config_ids)
    parser.add_argument("--dry-run", action="store_true", help="print the plan, load no model")
    parser.add_argument("--worker", choices=["torch", "mlx"], help=argparse.SUPPRESS)
    parser.add_argument("--spec", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker:
        worker_main(args.worker, args.spec)
        return
    if args.model_size is None:
        parser.error("the following arguments are required: --model-size")
    if args.dry_run:
        try:
            dry_run(args.model_size, args.dtype_mode, args.torch_device, args.configs)
        except RunRuleError as error:
            raise SystemExit(str(error)) from error
        return
    CROSS_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(CROSS_DIR / "driver.log")],
    )
    with heavy_job("scripts/cross_check.py"):
        try:
            code = run(args.model_size, args.dtype_mode, args.torch_device, args.configs)
        except RunRuleError as error:
            logger.error(f"refused: {error}")  # the refusal goes to driver.log
            raise SystemExit(str(error)) from error
    sys.exit(code)


if __name__ == "__main__":
    main()
