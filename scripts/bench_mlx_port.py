"""The small paired benchmark of Step 4: ViT-S/16 in fp32 at 224 px on mlx and on torch-mps.

Report only, no target (Section 7 of specs/04-mlx-port/plan.html). The throughput of one
configuration on this Mac changes by up to 2x from one run to the next (finding F1 of
specs/03-baseline/verify.html). So the mlx workers and the torch-mps workers run in alternation in
one session, round after round, and verify.html takes the median of the paired ratios.

Each worker runs in its own subprocess, one after the other. Before each worker the driver reads
the machine state, stops on battery power or in Low Power Mode, and checks the free memory. A
worker that does not fit gives a "skipped" line. The mlx worker imports no torch and runs with
MLX_ENABLE_TF32=0 (D7). The torch-mps worker is the eager worker of Step 3 with the MPS
watermarks 0.8 and 0.6.

Usage:
    uv run python scripts/bench_mlx_port.py --dry-run     # the plan and the needs, no model load
    uv run python scripts/bench_mlx_port.py               # 3 rounds, 12 workers, about 4 min
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from dinov3_apple.bench import (
    TIMED,
    WARM_UP,
    BenchConfig,
    TimeBudget,
    check_validity,
    keep_going,
    make_batch,
    reference_tokens,
    required_bench_bytes,
    result_line,
    summarize,
)
from dinov3_apple.footprint import read_footprint
from dinov3_apple.machine import (
    MachineStateError,
    check_machine_state,
    read_machine_state,
    read_versions,
)
from dinov3_apple.memory import GB, InsufficientMemoryError, available_memory_bytes
from dinov3_apple.outputs import NUM_PREFIX_TOKENS
from dinov3_apple.stored import model_slug, read_inputs

if TYPE_CHECKING:  # the module imports mlx only in the mlx worker
    from dinov3_apple.mlx_model import MlxBackbone

REPO_ROOT = Path(__file__).parents[1]
RESULTS = REPO_ROOT / "results" / "bench-mlx-port.jsonl"
BENCH_DIR = REPO_ROOT / "tests" / "fixtures" / "generated" / "bench" / "mlx-port"
DRIVER_LOG = BENCH_DIR / "driver.log"
WORKER_TIMEOUT_S = 600
STDERR_LINES = 20

MODEL_SIZE = "ViT-S/16"
DTYPE_MODE = "fp32"
SIDE = 224
BATCH_SIZES = (8, 1)  # 8: the priority configuration P1
RUN = "step4"

WATERMARKS = {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.8", "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.6"}
# (backend, tf32). TF32 off in the fp32 dtype mode of mlx (D7); None: not an mlx worker.
WORKERS: list[tuple[str, bool | None]] = [("torch-mps", None), ("mlx", False)]

logger = logging.getLogger("bench_mlx_port")


def worker_plan(rounds: int = 3) -> list[tuple[int, int, str, bool | None]]:
    """Return (round, batch size, backend, tf32) for each worker, in the order of the run.

    The order of the two workers alternates from round to round (finding F1): round 1 torch-mps
    first, round 2 mlx first, round 3 torch-mps first.

    Args:
        rounds: The number of rounds.

    Returns:
        rounds x 2 batch sizes x 2 workers entries.
    """
    plan = []
    for r in range(rounds):
        order = WORKERS[r % 2 :] + WORKERS[: r % 2]
        for batch_size in BATCH_SIZES:
            plan += [(r + 1, batch_size, backend, tf32) for backend, tf32 in order]
    return plan


def worker_env(backend: str, tf32: bool | None, base: Mapping[str, str]) -> dict[str, str]:
    """Return the environment of one worker subprocess.

    Args:
        backend: "torch-mps" or "mlx".
        tf32: The TF32 state of an mlx worker. None for torch-mps.
        base: The environment of the driver.

    Returns:
        torch-mps: base with the MPS watermarks. mlx: base without the MPS watermarks, with
        MLX_ENABLE_TF32 set to the TF32 state (MLX reads it once, at its first use).
    """
    env = {k: v for k, v in base.items() if k not in WATERMARKS and k != "MLX_ENABLE_TF32"}
    if backend == "mlx":
        return env | {"MLX_ENABLE_TF32": str(int(bool(tf32)))}
    return env | WATERMARKS


def bench_config(round_: int, batch_size: int) -> BenchConfig:
    """Return the configuration of one worker. The repeat number is the round minus 1."""
    return BenchConfig(RUN, MODEL_SIZE, DTYPE_MODE, SIDE, batch_size, repeat=round_ - 1)


def time_batch_mlx(
    model: "MlxBackbone", batch: np.ndarray
) -> tuple[tuple[float, float, float], np.ndarray, np.ndarray]:
    """Run one batch host to host on mlx and time it (Table 4 of the Step 3 plan).

    Args:
        model: An MlxBackbone.
        batch: fp32 pixels with shape (B, 3, H, W) in NumPy.

    Returns:
        The device time, the frame time and the patch time in seconds, then the frame embeddings
        (B, D) and the patch embeddings (B, N, D) in NumPy.
    """
    import mlx.core as mx

    t0 = time.perf_counter()
    tokens = model(mx.array(batch))
    mx.eval(tokens)  # mlx synchronizes with mx.eval
    t1 = time.perf_counter()
    frame = np.array(tokens[:, 0])
    t2 = time.perf_counter()
    patches = np.array(tokens[:, NUM_PREFIX_TOKENS:])
    t3 = time.perf_counter()
    return (t1 - t0, t2 - t0, (t1 - t0) + (t3 - t2)), frame, patches


def _phase(
    run: Callable[[], tuple[float, float, float]], budget: TimeBudget
) -> list[tuple[float, float, float]]:
    times: list[tuple[float, float, float]] = []
    elapsed = 0.0
    while keep_going(budget, len(times), elapsed):
        batch_times = run()
        times.append(batch_times)
        elapsed += batch_times[2]
    return times


def run_mlx_worker(
    config: BenchConfig,
    warm_up: TimeBudget = WARM_UP,
    timed: TimeBudget = TIMED,
    bench_dir: Path = BENCH_DIR / "mlx",
) -> dict:
    """Load the MLX backbone, check its first output, warm up, time the batches, return the result.

    The same phases and fields as torch_bench.run_worker of Step 3. TF32 is off (D7).

    Args:
        config: The configuration.
        warm_up: The time budget of the warm-up batches.
        timed: The time budget of the timed batches.
        bench_dir: The folder for the per-batch times.

    Returns:
        The worker fields of LINE_FIELDS with status "done", plus mlx_peak_bytes,
        mlx_memory_limit_bytes (the value that the load gave to mx.set_memory_limit) and tf32.

    Raises:
        TF32ModeError: If MLX_ENABLE_TF32 is not 0 in the environment.
        InsufficientMemoryError: If the memory check of the load fails.
        WeightsNotFoundError: If the weights are not on this Mac.
    """
    import mlx.core as mx

    from dinov3_apple.mlx_weights import load_mlx_backbone

    start_time = datetime.now(UTC).isoformat(timespec="seconds")
    base = read_footprint()
    start = time.perf_counter()
    model = load_mlx_backbone(config.model_size, config.dtype_mode, tf32=False)
    load_time_s = time.perf_counter() - start
    load_peak = read_footprint().lifetime_max_bytes

    batch = make_batch(read_inputs(), config.side, config.batch_size)
    (_, _, first_call_s), frame, patches = time_batch_mlx(model, batch)
    validity = check_validity(
        frame[0], patches[0], reference_tokens(config.model_size, config.side), config.dtype_mode
    )
    output_dtype = str(frame.dtype)
    del frame, patches

    warm = _phase(lambda: time_batch_mlx(model, batch)[0], warm_up)
    timed_times = _phase(lambda: time_batch_mlx(model, batch)[0], timed)
    end = read_footprint()

    device, frame_t, patch_t = (list(column) for column in zip(*timed_times, strict=True))
    bench_dir.mkdir(parents=True, exist_ok=True)
    slug = model_slug(config.model_size)
    name = f"{slug}_{config.dtype_mode}_{config.side}_b{config.batch_size}_r{config.repeat}.json"
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
        "mps_driver_max_bytes": None,
        "versions": read_versions(),
        "attn_implementation": "mx.fast.scaled_dot_product_attention",
        "start_time": start_time,
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
        "mlx_peak_bytes": mx.get_peak_memory(),
        "mlx_memory_limit_bytes": mx.get_memory_limit(),
        "tf32": False,
    }


def worker_main(spec_json: str) -> None:
    """Run one worker in this process and print the result as one JSON line."""
    spec = json.loads(spec_json)
    backend = spec.pop("backend")
    config = BenchConfig(**spec)
    try:
        if backend == "mlx":
            result = run_mlx_worker(config)
        else:
            from dinov3_apple.torch_bench import run_worker

            result = run_worker(config, bench_dir=BENCH_DIR / "torch-mps")
            result |= {"mlx_peak_bytes": None, "mlx_memory_limit_bytes": None, "tf32": None}
    except InsufficientMemoryError as error:
        result = {"status": "skipped", "reason": f"load check in the worker: {error}"}
    watermarks = {k: os.environ[k] for k in WATERMARKS if k in os.environ}
    result["mps_watermarks"] = watermarks or None
    result["torch_imported"] = "torch" in sys.modules
    print(json.dumps(result))


def run_worker_subprocess(config: BenchConfig, backend: str, tf32: bool | None) -> dict:
    """Run one worker in a new subprocess and return its result.

    A worker that exits with an error gives status "error" and the last lines of its stderr.
    """
    spec = json.dumps(asdict(config) | {"backend": backend})
    command = [sys.executable, __file__, "--worker", spec]
    try:
        done = subprocess.run(
            command,
            check=False,
            env=worker_env(backend, tf32, os.environ),
            capture_output=True,
            text=True,
            timeout=WORKER_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as error:
        return {"status": "error", "reason": f"timeout after {error.timeout} s"}
    if done.returncode == 0:
        for text in reversed(done.stdout.splitlines()):
            if text.startswith("{"):
                return json.loads(text)
    tail = "\n".join(done.stderr.splitlines()[-STDERR_LINES:])
    return {"status": "error", "reason": f"exit code {done.returncode}: {tail}"}


def dry_run(plan: list[tuple[int, int, str, bool | None]]) -> None:
    """Print each worker of the plan with its memory need. Load no model."""
    available = available_memory_bytes()
    print(f"available memory: {available / GB:.1f} GB")
    for round_, batch_size, backend, tf32 in plan:
        config = bench_config(round_, batch_size)
        need = required_bench_bytes(config)
        fits = "fits" if need <= available else "does not fit"
        print(
            f"round {round_} {backend:9s} tf32 {tf32!s:5} {config.label:30s} "
            f"need {need / GB:4.2f} GB  {fits}"
        )


def run_plan(plan: list[tuple[int, int, str, bool | None]], results: Path) -> Counter:
    """Run the workers one after the other and append one line for each.

    Raises:
        MachineStateError: On battery power or in Low Power Mode. No line is written.
    """
    counts: Counter = Counter()
    for index, (round_, batch_size, backend, tf32) in enumerate(plan, start=1):
        config = bench_config(round_, batch_size)
        machine = read_machine_state()
        check_machine_state(machine)
        need = required_bench_bytes(config)
        available = available_memory_bytes()
        fits = available >= need
        label = f"{backend} {config.label}"
        logger.info(
            f"memory check: {label}: needs {need / GB:.1f} GB, "
            f"{available / GB:.1f} GB available, {'ok' if fits else 'does not fit'}"
        )
        if fits:
            worker = run_worker_subprocess(config, backend, tf32)
        else:
            reason = f"needs {need / GB:.1f} GB, {available / GB:.1f} GB available"
            worker = {"status": "skipped", "reason": reason}
        line = result_line(config, worker, need, available, machine)
        position = 1 + [entry[2] for entry in plan if entry[:2] == (round_, batch_size)].index(
            backend
        )
        line |= {
            "backend": backend,
            "round": round_,
            "position": position,
            "tf32": tf32,
            "mlx_peak_bytes": worker.get("mlx_peak_bytes"),
            "torch_imported": worker.get("torch_imported"),
            "mlx_memory_limit_bytes": worker.get("mlx_memory_limit_bytes"),
            "mps_watermarks": worker.get("mps_watermarks"),
        }
        results.parent.mkdir(parents=True, exist_ok=True)
        with results.open("a") as file:
            file.write(json.dumps(line) + "\n")
        counts[line["status"]] += 1
        summary = line["reason"].splitlines()[-1] if line["reason"] else ""
        if line["status"] == "done":
            summary = (
                f"valid {line['valid']}, frame {line['frame']['throughput_ips']:.1f} img/s, "
                f"patch {line['patch']['throughput_ips']:.1f} img/s, "
                f"peak {line['peak_bytes'] / GB:.2f} GB"
            )
        logger.info(f"[{index}/{len(plan)}] {label}: {line['status']}: {summary}")
    return counts


def main() -> None:
    """Parse the arguments and run the benchmark."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--rounds", type=int, default=3, help="rounds of 4 workers each")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, load no model")
    parser.add_argument("--results", type=Path, default=RESULTS, help=argparse.SUPPRESS)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        worker_main(args.worker)
        return
    plan = worker_plan(args.rounds)
    if args.dry_run:
        dry_run(plan)
        return
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(DRIVER_LOG)],
    )
    logger.info(f"run of {len(plan)} workers")
    try:
        counts = run_plan(plan, args.results)
    except MachineStateError as error:
        logger.error(f"stop: {error}")
        sys.exit(1)
    logger.info(f"end of run: {dict(counts)}")


if __name__ == "__main__":
    main()
