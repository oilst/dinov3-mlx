"""The probe D2 C of Step 7b (specs/07b-neural-engine-backend/explore.html): the two-stream run
with two threads in one process against two processes, at ViT-S/16, 224 px, batch size 8.

Each case runs in new subprocesses: "mlx" (mlx alone), "ne" (the Neural Engine alone), "threads"
(both sides in one process, one thread each) and "processes" (each side in its own process). Each
worker loads and warms up, writes a ready file and waits. The driver then writes a start time
LEAD_S ahead, and each side counts its batches for SECONDS. The order of the cases rotates from
round to round.

The mlx side runs the compiled forward pass in the default dtype mode of the API (fp16-autocast)
and copies both outputs to the host. The Neural Engine side runs the ne3 package of batch size 1
with the batch API, as the two-stream run of Step 6d. The plan of Step 7b takes threads if their
median stream ratio reaches THREADS_SHARE of the median stream ratio of two processes.

Results: results/streams-7b.jsonl; ready and start files in tests/fixtures/generated/bench/streams-7b/.

Usage (with the coreml dependency group; the driver takes the GPU lock):
    uv run --group coreml python scripts/bench_streams.py --rounds 3
    uv run --group coreml python scripts/bench_streams.py --dry-run
"""

import argparse
import json
import logging
import os
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from dinov3_mlx.memory import GB, InsufficientMemoryError, check_free_memory, send_checks_to

REPO_ROOT = Path(__file__).parents[1]
RESULTS = REPO_ROOT / "results" / "streams-7b.jsonl"
BENCH_DIR = REPO_ROOT / "tests" / "fixtures" / "generated" / "bench" / "streams-7b"
CASES = ["mlx", "ne", "threads", "processes"]
SIDES_OF_CASE = {"mlx": ["mlx"], "ne": ["ne"], "threads": ["threads"], "processes": ["ne", "mlx"]}
MODEL_SIZE = "ViT-S/16"
SIDE = 224
BATCH_SIZE = 8
DTYPE_MODE = "fp16-autocast"  # the default dtype mode of the API (ADR 0019)
SECONDS = 10.0
LEAD_S = 2.0
WARM_UP_CALLS = 5
READY_TIMEOUT_S = 600.0
THREADS_SHARE = 0.95  # D2 C: threads if they reach this share of the stream ratio of processes
NEED_BYTES = 6 * GB  # ViT-S/16 mlx (4.1 GB in the Step 6 check) plus the package (0.4 GB)
WATERMARKS = {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.8", "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.6"}

logger = logging.getLogger("bench_streams")


def schedule(rounds: int) -> list[list[str]]:
    """Return the cases of each round. The order rotates by one from round to round."""
    return [CASES[index % len(CASES) :] + CASES[: index % len(CASES)] for index in range(rounds)]


def stream_ratios(lines: list[dict]) -> dict[str, list[float]]:
    """Return the stream ratio of each round for threads and for processes.

    Stream ratio = (throughput of both sides) / (throughput of mlx alone in the same round).
    A round without a done line of mlx alone gives no ratio.

    Args:
        lines: The result lines of the probe.

    Returns:
        The ratios in round order, keyed by "threads" and "processes".
    """
    done = [line for line in lines if line.get("status") == "done"]
    alone = {
        line["round"]: line["sides"]["mlx"]["throughput_ips"]
        for line in done
        if line["case"] == "mlx"
    }
    ratios: dict[str, list[float]] = {"threads": [], "processes": []}
    for line in sorted(done, key=lambda line: line["round"]):
        if line["case"] in ratios and line["round"] in alone:
            total = sum(side["throughput_ips"] for side in line["sides"].values())
            ratios[line["case"]].append(total / alone[line["round"]])
    return ratios


def choice(ratios: dict[str, list[float]]) -> str:
    """Return "threads" or "process" by the rule of D2 C.

    Raises:
        ValueError: If threads or processes have no stream ratio.
    """
    if not ratios["threads"] or not ratios["processes"]:
        raise ValueError(f"no stream ratio for threads or processes: {ratios}")
    threads = statistics.median(ratios["threads"])
    processes = statistics.median(ratios["processes"])
    return "threads" if threads >= THREADS_SHARE * processes else "process"


# The worker side.


def ne_step() -> Callable[[], object]:
    """Load the ne3 package of batch size 1 on the Neural Engine and return one batch-API call."""
    from dinov3_mlx.bench import make_batch
    from dinov3_mlx.coreml import load_package, package_path
    from dinov3_mlx.stored import read_inputs

    model, _, _ = load_package(package_path(MODEL_SIZE, SIDE, 1), "CPU_AND_NE")
    batch = make_batch(read_inputs(), SIDE, BATCH_SIZE)
    feed = [{"pixels": batch[i : i + 1]} for i in range(BATCH_SIZE)]
    return lambda: model.predict(feed)


def mlx_step() -> Callable[[], object]:
    """Load and compile ViT-S/16 in mlx and return one host-to-host batch."""
    from dinov3_mlx.bench import make_batch
    from dinov3_mlx.mlx_bench import time_batch_mlx
    from dinov3_mlx.mlx_model import compile_backbone
    from dinov3_mlx.mlx_weights import load_mlx_backbone
    from dinov3_mlx.stored import read_inputs

    forward = compile_backbone(load_mlx_backbone(MODEL_SIZE, DTYPE_MODE, tf32=False))
    batch = make_batch(read_inputs(), SIDE, BATCH_SIZE)
    return lambda: time_batch_mlx(forward, batch, DTYPE_MODE)


def count_batches(step: Callable[[], object], start_at: float) -> dict:
    """Wait for start_at, then call step for SECONDS and return the throughput."""
    while time.time() < start_at:
        time.sleep(0.001)
    end_at = start_at + SECONDS
    batches = 0
    while time.time() < end_at:
        step()
        batches += 1
    elapsed = time.time() - start_at
    return {
        "batches": batches,
        "elapsed_s": elapsed,
        "throughput_ips": batches * BATCH_SIZE / elapsed,
    }


def wait_for_start(ready_file: Path, start_file: Path) -> float:
    """Write the ready file and return the start time that the driver writes."""
    ready_file.write_text("ready")
    while not start_file.exists():
        time.sleep(0.01)
    return float(start_file.read_text())


def worker_main(spec: dict) -> dict:
    """Run one side ("mlx" or "ne") or both sides in two threads ("threads").

    Returns:
        The result of each side, keyed by side.
    """
    names = ["ne", "mlx"] if spec["side"] == "threads" else [spec["side"]]
    steps = {name: (ne_step if name == "ne" else mlx_step)() for name in names}
    for step in steps.values():
        for _ in range(WARM_UP_CALLS):
            step()
    start_at = wait_for_start(Path(spec["ready_file"]), Path(spec["start_file"]))
    if len(steps) == 1:
        return {name: count_batches(step, start_at) for name, step in steps.items()}
    results: dict[str, dict] = {}

    def run(name: str) -> None:
        results[name] = count_batches(steps[name], start_at)

    threads = [threading.Thread(target=run, args=(name,)) for name in steps]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


# The driver side.


def run_case(round_: int, case: str, bench_dir: Path) -> dict:
    """Run one case in new subprocesses and return its result line."""
    folder = bench_dir / f"r{round_}-{case}"
    folder.mkdir(parents=True, exist_ok=True)
    start_file = folder / "start"
    start_file.unlink(missing_ok=True)
    line = {
        "run": "S7b-D2",
        "round": round_,
        "case": case,
        "model_size": MODEL_SIZE,
        "side": SIDE,
        "batch_size": BATCH_SIZE,
        "dtype_mode": DTYPE_MODE,
        "seconds": SECONDS,
    }
    try:
        check_free_memory(NEED_BYTES, f"{MODEL_SIZE} streams {case}")
    except InsufficientMemoryError as error:  # record the skip (run rules of CLAUDE.md)
        return line | {"status": "skipped", "reason": str(error)}
    processes = {}
    for side in SIDES_OF_CASE[case]:
        ready = folder / f"ready-{side}"
        ready.unlink(missing_ok=True)
        spec = {"side": side, "ready_file": str(ready), "start_file": str(start_file)}
        processes[side] = subprocess.Popen(
            [sys.executable, __file__, "--worker", json.dumps(spec)],
            env={**os.environ, **WATERMARKS, "MLX_ENABLE_TF32": "0"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    deadline = time.time() + READY_TIMEOUT_S
    while not all((folder / f"ready-{side}").exists() for side in processes):
        if time.time() > deadline or any(p.poll() is not None for p in processes.values()):
            for process in processes.values():
                process.kill()
            errors = {side: p.communicate()[1][-2000:] for side, p in processes.items()}
            return line | {"status": "error", "reason": f"a side did not get ready: {errors}"}
        time.sleep(0.05)
    start_file.write_text(repr(time.time() + LEAD_S))
    sides: dict[str, dict] = {}
    for side, process in processes.items():
        out, err = process.communicate(timeout=SECONDS + 600)
        found = [json.loads(text) for text in out.splitlines() if text.startswith("{")]
        if process.returncode != 0 or not found:
            return line | {
                "status": "error",
                "reason": f"{side}: exit {process.returncode}: {err[-2000:]}",
            }
        sides |= found[-1]
    return line | {
        "status": "done",
        "sides": sides,
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Return the command-line options."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--dry-run", action="store_true", help="print the plan, load no model")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the probe, write one line for each case and print the stream ratios and the choice."""
    args = parse_args(argv)
    if args.worker:
        print(json.dumps(worker_main(json.loads(args.worker))), flush=True)
        return
    plan = [(round_, case) for round_, cases in enumerate(schedule(args.rounds)) for case in cases]
    if args.dry_run:
        for round_, case in plan:
            print(f"round {round_} {case}: sides {SIDES_OF_CASE[case]}")
        print(f"{len(plan)} cases, {MODEL_SIZE} {SIDE}px b{BATCH_SIZE} {DTYPE_MODE}")
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    send_checks_to(sys.stderr)
    from dinov3_mlx.gpu_lock import heavy_job

    lines = []
    with heavy_job("scripts/bench_streams.py"):
        for round_, case in plan:
            line = run_case(round_, case, BENCH_DIR)
            logger.info(
                "round %s %s: %s %s",
                round_,
                case,
                line["status"],
                line.get("sides", line.get("reason")),
            )
            lines.append(line)
            with args.results.open("a") as file:
                file.write(json.dumps(line) + "\n")
    ratios = stream_ratios(lines)
    print(json.dumps({"stream_ratios": ratios, "choice": choice(ratios)}))


if __name__ == "__main__":
    main()
