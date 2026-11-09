"""Benchmark B1 of specs/07-release/plan.html: the overhead of the public API.

Two forms on the same uint8 batch, in one worker for each configuration:

- api: Embedder.embed_frames or Embedder.embed_patches (batch loop, padded batch, host output);
- direct: preprocess_frames, the compiled forward pass of the same Embedder and the cast of the
  one output to the host, as time_batch_mlx of Step 6 does it for that output.

The forms rotate from round to round: 5 rounds, the budgets WARM_UP and TIMED of bench.py. The
ratio of a round is api ÷ direct of the throughput; the result is the lower median of the 5 round
ratios (ADR 0004). The driver checks the machine state and the free memory before each worker and
holds the GPU lock. The absolute throughput is report only (V2 of Step 6).

    uv run python -m dinov3_mlx.gpu_lock -- uv run python scripts/bench_api.py [--dry-run]
"""

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from dinov3_mlx.bench import (
    TIMED,
    WARM_UP,
    BenchConfig,
    TimeBudget,
    keep_going,
    required_bench_bytes,
    run_worker_subprocess,
)
from dinov3_mlx.gpu_lock import heavy_job
from dinov3_mlx.machine import check_machine_state, read_machine_state, read_versions
from dinov3_mlx.memory import GB, available_memory_bytes
from dinov3_mlx.speedup import _lower_median

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results" / "api-overhead.jsonl"
TABLE = REPO_ROOT / "results" / "api-table.md"
RUN = "S7"
ROUNDS = 5
TARGET = 0.97  # AC9
WORKER_TIMEOUT_S = 1800
# The frame that the API gets: a video frame of 1280 x 720 px, from the first test image.
FRAME_SIZE = (1280, 720)

# Table 4 of the plan.
CONFIGS = [
    {"name": "P1", "model_size": "ViT-S/16", "side": 224, "output": "frame", "frames": 8},
    {"name": "P2", "model_size": "ViT-H+/16", "side": 512, "output": "patch", "frames": 8},
    {"name": "P1-pad", "model_size": "ViT-S/16", "side": 224, "output": "frame", "frames": 13},
]
DTYPE_MODE = "fp16-autocast"
BATCH_SIZE = 8

logger = logging.getLogger("bench_api")


def video_frames(count: int) -> np.ndarray:
    """Return count uint8 RGB frames of FRAME_SIZE: the test images, resized, in turn."""
    from PIL import Image

    from dinov3_mlx.parity import TEST_IMAGES
    from dinov3_mlx.stored import IMAGES_DIR

    frames = []
    for name in TEST_IMAGES:
        with Image.open(IMAGES_DIR / f"{name}.jpg") as image:
            frames.append(np.asarray(image.convert("RGB").resize(FRAME_SIZE, Image.BILINEAR)))
    return np.stack([frames[i % len(frames)] for i in range(count)])


def timed_throughput(
    run: Callable[[], None], frames: int, warm_up: TimeBudget, timed: TimeBudget
) -> float:
    """Run a warm-up and a timed phase of calls; return the frames per second of the timed phase."""
    for budget, keep in ((warm_up, False), (timed, True)):
        times: list[float] = []
        while keep_going(budget, len(times), sum(times)):
            start = time.perf_counter()
            run()
            times.append(time.perf_counter() - start)
        if keep:
            return frames * len(times) / sum(times)
    raise AssertionError("unreachable")


def run_worker(spec: dict, warm_up: TimeBudget = WARM_UP, timed: TimeBudget = TIMED) -> dict:
    """Time the two forms of one configuration in this process."""
    import mlx.core as mx

    from dinov3_mlx import Embedder
    from dinov3_mlx.mlx_model import patch_dtype
    from dinov3_mlx.mlx_preprocess import preprocess_frames
    from dinov3_mlx.outputs import NUM_PREFIX_TOKENS

    mx.set_memory_limit(spec["need_bytes"])
    model = Embedder(spec["model_size"], DTYPE_MODE, batch_size=BATCH_SIZE)
    batch = video_frames(spec["frames"])
    size = (spec["side"], spec["side"])
    half = patch_dtype(DTYPE_MODE)

    def api() -> None:
        if spec["output"] == "frame":
            model.embed_frames(batch, size=size)
        else:
            model.embed_patches(batch, size=size)

    def direct() -> None:
        for start in range(0, len(batch), BATCH_SIZE):
            tokens = model._forward(preprocess_frames(batch[start : start + BATCH_SIZE], size))
            if spec["output"] == "frame":
                np.array(tokens[:, 0].astype(mx.float32))
            else:
                np.array(tokens[:, NUM_PREFIX_TOKENS:].astype(half))

    forms = {"api": api, "direct": direct}
    rounds = []
    for r in range(ROUNDS):
        order = ["api", "direct"] if r % 2 == 0 else ["direct", "api"]
        ips = {form: timed_throughput(forms[form], len(batch), warm_up, timed) for form in order}
        rounds.append({"round": r, "order": order, **ips, "ratio": ips["api"] / ips["direct"]})
    return {
        "status": "done",
        "reason": "",
        "rounds": rounds,
        "ratio": _lower_median([r["ratio"] for r in rounds]),
        "mlx_memory_limit_bytes": mx.get_memory_limit(),
        "versions": read_versions(),
    }


def driver(results: Path, names: list[str]) -> None:
    """Run one worker for each configuration, after the machine and memory checks."""
    for spec in [c for c in CONFIGS if c["name"] in names]:
        config = BenchConfig(RUN, spec["model_size"], DTYPE_MODE, spec["side"], BATCH_SIZE, True)
        machine = read_machine_state()
        check_machine_state(machine)
        need = required_bench_bytes(config, backend="mlx")
        available = available_memory_bytes()
        line = {"run": RUN, **spec, "dtype_mode": DTYPE_MODE, "batch_size": BATCH_SIZE}
        line |= {"need_bytes": need, "available_bytes": available, "target": TARGET}
        logger.info(
            f"memory check: {spec['name']}: needs {need / GB:.1f} GB, {available / GB:.1f} GB available"
        )
        if available < need:
            worker = {
                "status": "skipped",
                "reason": f"needs {need / GB:.1f} GB, {available / GB:.1f} GB available",
            }
        else:
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", json.dumps(line)]
            env = os.environ | {"MLX_ENABLE_TF32": "0"}
            worker = run_worker_subprocess(command, env, WORKER_TIMEOUT_S)
        line |= worker
        line["machine"] = {
            "power_source": machine.power_source,
            "chip": machine.chip,
            "thermal": machine.thermal,
        }
        line["time"] = datetime.now(UTC).isoformat(timespec="seconds")
        results.parent.mkdir(parents=True, exist_ok=True)
        with results.open("a") as file:
            file.write(json.dumps(line) + "\n")
        logger.info(f"{spec['name']}: {line['status']} {line.get('ratio', line['reason'])}")


def write_table(results: Path, table: Path) -> None:
    """Write the Markdown table of the latest line of each configuration."""
    latest = {}
    for text in results.read_text().splitlines():
        line = json.loads(text)
        latest[line["name"]] = line
    rows = [
        "| Configuration | Model size | Input resolution | Batch size | Output | Frames per call | api img/s (median) | direct img/s (median) | Round ratios | Lower median | Target | Result |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for spec in CONFIGS:
        line = latest.get(spec["name"])
        if line is None:
            continue
        if line["status"] != "done":
            rows.append(
                f"| {spec['name']} | {spec['model_size']} | {spec['side']} | {BATCH_SIZE} | {spec['output']} | {spec['frames']} | – | – | – | – | – | {line['status']}: {line['reason']} |"
            )
            continue
        api = np.median([r["api"] for r in line["rounds"]])
        direct = np.median([r["direct"] for r in line["rounds"]])
        ratios = ", ".join(f"{r['ratio']:.3f}" for r in line["rounds"])
        report_only = spec["name"] == "P1-pad"
        target = "report only" if report_only else f"≥ {TARGET}"
        result = "report only" if report_only else ("pass" if line["ratio"] >= TARGET else "fail")
        rows.append(
            f"| {spec['name']} | {spec['model_size']} | {spec['side']} | {BATCH_SIZE} | {spec['output']} | {spec['frames']} | {api:.1f} | {direct:.1f} | {ratios} | {line['ratio']:.3f} | {target} | {result} |"
        )
    text = f"""# Overhead of the public API

Made by `scripts/bench_api.py` on {datetime.now(UTC).date()} from `results/api-overhead.jsonl`. Benchmark B1 of `specs/07-release/plan.html`: `Embedder` against the direct path (preprocess_frames, the compiled forward pass, the cast of the output to the host) on the same uint8 batch of {FRAME_SIZE[0]} × {FRAME_SIZE[1]} px frames, {DTYPE_MODE}. Ratio = api ÷ direct throughput of one round; the result is the lower median of {ROUNDS} rounds (ADR 0004). The absolute throughput is report only (V2 of Step 6).

{chr(10).join(rows)}

Table 1. The API overhead (AC9). P1-pad calls the API with 13 frames, so the second batch has 3 padded rows; its direct form runs 8 + 5 frames without padding.
"""
    table.write_text(text)


def main(argv: list[str] | None = None) -> None:
    """Parse the arguments and run the driver, a worker or the table."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument(
        "--only", nargs="*", default=[c["name"] for c in CONFIGS], help="configuration names"
    )
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument(
        "--table-only", action="store_true", help="write the table from the results"
    )
    args = parser.parse_args(argv)
    if args.worker:
        print(json.dumps(run_worker(json.loads(args.worker))))
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if not args.table_only:
        with heavy_job("scripts/bench_api.py"):
            driver(args.results, args.only)
    write_table(args.results, TABLE if args.results == RESULTS else args.results.with_suffix(".md"))


if __name__ == "__main__":
    main()
