"""Time each part of block 1 of ViT-S/16 and ViT-H+/16 on torch-mps (B5, Section 3.11 of
specs/03-baseline/plan.html).

The driver runs one subprocess for each model size and dtype mode, one after the other, after a
memory check. Each subprocess profiles 224 and 512 px at batch sizes 1 and 8 and appends one JSON
line for each configuration to results/block-profile-torch-mps.jsonl.

Usage:
    uv run python scripts/profile_block.py
"""

import os

# The MPS watermarks must be set before torch starts (CLAUDE.md run rules).
WATERMARKS = {"PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.8", "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.6"}
os.environ.update(WATERMARKS)

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from dinov3_apple.bench import BenchConfig, make_batch, required_bench_bytes
from dinov3_apple.flops import MODEL_SHAPES, part_bytes, part_flops, tokens
from dinov3_apple.gpu_lock import heavy_job
from dinov3_apple.machine import check_machine_state, read_machine_state
from dinov3_apple.memory import GB, available_memory_bytes

REPO_ROOT = Path(__file__).parents[1]
RESULTS = REPO_ROOT / "results" / "block-profile-torch-mps.jsonl"
LOG = REPO_ROOT / "tests" / "fixtures" / "generated" / "bench" / "profile_block.log"
MODEL_SIZES = ["ViT-S/16", "ViT-H+/16"]
DTYPE_MODES = ["fp32", "fp16-cast"]
SIDES = [224, 512]
BATCH_SIZES = [1, 8]
STDERR_LINES = 20

logger = logging.getLogger("profile_block")


def child(model_size: str, dtype_mode: str) -> None:
    """Load one backbone and profile all input resolutions and batch sizes. Print JSON lines."""
    import torch

    from dinov3_apple.block_profile import profile_block
    from dinov3_apple.reference import DTYPE_MODES as MODES
    from dinov3_apple.reference import load_torch_backbone, read_inputs

    model = load_torch_backbone(model_size, "mps", dtype_mode)
    inputs = read_inputs()
    shape = MODEL_SHAPES[model_size]
    bytes_per_value = MODES[dtype_mode].weights.itemsize
    for side in SIDES:
        for batch_size in BATCH_SIZES:
            batch = make_batch(inputs, side, batch_size)
            pixels = torch.from_numpy(batch).to("mps", MODES[dtype_mode].weights)
            result = profile_block(model, pixels, dtype_mode)
            num_tokens = tokens(side)
            line = {
                "model_size": model_size,
                "backend": "torch-mps",
                "dtype_mode": dtype_mode,
                "side": side,
                "batch_size": batch_size,
                **result,
                "part_flops": part_flops(shape, batch_size, num_tokens),
                "part_bytes": part_bytes(shape, batch_size, num_tokens, bytes_per_value),
                "attn_implementation": model.config._attn_implementation,
                "time": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            print(json.dumps(line), flush=True)


def subprocess_lines(
    model_size: str, dtype_mode: str, returncode: int, stdout: str, stderr: str, extra: dict
) -> list[dict]:
    """Return the result lines of one subprocess, or one "error" line if it failed.

    Args:
        model_size: The model size of the subprocess.
        dtype_mode: The dtype mode of the subprocess.
        returncode: The exit code of the subprocess.
        stdout: Its standard output, with one JSON line for each configuration.
        stderr: Its standard error.
        extra: Fields for every line, for example the need and the machine state.

    Returns:
        The lines to append to the results file.
    """
    if returncode != 0:
        tail = "\n".join(stderr.splitlines()[-STDERR_LINES:])
        return [
            {
                "model_size": model_size,
                "dtype_mode": dtype_mode,
                "status": "error",
                "reason": f"exit code {returncode}: {tail}",
                "time": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            | extra
        ]
    found = [json.loads(text) for text in stdout.splitlines() if text.startswith("{")]
    return [line | {"status": "done"} | extra for line in found]


def main(argv: list[str] | None = None) -> None:
    """Run one subprocess for each model size and dtype mode, under the GPU lock.

    Args:
        argv: The arguments. None reads sys.argv.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--child", nargs=2, help=argparse.SUPPRESS)
    parser.add_argument("--results", type=Path, default=RESULTS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        child(*args.child)
        return

    LOG.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(LOG)],
    )
    with heavy_job("scripts/profile_block.py"):
        run_all(args.results)


def run_all(results: Path) -> None:
    """Run one subprocess for each model size and dtype mode and append its lines to results."""
    for model_size in MODEL_SIZES:
        for dtype_mode in DTYPE_MODES:
            machine = read_machine_state()
            check_machine_state(machine)
            # The largest configuration of the subprocess sets the need.
            need = required_bench_bytes(
                BenchConfig("B5", model_size, dtype_mode, max(SIDES), max(BATCH_SIZES)),
                backend="torch-mps",
            )
            available = available_memory_bytes()
            fits = available >= need
            logger.info(
                f"memory check: {model_size} {dtype_mode} B5: needs {need / GB:.1f} GB, "
                f"{available / GB:.1f} GB available, {'ok' if fits else 'does not fit'}"
            )
            if not fits:
                skip = {
                    "model_size": model_size,
                    "dtype_mode": dtype_mode,
                    "status": "skipped",
                    "reason": f"needs {need / GB:.1f} GB, {available / GB:.1f} GB available",
                    "time": datetime.now(UTC).isoformat(timespec="seconds"),
                }
                with results.open("a") as file:
                    file.write(json.dumps(skip) + "\n")
                continue
            done = subprocess.run(
                [sys.executable, __file__, "--child", model_size, dtype_mode],
                check=False,
                env=os.environ | WATERMARKS,
                capture_output=True,
                text=True,
            )
            extra = {"need_bytes": need, "available_bytes": available, "machine": asdict(machine)}
            lines = subprocess_lines(
                model_size, dtype_mode, done.returncode, done.stdout, done.stderr, extra
            )
            with results.open("a") as file:
                for line in lines:
                    file.write(json.dumps(line) + "\n")
            for line in lines:
                if line["status"] != "done":
                    logger.error(f"{model_size} {dtype_mode}: {line['reason']}")
                    continue
                ratio = sum(line["parts_s"].values()) / line["block_s"]
                logger.info(
                    f"{model_size} {dtype_mode} {line['side']}px b{line['batch_size']}: "
                    f"block {1e3 * line['block_s']:.3f} ms, parts / block {ratio:.3f}"
                )


if __name__ == "__main__":
    main()
