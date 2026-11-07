"""Make the stored pixels and the reference output of each model size on torch-cpu in fp32.

The script writes inputs.safetensors first. Then it runs each model size in its own subprocess,
from the smallest to the largest, one after the other. Before each subprocess it checks the run
rules of CLAUDE.md and then the free memory: LOAD_RATIO x the fp32 weight bytes + 4 GB. A model
size that breaks a run rule (ViT-7B/16 never loads with fp32 weights) or does not fit is skipped,
and runs.jsonl records the reason. Section 3.5 of specs/02-reference/plan.html gives the rules.

With --streamed, each model size runs through run_torch_streamed instead: the fp32 weights of one
block at a time, never the whole model (path D4-Y, Section 3.8 of specs/05-all-sizes/plan.html).
Its memory check asks LOAD_RATIO x (the largest block + the parts outside the blocks) + 4 GB.
With --check, the streamed output of each model size with a reference file is compared with the
stored file; no file is written.

Usage:
    uv run python scripts/make_reference.py                # all model sizes
    uv run python scripts/make_reference.py ViT-L/16       # one model size again
    uv run python scripts/make_reference.py --streamed ViT-7B/16
    uv run python scripts/make_reference.py --streamed --check
"""

import argparse
import hashlib
import json
import logging
import platform
import resource
import subprocess
import sys
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

from dinov3_apple.gpu_lock import heavy_job
from dinov3_apple.hub import CONVERTED_DIR, MODEL_IDS, weights_dir
from dinov3_apple.memory import (
    GB,
    NO_FP32_WEIGHTS,
    InsufficientMemoryError,
    RunRuleError,
    check_free_memory,
    check_run_rule,
)
from dinov3_apple.preprocess import IMAGENET_MEAN, IMAGENET_STD, load_image, preprocess
from dinov3_apple.reference import (
    HIDDEN_STATES_CONFIG,
    IMAGES_DIR,
    INPUTS_FILE,
    LOAD_RATIO,
    REFERENCE_DIR,
    STREAMED_ATTN_IMPLEMENTATION,
    TEST_IMAGES,
    ShardIndex,
    configurations,
    load_torch_backbone,
    parameter_count,
    read_inputs,
    read_reference,
    reference_path,
    required_load_bytes,
    required_streamed_bytes,
    run_torch,
    run_torch_hidden_states,
    run_torch_streamed,
    streamed_label,
)

RUNS_LOG = REFERENCE_DIR / "runs.jsonl"
RUN_LOG = REFERENCE_DIR / "make_reference.log"
# The load ratio is measured on model sizes with at least this many fp32 weight bytes. The fixed
# overhead of torch dominates the peak of a small model and does not scale to ViT-7B/16.
RATIO_MIN_WEIGHT_BYTES = 10**9
HIDDEN_STATES_KEY = f"{HIDDEN_STATES_CONFIG}/hidden_states"
# --check: the largest difference of the streamed output to a stored file (R24 of
# specs/05-all-sizes/plan.html). Expected: 0.0.
CHECK_TARGET = 1e-6
# The reading of the run rule of CLAUDE.md for the streamed path, in each of its runs.jsonl lines.
STREAMED_RUN_RULE = (
    "streamed: the fp32 weights of one block at a time, never the whole model "
    "(the user's yes of 2026-11-07, Table 20 of specs/05-all-sizes/plan.html)"
)
PREPROCESSING = {
    "decode": "Pillow, RGB, uint8",
    "scale": "divide by 255 to float64",
    "resize": "torch.nn.functional.interpolate, bilinear, antialias=True, float64",
    "normalize": "float64, then cast to float32",
    "mean": IMAGENET_MEAN.tolist(),
    "std": IMAGENET_STD.tolist(),
}

logger = logging.getLogger("make_reference")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _peak_rss_bytes() -> int:
    # macOS reports ru_maxrss in bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _common_metadata() -> dict:
    return {
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": version("transformers"),
            "safetensors": version("safetensors"),
            "pillow": version("pillow"),
            "numpy": np.__version__,
        },
        "image_sha256": {image: _sha256(IMAGES_DIR / f"{image}.jpg") for image in TEST_IMAGES},
        "preprocessing": PREPROCESSING,
        "date": _now(),
    }


def write_inputs() -> None:
    """Write the normalized fp32 pixels of the 12 configurations to inputs.safetensors."""
    tensors = {
        f"{config.id}/pixels": preprocess(
            load_image(config.image_path), (config.height, config.width)
        )
        .numpy()
        .copy()
        for config in configurations()
    }
    INPUTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, INPUTS_FILE, metadata={"inputs": json.dumps(_common_metadata())})
    logger.info(f"wrote {len(tensors)} pixel tensors to {INPUTS_FILE}")


def reference_metadata(
    model_size: str, num_blocks: int, attn_implementation: str, run: Mapping[str, object]
) -> dict:
    """Return the header metadata of the reference file of one model size.

    Args:
        model_size: A key of MODEL_IDS.
        num_blocks: The number of blocks.
        attn_implementation: The attention of the run, for example "sdpa".
        run: The fields of the run: load_time_s, run_time_s, peak_memory_bytes,
            rss_before_load_bytes and, for the streamed path, "streamed": true.

    Returns:
        The fields of REQUIRED_METADATA and the fields of the run.
    """
    folder = weights_dir(model_size)
    weight_files = sorted(folder.glob("*.safetensors"))
    return (
        _common_metadata()
        | {
            "model_size": model_size,
            "model_id": MODEL_IDS[model_size],
            "num_blocks": num_blocks,
            "weight_source": (
                "converted folder" if folder.is_relative_to(CONVERTED_DIR) else "Hugging Face cache"
            ),
            "weight_sha256": {file.name: _sha256(file.resolve()) for file in weight_files},
            "attn_implementation": attn_implementation,
        }
        | dict(run)
    )


def _save_reference(model_size: str, tensors: dict[str, np.ndarray], metadata: dict) -> dict:
    """Write the reference file through a partial file and return its name and size."""
    out = reference_path(model_size)
    partial = out.with_suffix(".partial")
    save_file(tensors, partial, metadata={"reference": json.dumps(metadata)})
    partial.rename(out)
    return {"file": out.name, "file_bytes": out.stat().st_size}


def run_child(model_size: str, ratio: float) -> None:
    """Make the reference file of one model size. Runs inside its own subprocess."""
    rss_before_load = _peak_rss_bytes()
    start = time.perf_counter()
    model = load_torch_backbone(model_size, "cpu", "fp32", ratio)
    load_s = time.perf_counter() - start
    inputs = read_inputs()

    tensors = {}
    for config in configurations():
        tokens = run_torch(model, inputs[config.id], "fp32")[0]
        tensors[f"{config.id}/tokens"] = np.ascontiguousarray(tokens)
        logger.info(f"{model_size} {config.id}: tokens {tokens.shape}")
    hidden = run_torch_hidden_states(model, inputs[HIDDEN_STATES_CONFIG])
    tensors[HIDDEN_STATES_KEY] = np.ascontiguousarray(hidden)
    run_time_s = time.perf_counter() - start
    peak = _peak_rss_bytes()

    run = {
        "load_time_s": round(load_s, 2),
        "run_time_s": round(run_time_s, 2),
        "peak_memory_bytes": peak,
        "rss_before_load_bytes": rss_before_load,
    }
    num_blocks = model.config.num_hidden_layers
    metadata = reference_metadata(model_size, num_blocks, model.config._attn_implementation, run)
    print(json.dumps(run | _save_reference(model_size, tensors, metadata)), flush=True)


def streamed_differences(
    stored: Mapping[str, np.ndarray], tokens: Mapping[str, np.ndarray], hidden: np.ndarray
) -> dict[str, float | int]:
    """Return the largest absolute differences of a streamed output to a reference file.

    Args:
        stored: The tensors of read_reference: the tokens of each configuration and the key
            HIDDEN_STATES_KEY.
        tokens: The streamed tokens of each configuration.
        hidden: The streamed hidden states of HIDDEN_STATES_CONFIG.

    Returns:
        The number of configurations, the largest difference over all tokens and the largest
        difference over all hidden states.

    Raises:
        ValueError: If the configurations or a shape differ.
    """
    stored_configs = sorted(key for key in stored if key != HIDDEN_STATES_KEY)
    if stored_configs != sorted(tokens):
        raise ValueError(f"other configurations: {stored_configs} and {sorted(tokens)}")
    pairs = [(stored[config], tokens[config]) for config in stored_configs]
    pairs_hidden = (stored[HIDDEN_STATES_KEY], hidden)
    for expected, output in [*pairs, pairs_hidden]:
        if expected.shape != output.shape:
            raise ValueError(f"other shape: {expected.shape} and {output.shape}")

    def largest(expected: np.ndarray, output: np.ndarray) -> float:
        return float(np.abs(output.astype(np.float64) - expected.astype(np.float64)).max())

    return {
        "configurations": len(pairs),
        "tokens_max_abs": max(largest(*pair) for pair in pairs),
        "hidden_states_max_abs": largest(*pairs_hidden),
    }


def run_streamed_child(model_size: str, check: bool) -> None:
    """Make or check the reference output of one model size with the streamed path. Runs inside
    its own subprocess. It never calls load_torch_backbone.

    Args:
        model_size: A key of MODEL_IDS.
        check: False writes the reference file with "streamed": true in its header. True compares
            the output with the stored reference file and writes no file.
    """
    rss_before = _peak_rss_bytes()
    inputs = read_inputs()
    pixels = {config.id: inputs[config.id] for config in configurations()}
    shards = ShardIndex(weights_dir(model_size))
    start = time.perf_counter()
    tokens, hidden = run_torch_streamed(model_size, pixels, HIDDEN_STATES_CONFIG, shards)
    run = {
        "load_time_s": round(shards.read_seconds, 2),
        "run_time_s": round(time.perf_counter() - start, 2),
        "peak_memory_bytes": _peak_rss_bytes(),
        "rss_before_load_bytes": rss_before,
    }
    sizes = {
        "largest_block_bytes": shards.streamed_weight_bytes() - shards.outer_bytes(),
        "outer_bytes": shards.outer_bytes(),
    }
    if check:
        stored, metadata = read_reference(model_size)
        result = run | sizes | streamed_differences(stored, tokens, hidden)
        result["stored_attn_implementation"] = metadata.get("attn_implementation")
    else:
        tensors = {f"{config}/tokens": np.ascontiguousarray(t) for config, t in tokens.items()}
        tensors[HIDDEN_STATES_KEY] = np.ascontiguousarray(hidden)
        metadata = reference_metadata(
            model_size, len(hidden) - 1, STREAMED_ATTN_IMPLEMENTATION, run | {"streamed": True}
        )
        result = run | sizes | _save_reference(model_size, tensors, metadata)
    # The peak of the run is in the header; this one also covers the compare or the save.
    result["lifetime_peak_bytes"] = _peak_rss_bytes()
    print(json.dumps(result), flush=True)


def _append_run(line: dict) -> None:
    with RUNS_LOG.open("a") as file:
        file.write(json.dumps(line) + "\n")


def load_ratio(runs_log: Path = RUNS_LOG) -> float:
    """Return the load ratio for the memory check: the largest measured ratio of earlier runs.

    Only runs of model sizes with at least RATIO_MIN_WEIGHT_BYTES of fp32 weights count. The result
    is never below LOAD_RATIO.

    Args:
        runs_log: The JSON lines of earlier runs of this script.

    Returns:
        The load ratio.
    """
    ratio = LOAD_RATIO
    if not runs_log.exists():
        return ratio
    for text in runs_log.read_text().splitlines():
        line = json.loads(text)
        # A streamed run holds one block at a time: its peak says nothing about a full load.
        if line.get("streamed") or line.get("status") != "done" or "measured_ratio" not in line:
            continue
        weight_bytes = line.get("fp32_weight_bytes")
        if weight_bytes is None and line.get("model_size") in MODEL_IDS:
            weight_bytes = parameter_count(line["model_size"]) * 4
        if weight_bytes is not None and weight_bytes >= RATIO_MIN_WEIGHT_BYTES:
            ratio = max(ratio, line["measured_ratio"])
    return ratio


def run_all(model_sizes: list[str]) -> None:
    """Run each model size in its own subprocess, one after the other.

    The run rule of every requested model size comes first, before load_ratio reads runs.jsonl
    and the headers of earlier model sizes (F11 of specs/05-all-sizes/verify.html). A refused
    model size gets its skip line at once, without a load ratio. Each other model size then
    passes the memory check. A skip goes to runs.jsonl with its reason.
    """
    allowed = []
    for model_size in model_sizes:
        try:
            check_run_rule(model_size, 4, f"{model_size} torch-cpu fp32 (make_reference.py)")
        except RunRuleError as error:
            line = {"model_size": model_size, "time": _now()}
            _append_run(line | {"status": "skipped", "reason": str(error)})
            logger.warning(f"{model_size}: skipped. {error}")
            continue
        allowed.append(model_size)
    if not allowed:
        return
    ratio = load_ratio()
    logger.info(f"the memory check uses the load ratio {ratio:.2f}")
    for model_size in allowed:
        fp32_bytes = parameter_count(model_size) * 4
        line = {
            "model_size": model_size,
            "time": _now(),
            "ratio": round(ratio, 3),
            "fp32_weight_bytes": fp32_bytes,
        }
        try:
            available = check_free_memory(
                required_load_bytes(model_size, torch.float32, ratio),
                f"{model_size} torch-cpu fp32 (make_reference.py)",
            )
        except InsufficientMemoryError as error:
            _append_run(line | {"status": "skipped", "reason": str(error)})
            logger.warning(f"{model_size}: skipped. {error}")
            continue
        line["available_before_bytes"] = available
        command = [sys.executable, __file__, "--child", model_size, "--ratio", str(ratio)]
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=False)
        if result.returncode != 0:
            _append_run(line | {"status": "failed", "returncode": result.returncode})
            raise SystemExit(f"{model_size}: the subprocess failed ({result.returncode}).")
        child = json.loads(result.stdout.strip().splitlines()[-1])
        measured = (child["peak_memory_bytes"] - child["rss_before_load_bytes"]) / fp32_bytes
        _append_run(line | {"status": "done", "measured_ratio": round(measured, 3)} | child)
        logger.info(
            f"{model_size}: done in {child['run_time_s']:.1f} s, peak "
            f"{child['peak_memory_bytes'] / GB:.1f} GB, load ratio {measured:.2f}"
        )
        if fp32_bytes >= RATIO_MIN_WEIGHT_BYTES and measured > ratio:
            ratio = measured
            logger.info(f"the memory check now uses the measured load ratio {ratio:.2f}")


def run_streamed(model_sizes: list[str], check: bool) -> None:
    """Run each model size through the streamed path in its own subprocess, one after the other.

    The run rule of CLAUDE.md applies in its streamed reading (STREAMED_RUN_RULE): the fp32
    weights of one block at a time. So ViT-7B/16 runs too. The memory check of each model size
    asks the streamed need (required_streamed_bytes), from the headers. The lines in runs.jsonl
    carry "streamed": true, the need and the peak; load_ratio ignores them.

    Args:
        model_sizes: Keys of MODEL_IDS, in the order of the runs.
        check: True compares the output of each model size that has a reference file with the
            file and writes no file. A model size without a file gets a skip line.

    Raises:
        SystemExit: If a subprocess fails, or after the last model size if a check found a
            largest difference above CHECK_TARGET.
    """
    above_target = []
    for model_size in model_sizes:
        line = {"model_size": model_size, "time": _now(), "streamed": True}
        line |= {"check": True} if check else {}
        line["run_rule"] = STREAMED_RUN_RULE
        if check and not reference_path(model_size).exists():
            reason = f"{reference_path(model_size).name} is missing"
            _append_run(line | {"status": "skipped", "reason": reason})
            logger.warning(f"{model_size}: check skipped. {reason}")
            continue
        need = required_streamed_bytes(model_size)
        line["need_bytes"] = need
        try:
            available = check_free_memory(need, f"{streamed_label(model_size)} (make_reference.py)")
        except InsufficientMemoryError as error:
            _append_run(line | {"status": "skipped", "reason": str(error)})
            logger.warning(f"{model_size}: skipped. {error}")
            continue
        line["available_before_bytes"] = available
        command = [sys.executable, __file__, "--child", model_size, "--streamed"]
        command += ["--check"] if check else []
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True, check=False)
        if result.returncode != 0:
            _append_run(line | {"status": "failed", "returncode": result.returncode})
            raise SystemExit(f"{model_size}: the subprocess failed ({result.returncode}).")
        child = json.loads(result.stdout.strip().splitlines()[-1])
        if check:
            largest = max(child["tokens_max_abs"], child["hidden_states_max_abs"])
            passed = largest <= CHECK_TARGET
            verdict = {"largest_difference": largest, "target": CHECK_TARGET, "passed": passed}
            _append_run(line | {"status": "checked"} | verdict | child)
            logger.info(
                f"{model_size}: streamed check in {child['run_time_s']:.1f} s, largest difference "
                f"{largest:g} (target {CHECK_TARGET:g}), peak "
                f"{child['peak_memory_bytes'] / GB:.2f} GB, need {need / GB:.2f} GB"
            )
            if not passed:
                above_target.append(f"{model_size} {largest:g}")
        else:
            _append_run(line | {"status": "done"} | child)
            logger.info(
                f"{model_size}: streamed reference in {child['run_time_s']:.1f} s, peak "
                f"{child['peak_memory_bytes'] / GB:.2f} GB, need {need / GB:.2f} GB"
            )
    if above_target:
        raise SystemExit(f"largest difference above {CHECK_TARGET:g}: {', '.join(above_target)}")


def model_sizes_to_run(model_sizes: list[str], streamed: bool, check: bool) -> list[str]:
    """Return the model sizes of one run, from the smallest to the largest (V2 of
    specs/05-all-sizes/verify.html).

    Args:
        model_sizes: The model sizes of the command line. [] gives the default of the path.
        streamed: True for the streamed path.
        check: With streamed, True compares with the stored files.

    Returns:
        The given model sizes, sorted. Without model sizes: all six on the full-model path; with
        --streamed only the model sizes without fp32 weights (ViT-7B/16), whose files only the
        streamed path makes; with --streamed --check the five model sizes of the full-model files.
    """
    if not model_sizes:
        if not streamed:
            model_sizes = list(MODEL_IDS)
        elif check:
            model_sizes = [size for size in MODEL_IDS if size not in NO_FP32_WEIGHTS]
        else:
            model_sizes = sorted(NO_FP32_WEIGHTS)
    return sorted(model_sizes, key=list(MODEL_IDS).index)


def main(argv: list[str] | None = None) -> None:
    """Parse the arguments and make the stored files. A run that is not a child process takes the
    GPU lock.

    Args:
        argv: The arguments. None reads sys.argv.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("model_sizes", nargs="*", help=f"some of {', '.join(MODEL_IDS)}")
    parser.add_argument(
        "--force-inputs", action="store_true", help="write inputs.safetensors again"
    )
    parser.add_argument(
        "--streamed",
        action="store_true",
        help="run through run_torch_streamed: the fp32 weights of one block at a time",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="with --streamed: compare with the stored reference files, write no file",
    )
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--ratio", type=float, default=LOAD_RATIO, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.check and not args.streamed:
        parser.error("--check checks the streamed path: add --streamed")

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(RUN_LOG)],
    )
    if args.child:
        if args.streamed:
            run_streamed_child(args.child, args.check)
        else:
            run_child(args.child, args.ratio)
        return
    unknown = set(args.model_sizes) - set(MODEL_IDS)
    if unknown:
        parser.error(f"unknown model sizes: {sorted(unknown)}")
    model_sizes = model_sizes_to_run(args.model_sizes, args.streamed, args.check)
    with heavy_job("scripts/make_reference.py"):
        if args.force_inputs or not INPUTS_FILE.exists():
            write_inputs()
        if args.streamed:
            run_streamed(model_sizes, args.check)
        else:
            run_all(model_sizes)


if __name__ == "__main__":
    main()
