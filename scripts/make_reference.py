"""Make the stored pixels and the reference output of each model size on torch-cpu in fp32.

The script writes inputs.safetensors first. Then it runs each model size in its own subprocess,
from the smallest to the largest, one after the other. Before each subprocess it checks the free
memory: LOAD_RATIO x the fp32 weight bytes + 4 GB. A model size that does not fit is skipped, and
runs.jsonl records the reason. Section 3.5 of specs/02-reference/plan.html gives the rules.

Usage:
    uv run python scripts/make_reference.py                # all model sizes
    uv run python scripts/make_reference.py ViT-7B/16      # one model size again
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
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file

from dinov3_apple.hub import CONVERTED_DIR, MODEL_IDS, weights_dir
from dinov3_apple.memory import GB, InsufficientMemoryError, check_free_memory
from dinov3_apple.preprocess import IMAGENET_MEAN, IMAGENET_STD, load_image, preprocess
from dinov3_apple.reference import (
    HIDDEN_STATES_CONFIG,
    IMAGES_DIR,
    INPUTS_FILE,
    LOAD_RATIO,
    REFERENCE_DIR,
    TEST_IMAGES,
    configurations,
    load_torch_backbone,
    parameter_count,
    read_inputs,
    reference_path,
    required_load_bytes,
    run_torch,
    run_torch_hidden_states,
)

RUNS_LOG = REFERENCE_DIR / "runs.jsonl"
RUN_LOG = REFERENCE_DIR / "make_reference.log"
# The load ratio is measured on model sizes with at least this many fp32 weight bytes. The fixed
# overhead of torch dominates the peak of a small model and does not scale to ViT-7B/16.
RATIO_MIN_WEIGHT_BYTES = 10**9
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
    tensors[f"{HIDDEN_STATES_CONFIG}/hidden_states"] = np.ascontiguousarray(hidden)
    run_time_s = time.perf_counter() - start
    peak = _peak_rss_bytes()

    folder = weights_dir(model_size)
    weight_files = sorted(folder.glob("*.safetensors"))
    metadata = _common_metadata() | {
        "model_size": model_size,
        "model_id": MODEL_IDS[model_size],
        "num_blocks": model.config.num_hidden_layers,
        "weight_source": (
            "converted folder" if folder.is_relative_to(CONVERTED_DIR) else "Hugging Face cache"
        ),
        "weight_sha256": {file.name: _sha256(file.resolve()) for file in weight_files},
        "attn_implementation": model.config._attn_implementation,
        "load_time_s": round(load_s, 2),
        "run_time_s": round(run_time_s, 2),
        "peak_memory_bytes": peak,
        "rss_before_load_bytes": rss_before_load,
    }
    out = reference_path(model_size)
    partial = out.with_suffix(".partial")
    save_file(tensors, partial, metadata={"reference": json.dumps(metadata)})
    partial.rename(out)
    print(
        json.dumps(
            {
                "run_time_s": metadata["run_time_s"],
                "load_time_s": metadata["load_time_s"],
                "peak_memory_bytes": peak,
                "rss_before_load_bytes": rss_before_load,
                "file": out.name,
                "file_bytes": out.stat().st_size,
            }
        ),
        flush=True,
    )


def _append_run(line: dict) -> None:
    with RUNS_LOG.open("a") as file:
        file.write(json.dumps(line) + "\n")


def run_all(model_sizes: list[str]) -> None:
    """Run each model size in its own subprocess, one after the other, after a memory check."""
    ratio = LOAD_RATIO
    for model_size in model_sizes:
        fp32_bytes = parameter_count(model_size) * 4
        line = {"model_size": model_size, "time": _now(), "ratio": round(ratio, 3)}
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


def main() -> None:
    """Parse the arguments and make the stored files."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("model_sizes", nargs="*", help=f"some of {', '.join(MODEL_IDS)}")
    parser.add_argument(
        "--force-inputs", action="store_true", help="write inputs.safetensors again"
    )
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--ratio", type=float, default=LOAD_RATIO, help=argparse.SUPPRESS)
    args = parser.parse_args()

    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(message)s",
        handlers=[logging.StreamHandler(sys.stderr), logging.FileHandler(RUN_LOG)],
    )
    if args.child:
        run_child(args.child, args.ratio)
        return
    if args.force_inputs or not INPUTS_FILE.exists():
        write_inputs()
    unknown = set(args.model_sizes) - set(MODEL_IDS)
    if unknown:
        parser.error(f"unknown model sizes: {sorted(unknown)}")
    model_sizes = args.model_sizes or list(MODEL_IDS)
    run_all(sorted(model_sizes, key=list(MODEL_IDS).index))


if __name__ == "__main__":
    main()
