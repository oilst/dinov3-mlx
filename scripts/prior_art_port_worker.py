"""The port worker of Step 6b: one Python MLX port, one dtype, one batch size (job J5, gated by
D2 of specs/06b-prior-art/explore.html).

The driver scripts/bench_prior_art.py starts this file only with --allow-third-party-python, the
user's yes to question Q1. It runs in the port venv with -I:

    ~/.cache/dinov3-apple/prior-art/venv-mlx/bin/python -I prior_art_port_worker.py '<spec>'

It imports the standard library, NumPy, safetensors and mlx; the code of the candidate (mlxim of
C1, dinov3 of C2) comes from the clone that the spec names, inside the functions. It imports no
module of the repo and loads no pickle file: C1 loads the safetensors file of the mlx-vision org,
C2 loads our own weights with the Meta names (scripts/prior_art_meta_names.py). The output
contract is the contract of prior_art_coreml_worker.py: the outputs of the 3 test images at batch
size 1 in a safetensors file, and one JSON line with the batch times.

The spec (JSON) has the fields: candidate ("C1" or "C2"), model ("ViT-S/16", "ViT-H+/16" or
"dinov2"), dtype ("fp32", "fp16" or "bf16"), side, batch_size, pixels ("stored" or "random"),
clone (the source folder of the clone), weights (a safetensors file), inputs_file, outputs_file,
warm_up and timed (the fields of TimeBudget), limit_bytes (the MLX memory limit).
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

TEST_IMAGES = ["iss-kibo-lab", "forest", "iss-conference"]  # the order of parity.TEST_IMAGES
RANDOM_SEED = 0
NUM_PREFIX_TOKENS = 5  # 1 CLS token and 4 register tokens
# The model names of mlx-image (C1).
MLXIM_NAMES = {"ViT-S/16": "vit_small_patch16_224.dinov3", "dinov2": "vit_small_patch14_518.dinov2"}


def keep_going(budget: dict, batches: int, elapsed_s: float) -> bool:
    """Return True while the phase needs another batch: keep_going of bench.py, copied."""
    max_seconds = budget.get("max_seconds")
    stop_time = max_seconds is not None and elapsed_s >= max_seconds
    if stop_time and batches >= budget.get("floor_batches", 0):
        return False
    return batches < budget["min_batches"] or elapsed_s < budget["min_seconds"]


def run_phase(call: object, budget: dict) -> list[float]:
    """Run calls until the time budget ends and return the time of each call."""
    times: list[float] = []
    while keep_going(budget, len(times), sum(times)):
        start = time.perf_counter()
        call()
        times.append(time.perf_counter() - start)
    return times


def build_c1(spec: dict) -> object:
    """Return the model of mlx-image (C1) with the safetensors weights of the mlx-vision org."""
    sys.path.insert(0, spec["clone"])
    from mlxim.model import create_model

    name = MLXIM_NAMES[spec["model"]]
    return create_model(name, weights=spec["weights"], num_classes=0, strict=True)


def build_c2(spec: dict) -> object:
    """Return the ViT-H+/16 model of mlx_dinov3 (C2) with our own weights in the Meta names."""
    import mlx.core as mx
    from mlx.utils import tree_flatten

    sys.path.insert(0, spec["clone"])
    from dinov3.models import vit_huge2

    model = vit_huge2(
        patch_size=16,
        n_storage_tokens=4,
        layerscale_init=1e-5,
        mask_k_bias=True,
        ffn_layer="swiglu",
        ffn_ratio=6,
    )
    params = dict(tree_flatten(model.parameters()))
    loaded = mx.load(spec["weights"])
    weights = [(key, value.astype(params[key].dtype)) for key, value in loaded.items()]
    weights += [(key, params[key]) for key in sorted(set(params) - set(loaded))]
    model.load_weights(weights, strict=True)
    return model


def forward_function(model: object, spec: dict) -> object:
    """Return forward(pixels) -> (frame embeddings, patch embeddings) of the port."""
    import mlx.core as mx

    if spec["model"] == "dinov2":

        def forward(x: mx.array) -> tuple[mx.array, mx.array]:
            tokens = model.encoder(model._prepare_tokens(x.transpose(0, 2, 3, 1)))[0]
            return tokens[:, 0], tokens[:, NUM_PREFIX_TOKENS:]

        return forward

    def forward(x: mx.array) -> tuple[mx.array, mx.array]:
        out = model.forward_features(x)
        return out["x_norm_clstoken"], out["x_norm_patchtokens"]

    return forward


def read_pixels(spec: dict) -> list[np.ndarray]:
    """Return the pixels (1, 3, H, W) of the 3 test images, or 1 random image."""
    side = spec["side"]
    if spec["pixels"] == "random":
        rng = np.random.default_rng(RANDOM_SEED)
        return [rng.standard_normal((1, 3, side, side)).astype(np.float32)]
    with safe_open(spec["inputs_file"], "np") as tensors:
        return [tensors.get_tensor(f"{image}/{side}x{side}/pixels") for image in TEST_IMAGES]


def run(spec: dict) -> dict:
    """Load the port, run the test images at batch size 1, then time the batches host to host."""
    import mlx.core as mx

    mx.set_memory_limit(spec["limit_bytes"])
    dtype = {"fp32": mx.float32, "fp16": mx.float16, "bf16": mx.bfloat16}[spec["dtype"]]
    start = time.perf_counter()
    model = build_c1(spec) if spec["candidate"] == "C1" else build_c2(spec)
    if dtype != mx.float32:
        model.set_dtype(dtype)  # the standard cast of MLX
    model.eval()
    mx.eval(model.parameters())
    load_s = time.perf_counter() - start
    forward = forward_function(model, spec)

    images = read_pixels(spec)
    outputs = {}
    for index, pixels in enumerate(images):
        frame, patches = forward(mx.array(pixels).astype(dtype))
        key = TEST_IMAGES[index] if spec["pixels"] == "stored" else "random"
        outputs[f"{key}/embedding"] = np.array(frame.astype(mx.float32))
        outputs[f"{key}/patch_embeddings"] = np.array(patches.astype(mx.float32))
    finite = all(bool(np.isfinite(value).all()) for value in outputs.values())
    Path(spec["outputs_file"]).parent.mkdir(parents=True, exist_ok=True)
    save_file(outputs, spec["outputs_file"])

    batch_size = spec["batch_size"]
    batch = np.concatenate([images[i % len(images)] for i in range(batch_size)])
    x = mx.array(batch).astype(dtype)
    mx.eval(x)

    def call() -> None:
        frame, patches = forward(x)
        mx.eval(frame, patches)
        np.array(frame.astype(mx.float32))
        np.array(patches.astype(mx.float16 if dtype != mx.float32 else mx.float32))

    warm = run_phase(call, spec["warm_up"])
    times = run_phase(call, spec["timed"])
    values = np.asarray(times, dtype=np.float64)
    return {
        "status": "done",
        "reason": "",
        "load_s": load_s,
        "outputs_file": spec["outputs_file"],
        "output_names": ["embedding", "patch_embeddings"],
        "finite": finite,
        "warm_up_batches": len(warm),
        "warm_up_s": sum(warm),
        "timed_batches": len(times),
        "timed_s": sum(times),
        "times_s": times,
        "patch": {
            "throughput_ips": float(batch_size * len(values) / values.sum()),
            "median_s": float(np.median(values)),
            "p90_s": float(np.percentile(values, 90)),
        },
        "mlx": mx.__version__,
        "mlx_peak_bytes": mx.get_peak_memory(),
        "python": sys.version.split()[0],
    }


def main() -> None:
    """Run the spec of argv[1] and print one JSON line. An error gives the status "error"."""
    spec = json.loads(sys.argv[1])
    try:
        line = run(spec)
    except (
        RuntimeError,
        ValueError,
        TypeError,
        KeyError,
        OSError,
    ) as error:  # the driver records it
        line = {"status": "error", "reason": f"{type(error).__name__}: {error}"}
    print(json.dumps(line))


if __name__ == "__main__":
    main()
