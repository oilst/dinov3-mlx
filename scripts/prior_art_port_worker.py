"""The port worker of Step 6b: one Python MLX port, one dtype, one batch size (job J5, gated by
D2 of specs/06b-prior-art/explore.html).

The driver scripts/bench_prior_art.py starts this file only with --allow-third-party-python, the
user's yes to question Q1. It runs in the port venv with -I:

    ~/.cache/dinov3-mlx/prior-art/venv-mlx/bin/python -I prior_art_port_worker.py '<spec>'

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
import re
import sys
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

TEST_IMAGES = ["iss-kibo-lab", "forest", "iss-conference"]  # the order of parity.TEST_IMAGES
RANDOM_SEED = 0
NUM_PREFIX_TOKENS = 5  # 1 CLS token and 4 register tokens
DINOV2_PREFIX_TOKENS = 1  # the dinov2 model of mlx-image has no register tokens
# The model names of mlx-image (C1).
MLXIM_NAMES = {"ViT-S/16": "vit_small_patch16_224.dinov3", "dinov2": "vit_small_patch14_518.dinov2"}
# The parameters of C2 that our weights file does not have: the port keeps its own values for them.
PORT_OWN_KEYS = re.compile(r"^rope_embed\.periods$|\.bias_mask$")
# The parameters of the dinov2 model of C1 that the mlx-vision file does not have: the layer
# scales. The port keeps its init value 1e-5 for them; the dinov2 cells run on random pixels and
# measure the speed only.
DINOV2_OWN_KEYS = re.compile(r"\.ls[12]\.gamma$")


def keep_going(budget: dict, batches: int, elapsed_s: float) -> bool:
    """Return True while the phase needs another batch: keep_going of bench.py, copied."""
    max_seconds = budget.get("max_seconds")
    stop_time = max_seconds is not None and elapsed_s >= max_seconds
    if stop_time and batches >= budget.get("floor_batches", 0):
        return False
    return batches < budget["min_batches"] or elapsed_s < budget["min_seconds"]


def run_phase(call: object, budget: dict) -> list[tuple[float, float, float]]:
    """Run batches until the time budget ends: phase of mlx_bench.py, copied. The elapsed time is
    the sum of the patch times."""
    times: list[tuple[float, float, float]] = []
    elapsed = 0.0
    while keep_going(budget, len(times), elapsed):
        batch_times = call()
        times.append(batch_times)
        elapsed += batch_times[2]
    return times


def summarize(times: list[float], batch_size: int) -> dict[str, float]:
    """Return the throughput, the median and the p90 of the batch times: summarize of bench.py."""
    values = np.asarray(times, dtype=np.float64)
    return {
        "throughput_ips": float(batch_size * len(values) / values.sum()),
        "median_s": float(np.median(values)),
        "p90_s": float(np.percentile(values, 90)),
    }


def port_weights(
    params: dict, loaded: dict, own_keys: re.Pattern = PORT_OWN_KEYS
) -> tuple[list[tuple[str, object]], list[str]]:
    """Return the weight pairs of a strict load and the keys that keep the values of the port.

    Only the keys of own_keys keep the values of the port: the RoPE periods and the k-bias masks
    of C2 (PORT_OWN_KEYS), or the layer scales of dinov2 (DINOV2_OWN_KEYS). Any other key that
    the weights file does not have stops the worker, so a wrong key map cannot give random
    weights (finding V8 of specs/06b-prior-art/verify.html).

    Raises:
        KeyError: For a missing key outside own_keys.
    """
    missing = sorted(set(params) - set(loaded))
    unknown = [key for key in missing if not own_keys.search(key)]
    if unknown:
        raise KeyError(f"not in our weights file: {unknown[:5]}")
    return list(loaded.items()) + [(key, params[key]) for key in missing], missing


def build_c1(spec: dict) -> object:
    """Return the model of mlx-image (C1) with the safetensors weights of the mlx-vision org.

    The dinov2 file has separate q, k and v projections and no layer scales. The loader of
    mlx-image does not join q, k and v, so the worker joins them with Attention.sanitize of the
    port and loads the weights itself (J5 run 1).
    """
    sys.path.insert(0, spec["clone"])
    from mlxim.model import create_model

    name = MLXIM_NAMES[spec["model"]]
    if spec["model"] != "dinov2":
        return create_model(name, weights=spec["weights"], num_classes=0, strict=True), []
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlxim.model.vit.vit import Attention

    model = create_model(name, weights=False, num_classes=0)
    params = dict(tree_flatten(model.parameters()))
    loaded = Attention.sanitize(mx.load(spec["weights"]))
    weights, own = port_weights(params, loaded, DINOV2_OWN_KEYS)
    model.load_weights(weights, strict=True)  # strict: an extra key of the file stops the load
    return model, own


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
    loaded = {
        key: value.astype(params[key].dtype) if key in params else value
        for key, value in mx.load(spec["weights"]).items()
    }
    weights, own = port_weights(params, loaded)
    model.load_weights(weights, strict=True)  # strict: an extra key of our file stops the load
    return model, own


def forward_function(model: object, spec: dict) -> object:
    """Return forward(pixels) -> (frame embeddings, patch embeddings) of the port."""
    import mlx.core as mx

    if spec["model"] == "dinov2":

        def forward(x: mx.array) -> tuple[mx.array, mx.array]:
            tokens = model.encoder(model._prepare_tokens(x.transpose(0, 2, 3, 1)))[0]
            return tokens[:, 0], tokens[:, DINOV2_PREFIX_TOKENS:]

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
    model, own = build_c1(spec) if spec["candidate"] == "C1" else build_c2(spec)
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
    patch_dtype = mx.float32 if dtype == mx.float32 else mx.float16

    def call() -> tuple[float, float, float]:
        """One batch host to host with the three times of time_batch_mlx in mlx_bench.py: the
        device time, the frame time and the patch time (finding V3 of verify.html)."""
        t0 = time.perf_counter()
        frame, patches = forward(mx.array(batch).astype(dtype))  # the copy and the cast inside
        mx.eval(frame, patches)
        t1 = time.perf_counter()
        np.array(frame.astype(mx.float32))
        t2 = time.perf_counter()
        np.array(patches.astype(patch_dtype))
        t3 = time.perf_counter()
        return t1 - t0, t2 - t0, (t1 - t0) + (t3 - t2)

    warm = run_phase(call, spec["warm_up"])
    times = run_phase(call, spec["timed"])
    device, frame_t, patch_t = (list(column) for column in zip(*times, strict=True))
    return {
        "status": "done",
        "reason": "",
        "load_s": load_s,
        "outputs_file": spec["outputs_file"],
        "output_names": ["embedding", "patch_embeddings"],
        "finite": finite,
        "warm_up_batches": len(warm),
        "warm_up_s": sum(t[2] for t in warm),
        "timed_batches": len(times),
        "timed_s": sum(patch_t),
        "patch_times_s": patch_t,
        "frame": summarize(frame_t, batch_size),
        "patch": summarize(patch_t, batch_size),
        "device": summarize(device, batch_size),
        "own_values": own,
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
