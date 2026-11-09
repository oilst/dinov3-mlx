"""The Core ML worker of Step 6b: one Core ML export, one compute-unit setting, one batch size
(Section 3.5 of specs/06b-prior-art/plan.html).

The driver scripts/bench_prior_art.py starts this file in the Core ML venv with -I:

    ~/.cache/dinov3-mlx/prior-art/venv-coreml/bin/python -I prior_art_coreml_worker.py '<spec>'

It imports only the standard library, NumPy, safetensors, coremltools and Pillow; coremltools and
Pillow only inside the functions. It imports no module of the repo, so it repeats the small batch
code of bench.py (keep_going). It reads the stored pixels (safetensors) and writes:

- one safetensors file with the outputs of the 3 test images at batch size 1 (the driver computes
  the parity with dinov3_mlx.parity), and
- one JSON line on stdout with the batch times.

The spec (JSON) has the fields: package, input ("spec-array", "array" or "image"), units,
batch_size, side, pixels ("stored" or "random"), inputs_file, outputs_file, warm_up and timed
(the fields of TimeBudget), parity_only.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

TEST_IMAGES = ["iss-kibo-lab", "forest", "iss-conference"]  # the order of parity.TEST_IMAGES
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
RANDOM_SEED = 0


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


def load_export(path: Path, input_kind: str, units: str) -> object:
    """Load one export. With the spec-array input, the image input of the loaded spec becomes a
    FLOAT32 multiArray (1, 3, H, W); the package on disk does not change (D3)."""
    import coremltools as ct

    unit = getattr(ct.ComputeUnit, units)
    if input_kind != "spec-array":
        return ct.models.MLModel(str(path), compute_units=unit)
    spec = ct.utils.load_spec(str(path))
    feature = spec.description.input[0]  # imageType (H, W, RGB) in the package
    height, width = feature.type.imageType.height, feature.type.imageType.width
    feature.type.multiArrayType.shape[:] = [1, 3, height, width]  # the oneof moves to multiArray
    feature.type.multiArrayType.dataType = ct.proto.FeatureTypes_pb2.ArrayFeatureType.FLOAT32
    weights = path / "Data" / "com.apple.CoreML" / "weights"
    return ct.models.MLModel(spec, weights_dir=str(weights), compute_units=unit)


def to_input(pixels: np.ndarray, input_kind: str, name: str) -> dict:
    """Return the input dict of one image: normalized pixels (array), 0..255 floats without
    rounding (spec-array) or an 8-bit RGB image (image)."""
    if input_kind == "array":
        return {name: pixels.astype(np.float32)}
    rgb = (pixels * STD + MEAN) * 255.0
    if input_kind == "spec-array":
        return {name: rgb.astype(np.float32)}
    from PIL import Image

    hwc = np.clip(np.rint(rgb[0]), 0, 255).astype(np.uint8).transpose(1, 2, 0)
    return {name: Image.fromarray(hwc, mode="RGB")}


def read_pixels(spec: dict) -> list[np.ndarray]:
    """Return the pixels (1, 3, H, W) of the 3 test images, or 1 random image."""
    side = spec["side"]
    if spec["pixels"] == "random":
        rng = np.random.default_rng(RANDOM_SEED)
        return [rng.standard_normal((1, 3, side, side)).astype(np.float32)]
    with safe_open(spec["inputs_file"], "np") as tensors:
        return [tensors.get_tensor(f"{image}/{side}x{side}/pixels") for image in TEST_IMAGES]


def run(spec: dict) -> dict:
    """Load the export, predict the test images at batch size 1, then time the batches."""
    import coremltools as ct

    start = time.perf_counter()
    model = load_export(Path(spec["package"]), spec["input"], spec["units"])
    load_s = time.perf_counter() - start
    name = model.get_spec().description.input[0].name
    images = read_pixels(spec)
    outputs = {}
    start = time.perf_counter()
    for index, pixels in enumerate(images):
        result = model.predict(to_input(pixels, spec["input"], name))
        key = TEST_IMAGES[index] if spec["pixels"] == "stored" else "random"
        for output, value in result.items():
            outputs[f"{key}/{output}"] = np.asarray(value, dtype=np.float32)
    first_predict_s = time.perf_counter() - start
    finite = all(bool(np.isfinite(value).all()) for value in outputs.values())
    Path(spec["outputs_file"]).parent.mkdir(parents=True, exist_ok=True)
    save_file(outputs, spec["outputs_file"])

    batch_size = spec["batch_size"]
    inputs = [to_input(images[i % len(images)], spec["input"], name) for i in range(batch_size)]
    if batch_size == 1:

        def call() -> object:
            return model.predict(inputs[0])
    else:

        def call() -> object:
            return model.predict(inputs)  # the batch API: one call with a list of inputs

    warm = run_phase(call, spec["warm_up"])
    times = run_phase(call, spec["timed"])
    line = {
        "status": "done",
        "reason": "",
        "load_s": load_s,
        "first_predict_s": first_predict_s,
        "input_name": name,
        "output_names": sorted({key.split("/", 1)[1] for key in outputs}),
        "outputs_file": spec["outputs_file"],
        "finite": finite,
        "warm_up_batches": len(warm),
        "warm_up_s": sum(warm),
        "timed_batches": len(times),
        "timed_s": sum(times),
        "times_s": times,
        "coremltools": ct.__version__,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
    }
    if times:
        values = np.asarray(times, dtype=np.float64)
        line["patch"] = {
            "throughput_ips": float(batch_size * len(values) / values.sum()),
            "median_s": float(np.median(values)),
            "p90_s": float(np.percentile(values, 90)),
        }
    return line


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
