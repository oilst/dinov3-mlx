"""Tests T3, T4 and T6 to T9 of specs/07-release/plan.html: the public API with the MLX small
model. The tests need no weights, so they also run in CI."""

import json
import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image
from small_mlx import load_small_mlx
from small_model import small_hf_model
from under_test import SLOW

from dinov3_mlx import api, hub
from dinov3_mlx.hub import MODEL_IDS, WeightsNotFoundError
from dinov3_mlx.memory import RunRuleError
from dinov3_mlx.mlx_model import MLX_DTYPE_MODES, TF32ModeError

pytestmark = pytest.mark.mlx

TESTS_DIR = Path(__file__).parent
# Small frames and a small input resolution keep each test fast.
FRAME_SHAPE = (40, 56, 3)
SIZE = (32, 48)
GRID = (SIZE[0] // 16, SIZE[1] // 16)
WIDTH = 384  # the hidden size of the small model "vits16"
PADDED_COSINE = 0.99999  # AC3

Make = Callable[..., api.Embedder]


@pytest.fixture(scope="module")
def small_folder(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The saved weights of the small model "vits16", in the key spelling of the Hub files."""
    from small_model import save_small_model

    folder = tmp_path_factory.mktemp("small") / "vits16"
    save_small_model(small_hf_model(), folder, files=1)
    return folder


@pytest.fixture
def make(small_folder: Path, monkeypatch: pytest.MonkeyPatch) -> Make:
    """Return a constructor of Embedder objects that load the small model."""
    monkeypatch.setattr(api, "ensure_weights", lambda model_size: small_folder)
    monkeypatch.setattr(
        api,
        "load_mlx_backbone",
        lambda model_size, dtype_mode, bits=None: load_small_mlx(small_folder, dtype_mode),
    )

    def construct(**kwargs: object) -> api.Embedder:
        return api.Embedder(**kwargs)

    return construct


def frames(count: int, seed: int = 0, shape: tuple[int, ...] = FRAME_SHAPE) -> np.ndarray:
    """Random uint8 RGB frames (count, h, w, 3)."""
    return np.random.default_rng(seed).integers(0, 256, (count, *shape), dtype=np.uint8)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The cosine along the last axis, in fp64."""
    a, b = a.astype(np.float64), b.astype(np.float64)
    return (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1))


def run_python(code: str, env: dict[str, str] | None = None, *args: str) -> dict:
    """Run code in a new process and return the JSON of its last output line."""
    environ = {k: v for k, v in os.environ.items() if k != "MLX_ENABLE_TF32"} | (env or {})
    environ["PYTHONPATH"] = str(TESTS_DIR)
    result = subprocess.run(
        [sys.executable, "-c", code, *args],
        check=False,
        capture_output=True,
        text=True,
        env=environ,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_padded_batch_equals_unpadded(make: Make) -> None:
    """T3: 5 frames with batch size 8 (3 padded rows) against batch size 5."""
    batch = frames(5)
    padded = make(batch_size=8).embed(batch, size=SIZE)
    plain = make(batch_size=5).embed(batch, size=SIZE)
    assert padded[0].shape[0] == padded[1].shape[0] == 5
    assert cosine(padded[0], plain[0]).min() >= PADDED_COSINE
    patch_mean = cosine(padded[1], plain[1]).reshape(5, -1).mean(axis=1)
    assert patch_mean.min() >= PADDED_COSINE


# T4: a process that refuses torch and transformers embeds frames through the public API.
NO_TORCH = """
import json, sys
from pathlib import Path

BLOCKED = ("torch", "transformers")


class Refuse:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError(f"{name} is blocked")
        return None


sys.meta_path.insert(0, Refuse())
import numpy as np
import dinov3_mlx
from dinov3_mlx import api
from small_mlx import load_small_mlx

folder = Path(sys.argv[1])
api.ensure_weights = lambda model_size: folder
api.load_mlx_backbone = lambda model_size, dtype_mode, bits=None: load_small_mlx(folder, dtype_mode)
model = dinov3_mlx.Embedder("ViT-S/16")
batch = np.random.default_rng(0).integers(0, 256, (3, 40, 56, 3), dtype=np.uint8)
frame, patches = model.embed(batch, size=(32, 48))
print(json.dumps({
    "frame": list(frame.shape),
    "patches": list(patches.shape),
    "finite": bool(np.isfinite(frame).all() and np.isfinite(patches).all()),
    "imported": [name for name in BLOCKED if name in sys.modules],
}))
"""


def test_api_imports_without_torch(small_folder: Path) -> None:
    """T4: the API imports and embeds without torch and transformers."""
    line = run_python(NO_TORCH, None, str(small_folder))
    assert line == {
        "frame": [3, WIDTH],
        "patches": [3, *GRID, WIDTH],
        "finite": True,
        "imported": [],
    }


IMPORT_ONLY = """
import json, os, sys
import dinov3_mlx
print(json.dumps({
    "MLX_ENABLE_TF32": os.environ.get("MLX_ENABLE_TF32"),
    "imported": [name for name in ("mlx", "torch", "transformers") if name in sys.modules],
}))
"""


def test_tf32_import_sets_the_default() -> None:
    """T6a: import dinov3_mlx sets MLX_ENABLE_TF32=0 and imports no mlx and no torch."""
    assert run_python(IMPORT_ONLY) == {"MLX_ENABLE_TF32": "0", "imported": []}


PROBE = """
import json
import dinov3_mlx
from dinov3_mlx.mlx_model import tf32_active
print(json.dumps({"active": tf32_active()}))
"""


def test_tf32_probe_finds_the_state() -> None:
    """T6b: the probe finds TF32 on and off, in two new processes."""
    assert run_python(PROBE, {"MLX_ENABLE_TF32": "1"}) == {"active": True}
    assert run_python(PROBE, {"MLX_ENABLE_TF32": "0"}) == {"active": False}


def test_tf32_embedder_refuses_tf32(make: Make, monkeypatch: pytest.MonkeyPatch) -> None:
    """T6c: the Embedder raises TF32ModeError with the fix when the probe finds TF32 on."""
    monkeypatch.setattr(api, "tf32_active", lambda: True)
    with pytest.raises(TF32ModeError) as error:
        make()
    assert "import dinov3_mlx before mlx" in str(error.value)
    assert "MLX_ENABLE_TF32=0" in str(error.value)


def test_tf32_explicit_value_stays() -> None:
    """T6d: an explicit MLX_ENABLE_TF32=1 stays after the import."""
    line = run_python(IMPORT_ONLY, {"MLX_ENABLE_TF32": "1"})
    assert line["MLX_ENABLE_TF32"] == "1"


def test_embedder_keeps_the_memory_limit(
    small_folder: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T7: the memory limit after the constructor equals the limit before it."""

    def loader(model_size: str, dtype_mode: str, bits: int | None = None) -> object:
        mx.set_memory_limit(123_456_789)  # as load_mlx_backbone sets the need of its load
        return load_small_mlx(small_folder, dtype_mode)

    monkeypatch.setattr(api, "ensure_weights", lambda model_size: small_folder)
    monkeypatch.setattr(api, "load_mlx_backbone", loader)
    before = mx.get_memory_limit()
    api.Embedder()
    assert mx.get_memory_limit() == before


def test_embedder_inputs_give_the_same_outputs(make: Make) -> None:
    """T8: an array, a list of PIL images and a list of arrays give the same outputs."""
    model = make()
    batch = frames(3)
    expected = model.embed(batch, size=SIZE)
    for form in ([Image.fromarray(frame) for frame in batch], list(batch)):
        frame, patches = model.embed(form, size=SIZE)
        assert np.array_equal(frame, expected[0])
        assert np.array_equal(patches, expected[1])


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(frames(2).astype(np.float32), id="float-array"),
        pytest.param(list(frames(2).astype(np.float32)), id="float-list"),
        pytest.param([b"frame"], id="bytes"),
    ],
)
def test_embedder_inputs_refuse_other_dtypes(make: Make, bad: object) -> None:
    """T8: another dtype or another type raises TypeError."""
    with pytest.raises(TypeError):
        make().embed(bad, size=SIZE)


@pytest.mark.parametrize(
    ("bad", "size"),
    [
        pytest.param([frames(1)[0], frames(1, shape=(48, 56, 3))[0]], SIZE, id="mixed-sizes"),
        pytest.param(frames(2), (30, 48), id="side-30"),
        pytest.param(frames(2), 100, id="side-100"),
        pytest.param(frames(2, shape=(40, 56, 4)), SIZE, id="four-channels"),
        pytest.param(frames(0), SIZE, id="no-frames"),
    ],
)
def test_embedder_inputs_refuse_bad_shapes(make: Make, bad: object, size: object) -> None:
    """T8: frames of different sizes and a side that is not a multiple of 16 raise ValueError."""
    with pytest.raises(ValueError):
        make().embed(bad, size=size)


@pytest.mark.parametrize("dtype_mode", list(MLX_DTYPE_MODES))
def test_embedder_inputs_shapes_and_dtypes(make: Make, dtype_mode: str) -> None:
    """T8: the shapes and dtypes of Table 2 in each dtype mode."""
    model = make(dtype_mode=dtype_mode)
    batch = frames(3)
    patch_dtype = np.float32 if dtype_mode == "fp32" else np.float16
    frame, patches = model.embed(batch, size=SIZE)
    assert (frame.shape, frame.dtype) == ((3, WIDTH), np.float32)
    assert (patches.shape, patches.dtype) == ((3, *GRID, WIDTH), patch_dtype)
    assert model.embed_frames(batch, size=SIZE).shape == (3, WIDTH)
    assert model.embed_patches(batch, size=SIZE).dtype == patch_dtype
    assert model.embed_patches(batch, size=SIZE, patch_fp32=True).dtype == np.float32
    on_gpu = model.embed(batch, size=SIZE, as_numpy=False)
    assert all(isinstance(output, mx.array) for output in on_gpu)
    assert isinstance(model.embed_frames(batch, size=SIZE, as_numpy=False), mx.array)
    assert np.array_equal(np.array(on_gpu[0]), frame)


def test_embedder_inputs_default_size(make: Make) -> None:
    """T8: the default input resolution is 224 x 224."""
    patches = make().embed_patches(frames(1, shape=(60, 80, 3)))
    assert patches.shape == (1, 14, 14, WIDTH)


def test_embedder_inputs_model_size_names(make: Make) -> None:
    """T8: a model size name or its Hugging Face id; another name raises ValueError."""
    assert make(model_size=MODEL_IDS["ViT-S/16"]).model_size == "ViT-S/16"
    with pytest.raises(ValueError):
        make(model_size="ViT-X/16")
    with pytest.raises(ValueError):
        make(dtype_mode="fp8")


def test_embedder_inputs_backend(make: Make) -> None:
    """T8: backend accepts only "mlx"; "mlx+ne" names Step 7b."""
    with pytest.raises(ValueError, match="Step 7b"):
        make(backend="mlx+ne")
    with pytest.raises(ValueError):
        make(backend="torch")


def test_embedder_run_rule_before_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """ViT-7B/16 in fp32 stops at the run rule, before a weight look-up or a download."""

    def no_weights(model_size: str) -> Path:
        raise AssertionError("no weight look-up before the run rule")

    monkeypatch.setattr(api, "ensure_weights", no_weights)
    with pytest.raises(RunRuleError):
        api.Embedder("ViT-7B/16", dtype_mode="fp32")


def test_embedder_tf32_before_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """V8 of the Step 7 verify: the TF32 probe runs before a weight look-up or a download."""

    def no_weights(model_size: str) -> Path:
        raise AssertionError("no weight look-up before the TF32 probe")

    monkeypatch.setattr(api, "tf32_active", lambda: True)
    monkeypatch.setattr(api, "ensure_weights", no_weights)
    with pytest.raises(TF32ModeError):
        api.Embedder()


@pytest.mark.parametrize(
    ("bits", "dtype_mode"),
    [(4, "fp16-autocast"), (2, "fp16-autocast"), (8, "fp32"), (8, "bf16-cast")],
)
def test_embedder_bits_only_8_in_autocast(
    bits: int, dtype_mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """V1 of the Step 7 verify: bits takes only None and 8, and 8 only in the two autocast modes
    (Table 2 of the plan; 4 bits are not valid in 72 of 72 rows of Step 6). The check runs before
    a weight look-up or a download."""

    def no_weights(model_size: str) -> Path:
        raise AssertionError("no weight look-up before the bits check")

    monkeypatch.setattr(api, "ensure_weights", no_weights)
    with pytest.raises(ValueError, match="bits"):
        api.Embedder(dtype_mode=dtype_mode, bits=bits)


def gated(*args: object, **kwargs: object) -> None:
    """A fake Hub download that refuses access to a gated repo."""
    import httpx2
    from huggingface_hub.errors import GatedRepoError

    request = httpx2.Request("GET", "https://huggingface.co/api/models")
    raise GatedRepoError("401 gated", response=httpx2.Response(401, request=request))


def test_gated_repo_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """T9: a refused download raises WeightsAccessError with the fix."""

    def missing(model_size: str) -> Path:
        raise WeightsNotFoundError(model_size)

    monkeypatch.setattr(hub, "weights_dir", missing)
    monkeypatch.setattr(hub, "snapshot_download", gated)
    with pytest.raises(api.WeightsAccessError) as error:
        api.Embedder("ViT-S/16")
    text = str(error.value)
    assert "https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m" in text
    assert "DINOv3 License" in text
    assert "hf auth login" in text


COUNT_DOWNLOADS = """
import json
import huggingface_hub

calls = []
huggingface_hub.snapshot_download = lambda *args, **kwargs: calls.append(args)
import dinov3_mlx
from dinov3_mlx import Embedder, WeightsAccessError
print(json.dumps({"downloads": len(calls)}))
"""


def test_gated_repo_no_download_on_import() -> None:
    """T9: the import of the package and of the API downloads nothing."""
    assert run_python(COUNT_DOWNLOADS) == {"downloads": 0}


@pytest.mark.skipif(not SLOW, reason="the wheel test belongs to the slow session (DINOV3_SLOW=1)")
def test_wheel_installs_without_torch(small_folder: Path, tmp_path: Path) -> None:
    """T5: the wheel installs with the core packages only, and T4 runs in its venv."""
    root = TESTS_DIR.parent
    dist, venv = tmp_path / "dist", tmp_path / "venv"
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(dist)], cwd=root, check=True)
    wheel = next(dist.glob("dinov3_mlx-*.whl"))
    subprocess.run(["uv", "venv", "--python", sys.executable, str(venv)], check=True)
    python = str(venv / "bin" / "python")
    subprocess.run(["uv", "pip", "install", "--python", python, str(wheel)], check=True)
    listed = subprocess.run(
        ["uv", "pip", "list", "--python", python], capture_output=True, text=True, check=True
    ).stdout
    assert not {"torch", "transformers"} & {line.split()[0] for line in listed.splitlines()[2:]}
    environ = {k: v for k, v in os.environ.items() if k not in ("MLX_ENABLE_TF32", "VIRTUAL_ENV")}
    result = subprocess.run(
        [python, "-c", NO_TORCH, str(small_folder)],
        capture_output=True,
        text=True,
        env=environ | {"PYTHONPATH": str(TESTS_DIR)},
        cwd=tmp_path,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    line = json.loads(result.stdout.strip().splitlines()[-1])
    assert line["imported"] == [] and line["finite"]
