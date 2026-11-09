"""Tests T1 and T2 of specs/07-release/plan.html: the public API from the image files against the
reference output, in the 12 configurations.

T1 runs ViT-S/16 in the default session and all six model sizes in the slow session
(DINOV3_SLOW=1). The tests skip without the weights or the reference file: the Embedder must not
download in a test.
"""

from collections.abc import Callable, Iterator

import numpy as np
import pytest
from parity_checks import assert_thresholds, record_skip, run_rule_skip
from under_test import SLOW

from dinov3_mlx import api
from dinov3_mlx.hub import MODEL_IDS, WeightsNotFoundError, weights_dir
from dinov3_mlx.memory import InsufficientMemoryError
from dinov3_mlx.outputs import NUM_PREFIX_TOKENS
from dinov3_mlx.parity import measure_parity
from dinov3_mlx.preprocess import load_image
from dinov3_mlx.stored import configurations

pytestmark = pytest.mark.mlx

BACKEND = "mlx"
KIND = "api"
CONFIGS = configurations()
T1_MODEL_SIZES = list(MODEL_IDS) if SLOW else ["ViT-S/16"]


@pytest.fixture(scope="module")
def embedder(backbone_slot: object) -> Iterator[Callable[[str, str], api.Embedder]]:
    """Return a getter that keeps one Embedder in memory. It frees the backbone slot first."""
    held: dict[tuple[str, str], api.Embedder] = {}

    def get(model_size: str, dtype_mode: str) -> api.Embedder:
        if (model_size, dtype_mode) not in held:
            held.clear()
            backbone_slot.free()
            try:
                weights_dir(model_size)  # no download in a test
                held[(model_size, dtype_mode)] = api.Embedder(model_size, dtype_mode)
            except (WeightsNotFoundError, InsufficientMemoryError) as error:
                pytest.skip(str(error))
        return held[(model_size, dtype_mode)]

    yield get
    held.clear()
    backbone_slot.free()


def check_parity(
    model_size: str,
    dtype_mode: str,
    embedder: Callable[[str, str], api.Embedder],
    reference: Callable[[str], dict[str, np.ndarray]],
    record: Callable[..., None],
) -> None:
    """Embed each test image through Embedder.embed and assert the thresholds of the dtype mode
    in each of the 12 configurations."""
    key = (KIND, model_size, dtype_mode, "all")
    if reason := run_rule_skip(model_size, dtype_mode, BACKEND):
        record(*key, None, "skipped", reason, backend=BACKEND)
        pytest.skip(reason)
    with record_skip(record, *key, backend=BACKEND):
        expected = reference(model_size)
        model = embedder(model_size, dtype_mode)
    failed = []
    for config in CONFIGS:
        frame, patches = model.embed(
            load_image(config.image_path)[None], size=(config.height, config.width)
        )
        stored = expected[config.id]
        tokens = np.concatenate(
            [frame, stored[1:NUM_PREFIX_TOKENS], patches[0].reshape(-1, frame.shape[-1])]
        )
        parity = measure_parity(stored, tokens)
        record(KIND, model_size, dtype_mode, config.id, parity, "measured", backend=BACKEND)
        try:
            assert_thresholds(parity, dtype_mode)
        except AssertionError:
            failed.append((config.id, parity))
    assert failed == []


@pytest.mark.parametrize(
    "model_size", [pytest.param(size, id=f"{BACKEND}-{size}") for size in T1_MODEL_SIZES]
)
def test_embedder_parity(
    model_size: str,
    embedder: Callable[[str, str], api.Embedder],
    reference: Callable[[str], dict[str, np.ndarray]],
    record: Callable[..., None],
) -> None:
    """T1: the default dtype mode fp16-autocast, 12 configurations."""
    check_parity(model_size, "fp16-autocast", embedder, reference, record)


@pytest.mark.parametrize("dtype_mode", ["fp32", "bf16-cast"])
def test_embedder_parity_other_modes(
    dtype_mode: str,
    embedder: Callable[[str, str], api.Embedder],
    reference: Callable[[str], dict[str, np.ndarray]],
    record: Callable[..., None],
) -> None:
    """T2: ViT-S/16 in fp32 and bf16-cast, with the thresholds of each dtype mode."""
    check_parity("ViT-S/16", dtype_mode, embedder, reference, record)
