"""Tests T1 to T6 and T8 of specs/06e-kernels/plan.html: the joined projections of part A.

The tests use the small models of tests/small_model.py, so they need no weights. A joined form
must give the same values as the separate form bit for bit (D5).
"""

from collections.abc import Callable

import mlx.core as mx
import numpy as np
import pytest
from mlx import nn
from mlx.utils import tree_flatten
from small_model import PIXEL_SHAPES, small_hf_model, small_mlx_model, small_pixels

from dinov3_mlx.bench import TimeBudget
from dinov3_mlx.mlx_model import (
    MLX_DTYPE_MODES,
    MlxBackbone,
    SwiGLU,
    compile_backbone,
    set_joined_projections,
)

pytestmark = pytest.mark.mlx

DTYPE_MODES = list(MLX_DTYPE_MODES)
SMALL_NAMES = ["vits16", "vits16plus", "vit7b16-shape"]
GATED_NAMES = ["vits16plus", "vit7b16-shape"]
# In fp32 with TF32 off, MLX 0.32.4 has no NAX kernel. A matmul with one batch and
# K >= max(M, N) then takes the split-K kernel (Case 1 of steel_matmul_axpby in
# mlx/backend/metal/matmul.cpp). The joined matmul has N = 3 D > K and takes the regular kernel,
# which sums K in another order (D5). The largest difference on the small models is 4.8e-6.
SPLIT_K_TOLERANCE = 1e-5


def _model(name: str, dtype_mode: str) -> MlxBackbone:
    return small_mlx_model(small_hf_model(seed=0, name=name), dtype_mode)


def _tokens(model: MlxBackbone, seed: int = 2) -> mx.array:
    """Normal tokens (2, 5 + 196, D) in the matmul dtype: the input of the projections."""
    values = np.random.default_rng(seed).standard_normal((2, 201, model.config.dim))
    return mx.array(values.astype(np.float32)).astype(model.dtypes.matmul)


def _difference(a: mx.array, b: mx.array) -> float:
    assert a.shape == b.shape and a.dtype == b.dtype
    return float(np.abs(np.array(a.astype(mx.float32)) - np.array(b.astype(mx.float32))).max())


def _equal(a: mx.array, b: mx.array) -> bool:
    return _difference(a, b) == 0.0


def split_k_separate(model: MlxBackbone, shape: tuple[int, ...]) -> bool:
    """Return True if the separate q, k or v matmul of one pixel shape takes the fp32 split-K
    kernel of MLX: fp32, batch size 1 and the token count M <= D = K = N."""
    batch, _, height, width = shape
    tokens = 5 + (height // 16) * (width // 16)
    fp32 = model.dtypes.matmul == mx.float32
    return fp32 and batch == 1 and tokens <= model.config.dim


@pytest.mark.parametrize("dtype_mode", DTYPE_MODES)
@pytest.mark.parametrize("name", SMALL_NAMES)
def test_joined_qkv_equals_separate(name: str, dtype_mode: str) -> None:
    """T1 (D1, D3): the same q, k and v bit for bit, eager and compiled."""
    model = _model(name, dtype_mode)
    if name != "vit7b16-shape":  # D3: a q bias, a v bias and no k bias
        assert model.config.query_bias and not model.config.key_bias
    attention = model.layer[0].attention
    x = _tokens(model)
    separate = attention.qkv(x)
    compiled = mx.compile(attention.qkv)
    separate_compiled = compiled(x)
    attention.joined = True
    joined = attention.qkv(x)
    joined_compiled = mx.compile(attention.qkv)(x)
    for values in (separate_compiled, joined, joined_compiled):
        assert len(values) == 3
        for a, b in zip(separate, values, strict=True):
            assert _equal(a, b)


@pytest.mark.parametrize("dtype_mode", DTYPE_MODES)
@pytest.mark.parametrize("name", GATED_NAMES)
def test_joined_gate_up_equals_separate(name: str, dtype_mode: str) -> None:
    """T2 (D1): SwiGLU.hidden bit for bit, eager and compiled."""
    model = _model(name, dtype_mode)
    mlp = model.layer[0].mlp
    assert isinstance(mlp, SwiGLU)
    x = _tokens(model)
    separate = mlp.hidden(x)
    separate_compiled = mx.compile(mlp.hidden)(x)
    mlp.joined = True
    assert _equal(separate, mlp.hidden(x))
    assert _equal(separate, mx.compile(mlp.hidden)(x))
    assert _equal(separate, separate_compiled)


@pytest.mark.parametrize("dtype_mode", DTYPE_MODES)
@pytest.mark.parametrize("name", SMALL_NAMES)
def test_joined_forward_equals_separate(
    name: str, dtype_mode: str, report: Callable[..., None]
) -> None:
    """T3: eager and compiled, with and without set_joined_projections, the same tokens bit for
    bit at the 3 pixel shapes. Exception: where the separate matmul takes the fp32 split-K kernel
    (split_k_separate), the largest difference is at most SPLIT_K_TOLERANCE and goes to the
    report."""
    model = _model(name, dtype_mode)
    pixels = [mx.array(small_pixels(shape)) for shape in PIXEL_SHAPES]
    forward = compile_backbone(model)
    separate = [model(p) for p in pixels]
    separate_compiled = [forward(p) for p in pixels]
    mx.eval(separate, separate_compiled)
    set_joined_projections(model, qkv=True, ff=True)
    assert all(block.attention.joined for block in model.layer)
    joined_forward = compile_backbone(model)
    for shape, p, s, sc in zip(PIXEL_SHAPES, pixels, separate, separate_compiled, strict=True):
        assert _equal(s, sc)
        largest = max(_difference(s, model(p)), _difference(s, joined_forward(p)))
        split_k = split_k_separate(model, shape)
        target = SPLIT_K_TOLERANCE if split_k else 0.0
        report(
            "test_joined_forward_equals_separate",
            model=name,
            dtype_mode=dtype_mode,
            shape=list(shape),
            split_k=split_k,
            difference=largest,
            target=target,
        )
        assert largest <= target


def test_set_joined_projections_sets_each_block() -> None:
    """D2: the flags of each block; the MLP has no gate, so only the qkv flag applies to it."""
    gated = _model("vits16plus", "fp32")
    set_joined_projections(gated, qkv=False, ff=True)
    assert [b.attention.joined for b in gated.layer] == [False, False]
    assert [b.mlp.joined for b in gated.layer] == [True, True]
    plain = _model("vits16", "fp32")
    set_joined_projections(plain, qkv=True, ff=True)
    assert [b.attention.joined for b in plain.layer] == [True, True]
    assert not hasattr(plain.layer[0].mlp, "joined")


@pytest.mark.parametrize("name", SMALL_NAMES)
def test_joined_keeps_parameter_names(name: str) -> None:
    """T4 (D2): the same keys and shapes; load_weights(strict=True) still passes."""
    model = _model(name, "bf16-cast")
    before = {key: (value.shape, value.dtype) for key, value in tree_flatten(model.parameters())}
    set_joined_projections(model, qkv=True, ff=True)
    pairs = tree_flatten(model.parameters())
    assert {key: (value.shape, value.dtype) for key, value in pairs} == before
    model.load_weights(pairs, strict=True)
    with pytest.raises(ValueError):
        model.load_weights(pairs[1:], strict=True)


def test_joined_refuses_quantized_weights() -> None:
    """T5 (D4): ValueError for a model after nn.quantize; the separate form stays allowed."""
    model = _model("vits16plus", "fp16-autocast")
    nn.quantize(model, bits=8)
    with pytest.raises(ValueError, match="quantized"):
        set_joined_projections(model, qkv=True, ff=False)
    with pytest.raises(ValueError, match="quantized"):
        set_joined_projections(model, qkv=False, ff=True)
    set_joined_projections(model, qkv=False, ff=False)


def test_joined_follows_a_weight_change() -> None:
    """T6 (R35): the joined weights come from the parameters at each call, so a compiled joined
    forward pass follows a weight change."""
    model = _model("vits16plus", "fp32")
    set_joined_projections(model, qkv=True, ff=True)
    forward = compile_backbone(model)
    pixels = mx.array(small_pixels(PIXEL_SHAPES[0]))
    before = np.array(forward(pixels))
    block = model.layer[0]
    model.update(
        {
            "layer": [
                {
                    "attention": {"k_proj": {"weight": block.attention.k_proj.weight + 0.01}},
                    "mlp": {"up_proj": {"weight": block.mlp.up_proj.weight + 0.01}},
                }
            ]
        }
    )
    after = np.array(forward(pixels))
    assert np.array_equal(after, np.array(model(pixels)))
    assert not np.array_equal(after, before)


def test_change_worker_ratio_of_each_form() -> None:
    """T8 (A2): with four forms, the ratio of each form to the first form in each round and its
    median; the order of the forms rotates. Stub forward functions, no GPU."""
    from dinov3_mlx.mlx_bench import run_forms

    forms = ["R", "R-J", "R-Jqkv", "R-Jff"]
    seconds = {"R": 0.4, "R-J": 0.32, "R-Jqkv": 0.5, "R-Jff": 0.25}
    order: list[str] = []
    # Round 3 of R-J is slower, so the median differs from the mean.
    slow = {("R-J", 3): 2.0}

    def timer(forward: object, batch: object, dtype_mode: str, patch_fp32: bool) -> tuple:
        form = forward(None)
        order.append(form)
        round_ = (order.count(form) - 1) // 3 + 1  # 1 warm-up and 2 timed batches each round
        t = seconds[form] * slow.get((form, round_), 1.0)
        return (t, t, t), None, None

    functions = {form: ((lambda p, form=form: form), False) for form in forms}
    result = run_forms(
        functions,
        forms,
        rounds=5,
        batch=None,
        dtype_mode="fp16-cast",
        batch_size=8,
        warm_up=TimeBudget(1, 0.0),
        timed=TimeBudget(2, 0.0),
        timer=timer,
    )
    starts = order[::3]
    assert starts[:4] == forms
    assert starts[4:8] == forms[1:] + forms[:1]
    ratios = result["form_ratios"]
    assert set(ratios) == {"R-J", "R-Jqkv", "R-Jff"}
    for form in ratios:
        for output in ("frame", "patch", "device"):
            assert len(ratios[form][output]) == 5
    assert ratios["R-Jqkv"]["patch"] == pytest.approx([0.8] * 5)
    assert ratios["R-Jff"]["frame"] == pytest.approx([1.6] * 5)
    assert ratios["R-J"]["patch"][2] == pytest.approx(0.4 / 0.64)
    assert result["form_ratio"]["R-J"]["patch"]["median"] == pytest.approx(1.25)
    assert result["form_ratio"]["R-J"]["patch"]["min"] == pytest.approx(0.625)
    # The two-form fields stay: the second form against the first.
    assert result["ratios"] == ratios["R-J"]
    assert result["ratio"] == result["form_ratio"]["R-J"]
    assert set(result["throughput"]) == set(forms)
