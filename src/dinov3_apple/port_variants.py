"""The port-change variants of Step 6b: our backbone with one change of a port at a time, to
measure the cost of that change in parity and speed (D8 of specs/06b-prior-art/explore.html).

V0 is our compiled forward pass. V1 is the RoPE of the Meta-based MLX ports C1 and C2 (bf16
tables, q and k rotated in bf16), V2 its bf16 tables with our fp32 rotation, V3 the Meta periods
(the fp32 periods rounded to bf16), V4 the unfused attention, V5 the 8-bit image input. We wrote
this code from a reading of the ports (Table 2 of explore.html); no code of a candidate runs here.

The fp32 dtype mode needs MLX_ENABLE_TF32=0, as in mlx_model.py.
"""

import math
import time
from collections.abc import Callable
from datetime import UTC, datetime

import mlx.core as mx
import numpy as np

from dinov3_apple import stored
from dinov3_apple.bench import (
    BenchConfig,
    TimeBudget,
    check_validity,
    make_batch,
    required_bench_bytes,
)
from dinov3_apple.machine import read_versions
from dinov3_apple.mlx_model import (
    Attention,
    MlxBackbone,
    apply_rope,
    compile_backbone,
    rope_layout,
)
from dinov3_apple.outputs import NUM_PREFIX_TOKENS
from dinov3_apple.parity import THRESHOLDS, measure_parity
from dinov3_apple.prior_art import VARIANTS, eight_bit_pixels, parity_passes
from dinov3_apple.stored import read_inputs, read_reference

Forward = Callable[[mx.array], mx.array]


def port_rope_tables(
    grid_h: int, grid_w: int, head_dim: int, base: float
) -> tuple[mx.array, mx.array]:
    """Return (cos, sin), each (Hp · Wp, head_dim) in bf16, with the steps of the Meta-based ports.

    The periods, the coordinates, the angles, sin and cos are bf16, as with the default
    pos_embed_rope_dtype="bf16" of C1 and C2. The ports return (sin, cos); this function returns
    our order. The tables have no rows for the 5 prefix tokens.

    Args:
        grid_h: The number of patches along the height.
        grid_w: The number of patches along the width.
        head_dim: The number of values of one head.
        base: The base of the periods (rope_theta, 100 in DINOv3).
    """
    bf16 = mx.bfloat16
    periods = base ** (2 * mx.arange(head_dim // 4, dtype=bf16) / (head_dim // 2))
    coords_h = mx.arange(0.5, grid_h, dtype=bf16) / grid_h
    coords_w = mx.arange(0.5, grid_w, dtype=bf16) / grid_w
    mesh = mx.stack(mx.meshgrid(coords_h, coords_w, indexing="ij"), axis=-1).reshape(-1, 2)
    mesh = 2.0 * mesh - 1.0
    angles = 2 * math.pi * mesh[:, :, None] / periods[None, None, :]
    angles = mx.tile(angles.reshape(angles.shape[0], -1), (1, 2))
    return mx.cos(angles), mx.sin(angles)


def port_rotate(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate the patch tokens of x as the ports do: x · cos + rotate_half(x) · sin, in the dtype
    of x. The prefix tokens (the first T − N tokens) keep their values without a rotation.

    Args:
        x: Queries or keys (B, heads, T, head_dim).
        cos: The cos table (N, head_dim) of the patch tokens.
        sin: The sin table (N, head_dim) of the patch tokens.
    """
    prefix = x.shape[-2] - cos.shape[0]
    patches = x[..., prefix:, :]
    half = x.shape[-1] // 2
    rotated = mx.concatenate([-patches[..., half:], patches[..., :half]], axis=-1)
    return mx.concatenate([x[..., :prefix, :], patches * cos + rotated * sin], axis=-2)


def port_apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate queries or keys as the Meta-based ports do: all tokens to bf16 (the prefix tokens
    too), port_rotate in bf16, then back to the dtype of x.

    Args:
        x: Queries or keys (B, heads, T, head_dim).
        cos: The cos table of port_rope_tables, (N, head_dim), bf16.
        sin: The sin table of port_rope_tables, (N, head_dim), bf16.
    """
    return port_rotate(x.astype(mx.bfloat16), cos, sin).astype(x.dtype)


class Bf16RotationAttention(Attention):
    """Attention with the rotation of the Meta-based ports: q and k go to bf16 (all tokens, the 5
    prefix tokens too), the patch tokens rotate in bf16, then q and k go back (V1).

    cos and sin are the tables of port_rope_tables (patch tokens only).
    """

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        q, k, v = self.qkv(x)
        q, k = port_apply_rope(q, cos, sin), port_apply_rope(k, cos, sin)
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=None)
        return self.output(a)


class UnfusedAttention(Attention):
    """Attention without the fused kernel, as the ports and the Core ML programs run it (V4):
    softmax(matmul(q, kᵀ) · scale) @ v in the matmul dtype; the score matrix lives in memory."""

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        q, k, v = self.qkv(x)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        scores = mx.matmul(q, k.swapaxes(-1, -2)) * self.scale
        return self.output(mx.matmul(mx.softmax(scores, axis=-1), v))


def meta_periods(head_dim: int, base: float) -> np.ndarray:
    """Return the RoPE periods of the Meta checkpoints: base^(k / (head_dim / 4)), computed in
    fp64, rounded to fp32 and then to bf16 (round to nearest even), as fp32 values."""
    exact = base ** (np.arange(head_dim // 4, dtype=np.float64) / (head_dim // 4))
    bits = exact.astype(np.float32).view(np.uint32).astype(np.uint64)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.astype(np.uint32).view(np.float32)


def meta_rope_tables(
    grid_h: int, grid_w: int, head_dim: int, base: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the tables of V3: the steps of rope_tables in fp32, with the Meta periods.

    Returns:
        cos and sin, each (5 + Hp · Wp, head_dim), fp32, with identity rows for the prefix tokens.
    """
    periods = meta_periods(head_dim, base)
    coord_h = np.arange(0.5, grid_h, dtype=np.float32) / grid_h
    coord_w = np.arange(0.5, grid_w, dtype=np.float32) / grid_w
    coords = np.stack(np.meshgrid(coord_h, coord_w, indexing="ij"), axis=-1).reshape(-1, 2)
    coords = 2.0 * coords - 1.0
    angles = (2 * np.pi * coords[:, :, None] / periods[None, None, :]).reshape(len(coords), -1)
    angles = np.tile(angles, 2)
    prefix = np.ones((NUM_PREFIX_TOKENS, head_dim), np.float32)
    return np.concatenate([prefix, np.cos(angles)]), np.concatenate([0 * prefix, np.sin(angles)])


def _with_prefix(cos: np.ndarray, sin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prefix = np.ones((NUM_PREFIX_TOKENS, cos.shape[1]), np.float32)
    return np.concatenate([prefix, cos]), np.concatenate([0 * prefix, sin])


def variant_tables(
    model: MlxBackbone, variant: str, grid_h: int, grid_w: int
) -> tuple[mx.array, mx.array]:
    """Return the RoPE tables that the blocks of one variant get for one patch grid.

    V1: the bf16 tables of the ports (N, head_dim). V2: the same values in fp32, in our layout.
    V3: the Meta periods in our layout. V0, V4 and V5: the tables of the RoPE cache.
    """
    config = model.config
    if variant in ("V1", "V2"):
        cos, sin = port_rope_tables(grid_h, grid_w, config.head_dim, config.rope_theta)
        if variant == "V1":
            return cos, sin
        cos32, sin32 = (np.array(t.astype(mx.float32)) for t in (cos, sin))
        return tuple(mx.array(t) for t in rope_layout(*_with_prefix(cos32, sin32)))
    if variant == "V3":
        cos, sin = meta_rope_tables(grid_h, grid_w, config.head_dim, config.rope_theta)
        return tuple(mx.array(t) for t in rope_layout(cos, sin))
    return model.rope(grid_h, grid_w)


def variant_forward(model: MlxBackbone, variant: str) -> Forward:
    """Return the compiled forward pass of one port-change variant (Table 3 of the plan).

    The function replaces the attention module of each block where the variant needs it (V1,
    V4): the new module gets the parameters of the old one. So the backbone keeps the variant;
    load a new backbone for another variant. It compiles model.forward_with_rope with
    inputs=model.state, as compile_backbone does; the tables of the variant enter as arguments.

    Args:
        model: A loaded backbone.
        variant: A key of VARIANTS.

    Returns:
        forward(pixels) -> tokens, with the contract of model(pixels).

    Raises:
        ValueError: For an unknown variant.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}")
    attention = {"V1": Bf16RotationAttention, "V4": UnfusedAttention}.get(variant)
    if attention is not None:
        for block in model.layer:
            replacement = attention(model.config)
            replacement.update(block.attention.parameters())
            block.attention = replacement
    traced = mx.compile(lambda p, c, s: model.forward_with_rope(p, c, s), inputs=model.state)
    patch = model.config.patch_size
    tables: dict[tuple[int, int], tuple[mx.array, mx.array]] = {}

    def forward(pixels: mx.array) -> mx.array:
        grid = (pixels.shape[2] // patch, pixels.shape[3] // patch)
        if grid not in tables:
            tables[grid] = variant_tables(model, variant, *grid)
        if variant == "V5":
            pixels = mx.array(eight_bit_pixels(np.array(pixels.astype(mx.float32))))
        return traced(pixels, *tables[grid])

    return forward


def _tokens(forward: Forward, pixels: np.ndarray) -> np.ndarray:
    tokens = forward(mx.array(pixels)).astype(mx.float32)
    mx.eval(tokens)
    return np.array(tokens)


def run_variant_parity(
    model_size: str, dtype_mode: str, variant: str, configurations: list[str] | None = None
) -> dict:
    """Load one backbone, apply one variant and measure the parity of the 12 configurations at
    batch size 1 (J3).

    V0 changes no module, so the worker also compiles the same backbone with compile_backbone and
    gives the largest difference of the two on the first configuration (AC6): 0.0 means bit for
    bit.

    Args:
        model_size: A key of MODEL_IDS.
        dtype_mode: A key of MLX_DTYPE_MODES.
        variant: A key of VARIANTS.
        configurations: The configuration ids, for example "forest/224x224". None runs all 12.

    Returns:
        The worker fields: one row for each configuration with the parity, passes and finite.
    """
    from dinov3_apple import mlx_weights

    start = time.perf_counter()
    model = mlx_weights.load_mlx_backbone(model_size, dtype_mode, tf32=False)
    load_s = time.perf_counter() - start
    forward = variant_forward(model, variant)
    inputs = read_inputs()
    reference, _ = read_reference(model_size)
    ids = configurations or [c.id for c in stored.configurations()]
    rows = []
    for config_id in ids:
        tokens = _tokens(forward, inputs[config_id])[0]
        finite = bool(np.isfinite(tokens).all())
        parity = measure_parity(reference[config_id], tokens).__dict__
        rows.append(
            {
                "config": config_id,
                "parity": parity,
                "finite": finite,
                "passes": finite and parity_passes(parity, dtype_mode),
            }
        )
    compiled_difference = None
    if variant == "V0":
        pixels = inputs[ids[0]]
        compiled = _tokens(compile_backbone(model), pixels)
        compiled_difference = float(np.abs(compiled - _tokens(forward, pixels)).max())
    return {
        "status": "done",
        "reason": "",
        "configurations": rows,
        "compiled_difference": compiled_difference,
        "load_time_s": load_s,
        "thresholds": THRESHOLDS[dtype_mode].__dict__,
        "mlx_peak_bytes": mx.get_peak_memory(),
        "versions": read_versions(),
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def run_variant_speed(
    config: BenchConfig, variant: str, warm_up: TimeBudget, timed: TimeBudget
) -> dict:
    """Time the compiled forward pass of one variant at one configuration (J4).

    The phases and the time budgets of run_mlx_worker: a first call with the validity check of
    image 0, the warm-up batches, then the timed batches, host to host.

    Args:
        config: A BenchConfig with compile=True.
        variant: "V0", "V1" or "V4".
        warm_up: The time budget of the warm-up batches.
        timed: The time budget of the timed batches.
    """
    from dinov3_apple import mlx_weights
    from dinov3_apple.bench import SIDES, reference_tokens, summarize
    from dinov3_apple.mlx_bench import phase, time_batch_mlx

    start = time.perf_counter()
    model = mlx_weights.load_mlx_backbone(config.model_size, config.dtype_mode, tf32=False)
    load_s = time.perf_counter() - start
    forward = variant_forward(model, variant)
    batch = make_batch(read_inputs(), config.side, config.batch_size)
    _, frame, patches = time_batch_mlx(forward, batch, config.dtype_mode)
    reference = reference_tokens(config.model_size, config.side) if config.side in SIDES else None
    validity = check_validity(frame[0], patches[0], reference, config.dtype_mode)
    mx.set_memory_limit(required_bench_bytes(config, backend="mlx"))
    warm = phase(lambda: time_batch_mlx(forward, batch, config.dtype_mode)[0], warm_up)
    times = phase(lambda: time_batch_mlx(forward, batch, config.dtype_mode)[0], timed)
    device, frame_t, patch_t = (list(column) for column in zip(*times, strict=True))
    return {
        "status": "done",
        "reason": "",
        "valid": validity.valid,
        "finite": validity.finite,
        "parity": None if validity.parity is None else validity.parity.__dict__,
        "validity_note": validity.note,
        "load_time_s": load_s,
        "warm_up_batches": len(warm),
        "timed_batches": len(times),
        "timed_s": sum(patch_t),
        "frame": summarize(frame_t, config.batch_size),
        "patch": summarize(patch_t, config.batch_size),
        "device": summarize(device, config.batch_size),
        "mlx_peak_bytes": mx.get_peak_memory(),
        "versions": read_versions(),
        "time": datetime.now(UTC).isoformat(timespec="seconds"),
    }
