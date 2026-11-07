"""The RoPE rotation of Step 5 (rotate_half and a join), kept as the oracle of the concat-free
rotation of Step 6 (D2 of specs/06-speed/plan.html) and as form A of the change ratio C2."""

import mlx.core as mx


def rotate_half(x: mx.array) -> mx.array:
    """Return (-x2, x1) for the halves x1 and x2 of the last axis (the half-split layout)."""
    half = x.shape[-1] // 2
    return mx.concatenate([-x[..., half:], x[..., :half]], axis=-1)


def apply_rope_step5(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """Rotate queries or keys with the tables of rope_tables, (T, head_dim), in fp32 (Step 5)."""
    xf = x.astype(mx.float32)
    return (xf * cos + rotate_half(xf) * sin).astype(x.dtype)
