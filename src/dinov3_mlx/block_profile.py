"""Split block 1 of a backbone into its parts and time each part alone (D6 of
specs/03-baseline/explore.html).

A synchronize after each part costs about 170 µs on torch-mps, more than some parts take. So each
part runs alone in a loop with one synchronize at the end, on the inputs that one run of the
block gave it.
"""

import math
import statistics
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import DINOv3ViTModel
from transformers.models.dinov3_vit.modeling_dinov3_vit import DINOv3ViTLayer, apply_rotary_pos_emb

from dinov3_mlx.reference import DTYPE_MODES


@dataclass(frozen=True)
class Part:
    """One part of a block.

    Attributes:
        name: A name of PART_NAMES.
        fn: The computation. It takes the inputs and returns a tuple of the outputs.
        inputs: The names of the input tensors in the state.
        outputs: The names of the output tensors in the state.
    """

    name: str
    fn: Callable[..., tuple[torch.Tensor, ...]]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


def block_parts(layer: DINOv3ViTLayer) -> list[Part]:
    """Return the 11 parts of one block, with the Hugging Face modules and functions (Table 6).

    The state starts with "x" (the tokens), "cos" and "sin" (the RoPE tables). The last part
    writes "out", the output of the block.

    Args:
        layer: One block, for example model.model.layer[0].

    Returns:
        The parts in the order of the forward pass.
    """
    attention, mlp = layer.attention, layer.mlp
    heads, head_dim = attention.num_heads, attention.head_dim

    def qkv_proj(h: torch.Tensor) -> tuple[torch.Tensor, ...]:
        b, t, _ = h.shape
        return tuple(
            proj(h).view(b, t, heads, head_dim).transpose(1, 2)
            for proj in (attention.q_proj, attention.k_proj, attention.v_proj)
        )

    def sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor]:
        return (F.scaled_dot_product_attention(q, k, v, scale=attention.scaling),)

    def o_proj(a: torch.Tensor) -> tuple[torch.Tensor]:
        b, _, t, _ = a.shape
        return (attention.o_proj(a.transpose(1, 2).contiguous().reshape(b, t, -1)),)

    if hasattr(mlp, "gate_proj"):
        ff_in = Part("ff_in", lambda h: (mlp.gate_proj(h), mlp.up_proj(h)), ("h2",), ("gate", "up"))
        activation = Part(
            "activation", lambda gate, up: (mlp.act_fn(gate) * up,), ("gate", "up"), ("g",)
        )
    else:
        ff_in = Part("ff_in", lambda h: (mlp.up_proj(h),), ("h2",), ("up",))
        activation = Part("activation", lambda up: (mlp.act_fn(up),), ("up",), ("g",))

    return [
        Part("norm1", lambda x: (layer.norm1(x),), ("x",), ("h1",)),
        Part("qkv_proj", qkv_proj, ("h1",), ("q", "k", "v")),
        Part("rope", apply_rotary_pos_emb, ("q", "k", "cos", "sin"), ("qr", "kr")),
        Part("sdpa", sdpa, ("qr", "kr", "v"), ("a",)),
        Part("o_proj", o_proj, ("a",), ("o",)),
        Part("layer_scale1", lambda o, x: (layer.layer_scale1(o) + x,), ("o", "x"), ("x2",)),
        Part("norm2", lambda x: (layer.norm2(x),), ("x2",), ("h2",)),
        ff_in,
        activation,
        Part("ff_out", lambda g: (mlp.down_proj(g),), ("g",), ("y",)),
        Part("layer_scale2", lambda y, x: (layer.layer_scale2(y) + x,), ("y", "x2"), ("out",)),
    ]


def run_parts(parts: list[Part], state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Run the parts in order.

    Args:
        parts: The parts from block_parts.
        state: The input tensors by name: "x", "cos" and "sin".

    Returns:
        A new state with the inputs and the outputs of every part.
    """
    state = dict(state)
    for part in parts:
        outputs = part.fn(*(state[name] for name in part.inputs))
        state |= dict(zip(part.outputs, outputs, strict=True))
    return state


def _synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()


def time_part(
    fn: Callable[[], object],
    device: str,
    min_calls: int = 20,
    min_seconds: float = 0.5,
    loops: int = 5,
) -> float:
    """Return the median time of one call of fn, in seconds.

    A loop runs fn n times with one synchronize at the end. A first loop finds an n with at least
    min_calls calls and min_seconds; it is also the warm-up. Then the function times `loops` loops
    of n calls.

    Args:
        fn: The computation, with its inputs bound.
        device: "cpu" or "mps".
        min_calls: The smallest number of calls in one loop.
        min_seconds: The shortest time of one loop.
        loops: The number of timed loops.

    Returns:
        The median over the loops of the loop time divided by n.
    """

    def loop(n: int) -> float:
        _synchronize(device)
        start = time.perf_counter()
        for _ in range(n):
            fn()
        _synchronize(device)
        return time.perf_counter() - start

    n, elapsed = 1, loop(1)
    while n < min_calls or elapsed < min_seconds:
        n = max(min_calls, math.ceil(1.2 * n * min_seconds / max(elapsed, 1e-9)), n + 1)
        elapsed = loop(n)
    return statistics.median(loop(n) / n for _ in range(loops))


@torch.inference_mode()
def profile_block(
    model: DINOv3ViTModel, pixels: torch.Tensor, dtype_mode: str, **timing: float
) -> dict:
    """Time each part of block 1, the whole block, the parts outside the blocks and the forward pass.

    Args:
        model: A backbone on its device.
        pixels: Pixels on the same device, in the dtype of the weights.
        dtype_mode: A key of DTYPE_MODES.
        **timing: Keyword arguments for time_part, for example min_seconds.

    Returns:
        parts_s (seconds per part, by name), block_s, outer_s (patch_embedding, rope_tables,
        final_norm), forward_before_s and forward_s (the forward pass before and after the parts,
        so that a change of the speed of the Mac during the profile is visible; finding F4 of
        specs/03-baseline/verify.html) and blocks (the number of blocks L).
    """
    device = pixels.device.type
    autocast = DTYPE_MODES[dtype_mode].autocast
    context = nullcontext() if autocast is None else torch.autocast(device, dtype=autocast)
    layer = model.model.layer[0]
    parts = block_parts(layer)
    with context:
        forward_before_s = time_part(lambda: model(pixel_values=pixels), device, **timing)
        x = model.embeddings(pixels)
        cos, sin = model.rope_embeddings(pixels)
        state = run_parts(parts, {"x": x, "cos": cos, "sin": sin})

        def bound(part: Part) -> Callable[[], object]:
            args = [state[name] for name in part.inputs]
            return lambda: part.fn(*args)

        parts_s = {part.name: time_part(bound(part), device, **timing) for part in parts}
        position = (cos, sin)
        block_s = time_part(lambda: layer(x, position_embeddings=position), device, **timing)
        outer_s = {
            "patch_embedding": time_part(lambda: model.embeddings(pixels), device, **timing),
            "rope_tables": time_part(lambda: model.rope_embeddings(pixels), device, **timing),
            "final_norm": time_part(lambda: model.norm(state["out"]), device, **timing),
        }
        forward_s = time_part(lambda: model(pixel_values=pixels), device, **timing)
    return {
        "parts_s": parts_s,
        "block_s": block_s,
        "outer_s": outer_s,
        "forward_before_s": forward_before_s,
        "forward_s": forward_s,
        "blocks": model.config.num_hidden_layers,
    }
