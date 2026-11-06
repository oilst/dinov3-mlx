"""FLOPs and bytes moved for each part of a block, and FLOPs for one image. Pure Python.

One multiply-add counts as 2 FLOPs. The bias adds, the layer norms, RoPE, LayerScale, the
activation function, the softmax and the residual adds count as 0 FLOPs (Table 6 of
specs/03-baseline/plan.html). Their cost is memory traffic, which part_bytes estimates.
"""

from dataclasses import dataclass

from dinov3_apple.outputs import NUM_PREFIX_TOKENS, PATCH_SIZE

# The parts of one block, in the order of the forward pass (Table 6 of the plan).
PART_NAMES = [
    "norm1",
    "qkv_proj",
    "rope",
    "sdpa",
    "o_proj",
    "layer_scale1",
    "norm2",
    "ff_in",
    "activation",
    "ff_out",
    "layer_scale2",
]


@dataclass(frozen=True)
class ModelShape:
    """The shape of one model size, from its config.json.

    Attributes:
        dim: The embedding dimension D.
        blocks: The number of blocks L.
        heads: The number of attention heads.
        ff: The feed-forward size F (intermediate_size).
        gated: True for SwiGLU, False for an MLP with GELU.
    """

    dim: int
    blocks: int
    heads: int
    ff: int
    gated: bool


MODEL_SHAPES: dict[str, ModelShape] = {
    "ViT-S/16": ModelShape(dim=384, blocks=12, heads=6, ff=1536, gated=False),
    "ViT-S+/16": ModelShape(dim=384, blocks=12, heads=6, ff=1536, gated=True),
    "ViT-B/16": ModelShape(dim=768, blocks=12, heads=12, ff=3072, gated=False),
    "ViT-L/16": ModelShape(dim=1024, blocks=24, heads=16, ff=4096, gated=False),
    "ViT-H+/16": ModelShape(dim=1280, blocks=32, heads=20, ff=5120, gated=True),
    "ViT-7B/16": ModelShape(dim=4096, blocks=40, heads=32, ff=8192, gated=True),
}


def tokens(side: int) -> int:
    """Return the token count T of a square input resolution: prefix tokens + patch tokens."""
    return NUM_PREFIX_TOKENS + (side // PATCH_SIZE) ** 2


def part_flops(shape: ModelShape, batch_size: int, num_tokens: int) -> dict[str, int]:
    """Return the FLOPs of each part of one block.

    Args:
        shape: The model shape.
        batch_size: The batch size B.
        num_tokens: The token count T.

    Returns:
        The FLOPs keyed by the names of PART_NAMES.
    """
    bt, d, f = batch_size * num_tokens, shape.dim, shape.ff
    ff_in_matrices = 2 if shape.gated else 1
    flops = dict.fromkeys(PART_NAMES, 0)
    flops["qkv_proj"] = 3 * 2 * bt * d * d
    flops["sdpa"] = 4 * batch_size * num_tokens * num_tokens * d
    flops["o_proj"] = 2 * bt * d * d
    flops["ff_in"] = ff_in_matrices * 2 * bt * d * f
    flops["ff_out"] = 2 * bt * f * d
    return flops


def part_bytes(
    shape: ModelShape, batch_size: int, num_tokens: int, bytes_per_value: int
) -> dict[str, int]:
    """Return the approximate bytes that each part of one block reads and writes.

    The count includes the activations and the weight matrices. SDPA counts only its inputs and
    its output; a kernel that writes the full score matrix moves more.

    Args:
        shape: The model shape.
        batch_size: The batch size B.
        num_tokens: The token count T.
        bytes_per_value: 4 for fp32, 2 for fp16 and bf16.

    Returns:
        The bytes keyed by the names of PART_NAMES.
    """
    btd = batch_size * num_tokens * shape.dim
    btf = batch_size * num_tokens * shape.ff
    d2, df = shape.dim * shape.dim, shape.dim * shape.ff
    ff_in_matrices = 2 if shape.gated else 1
    values = {
        "norm1": 2 * btd,
        "qkv_proj": 4 * btd + 3 * d2,
        "rope": 6 * btd,
        "sdpa": 4 * btd,
        "o_proj": 2 * btd + d2,
        "layer_scale1": 3 * btd,
        "norm2": 2 * btd,
        "ff_in": btd + ff_in_matrices * (btf + df),
        "activation": (3 if shape.gated else 2) * btf,
        "ff_out": btf + btd + df,
        "layer_scale2": 3 * btd,
    }
    return {name: values[name] * bytes_per_value for name in PART_NAMES}


def image_flops(shape: ModelShape, side: int) -> tuple[int, float]:
    """Return the FLOPs of one square image and the share of the attention scores in them.

    The count is the patch embedding plus L blocks. It leaves out the final layer norm.

    Args:
        shape: The model shape.
        side: The input resolution in pixels.

    Returns:
        The FLOPs of one image and the share of the SDPA part, between 0 and 1.
    """
    num_tokens = tokens(side)
    patches = num_tokens - NUM_PREFIX_TOKENS
    block = part_flops(shape, 1, num_tokens)
    patch_embedding = 2 * 3 * PATCH_SIZE * PATCH_SIZE * shape.dim * patches
    total = patch_embedding + shape.blocks * sum(block.values())
    return total, shape.blocks * block["sdpa"] / total
