"""The DINOv3 backbone in MLX: all six model sizes, the MLP and SwiGLU, five dtype modes.

Source: we wrote this module from modeling_dinov3_vit.py of Hugging Face transformers 5.22.1
(Apache-2.0, Copyright 2025 Meta AI and The HuggingFace Inc. team) and the DINOv3 paper
(ADR 0002). The parameter names of the modules are the keys of the Hugging Face safetensors
files, so the key map of mlx_weights.py is almost the identity (D1 of
specs/04-mlx-port/explore.html).

Each dtype mode is a pair of dtypes: the matmul dtype and the stream dtype (D2 of
specs/05-all-sizes/explore.html). The backbone keeps the pair as a plain attribute, not as a
parameter, so load_weights, set_dtype and nn.quantize do not change it.

No torch import. The fp32 dtype mode needs TF32 off: MLX_ENABLE_TF32=0 in the environment before
the first use of mlx (D7 of specs/04-mlx-port/explore.html). This module does not change the
environment.
"""

import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import numpy as np
from mlx import nn

from dinov3_apple.outputs import NUM_PREFIX_TOKENS

if TYPE_CHECKING:  # mlx_weights imports this module
    from dinov3_apple.mlx_weights import LoadReport


@dataclass(frozen=True)
class MlxDtypeMode:
    """The two dtypes of one dtype mode of mlx (D2 of specs/05-all-sizes/explore.html).

    Attributes:
        matmul: The matmul dtype: the weight and bias of each nn.Linear, the inputs of the
            matmuls and of the SDPA.
        stream: The stream dtype: the residual stream, the layer norms, LayerScale, the CLS
            token and the register tokens.
    """

    matmul: mx.Dtype
    stream: mx.Dtype


# The keys are the keys of THRESHOLDS and DTYPE_OF_MODE, in the order of THRESHOLDS.
MLX_DTYPE_MODES: dict[str, MlxDtypeMode] = {
    "fp32": MlxDtypeMode(mx.float32, mx.float32),
    "fp16-cast": MlxDtypeMode(mx.float16, mx.float16),
    "bf16-cast": MlxDtypeMode(mx.bfloat16, mx.bfloat16),
    "fp16-autocast": MlxDtypeMode(mx.float16, mx.float32),
    "bf16-autocast": MlxDtypeMode(mx.bfloat16, mx.float32),
}
FP32 = MLX_DTYPE_MODES["fp32"]


class TF32ModeError(RuntimeError):
    """The TF32 state of a load differs from the value of MLX_ENABLE_TF32."""


def tf32_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return True if MLX runs fp32 matmuls with TF32, from MLX_ENABLE_TF32.

    MLX reads the variable once, at its first use, into a static value: the leading integer of
    the value (as atoi does), with the default 1. A process that changes the variable after its
    first mlx matmul defeats this check.

    Args:
        environ: The environment. None reads os.environ.

    Returns:
        False only if MLX_ENABLE_TF32 holds 0 or a value without a leading integer.
    """
    value = (os.environ if environ is None else environ).get("MLX_ENABLE_TF32")
    if value is None:
        return True
    match = re.match(r"\s*([+-]?\d+)", value)
    return bool(match and int(match.group(1)))


@dataclass(frozen=True)
class MlxConfig:
    """The shape of one model size, from config.json of its Hugging Face folder.

    Attributes:
        dim: The embedding dimension.
        heads: The number of attention heads.
        blocks: The number of blocks.
        ff: The hidden width of the feed-forward network.
        patch_size: The patch size in pixels.
        num_channels: The number of color channels of the input.
        num_register_tokens: The number of register tokens.
        eps: The epsilon of the layer norms.
        rope_theta: The base of the RoPE frequencies.
        query_bias: True if q_proj has a bias. The same for key_bias, value_bias and proj_bias.
        mlp_bias: True if the feed-forward network has biases.
        gated: True for SwiGLU (use_gated_mlp).
        hidden_act: The activation of the feed-forward network: "gelu" for the MLP, "silu" for
            SwiGLU.
    """

    dim: int
    heads: int
    blocks: int
    ff: int
    patch_size: int
    num_channels: int
    num_register_tokens: int
    eps: float
    rope_theta: float
    query_bias: bool
    key_bias: bool
    value_bias: bool
    proj_bias: bool
    mlp_bias: bool
    gated: bool
    hidden_act: str

    @property
    def head_dim(self) -> int:
        """The number of values of one head."""
        return self.dim // self.heads

    @classmethod
    def from_hf(cls, config: Mapping[str, Any]) -> "MlxConfig":
        """Return the config from the values of a Hugging Face config.json (or its to_dict)."""
        return cls(
            dim=config["hidden_size"],
            heads=config["num_attention_heads"],
            blocks=config["num_hidden_layers"],
            ff=config["intermediate_size"],
            patch_size=config["patch_size"],
            num_channels=config["num_channels"],
            num_register_tokens=config["num_register_tokens"],
            eps=config["layer_norm_eps"],
            rope_theta=config["rope_theta"],
            query_bias=config["query_bias"],
            key_bias=config["key_bias"],
            value_bias=config["value_bias"],
            proj_bias=config["proj_bias"],
            mlp_bias=config["mlp_bias"],
            gated=config["use_gated_mlp"],
            hidden_act=config["hidden_act"],
        )


def rope_tables(
    grid_h: int, grid_w: int, head_dim: int, theta: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the RoPE tables cos and sin of one patch grid, each (5 + Hp · Wp, head_dim), fp32.

    The steps and their order follow DINOv3ViTRopePositionEmbedding of Hugging Face, so the
    tables match the tables of the reference output (D3). NumPy 2 keeps a Python float weak, so
    each step stays fp32. Rows 0 to 4 are identity rows (cos 1, sin 0) for the prefix tokens.

    Args:
        grid_h: The number of patches along the height.
        grid_w: The number of patches along the width.
        head_dim: The number of values of one head. Each quarter gets one frequency.
        theta: The base of the frequencies (rope_theta, 100 in DINOv3).

    Returns:
        The cos table and the sin table. Token n = 5 + Wp · i + j has the angles of the patch
        center (y, x) of row i and column j: y for the pairs 0 to head_dim / 4 - 1, x for the
        next head_dim / 4 pairs (the half-split layout of rotate_half).
    """
    inv_freq = 1 / theta ** np.arange(0, 1, 4 / head_dim, dtype=np.float32)  # (head_dim / 4,)
    coord_h = np.arange(0.5, grid_h, dtype=np.float32) / grid_h
    coord_w = np.arange(0.5, grid_w, dtype=np.float32) / grid_w
    coords = np.stack(np.meshgrid(coord_h, coord_w, indexing="ij"), axis=-1).reshape(-1, 2)
    coords = 2.0 * coords - 1.0  # (N, 2): y, x in [-1, 1]
    angles = (2 * np.pi * coords[:, :, None] * inv_freq[None, None, :]).reshape(len(coords), -1)
    angles = np.tile(angles, 2)  # (N, head_dim)
    prefix = np.ones((NUM_PREFIX_TOKENS, head_dim), np.float32)
    return np.concatenate([prefix, np.cos(angles)]), np.concatenate([0 * prefix, np.sin(angles)])


def rope_layout(cos: np.ndarray, sin: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the RoPE tables in the layout of the concat-free rotation (D2 of
    specs/06-speed/plan.html).

    Args:
        cos: The cos table of rope_tables, (T, head_dim), fp32.
        sin: The sin table of rope_tables, (T, head_dim), fp32.

    Returns:
        cos2 and sin2, each (T, 2, head_dim / 2), fp32. sin2 holds -sin in its first half: the
        sign of rotate_half moves into the table, so the rotation needs no join.
    """
    t, head_dim = cos.shape
    half = head_dim // 2
    signed = np.concatenate([-sin[:, :half], sin[:, half:]], axis=-1)
    return cos.reshape(t, 2, half), signed.reshape(t, 2, half)


def apply_rope(x: mx.array, cos2: mx.array, sin2: mx.array) -> mx.array:
    """Rotate queries or keys with the RoPE tables, without a join (D2, option R2).

    out = x · cos + (x2, x1) · (−sin1, sin2). The pair (x2, x1) is a reversed view of an axis of
    size 2, so mx.compile fuses the rotation into one kernel. The rotation runs in fp32 and the
    result has the dtype of x. In IEEE arithmetic (−x2) · s equals x2 · (−s), so the result
    equals the Step 5 form with rotate_half bit for bit. The identity rows of the prefix tokens
    give x · 1 + r · 0 = x.

    Args:
        x: Queries or keys with shape (B, heads, 5 + N, head_dim).
        cos2: The cos table, (5 + N, 2, head_dim / 2), fp32.
        sin2: The signed sin table, (5 + N, 2, head_dim / 2), fp32.

    Returns:
        The rotated values with the shape and the dtype of x.
    """
    b, h, t, d = x.shape
    xr = x.reshape(b, h, t, 2, d // 2).astype(mx.float32)
    out = xr * cos2 + xr[..., ::-1, :] * sin2
    return out.astype(x.dtype).reshape(b, h, t, d)


class RopeCache:
    """The RoPE tables of each patch grid, in the layout of apply_rope (A3 of
    specs/06-speed/plan.html).

    A plain object, not a dict: mlx.nn.Module keeps a dict attribute in its module tree, so
    model.state and the inputs of the compiled forward pass would change with each new grid (R43).
    """

    def __init__(self, head_dim: int, theta: float) -> None:
        self.head_dim = head_dim
        self.theta = theta
        self._tables: dict[tuple[int, int], tuple[mx.array, mx.array]] = {}

    def tables(self, grid_h: int, grid_w: int) -> tuple[mx.array, mx.array]:
        """Return (cos2, sin2) of one patch grid. The first call computes them in NumPy."""
        key = (grid_h, grid_w)
        if key not in self._tables:
            cos, sin = rope_tables(grid_h, grid_w, self.head_dim, self.theta)
            cos2, sin2 = rope_layout(cos, sin)
            self._tables[key] = (mx.array(cos2), mx.array(sin2))
        return self._tables[key]

    def __len__(self) -> int:
        return len(self._tables)


def patchify(pixels: mx.array, patch_size: int) -> mx.array:
    """Cut NCHW pixels into patches. The order inside one patch is channel, row, column.

    The order matches the patch embedding weight (D, C, p, p) after a reshape to (D, C · p · p).
    The token order is row-major: token n = Wp · i + j, as flatten(2) of Hugging Face (D4).

    Args:
        pixels: Pixels with shape (B, C, H, W).
        patch_size: The patch size p.

    Returns:
        The patches with shape (B, Hp · Wp, C · p · p).

    Raises:
        ValueError: If H or W is not a multiple of patch_size.
    """
    b, c, h, w = pixels.shape
    if h % patch_size or w % patch_size:
        raise ValueError(f"input resolution {h} x {w} is not a multiple of {patch_size}")
    hp, wp = h // patch_size, w // patch_size
    x = pixels.reshape(b, c, hp, patch_size, wp, patch_size).transpose(0, 2, 4, 1, 3, 5)
    return x.reshape(b, hp * wp, c * patch_size * patch_size)


class Embeddings(nn.Module):
    """The patch embedding and the prefix tokens: CLS token, register tokens, patch tokens.

    The convolution with stride = kernel of Hugging Face is one matmul on the patches (D4). The
    mask token of the weights file is for training only; the key map drops it.
    """

    def __init__(self, config: MlxConfig, dtypes: MlxDtypeMode = FP32) -> None:
        super().__init__()
        self.dtypes = dtypes  # a plain attribute: no parameter, nn.quantize keeps it
        self.patch_size = config.patch_size
        self.cls_token = mx.zeros((1, 1, config.dim))
        self.register_tokens = mx.zeros((1, config.num_register_tokens, config.dim))
        self.patch_embeddings = nn.Linear(config.num_channels * config.patch_size**2, config.dim)

    def __call__(self, pixels: mx.array) -> mx.array:
        """Return the token sequence (B, 1 + 4 + N, D) of NCHW pixels (B, C, H, W).

        The pixels get the matmul dtype first, as in the Hugging Face forward pass. The patch
        tokens get the stream dtype: half in cast mode, fp32 in autocast mode. The cast uses the
        plain attribute dtypes, not the dtype of a weight: nn.quantize packs the patch embedding
        weight into uint32, and uint32 pixels give wrong tokens without an error (F1 of
        specs/04-mlx-port/verify.html).

        Returns:
            The tokens in the stream dtype.
        """
        pixels = pixels.astype(self.dtypes.matmul)
        patches = self.patch_embeddings(patchify(pixels, self.patch_size))
        patches = patches.astype(self.dtypes.stream)
        prefix = mx.concatenate([self.cls_token, self.register_tokens], axis=1)
        prefix = mx.broadcast_to(prefix, (pixels.shape[0], *prefix.shape[1:]))
        return mx.concatenate([prefix.astype(self.dtypes.stream), patches], axis=1)


class Attention(nn.Module):
    """Full attention over all tokens, with RoPE on the queries and keys of the patch tokens."""

    def __init__(self, config: MlxConfig) -> None:
        super().__init__()
        dim = config.dim
        self.heads = config.heads
        self.scale = config.head_dim**-0.5  # 0.125 for head dimension 64
        self.q_proj = nn.Linear(dim, dim, bias=config.query_bias)
        self.k_proj = nn.Linear(dim, dim, bias=config.key_bias)  # no bias in ViT-S/16
        self.v_proj = nn.Linear(dim, dim, bias=config.value_bias)
        self.o_proj = nn.Linear(dim, dim, bias=config.proj_bias)

    def qkv(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """Return the queries, keys and values (B, heads, T, head_dim) of tokens (B, T, D).

        Head h holds the channels head_dim · h to head_dim · (h + 1) - 1.
        """
        b, t, _ = x.shape

        def heads(y: mx.array) -> mx.array:
            return y.reshape(b, t, self.heads, -1).transpose(0, 2, 1, 3)

        return heads(self.q_proj(x)), heads(self.k_proj(x)), heads(self.v_proj(x))

    def output(self, a: mx.array) -> mx.array:
        """Merge the heads (B, heads, T, head_dim) to (B, T, D) and apply o_proj."""
        b, _, t, _ = a.shape
        return self.o_proj(a.transpose(0, 2, 1, 3).reshape(b, t, -1))

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        """Return the output of the attention branch (B, T, D), before LayerScale.

        Args:
            x: The tokens after norm1, (B, 5 + N, D).
            cos: The cos table of the patch grid, (5 + N, 2, head_dim / 2).
            sin: The signed sin table of the patch grid, (5 + N, 2, head_dim / 2).
        """
        q, k, v = self.qkv(x)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        a = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=None)
        return self.output(a)


class LayerScale(nn.Module):
    """A learned factor for each channel."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.lambda1 = mx.ones((dim,))

    def __call__(self, x: mx.array) -> mx.array:
        """Return x · lambda1."""
        return x * self.lambda1


class MLP(nn.Module):
    """The feed-forward network of ViT-S/16, ViT-B/16 and ViT-L/16: up_proj, exact GELU,
    down_proj."""

    def __init__(self, config: MlxConfig) -> None:
        super().__init__()
        if config.hidden_act != "gelu":
            raise NotImplementedError(f"activation {config.hidden_act!r} of the MLP")
        self.up_proj = nn.Linear(config.dim, config.ff, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.ff, config.dim, bias=config.mlp_bias)

    def activation(self, up: mx.array) -> mx.array:
        """Return gelu(up). It computes in fp32 and rounds once to the dtype of up.

        nn.gelu is the erf form, not nn.gelu_approx. In half precision the MLX activations round
        after each operation (Table 8 of specs/05-all-sizes/explore.html). In fp32 the casts do
        nothing.
        """
        return nn.gelu(up.astype(mx.float32)).astype(up.dtype)

    def hidden(self, x: mx.array) -> mx.array:
        """Return the feed-forward hidden values gelu(up_proj(x))."""
        return self.activation(self.up_proj(x))

    def __call__(self, x: mx.array) -> mx.array:
        """Return down_proj(gelu(up_proj(x))), as DINOv3ViTMLP."""
        return self.down_proj(self.hidden(x))


class SwiGLU(nn.Module):
    """The feed-forward network of ViT-S+/16, ViT-H+/16 and ViT-7B/16 (D1 of
    specs/05-all-sizes/explore.html). Three nn.Linear modules with the Hugging Face names."""

    def __init__(self, config: MlxConfig) -> None:
        super().__init__()
        if config.hidden_act != "silu":
            raise NotImplementedError(f"activation {config.hidden_act!r} of SwiGLU")
        self.gate_proj = nn.Linear(config.dim, config.ff, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.dim, config.ff, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.ff, config.dim, bias=config.mlp_bias)

    def activation(self, gate: mx.array, up: mx.array) -> mx.array:
        """Return silu(gate) · up. It computes in fp32 and rounds once to the dtype of gate."""
        product = nn.silu(gate.astype(mx.float32)) * up.astype(mx.float32)
        return product.astype(gate.dtype)

    def hidden(self, x: mx.array) -> mx.array:
        """Return the feed-forward hidden values silu(gate_proj(x)) · up_proj(x)."""
        return self.activation(self.gate_proj(x), self.up_proj(x))

    def __call__(self, x: mx.array) -> mx.array:
        """Return down_proj(silu(gate_proj(x)) · up_proj(x)), as DINOv3ViTGatedMLP."""
        return self.down_proj(self.hidden(x))


class Block(nn.Module):
    """One block: an attention branch and a feed-forward branch, each with a pre-norm, a
    LayerScale and a residual add.

    The branches run in the matmul dtype, the residual stream in the stream dtype. In an
    autocast mode a half branch output times the fp32 LayerScale factor gives fp32. In fp32 and
    in a cast mode both dtypes are the same, so the casts allocate nothing.
    """

    def __init__(self, config: MlxConfig, dtypes: MlxDtypeMode = FP32) -> None:
        super().__init__()
        self.dtypes = dtypes  # a plain attribute: no parameter, nn.quantize keeps it
        self.norm1 = nn.LayerNorm(config.dim, eps=config.eps)
        self.attention = Attention(config)
        self.layer_scale1 = LayerScale(config.dim)
        self.norm2 = nn.LayerNorm(config.dim, eps=config.eps)
        self.mlp = SwiGLU(config) if config.gated else MLP(config)
        self.layer_scale2 = LayerScale(config.dim)

    def attention_branch(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        """Return the attention branch output before LayerScale, in the matmul dtype."""
        return self.attention(self.norm1(x).astype(self.dtypes.matmul), cos, sin)

    def ff_hidden(self, x: mx.array) -> mx.array:
        """Return the feed-forward hidden values: the GELU output, or silu(gate) · up."""
        return self.mlp.hidden(self.norm2(x).astype(self.dtypes.matmul))

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        """Return the block output (B, 5 + N, D) of the tokens x and the RoPE tables.

        run_mlx_token_range calls the same methods in the same order.
        """
        x = self.layer_scale1(self.attention_branch(x, cos, sin)) + x
        return self.layer_scale2(self.mlp.down_proj(self.ff_hidden(x))) + x


class MlxBackbone(nn.Module):
    """The DINOv3 backbone in MLX. Its parameter names are the keys of the safetensors file.

    The RoPE tables are not parameters: they live in a RopeCache, a plain object outside the
    module tree, so load_weights(strict=True), set_dtype, nn.quantize and the inputs of the
    compiled forward pass do not see them (R43). The dtype mode, its dtypes and the load report
    are plain attributes, not parameters.
    """

    def __init__(self, config: MlxConfig, dtype_mode: str = "fp32") -> None:
        super().__init__()
        self.config = config
        self.dtype_mode = dtype_mode
        self.dtypes = MLX_DTYPE_MODES[dtype_mode]
        self.embeddings = Embeddings(config, self.dtypes)
        self.layer = [Block(config, self.dtypes) for _ in range(config.blocks)]
        self.norm = nn.LayerNorm(config.dim, eps=config.eps)
        self.rope_cache = RopeCache(config.head_dim, config.rope_theta)
        self.rope_box_jitter = False  # RoPE-box jittering of training: off; Step 8 decides (R17)
        self.load_report: LoadReport | None = None  # load_mlx_backbone sets it

    def rope(self, grid_h: int, grid_w: int) -> tuple[mx.array, mx.array]:
        """Return the RoPE tables (cos2, sin2) of one patch grid from the RoPE cache, in the
        layout of the concat-free rotation."""
        return self.rope_cache.tables(grid_h, grid_w)

    def forward_with_rope(self, pixels: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        """Return all tokens after the final layer norm, (B, 5 + N, D), with the given RoPE tables.

        The eager and the compiled forward pass share this code path (D1 of
        specs/06-speed/plan.html).
        """
        x = self.embeddings(pixels)
        for block in self.layer:
            x = block(x, cos, sin)
        return self.norm(x)

    def __call__(self, pixels: mx.array) -> mx.array:
        """The eager forward pass: all tokens after the final layer norm, (B, 5 + N, D), of NCHW
        pixels, with the RoPE tables of the patch grid from the cache."""
        p = self.config.patch_size
        return self.forward_with_rope(
            pixels, *self.rope(pixels.shape[2] // p, pixels.shape[3] // p)
        )


def compile_backbone(model: MlxBackbone) -> Callable[[mx.array], mx.array]:
    """Return the compiled forward pass of one backbone (D1 of specs/06-speed/plan.html).

    One trace for each input shape (B, H, W). The weights enter as implicit inputs
    (inputs=model.state), so the output follows a later weight change (R35). The RoPE tables enter
    as arguments; Python reads them from the cache, outside the trace. Call it after the last
    weight change (the load, nn.quantize). The caller holds the function: a function stored on the
    model keeps the weights until gc.collect() (R36).

    Args:
        model: A loaded backbone.

    Returns:
        forward(pixels) -> tokens, with the contract of model(pixels).
    """
    traced = mx.compile(lambda p, c, s: model.forward_with_rope(p, c, s), inputs=model.state)
    patch = model.config.patch_size

    def forward(pixels: mx.array) -> mx.array:
        cos, sin = model.rope(pixels.shape[2] // patch, pixels.shape[3] // patch)
        return traced(pixels, cos, sin)

    return forward


def linear_parameter_names(model: nn.Module) -> frozenset[str]:
    """Return the names of the parameters of each nn.Linear and nn.QuantizedLinear module: the
    matmul dtype (D2).

    nn.QuantizedLinear is not a subclass of nn.Linear (F1 of specs/06-speed/plan.html): its
    weight, scales, biases (the offsets) and bias count too. The names come from named_modules()
    and isinstance, not from name suffixes. All other parameters get the stream dtype.
    """
    return frozenset(
        f"{name}.{leaf}"
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear | nn.QuantizedLinear)
        for leaf in module.parameters()
    )


def quantized_weight_names(model: nn.Module) -> frozenset[str]:
    """Return the names of the weight of each nn.QuantizedLinear module: the tensors that the
    quantized load quantizes (D17 of specs/06-speed/plan.html)."""
    return frozenset(
        f"{name}.weight"
        for name, module in model.named_modules()
        if isinstance(module, nn.QuantizedLinear)
    )


def run_mlx(
    model: MlxBackbone,
    pixels: np.ndarray,
    dtype_mode: str,
    forward: Callable[[mx.array], mx.array] | None = None,
) -> np.ndarray:
    """Run the backbone on stored pixels and return all tokens after the final layer norm.

    The contract of run_torch in reference.py. The model casts the fp32 pixels to its matmul
    dtype.

    Args:
        model: A backbone from load_mlx_backbone.
        pixels: Normalized fp32 pixels with shape (B, 3, H, W), NCHW.
        dtype_mode: A key of MLX_DTYPE_MODES. It must be the dtype mode of the model.
        forward: The compiled forward pass of model, from compile_backbone (D4 of
            specs/06-speed/plan.html). None runs the eager forward pass (part tests, diagnostics).

    Returns:
        The tokens with shape (B, 1 + 4 + N, D), cast to fp32.

    Raises:
        ValueError: If dtype_mode is not the dtype mode of the model.
    """
    if dtype_mode != model.dtype_mode:
        raise ValueError(f"the model runs in {model.dtype_mode}, not in {dtype_mode}")
    tokens = (forward or model)(mx.array(pixels))
    tokens = tokens.astype(mx.float32)  # a bf16 mx.array does not convert to NumPy
    mx.eval(tokens)
    return np.array(tokens)


def patch_dtype(dtype_mode: str, patch_fp32: bool = False) -> mx.Dtype:
    """Return the host dtype of the patch embeddings of one dtype mode (D13 of
    specs/06-speed/plan.html): fp16 in the four fp16 and bf16 dtype modes, else fp32."""
    half = MLX_DTYPE_MODES[dtype_mode].matmul != mx.float32
    return mx.float16 if half and not patch_fp32 else mx.float32


def mlx_embeddings(
    tokens: mx.array, dtype_mode: str, patch_fp32: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return the frame embeddings and the patch embeddings of one batch on the host (D13 of
    specs/06-speed/plan.html).

    The frame embeddings are fp32. In the four fp16 and bf16 dtype modes the patch embeddings are
    fp16: half the host bytes. The extra parity loss against fp32 patch embeddings is at most
    5.84e-8 for the patch mean and at most 1.27e-6 for the patch 1st percentile (ViT-H+/16
    bf16-autocast, iss-kibo-lab/224x400; AC15 of specs/06-speed/verify.html, Table 8 of
    results/parity-mlx.md). NumPy has no bf16, so the bf16 modes cast to fp16 on the GPU; after
    the final layer norm no value is above 18.

    Args:
        tokens: All tokens after the final layer norm, (B, 5 + N, D).
        dtype_mode: The dtype mode of the backbone.
        patch_fp32: True for fp32 patch embeddings in every dtype mode.

    Returns:
        The frame embeddings (B, D) and the patch embeddings (B, N, D).
    """
    frame = tokens[:, 0].astype(mx.float32)
    patches = tokens[:, NUM_PREFIX_TOKENS:].astype(patch_dtype(dtype_mode, patch_fp32))
    mx.eval(frame, patches)
    return np.array(frame), np.array(patches)


def run_mlx_hidden_states(model: MlxBackbone, pixels: np.ndarray) -> np.ndarray:
    """Return the token sequence before block 1 and after each block, before the final layer norm.

    The contract of run_torch_hidden_states in reference.py: image 0 only.

    Args:
        model: A backbone from load_mlx_backbone.
        pixels: Normalized fp32 pixels with shape (1, 3, H, W).

    Returns:
        The hidden states with shape (L + 1, 1 + 4 + N, D), cast to fp32.
    """
    x = model.embeddings(mx.array(pixels[:1]))  # the model casts the fp32 pixels
    p = model.config.patch_size
    cos, sin = model.rope(pixels.shape[2] // p, pixels.shape[3] // p)
    states = [x[0]]
    for block in model.layer:
        x = block(x, cos, sin)
        states.append(x[0])
    hidden = mx.stack(states).astype(mx.float32)
    mx.eval(hidden)
    return np.array(hidden)


@dataclass(frozen=True)
class TokenRange:
    """The largest absolute values of one configuration, block by block (D6). Report only.

    Attributes:
        tokens: (L + 1,) fp32: the token sequence before block 1 and after each block.
        tokens_where: (L + 1,): the token index of each value of tokens. 0 is the CLS token; 1 to 4
            are the register tokens.
        tokens_finite: (L + 1,): True if every value of the token sequence is finite.
        attention: (L,) fp32: the attention branch output before LayerScale.
        ff_hidden: (L,) fp32: the feed-forward hidden values.
        ff_out: (L,) fp32: the feed-forward branch output before LayerScale.
        last_state: (5 + N, D) fp32: the token sequence after the last block.
    """

    tokens: np.ndarray
    tokens_where: np.ndarray
    tokens_finite: np.ndarray
    attention: np.ndarray
    ff_hidden: np.ndarray
    ff_out: np.ndarray
    last_state: np.ndarray

    @property
    def finite(self) -> bool:
        """True if every token sequence and every branch value is finite."""
        values = (self.tokens, self.attention, self.ff_hidden, self.ff_out)
        return bool(self.tokens_finite.all() and all(np.isfinite(v).all() for v in values))

    def report_values(self) -> dict[str, list]:
        """Return all fields except last_state as lists, for one JSON line."""
        return {
            "tokens": self.tokens.tolist(),
            "tokens_where": self.tokens_where.tolist(),
            "tokens_finite": self.tokens_finite.tolist(),
            "attention": self.attention.tolist(),
            "ff_hidden": self.ff_hidden.tolist(),
            "ff_out": self.ff_out.tolist(),
        }


def _largest(t: mx.array) -> mx.array:
    return mx.abs(t.astype(mx.float32)).max()  # inf stays inf


def _token_values(x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """The largest absolute value of a token sequence (1, T, D), its token and the finite flag."""
    magnitude = mx.abs(x[0].astype(mx.float32)).max(axis=-1)
    return magnitude.max(), mx.argmax(magnitude), mx.isfinite(x).all()


def run_mlx_token_range(model: MlxBackbone, pixels: np.ndarray) -> TokenRange:
    """Run image 0 block by block and keep only the largest absolute values (D6). Report only.

    The calls and their order are the calls of Block.__call__, so the last token sequence equals
    the last state of run_mlx_hidden_states. The function evaluates one block at a time.

    Args:
        model: A backbone from load_mlx_backbone.
        pixels: Normalized fp32 pixels with shape (B, 3, H, W). Only image 0 runs.

    Returns:
        The token range of image 0.
    """
    x = model.embeddings(mx.array(pixels[:1]))
    p = model.config.patch_size
    cos, sin = model.rope(pixels.shape[2] // p, pixels.shape[3] // p)
    states = [_token_values(x)]
    mx.eval(x, states[-1])
    branches = []
    for block in model.layer:
        a = block.attention_branch(x, cos, sin)
        x = block.layer_scale1(a) + x
        h = block.ff_hidden(x)
        f = block.mlp.down_proj(h)
        x = block.layer_scale2(f) + x
        states.append(_token_values(x))
        branches.append((_largest(a), _largest(h), _largest(f)))
        mx.eval(x, states[-1], branches[-1])  # one block at a time
    largest, where, finite = (np.array([s[i].item() for s in states]) for i in range(3))
    attention, ff_hidden, ff_out = (
        np.array([b[i].item() for b in branches], np.float32) for i in range(3)
    )
    return TokenRange(
        tokens=largest.astype(np.float32),
        tokens_where=where.astype(np.int64),
        tokens_finite=finite.astype(bool),
        attention=attention,
        ff_hidden=ff_hidden,
        ff_out=ff_out,
        last_state=np.array(x[0].astype(mx.float32)),
    )
