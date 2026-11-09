"""The MLX weights: the key map from the Hugging Face safetensors and the load of the MLX backbone.

The key map runs at load time; no converted file (D2 of specs/04-mlx-port/explore.html). Each
load checks the run rule and the free memory first and raises the errors of the torch loader, so
the test fixture skips in the same way. The load casts and evaluates one tensor at a time, so it
references at most one fp32 source; MLX can release it after the next read (D5 of
specs/05-all-sizes/explore.html; F4 of specs/05-all-sizes/verify.html). No torch import.
"""

import dataclasses
import json
import re
from collections.abc import Callable, Iterator, MutableMapping
from dataclasses import dataclass
from functools import cache

import mlx.core as mx
from mlx import nn
from mlx.utils import tree_flatten, tree_map

from dinov3_mlx.bench import QUANTIZED_MODES, quantized_weight_bytes
from dinov3_mlx.footprint import read_footprint
from dinov3_mlx.hub import parameter_count, weights_dir
from dinov3_mlx.memory import (
    LOAD_RATIO,
    check_free_memory,
    check_run_rule,
    load_bytes,
    record_load_peak,
)
from dinov3_mlx.mlx_model import (
    MLX_DTYPE_MODES,
    Block,
    MlxBackbone,
    MlxConfig,
    MlxDtypeMode,
    TF32ModeError,
    linear_parameter_names,
    quantized_weight_names,
    tf32_enabled,
)

# Quantized weights (D14 of specs/06-speed/plan.html): the affine format, 8 and 4 bits, one
# scale and one offset for each group of 64 weights of a row.
GROUP_SIZE = 64
QUANTIZED_BITS = (8, 4)

DROPPED_KEYS = frozenset({"embeddings.mask_token"})  # training only (bool_masked_pos)
PATCH_WEIGHT = "embeddings.patch_embeddings.weight"
OUTER_KEYS = frozenset(
    {
        "embeddings.cls_token",
        "embeddings.register_tokens",
        PATCH_WEIGHT,
        "embeddings.patch_embeddings.bias",
        "norm.weight",
        "norm.bias",
    }
)
# layer.{i}.{parameter} in the safetensors file, model.layer.{i}.{parameter} in a transformers 5
# model in memory. {parameter} must be a parameter name of the MLX block (block_parameter_names).
BLOCK_KEY = re.compile(r"(?:model\.)?(layer\.\d+\.(?P<parameter>.+))")


class UnknownWeightError(KeyError):
    """A tensor name of the weights has no place in the MLX backbone."""


@dataclass(frozen=True)
class LoadReport:
    """The memory of one mlx load (D5 of specs/05-all-sizes/explore.html).

    Attributes:
        need_bytes: The need of the memory check and the MLX memory limit of the load.
        peak_bytes: The load peak: the largest MLX active memory during the load, minus the
            active memory before the load. It does not include the MLX cache.
        cache_bytes: The MLX cache memory after the load.
        footprint_bytes: The process footprint after the load.
        bits: None for unquantized weights; 8 or 4 for quantized block weights.
    """

    need_bytes: int
    peak_bytes: int
    cache_bytes: int
    footprint_bytes: int
    bits: int | None = None


@cache
def block_parameter_names() -> frozenset[str]:
    """Return the parameter names of an MLP block and of a SwiGLU block, both with all biases.

    The union has 20 names. The names come from the Block module itself, so the key map and the
    module cannot differ. map_key stays without the config; load_weights(strict=True) stops a
    gate_proj key of an MLP model size. The probe blocks have a small width. Their arrays stay
    lazy and are never computed; their random init advances the global key of mx.random.
    """
    mlp = MlxConfig(
        dim=8,
        heads=1,
        blocks=1,
        ff=8,
        patch_size=1,
        num_channels=1,
        num_register_tokens=0,
        eps=1e-6,
        rope_theta=100.0,
        query_bias=True,
        key_bias=True,
        value_bias=True,
        proj_bias=True,
        mlp_bias=True,
        gated=False,
        hidden_act="gelu",
    )
    gated = dataclasses.replace(mlp, gated=True, hidden_act="silu")
    return frozenset(
        name for config in (mlp, gated) for name, _ in tree_flatten(Block(config).parameters())
    )


def map_key(name: str) -> str | None:
    """Return the MLX parameter name of one Hugging Face tensor name, or None for a dropped tensor.

    map_key checks the name and the parameter name of the block in a block key, not the block
    index: it does not know the number of blocks. load_weights(strict=True) stops a block index
    beyond the config, for example layer.12 of ViT-S/16, and a bias that the config does not have.

    Args:
        name: A tensor name of the safetensors file (layer.{i}.…) or of the state dict of a
            transformers 5 model (model.layer.{i}.…).

    Returns:
        The name in the MLX backbone, or None for embeddings.mask_token.

    Raises:
        UnknownWeightError: If the name is not in OUTER_KEYS or DROPPED_KEYS and is not
            layer.{i}. followed by a parameter name of the MLX block.
    """
    if name in DROPPED_KEYS:
        return None
    if name in OUTER_KEYS:
        return name
    if (match := BLOCK_KEY.fullmatch(name)) and match.group("parameter") in block_parameter_names():
        return match.group(1)
    raise UnknownWeightError(name)


def mlx_weights(
    tensors: MutableMapping[str, mx.array], dtypes: MlxDtypeMode, linear: frozenset[str]
) -> Iterator[tuple[str, mx.array]]:
    """Pop each tensor, map its key, reshape the patch weight and cast the tensor.

    The patch embedding weight (D, C, p, p) becomes (D, C · p · p): channel, row, column, as in
    patchify. The weight and bias of each nn.Linear (a name in linear) get the matmul dtype, all
    other parameters the stream dtype. The arrays stay lazy: the caller evaluates one at a time
    (D5). A generator: load_weights cannot take it, so each caller gives a list of the pairs.

    Args:
        tensors: The tensors by Hugging Face name, for example from mx.load. The generator pops
            each tensor, so the dict holds no reference to an fp32 source after its pair.
        dtypes: The dtypes of the dtype mode.
        linear: The parameter names of the matmul dtype, from linear_parameter_names.

    Yields:
        The pairs (MLX name, array), without the dropped tensors.

    Raises:
        UnknownWeightError: If map_key rejects a name. load_weights(strict=True) stops a name
            that map_key accepts and the backbone does not have (a block index beyond the config).
    """
    for name in list(tensors):
        array = tensors.pop(name)  # drop the last reference of the dict to the fp32 source
        if (key := map_key(name)) is None:
            continue  # the mask token
        if key == PATCH_WEIGHT:
            array = array.reshape(array.shape[0], -1)
        yield key, array.astype(dtypes.matmul if key in linear else dtypes.stream)


def required_mlx_load_bytes(
    model_size: str, dtype_mode: str = "fp32", ratio: float = LOAD_RATIO, bits: int | None = None
) -> int:
    """Return the memory that the memory check asks for before an mlx load of one model size.

    Args:
        model_size: A key of MODEL_IDS.
        dtype_mode: A key of MLX_DTYPE_MODES.
        ratio: The ratio of the load peak to the weight bytes.
        bits: None, or 8 or 4 for quantized block weights: the weight bytes are then the
            quantized bytes of Table 9 of specs/06-speed/plan.html.

    Returns:
        ratio x the weight bytes in the matmul dtype + LOAD_RESERVE_BYTES. ViT-S/16 in fp32:
        1.2 x 86.4 MB + 4 GB = 4.10 GB. The autocast modes have the need of the cast modes,
        because their weights are half on mlx.

    Raises:
        WeightsNotFoundError: If the weights are not on this Mac.
    """
    if bits is not None:
        return load_bytes(
            quantized_weight_bytes(model_size, bits, parameter_count(model_size)), ratio
        )
    bytes_per_weight = MLX_DTYPE_MODES[dtype_mode].matmul.size
    return load_bytes(parameter_count(model_size) * bytes_per_weight, ratio)


def block_linear_predicate(
    bits: int, group_size: int = GROUP_SIZE
) -> Callable[[str, nn.Module], dict | bool]:
    """Return the class predicate of nn.quantize for the block nn.Linear modules (D16).

    The patch embedding keeps the matmul dtype: it is the most sensitive part (probe Q5) and the
    path of F1 of Step 4. The predicate returns the parameters of each module, so a later mixed
    row (other bits for one module group) changes only this function.

    Args:
        bits: 8 or 4.
        group_size: The number of weights of one row that share one scale and one offset.
    """

    def predicate(path: str, module: nn.Module) -> dict | bool:
        if isinstance(module, nn.Linear) and path.startswith("layer."):
            return {"group_size": group_size, "bits": bits}
        return False

    return predicate


def placeholder_skeleton(model: nn.Module) -> None:
    """Replace each parameter of a model before a load by a zero-stride placeholder (R50).

    The placeholder has the shape and the dtype of the parameter, so load_weights(strict=True)
    checks the same names and shapes. An evaluated placeholder holds one value. So an mx.eval of
    the skeleton, by mistake, allocates bytes, not the random fp32 weights of the whole model
    (about 27 GB for ViT-7B/16, against ADR 0008).
    """
    model.update(
        tree_map(lambda p: mx.broadcast_to(mx.zeros((), p.dtype), p.shape), model.parameters())
    )


def load_mlx_backbone(
    model_size: str,
    dtype_mode: str = "fp32",
    tf32: bool = False,
    ratio: float = LOAD_RATIO,
    bits: int | None = None,
    group_size: int = GROUP_SIZE,
) -> MlxBackbone:
    """Load the MLX backbone of one model size in one dtype mode (Section 3.5 of
    specs/05-all-sizes/plan.html and of specs/06-speed/plan.html).

    The order of the actions: the run rule, the check of bits, the TF32 check, the memory check,
    the memory limit, the skeleton (with bits: nn.quantize of the block nn.Linear modules), the
    placeholders, the lazy files, then one tensor at a time (with bits: cast, quantize and
    evaluate each block weight), the strict load and an empty cache. The backbone keeps a
    LoadReport in its plain attribute load_report, and one load-peak line goes to the log of the
    memory checks.

    Args:
        model_size: A key of MODEL_IDS.
        dtype_mode: A key of MLX_DTYPE_MODES.
        tf32: The TF32 state that the fp32 dtype mode asks for. It must equal the state of
            MLX_ENABLE_TF32 in the environment.
        ratio: The ratio of the load peak to the weight bytes for the memory check.
        bits: None for unquantized weights; 8 or 4 for quantized block weights (D14, D15).
        group_size: The group size of the quantized weights.

    Returns:
        The backbone with evaluated parameters, so the load cost stays out of the first call.

    Raises:
        RunRuleError: For ViT-7B/16 in fp32, before any file read.
        ValueError: For bits outside QUANTIZED_BITS, or bits in a dtype mode other than
            fp16-autocast and bf16-autocast (D15), before any file read.
        TF32ModeError: If the fp32 dtype mode asks for another TF32 state than MLX_ENABLE_TF32.
        InsufficientMemoryError: If the memory check fails.
        WeightsNotFoundError: If the weights are not on this Mac.
    """
    dtypes = MLX_DTYPE_MODES[dtype_mode]
    label = f"{model_size} mlx {dtype_mode}" + (f" {bits}-bit" if bits else "")
    check_run_rule(model_size, dtypes.matmul.size, label)
    if bits is not None:
        if bits not in QUANTIZED_BITS:
            raise ValueError(f"bits must be one of {QUANTIZED_BITS}, not {bits}")
        if dtype_mode not in QUANTIZED_MODES:
            raise ValueError(f"quantized bits load in {QUANTIZED_MODES} only, not in {dtype_mode}")
    if dtypes.matmul == mx.float32 and tf32_enabled() != tf32:
        raise TF32ModeError(f"set MLX_ENABLE_TF32={int(tf32)} before the first use of mlx")
    need = required_mlx_load_bytes(model_size, dtype_mode, ratio, bits)
    check_free_memory(need, label)
    mx.set_memory_limit(need)  # only a guideline (R13 of Step 4): the loop keeps the peak low
    folder = weights_dir(model_size)
    config = MlxConfig.from_hf(json.loads((folder / "config.json").read_text()))
    model = MlxBackbone(config, dtype_mode)  # the random init stays lazy and is never computed
    if bits is not None:
        nn.quantize(model, class_predicate=block_linear_predicate(bits, group_size))
    linear = linear_parameter_names(model)  # nn.Linear and nn.QuantizedLinear (F1)
    quantized = quantized_weight_names(model)  # the .weight of each QuantizedLinear
    placeholder_skeleton(model)  # no lazy random weight of the whole model is left (R50)
    tensors: dict[str, mx.array] = {}
    for file in sorted(folder.glob("*.safetensors")):
        tensors |= mx.load(str(file))  # lazy; a str, not a Path (R18)
    active_before = mx.get_active_memory()
    mx.reset_peak_memory()
    pairs = []
    for key, array in mlx_weights(tensors, dtypes, linear):
        if key in quantized:
            weight, scales, biases = mx.quantize(array, group_size, bits)
            mx.eval(weight, scales, biases)  # one source, its cast and its quantized output
            stem = key.removesuffix(".weight")
            pairs += [(key, weight), (f"{stem}.scales", scales), (f"{stem}.biases", biases)]
            continue
        mx.eval(array)  # reads one tensor from its file and casts it
        pairs.append((key, array))
    model.load_weights(pairs, strict=True)
    del pairs
    mx.clear_cache()  # the freed fp32 buffers leave the MLX cache
    model.load_report = LoadReport(
        need_bytes=need,
        peak_bytes=mx.get_peak_memory() - active_before,
        cache_bytes=mx.get_cache_memory(),
        footprint_bytes=read_footprint().current_bytes,
        bits=bits,
    )
    record_load_peak(label, model.load_report.peak_bytes, model.load_report.cache_bytes)
    return model
