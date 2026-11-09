"""The public API: frame embeddings and patch embeddings from uint8 RGB frames (Table 2 of
specs/07-release/plan.html).

A change to the contract of Embedder needs a new version. No torch import.
"""

from collections.abc import Sequence

import mlx.core as mx
import numpy as np
from PIL import Image

from dinov3_mlx.hub import MODEL_IDS, WeightsAccessError, ensure_weights
from dinov3_mlx.memory import check_run_rule
from dinov3_mlx.mlx_model import (
    MLX_DTYPE_MODES,
    TF32ModeError,
    compile_backbone,
    patch_dtype,
    tf32_active,
)
from dinov3_mlx.mlx_preprocess import preprocess_frames
from dinov3_mlx.mlx_weights import load_mlx_backbone
from dinov3_mlx.outputs import NUM_PREFIX_TOKENS

__all__ = ["Embedder", "Frames", "Size", "TF32ModeError", "WeightsAccessError"]

Frames = np.ndarray | Sequence[np.ndarray | Image.Image]
Size = int | tuple[int, int]
Output = np.ndarray | mx.array

BACKENDS = ("mlx",)
# The input resolution of the Hugging Face image processor of DINOv3.
DEFAULT_SIZE = 224


def as_uint8_batch(frames: Frames) -> np.ndarray:
    """Return the frames as one uint8 array (B, h, w, 3).

    Args:
        frames: A uint8 array (B, h, w, 3), or a list of PIL images or uint8 arrays (h, w, 3).

    Returns:
        The frames as one array.

    Raises:
        TypeError: For another dtype or another type of frame.
        ValueError: For no frames, frames of different sizes or another shape.
    """
    if isinstance(frames, np.ndarray):
        batch = frames
    elif isinstance(frames, Sequence) and not isinstance(frames, str | bytes):
        arrays = []
        for frame in frames:
            if isinstance(frame, Image.Image):
                frame = np.asarray(frame.convert("RGB"))
            if not isinstance(frame, np.ndarray):
                raise TypeError(f"a frame must be a PIL image or a NumPy array, not {type(frame)}")
            if frame.dtype != np.uint8:
                raise TypeError(f"the frames must be uint8, not {frame.dtype}")
            arrays.append(frame)
        if len({frame.shape for frame in arrays}) > 1:
            raise ValueError("all frames must have one size")
        batch = np.stack(arrays) if arrays else np.zeros((0, 0, 0, 3), np.uint8)
    else:
        raise TypeError(f"frames must be a NumPy array or a list, not {type(frames)}")
    if batch.dtype != np.uint8:
        raise TypeError(f"the frames must be uint8, not {batch.dtype}")
    if batch.ndim != 4 or batch.shape[-1] != 3:
        raise ValueError(f"the frames must have the shape (B, h, w, 3), not {batch.shape}")
    if len(batch) == 0:
        raise ValueError("no frames")
    return batch


def input_resolution(size: Size) -> tuple[int, int]:
    """Return the input resolution (H, W) of size.

    Raises:
        ValueError: If a side is not a positive multiple of 16.
    """
    height, width = (size, size) if isinstance(size, int) else tuple(size)
    if not all(side > 0 and side % 16 == 0 for side in (height, width)):
        raise ValueError(f"each side of size must be a positive multiple of 16, not {size}")
    return height, width


def model_size_of(name: str) -> str:
    """Return the model size of a model size name or of its Hugging Face id.

    Raises:
        ValueError: For another name.
    """
    if name in MODEL_IDS:
        return name
    for model_size, model_id in MODEL_IDS.items():
        if name == model_id:
            return model_size
    raise ValueError(f"unknown model size {name!r}: use one of {list(MODEL_IDS)}")


class Embedder:
    """DINOv3 frame embeddings and patch embeddings on the GPU with MLX.

    The constructor loads the weights (with a download on first use), loads the backbone and
    compiles its forward pass once. del frees the model.

    Example:
        >>> model = Embedder("ViT-S/16")
        >>> frame, patches = model.embed(frames)  # (B, 384), (B, 14, 14, 384)
    """

    def __init__(
        self,
        model_size: str = "ViT-S/16",
        dtype_mode: str = "fp16-autocast",
        batch_size: int = 8,
        bits: int | None = None,
        backend: str = "mlx",
    ) -> None:
        """Load and compile the backbone of one model size.

        Args:
            model_size: A model size name, for example "ViT-S/16", or its Hugging Face id.
            dtype_mode: "fp32", "fp16-cast", "bf16-cast", "fp16-autocast" or "bf16-autocast".
            batch_size: The number of frames of one forward pass. The last batch is padded.
            bits: None, or 8 for the 8-bit weights (fp16-autocast and bf16-autocast only).
            backend: "mlx". Step 7b adds "mlx+ne".

        Raises:
            ValueError: For an unknown model size, dtype mode or backend, or a bad batch size.
            RunRuleError: For ViT-7B/16 in fp32, before any file read.
            WeightsAccessError: If the Hub refuses the download of the weights.
            TF32ModeError: If mlx runs fp32 matmuls with TF32.
            InsufficientMemoryError: If the load does not fit in the free memory.
        """
        if backend == "mlx+ne":
            raise ValueError('backend "mlx+ne" (the Neural Engine) comes with Step 7b')
        if backend not in BACKENDS:
            raise ValueError(f"unknown backend {backend!r}: use one of {list(BACKENDS)}")
        if dtype_mode not in MLX_DTYPE_MODES:
            raise ValueError(
                f"unknown dtype mode {dtype_mode!r}: use one of {list(MLX_DTYPE_MODES)}"
            )
        if batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, not {batch_size}")
        self.model_size = model_size_of(model_size)
        self.dtype_mode = dtype_mode
        self.batch_size = batch_size
        self.backend = backend
        label = f"{self.model_size} mlx {dtype_mode}"
        check_run_rule(self.model_size, MLX_DTYPE_MODES[dtype_mode].matmul.size, label)
        ensure_weights(self.model_size)
        if tf32_active():
            raise TF32ModeError(
                "mlx runs fp32 matmuls with TF32: import dinov3_mlx before mlx, or set "
                "MLX_ENABLE_TF32=0 before the first use of mlx"
            )
        limit = mx.get_memory_limit()
        try:
            self._model = load_mlx_backbone(self.model_size, dtype_mode, bits=bits)
        finally:
            mx.set_memory_limit(limit)  # the loader sets the need of its load (Table 1 of explore)
        self._forward = compile_backbone(self._model)
        self._patch_size = self._model.config.patch_size

    def embed(
        self,
        frames: Frames,
        size: Size = DEFAULT_SIZE,
        as_numpy: bool = True,
        patch_fp32: bool = False,
    ) -> tuple[Output, Output]:
        """Return the frame embeddings and the patch embeddings from one forward pass.

        Args:
            frames: A uint8 array (B, h, w, 3) of RGB frames, or a list of PIL images or uint8
                arrays (h, w, 3) of one size.
            size: The input resolution: an int (square) or (H, W), each side a multiple of 16.
            as_numpy: False keeps the outputs as mx.array on the GPU.
            patch_fp32: True gives fp32 patch embeddings in every dtype mode.

        Returns:
            The frame embeddings (B, D) in fp32 and the patch embeddings (B, H/16, W/16, D): fp16
            in the four fp16 and bf16 dtype modes, else fp32.
        """
        return self._run(frames, size, as_numpy, patch_fp32, frame=True, patches=True)

    def embed_frames(
        self, frames: Frames, size: Size = DEFAULT_SIZE, as_numpy: bool = True
    ) -> Output:
        """Return the frame embeddings (B, D) in fp32. The arguments are those of embed."""
        return self._run(frames, size, as_numpy, False, frame=True, patches=False)[0]

    def embed_patches(
        self,
        frames: Frames,
        size: Size = DEFAULT_SIZE,
        as_numpy: bool = True,
        patch_fp32: bool = False,
    ) -> Output:
        """Return the patch embeddings (B, H/16, W/16, D). The arguments are those of embed."""
        return self._run(frames, size, as_numpy, patch_fp32, frame=False, patches=True)[1]

    def _run(
        self,
        frames: Frames,
        size: Size,
        as_numpy: bool,
        patch_fp32: bool,
        frame: bool,
        patches: bool,
    ) -> tuple[Output | None, Output | None]:
        """Run the batch loop (Section 5.3 of the plan). Each batch has batch_size frames: the
        last batch gets zero frames, so each input resolution needs one trace only."""
        batch = as_uint8_batch(frames)
        height, width = input_resolution(size)
        grid = (height // self._patch_size, width // self._patch_size)
        dtype = patch_dtype(self.dtype_mode, patch_fp32)
        frame_parts: list[Output] = []
        patch_parts: list[Output] = []
        for start in range(0, len(batch), self.batch_size):
            chunk = batch[start : start + self.batch_size]
            used = len(chunk)
            if used < self.batch_size:
                pad = np.zeros((self.batch_size - used, *chunk.shape[1:]), np.uint8)
                chunk = np.concatenate([chunk, pad])
            tokens = self._forward(preprocess_frames(chunk, (height, width)))
            outputs = []
            if frame:
                outputs.append(tokens[:used, 0].astype(mx.float32))
            if patches:
                outputs.append(
                    tokens[:used, NUM_PREFIX_TOKENS:].astype(dtype).reshape(used, *grid, -1)
                )
            mx.eval(*outputs)  # one batch at a time keeps the graph and the memory small
            if as_numpy:
                outputs = [np.array(output) for output in outputs]
            if frame:
                frame_parts.append(outputs.pop(0))
            if patches:
                patch_parts.append(outputs.pop(0))
        join = np.concatenate if as_numpy else mx.concatenate
        return (join(frame_parts) if frame else None, join(patch_parts) if patches else None)
