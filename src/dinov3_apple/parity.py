"""Cosine parity of the frame embedding and the patch embeddings. NumPy only, so mlx uses it too."""

from dataclasses import dataclass

import numpy as np

from dinov3_apple.outputs import NUM_PREFIX_TOKENS


@dataclass(frozen=True)
class Parity:
    """The parity of one output against the reference output.

    Attributes:
        frame: Cosine similarity of the frame embeddings.
        patch_mean: Mean cosine similarity over the patch embeddings.
        patch_p01: 1st percentile of the cosine similarity over the patch embeddings.
        patch_min: Minimum cosine similarity over the patch embeddings. Report only.
    """

    frame: float
    patch_mean: float
    patch_p01: float
    patch_min: float


_FP32 = Parity(frame=0.9999, patch_mean=0.9999, patch_p01=0.9999, patch_min=-1.0)
_FP16 = Parity(frame=0.999, patch_mean=0.999, patch_p01=0.99, patch_min=-1.0)
# bf16 got its own row on 2026-11-06. bf16 activations on torch-mps and torch-cpu alike give a
# frame parity down to 0.9984 and a patch 1st percentile down to 0.969 (ViT-H+/16 at 224 px).
_BF16 = Parity(frame=0.998, patch_mean=0.995, patch_p01=0.96, patch_min=-1.0)

# Table 6 of specs/02-reference/plan.html. patch_min is report only, so its limit is -1.
THRESHOLDS: dict[str, Parity] = {
    "fp32": _FP32,
    "fp16-cast": _FP16,
    "bf16-cast": _BF16,
    "fp16-autocast": _FP16,
    "bf16-autocast": _BF16,
}


def measure_parity(reference: np.ndarray, output: np.ndarray) -> Parity:
    """Compare two token sequences of shape (1 + 4 + N, D).

    The cosine runs in fp64, so two equal fp32 vectors give 1.0.

    Args:
        reference: The reference tokens after the final layer norm.
        output: The tokens of the backend under test.

    Returns:
        The parity of the frame embedding and of the patch embeddings.
    """
    ref, out = reference.astype(np.float64), output.astype(np.float64)
    cos = (ref * out).sum(-1) / (np.linalg.norm(ref, axis=-1) * np.linalg.norm(out, axis=-1))
    cos = np.minimum(cos, 1.0)
    patches = cos[NUM_PREFIX_TOKENS:]
    return Parity(
        frame=float(cos[0]),
        patch_mean=float(patches.mean()),
        patch_p01=float(np.quantile(patches, 0.01)),
        patch_min=float(patches.min()),
    )
