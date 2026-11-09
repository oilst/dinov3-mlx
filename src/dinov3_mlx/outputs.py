"""Token sequence to frame embedding and patch embeddings.

The sequence has 1 CLS token, 4 register tokens and N patch tokens. The register tokens are not
part of either output.
"""

import numpy as np

NUM_PREFIX_TOKENS = 5
PATCH_SIZE = 16


def split_tokens(tokens: np.ndarray, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Split the tokens after the final layer norm into the two outputs.

    Args:
        tokens: Shape (..., 1 + 4 + N, D).
        height: The input height in pixels.
        width: The input width in pixels.

    Returns:
        The frame embedding with shape (..., D) and the patch embeddings with shape
        (..., H / 16, W / 16, D).
    """
    frame = tokens[..., 0, :]
    patches = tokens[..., NUM_PREFIX_TOKENS:, :]
    grid = (height // PATCH_SIZE, width // PATCH_SIZE)
    return frame, patches.reshape(*tokens.shape[:-2], *grid, tokens.shape[-1])
