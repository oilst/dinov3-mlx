"""The model sizes under test. ViT-S/16 and ViT-B/16 by default, all six with DINOV3_SLOW=1.

The backend mlx also runs ViT-S+/16 by default, the smallest model size with SwiGLU (D7 of
specs/05-all-sizes/explore.html).
"""

import os

from dinov3_mlx.hub import MODEL_IDS

DEFAULT_MODEL_SIZES = ["ViT-S/16", "ViT-B/16"]
SLOW = os.environ.get("DINOV3_SLOW") == "1"
MODEL_SIZES_UNDER_TEST = list(MODEL_IDS) if SLOW else DEFAULT_MODEL_SIZES

# The model sizes of the backend mlx.
MLX_DEFAULT_MODEL_SIZES = ["ViT-S/16", "ViT-S+/16", "ViT-B/16"]
MLX_MODEL_SIZES_UNDER_TEST = MODEL_SIZES_UNDER_TEST if SLOW else MLX_DEFAULT_MODEL_SIZES
