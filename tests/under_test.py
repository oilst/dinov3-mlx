"""The model sizes under test. ViT-S/16 and ViT-B/16 by default, all six with DINOV3_SLOW=1."""

import os

from dinov3_apple.hub import MODEL_IDS

DEFAULT_MODEL_SIZES = ["ViT-S/16", "ViT-B/16"]
SLOW = os.environ.get("DINOV3_SLOW") == "1"
MODEL_SIZES_UNDER_TEST = list(MODEL_IDS) if SLOW else DEFAULT_MODEL_SIZES

# The model sizes of the backend mlx. Step 4 ports ViT-S/16 only. Step 5 sets this list to
# MODEL_SIZES_UNDER_TEST.
MLX_MODEL_SIZES_UNDER_TEST = ["ViT-S/16"]
