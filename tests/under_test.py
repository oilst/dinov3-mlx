"""The model sizes under test. ViT-S/16 and ViT-B/16 by default, all six with DINOV3_SLOW=1."""

import os

from dinov3_apple.hub import MODEL_IDS

DEFAULT_MODEL_SIZES = ["ViT-S/16", "ViT-B/16"]
SLOW = os.environ.get("DINOV3_SLOW") == "1"
MODEL_SIZES_UNDER_TEST = list(MODEL_IDS) if SLOW else DEFAULT_MODEL_SIZES
