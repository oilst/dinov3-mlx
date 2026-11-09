"""DINOv3 on Apple silicon with MLX.

The import sets MLX_ENABLE_TF32=0 if the variable is not set: MLX reads it once, at its first
use, and the fp32 matmuls of the preprocessing need TF32 off (ADR 0005). The import itself loads
no mlx and no torch: the public names load on first use (PEP 562).
"""

import os

os.environ.setdefault("MLX_ENABLE_TF32", "0")

__all__ = [
    "Embedder",
    "InsufficientMemoryError",
    "RunRuleError",
    "TF32ModeError",
    "WeightsAccessError",
]

# The module of each public name.
_MODULES = {
    "Embedder": "dinov3_mlx.api",
    "InsufficientMemoryError": "dinov3_mlx.memory",
    "RunRuleError": "dinov3_mlx.memory",
    "TF32ModeError": "dinov3_mlx.mlx_model",
    "WeightsAccessError": "dinov3_mlx.hub",
}


def __getattr__(name: str) -> object:
    """Import a public name on first use, so a torch process that imports a module of this
    package gets no mlx."""
    if name not in _MODULES:
        raise AttributeError(f"module 'dinov3_mlx' has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(_MODULES[name]), name)
