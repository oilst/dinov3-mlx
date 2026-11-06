import mlx.core as mx
import torch


def test_mlx_default_device_is_gpu() -> None:
    assert mx.default_device() == mx.gpu


def test_torch_mps_is_available() -> None:
    assert torch.backends.mps.is_available()


def test_transformers_has_dinov3_vit_model() -> None:
    from transformers import DINOv3ViTModel

    assert DINOv3ViTModel.__name__ == "DINOv3ViTModel"
