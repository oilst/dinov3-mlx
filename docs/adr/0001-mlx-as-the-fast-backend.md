# MLX is the fast backend

We build the fast DINOv3 path in MLX. PyTorch on MPS stays as the baseline, and PyTorch on the CPU gives the reference output. We chose MLX over Core ML and custom Metal kernels for three reasons. MLX keeps one Python API for inference and training, and Step 8 needs gradients. MLX runs any input resolution without a fixed-shape export. Apple reports that MLX uses the M5 GPU Neural Accelerators on macOS 26.2 or later. Core ML reaches the Neural Engine best with fixed input shapes and gives no practical path to adapter training.

Status: accepted on 2026-11-06 (the user's decision).
