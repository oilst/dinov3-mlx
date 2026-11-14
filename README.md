# dinov3-mlx

**DINOv3 on Apple silicon, in MLX: up to 1.4 times the throughput of PyTorch with `torch.compile` on MPS, and 1.16 to 1.37 times the throughput of the other MLX ports.** Frame embeddings and patch embeddings for all six model sizes, parity-tested against the Hugging Face reference, with no PyTorch at run time.

By Ueli Peter. The code is MIT; the weights stay under the DINOv3 License.

| | |
| --- | --- |
| Model sizes | ViT-S/16, ViT-S+/16, ViT-B/16, ViT-L/16, ViT-H+/16, ViT-7B/16 (ViT-7B/16 in bf16: 13.5 GB) |
| Parity | In the default dtype mode, the compiled forward pass of all six model sizes passes the fp16 parity thresholds against Hugging Face `transformers` in all 12 tested configurations. Worst frame cosine: 0.999986 |
| Speed against `torch.compile` | 1.41 at batch size 1 (ViT-S/16, fp16 autocast); 1.07 to 1.09 in fp32; 0.87 to 1.03 at batch size 8 in fp16 and bf16 |
| Distance to the optimum | ViT-H+/16 at 512 px reaches about 61 % of the matmul roof of the Mac; ViT-S/16 at 224 px about 36 % |
| Core | Five packages: mlx, numpy, safetensors, huggingface-hub, pillow |

Source: [results/parity-mlx.md](results/parity-mlx.md), [results/speedup-table.md](results/speedup-table.md), [results/prior-art-table.md](results/prior-art-table.md). Measured on an Apple M5 Max (64 GB).

## Quick start

```bash
uv add git+https://github.com/oilst/dinov3-mlx@v0.1.0
hf auth login  # after you accept the DINOv3 License on the model page
```

```python
import numpy as np
from dinov3_mlx import Embedder

model = Embedder("ViT-S/16")  # downloads the weights on first use
frames = np.stack(rgb_frames)  # uint8 RGB, (B, h, w, 3)
frame, patches = model.embed(frames, size=224)  # (B, 384) fp32, (B, 14, 14, 384) fp16
```

- `embed_frames` and `embed_patches` give one output. `size` is an int or (H, W); each side is a multiple of 16.
- Give a multiple of `batch_size` frames (default 8). A padded last batch costs throughput: 13 frames at batch size 8 (3 zero frames in 16 rows) gave 0.82 of the throughput of the direct path.
- `dtype_mode` sets the precision. The default `fp16-autocast` is the only fast mode that passes the fp16 parity thresholds for all six model sizes. `bf16-cast` is about 12 % faster at ViT-S/16, with a lower patch parity.
- The weights are gated. Accept the DINOv3 License on the page of each model size, for example [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m), then run `hf auth login`. Without access the constructor raises `WeightsAccessError` with these steps.

## How it compares

Ratio = throughput of this package ÷ throughput of the other option, on the same frames, in paired rounds. Above 1: this package is faster. P1 = ViT-S/16 at 224 px, P2 = ViT-H+/16 at 512 px, batch size 8, unless the row says otherwise.

| Other option | Ratio | Where | Parity of the other option |
| --- | --- | --- | --- |
| PyTorch MPS, `torch.compile`, fp16 autocast | 1.41 | ViT-S/16, 224 px, batch size 1 | passes |
| PyTorch MPS, `torch.compile`, fp32 | 1.07 / 1.09 | P1 / P2 | passes |
| PyTorch MPS, `torch.compile`, fp16 and bf16 | 0.87 to 0.90 / 1.01 to 1.03 | P1 / P2 | passes |
| mlx-image (MLX port) | 1.16 to 1.34 | P1, fp32, bf16 and fp16 | misses the fp32 thresholds narrowly; NaN in fp16 |
| mlx_dinov3 (MLX port) | 1.17 to 1.37 | P2, fp32 and bf16 | misses the fp32 thresholds narrowly |
| Core ML exports of DINOv3 on the GPU, FP32 | 1.06 to 1.09; 1.54 at ViT-S/16, batch size 8 | the fixed shape of each export | passes |
| Core ML exports of DINOv3 on the GPU, FP16 | 0.93 to 0.99 | the fixed shape of each export | passes; the INT8 export fails |

Source: [results/speedup-table.md](results/speedup-table.md) (PyTorch rows), [results/prior-art-table.md](results/prior-art-table.md) (MLX ports and Core ML exports).

- On MPS, `torch.compile` and this package reach the same matmul rate in fp16 and bf16. A small model at batch size 8 stays faster on PyTorch; large models are equal. At batch size 1 and in fp32, this package is faster.
- The other MLX ports lose 10 to 24 % of the throughput to unfused attention, and their bf16 RoPE misses the fp32 thresholds.
- A Core ML FP16 export of DINOv3 can be up to 8 % faster, but only at the one input shape that it was exported for.
- Core ML on the Neural Engine, in parallel with this package on the GPU, adds 8 to 32 % throughput ([results/coreml-6d-table.md](results/coreml-6d-table.md)). A later version brings it as `backend="mlx+ne"`.

## How far from the optimum

The upper limit: all FLOPs of a frame at the best matmul rate that we measured on this Mac, about 60 TFLOP/s in fp16 and bf16 (one `nn.Linear` 1,280 → 10,240). It is the best rate that this repo measured, not a hardware peak: a faster kernel would raise it.

| Configuration | GFLOP per frame | Upper limit (img/s) | Measured, best batch size (img/s) | Share of the limit |
| --- | --- | --- | --- | --- |
| ViT-S/16, 224 px, bf16-cast | 9.4 | about 6,400 | 2,271 (batch size 256) | about 36 % |
| ViT-H+/16, 512 px, bf16-cast | 1,901.9 | about 31.5 | 19.3 (batch size 2) | about 61 % |

Source: measured values from Table 6 of [results/speedup-table.md](results/speedup-table.md); matmul rate from Table 1 of [results/kernels-c1.md](results/kernels-c1.md); FLOPs from `dinov3_mlx.flops.image_flops`.

- A large model at a high input resolution spends its time in large matmuls, so it comes near the limit. A small model spends a large share in element-wise kernels, layer norms and the fixed cost of each kernel.
- The rest of the gap is in the matmul and attention kernels of MLX. Joined projections and other tile sizes did not close it ([ADR 0017](docs/adr/0017-separate-projections-and-the-tile-sizes-of-mlx-stay.md)). The [roofline explainer](docs/learn/06-5-roofline.html) puts each part of a block on the roofline.

## What this repo holds

- `src/dinov3_mlx/`: the package. The MLX backbone, the weight loader, the preprocessing on the GPU and the public API.
- `tests/`: parity tests against stored reference outputs, and unit tests on small random models.
- `scripts/` and `results/`: the benchmark harness and every measured table.
- `docs/learn/`: one explainer for each part of DINOv3 that we built, from tokens and RoPE to the roofline, the Neural Engine and the public API.
- `specs/` and `docs/adr/`: for each step the options, the plan with its acceptance criteria and the verification; the decisions.

## Documentation

- [The site of this repo](https://oilst.github.io/dinov3-mlx/): the plan, the explainers and the spec of each step.
- [The public API explainer](docs/learn/07-public-api.html): the path of one frame, the dtype modes, the padded batch and the errors.
- [Mission](mission.md), [roadmap](roadmap.md), [tech stack and tested versions](tech-stack.md), [glossary](GLOSSARY.md).
- Result tables: [parity](results/parity-mlx.md), [speedup](results/speedup-table.md), [prior art](results/prior-art-table.md), [API overhead](results/api-table.md), [Neural Engine](results/coreml-6d-table.md).

## Limits

- All numbers come from one Mac: an Apple M5 Max with 64 GB, macOS 26.5.2, on AC power. Other chips can give other ratios.
- The absolute throughput drifts by up to 53 % between jobs (V2); the paired ratios stay stable. The best batch size rests on one round (V4). One paired round does not decide the matrix cells near 1 (V17). These are findings of the [Step 6 verification](specs/06-speed/verify.html).
- ViT-7B/16 does not run in fp32: its weights alone need about 27 GB. fp16-cast overflows for ViT-L/16 and ViT-H+/16.
- `import dinov3_mlx` sets `MLX_ENABLE_TF32=0` if it is not set. This also turns TF32 off for your own fp32 matmuls in the same process.

## Tests

```bash
uv sync
uv run pytest                 # the tests that need the weights skip without them
DINOV3_SLOW=1 uv run pytest   # all six model sizes
```

## License

The code is MIT ([LICENSE](LICENSE)). The DINOv3 weights are under the [DINOv3 License](LICENSE-DINOv3.md); this repo holds no weights. If you publish research results that use DINOv3 through this package, clause 1.b.ii of the DINOv3 License asks you to acknowledge the use of the DINO Materials.
