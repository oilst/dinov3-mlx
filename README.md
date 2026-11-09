# dinov3-mlx

DINOv3 frame embeddings and patch embeddings for Apple silicon with MLX. All six model sizes, from ViT-S/16 to ViT-7B/16. By Ueli Peter.

- Why and what: [mission.md](mission.md)
- Steps and status: [roadmap.md](roadmap.md)
- The full plan: [docs/plan.html](docs/plan.html)
- The words we use: [GLOSSARY.md](GLOSSARY.md)

## What it does

The package runs the DINOv3 backbones of Meta on the GPU of a Mac with MLX. It gives two outputs for each frame:

- the frame embedding: the CLS token after the final layer norm, shape (D,);
- the patch embeddings: one token for each 16 × 16 px patch, shape (H/16, W/16, D).

The input is uint8 RGB frames. The package does the preprocessing of the Hugging Face image processor on the GPU: resize to 224 × 224 px (bilinear, with antialiasing), then the ImageNet mean and standard deviation. The forward pass is compiled once for each input resolution.

## Installation

Python 3.12 or newer on a Mac with Apple silicon.

```bash
uv add git+https://github.com/oilst/dinov3-mlx@v0.1.0
```

The core needs five packages: mlx, numpy, safetensors, huggingface-hub and pillow. It does not need torch. The extra `reference` adds torch and transformers for the reference outputs and the torch-mps baseline of this repo.

## Access to the weights

The DINOv3 weights are not part of this repo. They are gated on Hugging Face under the DINOv3 License.

1. Open the model page of a model size, for example [facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m).
2. Accept the DINOv3 License on that page.
3. Log in on your Mac: `hf auth login`.

The first `Embedder` of a model size downloads its weights into the Hugging Face cache. Without access, the constructor raises `WeightsAccessError` with these steps.

## Usage

```python
import numpy as np
from dinov3_mlx import Embedder

model = Embedder("ViT-S/16")  # fp16-autocast, batch size 8
frames = np.stack(rgb_frames)  # uint8, (B, h, w, 3)
frame = model.embed_frames(frames)  # (B, 384) fp32
frame = frame / np.linalg.norm(frame, axis=1, keepdims=True)  # L2 normalization, if needed
frame, patches = model.embed(frames, size=(224, 400))  # both from one forward pass
```

- `frames` is a uint8 array (B, h, w, 3), or a list of PIL images or uint8 arrays of one size.
- `size` is an int or (H, W). Each side must be a multiple of 16. The default is 224.
- The API cuts the frames into batches of `batch_size`. It pads the last batch with zero frames, so each input resolution needs one trace only.
- `as_numpy=False` keeps the outputs as `mx.array` on the GPU.
- `bits=8` loads 8-bit block weights (fp16-autocast and bf16-autocast only).
- The model size can also be its Hugging Face id. `del model` frees the memory.

## Dtype modes

`Embedder(model_size, dtype_mode=...)` takes one of five dtype modes. The default is fp16-autocast: it passes the fp16 parity thresholds for all six model sizes. bf16-cast is the fastest mode, with a lower patch parity.

| Dtype mode | ViT-S/16, 224 px, b8 (img/s) | ViT-S/16, 448 px, best (img/s) | Worst frame | Worst patch 1st percentile | Model sizes with finite output |
| --- | --- | --- | --- | --- | --- |
| fp32 | 905.6 | 164.7 | 1.000000 | 1.000000 | 5 of 6 (ViT-7B/16: run rule) |
| fp16-cast | 1772.1 | 411.5 | 0.999973 | 0.999654 | 4 of 6 (fp16 overflow of ViT-L/16 and ViT-H+/16) |
| bf16-cast | 1893.0 | 413.9 | 0.998847 | 0.982649 | 6 of 6 |
| fp16-autocast | 1693.6 | 378.7 | 0.999986 | 0.999679 | 6 of 6 |
| bf16-autocast | 1651.0 | 379.6 | 0.998387 | 0.983737 | 6 of 6 |

Source: throughput from Table 5 of [results/speedup-table.md](results/speedup-table.md) (compiled mlx, frame output, one round); parity from Table 1 of [results/parity-mlx.md](results/parity-mlx.md) (worst value over all model sizes and 12 configurations).

- "cast" modes run the whole network in fp16 or bf16. "autocast" modes keep the residual stream and the layer norms in fp32.
- fp32 runs with TF32 off (ADR 0005). ViT-7B/16 does not run in fp32 (its weights alone need about 27 GB).
- The frame embeddings are fp32. The patch embeddings are fp16 in the four fp16 and bf16 modes (`patch_fp32=True` gives fp32).

## Results

All numbers come from one Mac: an Apple M5 Max with 64 GB unified memory, macOS 26.5.2, on the power adapter. The reference output is the Hugging Face model on torch-cpu in fp32.

Parity: mlx passes the parity thresholds in every configuration of every model size and dtype mode, except the expected fp16 overflow of ViT-L/16 and ViT-H+/16 in fp16-cast (as on torch-mps).

| Model size | fp32 | fp16-autocast: worst frame | fp16-autocast: worst patch 1st percentile | bf16-cast: worst patch 1st percentile |
| --- | --- | --- | --- | --- |
| ViT-S/16 | pass, 1.000000 | 0.999997 | 0.999995 | 0.999560 |
| ViT-S+/16 | pass, 1.000000 | 0.999986 | 0.999904 | 0.990381 |
| ViT-B/16 | pass, 1.000000 | 0.999997 | 0.999993 | 0.999478 |
| ViT-L/16 | pass, 1.000000 | 0.999998 | 0.999966 | 0.996394 |
| ViT-H+/16 | pass, 1.000000 | 0.999999 | 0.999679 | 0.986725 |
| ViT-7B/16 | not run (run rule) | 0.999990 | 0.999762 | 0.982649 |

Source: Table 1 of [results/parity-mlx.md](results/parity-mlx.md), 12 configurations per row (3 test images × 4 input resolutions).

Speed against PyTorch on MPS: the paired ratio mlx ÷ torch-mps (above 1: mlx is faster), both sides compiled. P1 is ViT-S/16 at 224 px, P2 is ViT-H+/16 at 512 px, both at batch size 8.

| Priority | Dtype | mlx mode | torch-mps mode | Frame | Patch | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| P1 | fp32 | fp32 | fp32 (compile) | 1.067 | 1.068 | above 1 |
| P1 | fp16 | fp16-cast | fp16-cast (compile) | 0.867 | 0.900 | not above 1 |
| P1 | bf16 | bf16-cast | bf16-cast (compile) | 0.878 | 0.881 | not above 1 |
| P2 | fp32 | fp32 | fp32 (compile) | 1.092 | 1.092 | above 1 |
| P2 | fp16 | fp16-cast | fp16-cast (compile) | 1.032 | 1.033 | above 1 |
| P2 | bf16 | bf16-autocast | bf16-cast (compile) | 1.013 | 1.013 | above 1, inside the noise |

Source: Table 1 of [results/speedup-table.md](results/speedup-table.md), pooled median of two runs of 5 paired rounds. The matrix of 324 cells is Table 2 of the same file.

API overhead: the `Embedder` against the direct path (preprocessing, compiled forward pass and output cast, without the API) on the same uint8 frames, fp16-autocast, batch size 8.

| Configuration | Model size, px | Output | Frames per call | Ratio api ÷ direct |
| --- | --- | --- | --- | --- |
| P1 | ViT-S/16, 224 | frame | 8 | 0.996 |
| P2 | ViT-H+/16, 512 | patch | 8 | 1.009 |
| P1-pad | ViT-S/16, 224 | frame | 13 | 0.819 |

Source: Table 1 of [results/api-table.md](results/api-table.md), lower median of 5 rounds. P1-pad has one padded batch: 3 of its 16 rows are zero frames. Call the API with a multiple of `batch_size` frames, or set `batch_size` to the number of frames.

Prior art: against the two public Python MLX ports of DINOv3, this package is faster in all 8 measured cells.

| Cell | Port | Model, px | Dtype | Batch size | Ratio ours ÷ port |
| --- | --- | --- | --- | --- | --- |
| port-1 | mlx-image | ViT-S/16, 224 | fp32 | 8 | 1.162 |
| port-2 | mlx-image | ViT-S/16, 224 | bf16 | 8 | 1.274 |
| port-3 | mlx-image | ViT-S/16, 224 | fp16 | 8 | 1.335 |
| port-4 | mlx-image | ViT-S/16, 224 | fp32 | 1 | 1.199 |
| port-5 | mlx_dinov3 | ViT-H+/16, 512 | fp32 | 8 | 1.169 |

Source: Section 6 of [results/prior-art-table.md](results/prior-art-table.md), pooled median of two runs. The same file has the other cells, the Core ML exports and the parity of each port.

Neural Engine: Core ML on the Neural Engine gives valid tokens after the rewrite `ne3` (Step 6d). Alone it is slower than mlx above batch size 1. In parallel with mlx on the GPU it adds throughput.

| Group | Neural Engine alone (img/s) | mlx alone (img/s) | Both: total (img/s) | Stream ratio |
| --- | --- | --- | --- | --- |
| P1, ViT-S/16 224 px b8, run 1 | 677.3 | 1,543.4 | 2,044.0 | 1.324 |
| P1, ViT-S/16 224 px b8, run 2 | 676.4 | 1,939.9 | 2,583.1 | 1.318 |
| P3, ViT-L/16 224 px b8, run 2 | 65.4 | 246.9 | 292.2 | 1.184 |

Source: Table 6 of [results/coreml-6d-table.md](results/coreml-6d-table.md). Stream ratio = (both streams) ÷ (mlx alone). Step 7b brings this as `backend="mlx+ne"`.

## Limits

- V2: the absolute throughput of both backends drifts by up to 53 % between jobs. The paired ratios stay stable (run medians within 0.027 at P1). Compare ratios, not absolute img/s.
- V4: the best batch size of Table 5 of the speedup table rests on one round. The default batch size 8 is the best or close to it at ViT-S/16, 224 px.
- V17: one paired round per matrix cell does not decide the cells near 1. A second measurement changed 10 of 62 verdicts; 127 of 306 cells lie within 1 ± 0.06.
- One Mac only: every number comes from one M5 Max. Other chips can give other ratios.
- TF32: `import dinov3_mlx` sets `MLX_ENABLE_TF32=0` if the variable is not set. This also turns TF32 off for your own fp32 matmuls in the same process. If mlx runs with TF32 (for example because you used mlx before the import), the `Embedder` raises `TF32ModeError`.

V2, V4 and V17 are findings of the Step 6 verification ([specs/06-speed/verify.html](specs/06-speed/verify.html)); [ADR 0009](docs/adr/0009-compiled-torch-mps-rows-and-paired-groups-for-every-speedup.md) records them.

## Tested versions

macOS 26.5.2, Python 3.12.13, mlx 0.32.4, numpy 2.5.4, safetensors 0.8.0, huggingface-hub 2.4.0, pillow 13.0.0. Reference and baseline: torch 2.15.0, transformers 5.22.1. `uv.lock` holds the exact versions of all packages.

## Tests and benchmarks

```bash
uv sync
uv run pytest
```

- Tests that need the weights or the reference files skip without them. CI runs the tests without weights on each push.
- `DINOV3_SLOW=1 uv run pytest` runs the slow session: all six model sizes.
- `uv run python scripts/make_reference.py` writes the reference outputs (needs the extra `reference` and the weights).
- `scripts/bench_speed.py` runs the speed benchmarks of Step 6. Run one benchmark at a time: each configuration runs in its own process after a memory check.

## License

- The code is MIT: [LICENSE](LICENSE).
- The DINOv3 weights are under the DINOv3 License: [LICENSE-DINOv3.md](LICENSE-DINOv3.md). This repo holds no weights ([ADR 0002](docs/adr/0002-mit-for-code-dinov3-license-for-weights.md)).
- If you publish research results that use DINOv3 through this package, clause 1.b.ii of the DINOv3 License asks you to acknowledge the use of the DINO Materials.

## Pages

The plan, the explainers of each DINOv3 part and the spec of each step: [oilst.github.io/dinov3-mlx](https://oilst.github.io/dinov3-mlx/).
