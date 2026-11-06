# Mission

We make DINOv3 fast on Apple silicon, and we learn how DINOv3 works in detail.

## Why

We use DINOv3 backbones to classify video frames on Apple silicon Macs. We do not know how much speed PyTorch on the MPS backend leaves unused. MLX is Apple's array framework for Apple silicon. Apple reports that MLX uses the GPU Neural Accelerators of the M5 chip on macOS 26.2 or later.

## Goals

1. Run every DINOv3 ViT model size in MLX: ViT-S/16, ViT-S+/16, ViT-B/16, ViT-L/16, ViT-H+/16 and ViT-7B/16.
2. Give frame embeddings and patch embeddings that pass the parity tests.
3. Measure the throughput and the peak memory of every model size against the baseline, for frame embeddings and for patch embeddings.
4. Understand DINOv3 in detail. Write one explainer for each part that we build.
5. Make our reference fine-tuning setup fast on Apple silicon. Then compare better adapters, for example DoRA.
6. Publish the code, the tests, the benchmarks and the results as a public GitHub repo.

## Success measures

| Goal | Measure |
|---|---|
| Parity | Frame embedding cosine similarity is at least 0.9999 in fp32 and at least 0.999 in fp16 and bf16, for every model size. For patch embeddings, we report the mean and the minimum cosine similarity. Step 2 confirms these thresholds. |
| Throughput | For each model size and dtype, the speedup is above 1. If it is not, `verify.html` of the step explains why. |
| Understanding | Every DINOv3 part in `src/` has an explainer in `docs/learn/`. Every term that the code and the docs use is in `GLOSSARY.md`. |
| Fine-tuning | MLX reproduces the reference fine-tuning setup with the same quality on a public dataset and with more images per second. A table compares LoRA, DoRA and the other adapters. |

## Non-goals

- ConvNeXt model sizes. We add them only when a step needs them.
- Training DINOv3 from scratch.
- Core ML, the Apple Neural Engine and custom Metal kernels. We add them only when a profile shows that MLX cannot reach a goal.

## Constraints

- **Hardware.** We benchmark on one Mac: Apple M5 Max, 64 GB unified memory, macOS 26.5.2.
- **Weights.** We download the weights from Hugging Face after we accept the DINOv3 License. The weights stay outside the repo.
- **License.** The DINOv3 License covers derivative works of the DINOv3 code. We choose the repo license before the first push (Step 1).
- **Public repo.** The repo holds no client data, no private paths and no private weights. Private context goes into `CLAUDE.local.md`, which git ignores.

## Audience

- Developers who run DINOv3 on a Mac.
- Our team, which fine-tunes DINOv3 for video classification.
