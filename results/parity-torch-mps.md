# Parity of torch-mps

Made by `scripts/parity_table.py` on 2026-11-07 from the parity tests (torch 2.15.0). The reference output is torch-cpu in fp32. Each value is the minimum over the configurations of the row. The patch minimum is report only. Section 3 gives the thresholds.

## 1 Parity against the reference output

Each row covers 12 configurations: 3 test images × 4 input resolutions.

| Model size | Dtype mode | Measured | Non-finite | Frame | Patch mean | Patch 1st percentile | Patch minimum | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-S/16 | fp32 | 12 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-S/16 | fp16-cast | 12 | 0 | 0.999994 | 0.999997 | 0.999990 | 0.999945 | pass |
| ViT-S/16 | bf16-cast | 12 | 0 | 0.999607 | 0.999768 | 0.999275 | 0.998643 | pass |
| ViT-S/16 | fp16-autocast | 12 | 0 | 0.999998 | 0.999999 | 0.999995 | 0.999982 | pass |
| ViT-S/16 | bf16-autocast | 12 | 0 | 0.999867 | 0.999919 | 0.999618 | 0.999190 | pass |
| ViT-S+/16 | fp32 | 12 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-S+/16 | fp16-cast | 12 | 0 | 0.999989 | 0.999974 | 0.999935 | 0.999902 | pass |
| ViT-S+/16 | bf16-cast | 12 | 0 | 0.999411 | 0.998481 | 0.994487 | 0.994102 | pass |
| ViT-S+/16 | fp16-autocast | 12 | 0 | 0.999986 | 0.999956 | 0.999897 | 0.999685 | pass |
| ViT-S+/16 | bf16-autocast | 12 | 0 | 0.998365 | 0.995177 | 0.988774 | 0.980765 | pass |
| ViT-B/16 | fp32 | 12 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-B/16 | fp16-cast | 12 | 0 | 0.999995 | 0.999997 | 0.999990 | 0.999963 | pass |
| ViT-B/16 | bf16-cast | 12 | 0 | 0.999675 | 0.999781 | 0.999199 | 0.998260 | pass |
| ViT-B/16 | fp16-autocast | 12 | 0 | 0.999997 | 0.999998 | 0.999992 | 0.999949 | pass |
| ViT-B/16 | bf16-autocast | 12 | 0 | 0.999864 | 0.999900 | 0.999631 | 0.996915 | pass |
| ViT-L/16 | fp32 | 12 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-L/16 | fp16-cast | 0 | 12 | – | – | – | – | expected failure (fp16 overflow) |
| ViT-L/16 | bf16-cast | 12 | 0 | 0.999791 | 0.999596 | 0.997610 | 0.993536 | pass |
| ViT-L/16 | fp16-autocast | 12 | 0 | 0.999998 | 0.999997 | 0.999975 | 0.999935 | pass |
| ViT-L/16 | bf16-autocast | 12 | 0 | 0.999915 | 0.999763 | 0.996909 | 0.986853 | pass |
| ViT-H+/16 | fp32 | 12 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-H+/16 | fp16-cast | 6 | 6 | 0.999999 | 0.999998 | 0.999988 | 0.999972 | pass, 6 expected failures (fp16 overflow) |
| ViT-H+/16 | bf16-cast | 12 | 0 | 0.999860 | 0.998206 | 0.968878 | 0.845469 | pass |
| ViT-H+/16 | fp16-autocast | 12 | 0 | 0.999999 | 0.999985 | 0.999781 | 0.998451 | pass |
| ViT-H+/16 | bf16-autocast | 12 | 0 | 0.999946 | 0.999430 | 0.991417 | 0.923758 | pass |
| ViT-7B/16 | fp32 | 0 | 0 | – | – | – | – | skipped: ViT-7B/16 torch-mps fp32: ViT-7B/16 runs with fp16, bf16 or quantized weights only (CLAUDE.md run rules) |
| ViT-7B/16 | fp16-cast | 0 | 0 | – | – | – | – | skipped: reference-vit7b16.safetensors is missing. |
| ViT-7B/16 | bf16-cast | 0 | 0 | – | – | – | – | skipped: reference-vit7b16.safetensors is missing. |
| ViT-7B/16 | fp16-autocast | 0 | 0 | – | – | – | – | skipped: ViT-7B/16 torch-mps fp16-autocast: ViT-7B/16 runs with fp16, bf16 or quantized weights only (CLAUDE.md run rules) |
| ViT-7B/16 | bf16-autocast | 0 | 0 | – | – | – | – | skipped: ViT-7B/16 torch-mps bf16-autocast: ViT-7B/16 runs with fp16, bf16 or quantized weights only (CLAUDE.md run rules) |

Table 1. Parity of torch-mps against the reference output.

## 2 Batch consistency

A batch of 8 at 224 × 224 px against 8 runs with batch size 1, on torch-mps in the same dtype mode.

| Model size | Dtype mode | Measured | Non-finite | Frame | Patch mean | Patch 1st percentile | Patch minimum | Result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-S/16 | fp32 | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-S/16 | fp16-cast | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-S/16 | bf16-cast | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-S/16 | fp16-autocast | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-S/16 | bf16-autocast | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-S+/16 | fp32 | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-S+/16 | fp16-cast | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-S+/16 | bf16-cast | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-S+/16 | fp16-autocast | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-S+/16 | bf16-autocast | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-B/16 | fp32 | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-B/16 | fp16-cast | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-B/16 | bf16-cast | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-B/16 | fp16-autocast | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-B/16 | bf16-autocast | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-L/16 | fp32 | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-L/16 | fp16-cast | 0 | 1 | – | – | – | – | expected failure (fp16 overflow) |
| ViT-L/16 | bf16-cast | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-L/16 | fp16-autocast | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-L/16 | bf16-autocast | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-H+/16 | fp32 | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-H+/16 | fp16-cast | 0 | 1 | – | – | – | – | expected failure (fp16 overflow) |
| ViT-H+/16 | bf16-cast | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-H+/16 | fp16-autocast | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-H+/16 | bf16-autocast | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-7B/16 | fp32 | 0 | 0 | – | – | – | – | skipped: ViT-7B/16 torch-mps fp32: ViT-7B/16 runs with fp16, bf16 or quantized weights only (CLAUDE.md run rules) |
| ViT-7B/16 | fp16-cast | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-7B/16 | bf16-cast | 1 | 0 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| ViT-7B/16 | fp16-autocast | 0 | 0 | – | – | – | – | skipped: ViT-7B/16 torch-mps fp16-autocast: ViT-7B/16 runs with fp16, bf16 or quantized weights only (CLAUDE.md run rules) |
| ViT-7B/16 | bf16-autocast | 0 | 0 | – | – | – | – | skipped: ViT-7B/16 torch-mps bf16-autocast: ViT-7B/16 runs with fp16, bf16 or quantized weights only (CLAUDE.md run rules) |

Table 2. Batch consistency.

## 3 Thresholds

| Dtype mode | Frame | Patch mean | Patch 1st percentile |
| --- | --- | --- | --- |
| fp32 | 0.9999 | 0.9999 | 0.9999 |
| fp16-cast | 0.999 | 0.999 | 0.99 |
| bf16-cast | 0.998 | 0.995 | 0.96 |
| fp16-autocast | 0.999 | 0.999 | 0.99 |
| bf16-autocast | 0.998 | 0.995 | 0.96 |

Table 3. Thresholds of `src/dinov3_apple/parity.py`. On 2026-11-06 the user gave bf16 its own row: bf16 activations give lower parity on torch-mps and on torch-cpu alike.
