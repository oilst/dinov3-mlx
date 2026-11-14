# Tech stack

## Hardware

Apple M5 Max, 64 GB unified memory, macOS 26.5.2. All benchmark results in this repo come from this Mac unless a result file names another one.

## Software

| Part | Choice | Reason |
|---|---|---|
| Language | Python 3.12 | MLX, PyTorch and transformers support it. |
| Environment | uv (`uv sync`, `uv run`) | One lock file, fast installs. |
| Fast backend | MLX | See `docs/adr/0001-mlx-as-the-fast-backend.md`. |
| Reference output and baseline | PyTorch (CPU and MPS) with Hugging Face `transformers` (`DINOv3ViTModel`) | Our production pipeline uses this path. |
| Weights | Hugging Face Hub, safetensors | The DINOv3 repos are gated. Accept the DINOv3 License first. |
| Fine-tuning reference | PEFT on PyTorch | Our reference fine-tuning setup uses PEFT LoRA. |
| Tests | pytest | |
| Lint and format | ruff | |
| Specs and explainers | Single-file HTML | See the spec loop in `CLAUDE.md`. |

## Versions

`pyproject.toml` pins each package with `==`, and `uv.lock` pins the full tree. Step 1 recorded these versions on 2026-11-06.

| Part | Version |
|---|---|
| macOS | 26.5.2 (build 25F84) |
| Python | 3.12.14 (installed by uv) |
| uv | 0.12.15 |
| mlx | 0.32.4 |
| mlx-metal | 0.32.4. An indirect dependency of mlx: it holds `libmlx.dylib` and the Metal kernels. `uv.lock` pins it; `pyproject.toml` does not name it. Since Step 6, `machine.read_versions` records it, so each result line of `mlx` and of `torch-mps` has it (decision D19 of `specs/06-speed/explore.html`). |
| torch | 2.15.0 |
| transformers | 5.22.1 |
| safetensors | 0.8.0 |
| huggingface_hub | 2.4.0 |
| pillow | 13.0.0 |
| pytest | 9.2.0 |
| ruff | 0.17.4 |
| coremltools | 9.0 (dependency group `coreml`, Step 6c, 2026-11-07). It is tested with torch 2.7 only; with numpy 2.5 its torch converter needs one patch: `patch_coremltools_cast` in `src/dinov3_mlx/coreml.py` (ADR 0015). |
| Core ML | The framework of macOS (Darwin 25.5 on this Mac). Step 6c packages: ML Program, fp16 compute precision, macOS 15 target, fixed shapes, outside the repo in `~/.cache/dinov3-mlx/coreml/`. Compute units CPU_AND_NE (`coreml-ne`), CPU_AND_GPU (`coreml-gpu`) and ALL (`coreml-all`). ADR 0014 and ADR 0015. |

Change a version only in its own commit, and record the new version here.

## Neural Accelerators

Step 6 recorded the versions that run the NAX kernels (decision D19, option b, of `specs/06-speed/explore.html`). The versions and the switch come from the probe nax-record (Table 8 of that file). The fingerprint values come from the report line of the fingerprint test in the Verify runs (AC16 of `specs/06-speed/verify.html`). The throughput ratios come from the Learn data of Step 6.

| Part | Value |
|---|---|
| Chip | Apple M5 Max |
| GPU architecture | `applegpu_g17s` (`mx.device_info()["architecture"]`) |
| macOS | 26.5.2 |
| mlx and mlx-metal | 0.32.4 and 0.32.4 |
| `libmlx.dylib` | Built for macOS 26.2 and later (minos 26.2) |
| Switch to turn the NAX kernels off | `MLX_METAL_GPU_ARCH=applegpu_g16s` (Neural Accelerators off) |
| Relaxed precision | On in the NAX matmul kernel and the NAX attention kernel (`steel/gemm/nax.h`, `steel/attn/nax.h`). No environment variable turns it off. |
| SDPA fingerprint (test `test_mlx_sdpa_nax_fingerprint`, report only) | Neural Accelerators on: fp16 54.5 % (head dimension 64) and 54.6 % (head dimension 128), bf16 12.6 % (head dimension 64) and 12.7 % (head dimension 128). The two controls of the test: 0.21 % (fp16) and 0.03 % (bf16). Neural Accelerators off (probe nax-record): 0.21 % (fp16) and 0.023 % (bf16). |
| Throughput, Neural Accelerators on / off | Compiled forward pass, bf16-cast: 1.87 to 1.88 at P1, 2.79 to 2.82 at P2 (two rounds each; Learn data of Step 6, `docs/learn/06-2-attention-and-batch-size.html`). The probe nax-ab gave 1.81 and 2.69. |

Each bench line of `mlx` in Step 6 records `mlx_architecture` and `versions["mlx-metal"]`. The 2 token-range lines of `results/speed-learn.jsonl` have no `mlx_architecture` (finding V6 of `specs/06-speed/verify.html`). Each default session runs the fingerprint test. A new mlx or mlx-metal version that turns the NAX kernels off or changes the relaxed precision changes the fingerprint. Then measure the parity and the throughput again, and update this table.

## Weights

`uv run python scripts/download_weights.py` downloads the six model sizes into the Hugging Face cache, one at a time. All six need 32 GB of disk (fp32 safetensors). ViT-7B/16 alone needs 26.9 GB.

While the access to the gated repos waits, `uv run python scripts/convert_meta_checkpoint.py MODEL_SIZE CHECKPOINT.pth` converts a Meta checkpoint into `~/.cache/dinov3-mlx/converted/<Hugging Face id>`. The script checks the checksum and the first output values. On 2026-11-06 ViT-S/16 and ViT-H+/16 were converted, and their tensors are equal to the Hub tensors. The conversion of ViT-H+/16 needs 7.1 GB of memory. Since the access came, the converter is only a fallback.

## Model ids on Hugging Face

| Model size | Hugging Face id |
|---|---|
| ViT-S/16 | `facebook/dinov3-vits16-pretrain-lvd1689m` |
| ViT-S+/16 | `facebook/dinov3-vits16plus-pretrain-lvd1689m` |
| ViT-B/16 | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| ViT-L/16 | `facebook/dinov3-vitl16-pretrain-lvd1689m` |
| ViT-H+/16 | `facebook/dinov3-vith16plus-pretrain-lvd1689m` |
| ViT-7B/16 | `facebook/dinov3-vit7b16-pretrain-lvd1689m` |

## Code conventions

- Type hints on every parameter and every return value.
- Google-style docstrings on public functions.
- pathlib, f-strings and context managers.
- Specific exception types. No bare `except`.
- pytest tests with descriptive names for all logic.
