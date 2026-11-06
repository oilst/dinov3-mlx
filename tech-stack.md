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
| Python | 3.12.13 (installed by uv) |
| uv | 0.12.15 |
| mlx | 0.32.4 |
| torch | 2.15.0 |
| transformers | 5.22.1 |
| safetensors | 0.8.0 |
| huggingface_hub | 2.4.0 |
| pillow | 13.0.0 |
| pytest | 9.2.0 |
| ruff | 0.17.4 |

Change a version only in its own commit, and record the new version here.

## Weights

`uv run python scripts/download_weights.py` downloads the six model sizes into the Hugging Face cache, one at a time. All six need 32 GB of disk (fp32 safetensors). ViT-7B/16 alone needs 26.9 GB.

While the access to the gated repos waits, `uv run python scripts/convert_meta_checkpoint.py MODEL_SIZE CHECKPOINT.pth` converts a Meta checkpoint into `~/.cache/dinov3-apple/converted/<Hugging Face id>`. The script checks the checksum and the first output values. On 2026-11-06 ViT-S/16 and ViT-H+/16 are converted. The conversion of ViT-H+/16 needs 7.1 GB of memory.

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
