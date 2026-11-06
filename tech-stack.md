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

Step 1 pins the exact versions in `pyproject.toml`.

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
