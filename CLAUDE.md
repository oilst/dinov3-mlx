# dinov3-mlx

DINOv3 on Apple silicon with MLX: fast frame embeddings and patch embeddings for every model size, then fast fine-tuning. `mission.md` gives the goals, `roadmap.md` the current step, `docs/plan.html` the detail of every step, `tech-stack.md` the tools and model ids.

## Words

Use the terms of `GLOSSARY.md` in code, docs and messages. When a new term settles, add it to `GLOSSARY.md` at once. The glossary holds definitions only.

## Spec loop

Work on one roadmap step at a time, in its folder `specs/NN-slug/`. Each stage ends on its done condition.

1. **Explore.** When the step has real options, write `explore.html`: each option, its trade-off, a recommendation. Done: the user picks an option.
2. **Plan.** Write `plan.html` from `specs/_template.html`: data flow (SVG), key code snippets, the tests, the benchmarks and numbered acceptance criteria with target numbers. Done: the user approves the plan.
3. **Build.** A new session builds from the spec folder. Write each test first and see it go red, then make it green. Done: every acceptance criterion has a green test or a benchmark result.
4. **Verify.** A separate agent reads the spec folder and the diff, runs the tests and the benchmarks again, and writes `verify.html`: one row for each acceptance criterion, with the measured value and pass or fail. Done: every criterion has a row.
5. **Learn.** Write or extend the explainer in `docs/learn/` for the DINOv3 part that the step touched: a diagram, the annotated code, the surprises. Done: the explainer covers every function that the step added to `src/`.
6. **Close.** Set the step status in `roadmap.md`. Record a hard-to-reverse decision as an ADR in `docs/adr/`.

## HTML files

White background, black text, no dark mode. Numbered sections, figures and tables. Inline CSS and inline SVG, so each file stands alone. Write in ASD STE100 style: short sentences, active voice, one word for one meaning.

## Run rules

This Mac crashed from out-of-memory before.

- Run one heavy job at a time: a model load, a benchmark or a training run. Start the next one after the first one ends.
- Start each benchmark configuration in its own subprocess. Check the free memory first. Skip a configuration that does not fit, and record the skip in the results.
- PyTorch on MPS: set `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8` and `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.6`. MLX: set a limit with `mx.set_memory_limit`.
- Run ViT-7B/16 in fp16, bf16 or with quantized weights. In fp32 its weights alone need about 27 GB.
- Start every GPU job, also a light one, through the GPU lock: `uv run python -m dinov3_mlx.gpu_lock -- <command>`.

## Public repo

- Weights live in the Hugging Face cache, outside the repo.
- Commit code, docs, small test images and small result tables. Private context (client data, local paths, the production fine-tuning code) lives in `CLAUDE.local.md`, which git ignores.
- Commit messages describe the change only.
- The repo is by Ueli Peter. No company name and no company logo in any file, also not in the HTML files.

## Code

Type hints on every parameter and return value. Google-style docstrings on public functions. pathlib, f-strings, specific exceptions. Run `uv run ruff format` and `uv run pytest` before each commit.
