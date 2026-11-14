# A core of five packages with version ranges

Before Step 7, `pyproject.toml` required torch and transformers and pinned every package with `==`. The API path needs neither torch nor transformers.

Decision (D3 B of `specs/07-release/explore.html`, the user, 2026-11-09):

1. **The core** is the five packages of the API path, each with a range: mlx `>=0.32.4,<0.33`, numpy `>=2.5.4,<3`, safetensors `>=0.8.0,<0.9`, huggingface-hub `>=2.4.0,<3`, pillow `>=13.0.0,<14`.
2. **The extra `reference`** adds torch and transformers, for the reference outputs and the torch-mps baseline. The dev group pins them with `==`.
3. **`uv.lock`** keeps the exact versions of every package for this repo. `tech-stack.md` names the tested versions.

Why:

- Exact pins in a library clash with the pins of the project of the user, and torch is a large install that the API does not use.
- Test T4 imports the API with torch and transformers blocked in `sys.modules`; test T5 builds the wheel and installs it in a new venv with the core only.
- mlx gets a range below the next minor version (`<0.33`), because a new mlx version can change the NAX kernels and their relaxed precision. The fingerprint test of each release check shows such a change on this Mac (`tech-stack.md`, Section "Neural Accelerators").

We considered:

- **Keep torch and transformers required, with `==` pins** (D3 A). It installs torch for every user and clashes with other pins.

Consequences:

- A user can get a version of a core package that we did not measure. The README and `tech-stack.md` name the tested versions.
- Each new mlx minor version needs a release check (the default session with the weights and the fingerprint test) before the range moves.

Status: accepted on 2026-11-09 (D3 B); written at the Step 7 Close on 2026-11-14.
