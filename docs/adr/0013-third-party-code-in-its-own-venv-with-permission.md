# Third-party code runs outside the repo, in its own venv, with -I, and only with the user's permission

Step 6b measures code and model files that other people published (`specs/06b-prior-art/`). For each candidate:

1. The clone or the download goes into its own new folder below `~/.cache/dinov3-mlx/prior-art/`, outside the repo. The repo gets numbers only: no code, no model file and no weight of a candidate.
2. Each kind of third-party code gets its own venv there: `venv-coreml` (coremltools 9.0 for the Core ML exports) and `venv-mlx` (mlx 0.32.4 for the Python MLX ports). No candidate package goes into `pyproject.toml` or `uv.lock`.
3. The driver `scripts/bench_prior_art.py` starts a third-party worker as `<venv>/bin/python -I <our worker> '<spec>'`, inside the GPU lock. The worker is our own file in `scripts/`; the driver refuses a worker file inside the prior-art folder. `-I` keeps the PYTHON variables, the current folder and the script folder out of `sys.path`, so a planted module of a download cannot load. A worker that needs a clone inserts the path of the clone itself.
4. The workers import no module of the repo. They write one safetensors file of outputs and one JSON line; the driver computes the parity with `dinov3_mlx.parity`, so all sides share one parity code path.
5. No pickle file of a third party loads: not with `torch.load`, not in a worker. A port whose converter reads a pickle gets our own weights in its key names (`scripts/prior_art_meta_names.py`).
6. We read each Python file before it runs.
7. The run of third-party Python code (a port, not a model file) needs the user's yes. Until then the driver writes the port rows with the status "blocked", the reason and the command (`ports` without `--allow-third-party-python`).

Why:

- **The permission system refused the run.** On 2026-11-07 at 22:29 the permission system of the agent session refused a script that ran the Python ports C1 and C2, with the reason "Code from External". A run from another session or in another form would be a workaround. The user decides (question Q1 of `specs/06b-prior-art/plan.html`).
- **Model files are data, Python code is code.** Apple's coremltools loads the Core ML exports C7 to C9 with our own worker; no code of the candidate runs. These runs had no refusal.
- **Licenses.** C9 has no license file and C2 is under the DINOv3 License. Numbers in our tables need no license; copies of their files would.
- **The repo environment stays clean.** A third-party package in `uv.lock` would change the pinned tree of every test and benchmark of this repo.

We considered two other options:

- Install our package into the third-party venv, so the worker imports `dinov3_mlx` (D9 option A). It gives two copies of our code and puts our pinned tree into a third-party venv.
- Put the candidate packages into a dependency group of `pyproject.toml` (D9 option C). Simple, but against rule 2.

Consequences:

- The Core ML worker repeats about 20 lines of `bench.py` (the time budget). Test T12 scans both worker files for imports and forbidden calls.
- The 8 rows of the Python ports stay "blocked" until the user answers Q1. Then one command runs job J5 (about 35 min).
- Step 6c adds coremltools to the repo as the dependency group `coreml` for our own conversion (commit d9bb890). That is Apple's tool, not a candidate: rule 2 covers the candidates. The 6b worker keeps its own venv.

Status: accepted on 2026-11-08 by the Step 6b build, from D2 and D9 of `specs/06b-prior-art/explore.html` (the user asked the agent to decide). Q1 stays with the user.
