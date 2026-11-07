# The fp32 dtype mode of mlx runs with TF32 off

MLX turns TF32 on by default (`MLX_ENABLE_TF32=1`). With TF32, an fp32 matmul rounds its inputs to a 10-bit mantissa. The fp32 dtype mode of `mlx` runs with `MLX_ENABLE_TF32=0`:

- `tests/conftest.py` sets `MLX_ENABLE_TF32=0` before the first `mlx` import.
- The `mlx` benchmark worker sets it in its environment before `mlx` starts.
- `load_mlx_backbone` refuses a load in the fp32 dtype mode when its TF32 state differs from `MLX_ENABLE_TF32` (`TF32ModeError`). The fp32 dtype mode asks for TF32 off.

Why:

- The fp32 parity thresholds are 0.9999 for the frame embedding, the patch mean and the patch 1st percentile (ADR 0003). The TF32 probe of Step 4 gave a largest difference of 7.8e-7 with TF32 off and 7.2e-4 with TF32 on for one fp32 matmul against NumPy float64, about 900 times more. Over 12 blocks this error can cost the 0.9999 margin.
- With TF32 off, ViT-S/16 in fp32 passes all 12 stored configurations with a parity of 1 − 3.1e-11 or better (`results/parity-mlx.md`).
- An fp32 result that is not fp32 arithmetic would compare `mlx` and torch-mps at different precision. Speed with TF32 belongs to a separate dtype mode.

We considered two other options:

- Keep TF32 on and loosen the fp32 thresholds. This mixes two precisions under one name, and it hides real errors of the port.
- Run fp32 on the CPU stream of MLX. It is exact, but it does not measure the GPU, and it is slow.

Consequences:

- An `mlx` fp32 row is slower than it could be with TF32. Step 6 can add a separate dtype mode with TF32 on, with its own thresholds and its own rows. The fp16 and bf16 modes are not affected.
- A script that imports `mlx` before it sets the variable gets TF32. The loader check stops it.

Status: accepted on 2026-11-07 (Step 4, decision D7 of `specs/04-mlx-port/explore.html`; the user asked the agent to decide).
