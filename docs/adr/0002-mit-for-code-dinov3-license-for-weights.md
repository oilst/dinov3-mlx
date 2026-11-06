# MIT for our code, the DINOv3 License for the weights

Our original code is under the MIT License (file `LICENSE`). The DINOv3 weights stay under the DINOv3 License (copy in `LICENSE-DINOv3.md`). This repo never distributes the weights. Users download them from the gated Hugging Face repos.

Clause 1.b.i of the DINOv3 License puts derivative works of the DINO Materials under that license. So our code must not be a derivative work of the official DINOv3 code:

- We write the MLX model from the Hugging Face code (`modeling_dinov3_vit.py`, Apache-2.0) and from the paper. We do not copy code from `facebookresearch/dinov3`.
- A file that adapts Hugging Face code keeps an attribution line with the source file and the Apache-2.0 license.
- We read the official code only to check facts, for example in the explainers.

We considered the whole repo under the DINOv3 License (Option A of R1 in `docs/plan.html`). We rejected it because nobody could then reuse the code under a standard open-source license. MLXDINOv3 and mlx-image use the same split as this ADR (see `specs/01-scope/explore.html`).

This is not a legal opinion. A legal check must confirm this ADR before the first push (Step 7). If the check fails, we change to Option A.

Status: accepted on 2026-11-06 (the user's decision), subject to the legal check.
