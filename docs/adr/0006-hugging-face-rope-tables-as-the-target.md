# The Hugging Face RoPE tables are the target of the mlx RoPE

The RoPE tables of the `mlx` backbone follow the Hugging Face code (`DINOv3ViTRopePositionEmbedding` in transformers 5.22.1): 16 periods from base 100 in fp32, patch-center coordinates in (−1, 1), y for the first half of the pairs and x for the second half, a half-split rotation, and identity rows (cos 1, sin 0) for the 5 prefix tokens. `test_rope_tables` compares them with the Hugging Face tables (largest difference ≤ 1e-6). A report-only test gives the difference to the Meta tables.

Why:

- The reference output of Step 2 comes from the Hugging Face backbone. Parity with the reference output needs the same tables.
- The released Meta checkpoint of ViT-S/16 stores `rope_embed.periods` in bf16. Its angles differ from the Hugging Face angles by up to 0.0079 rad at 224 px (0.0083 rad at 512 px). The Hugging Face code computes the periods in fp32 and does not read them from the weights file.
- With these tables, ViT-S/16 in fp32 on `mlx` has a parity of 1 − 3.1e-11 or better against the reference output.

We considered one other option:

- The Meta tables (bf16 periods), as the official code uses them. They would move the `mlx` output away from the reference output by the same amount as the difference between the Meta and the Hugging Face model, and the parity tests would measure that difference, not the port.

Consequences:

- `docs/plan.html` Table 9 names the Hugging Face tables as the target of `test_rope_tables` (changed in Step 4).
- Table 1 of `docs/learn/01-prior-ports.html` said that the Meta hub model has the same angles. Step 4 corrected it.
- A user who wants the exact Meta behaviour must load the bf16 periods. The `mlx` backbone does not support it.

Status: accepted on 2026-11-07 (Step 4, decision D3 of `specs/04-mlx-port/explore.html`; the user asked the agent to decide).
