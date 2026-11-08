# Prior-art ratio: our compiled forward pass against each export at the export's own configuration, two runs of 5 paired rounds, the pooled lower median

Amended 2026-11-08. The first text of this ADR (commit `e82a7a9`) chose one run of 5 paired rounds with the verdict of `single_verdict`, and listed two runs of 5 rounds as a rejected option. Verify finding V1 showed that one run is not enough (see Why). The step then ran J1 a second time with the same plan and reports the pooled verdict of the two runs (Section 2 of `results/prior-art-table.md`). This amendment corrects items 4 and 5, the Why list and the options so that they record what the step did. The user has not reviewed the amendment.

Step 6b compares our backbone with the public ports and Core ML exports (`specs/06b-prior-art/`). The number of a comparison is the prior-art ratio of one cell:

1. A cell is one export at its own model size, input resolution and precision, at batch size 1 or 8. The side of ours is our compiled forward pass (`compile_backbone`) at the same model size and input resolution, in the matching dtype mode: fp32 (TF32 off, ADR 0005) against FP32 and INT8 exports, fp16-cast against FP16 exports. Batch size 8 of an export is one `predict` call with a list of 8 inputs; the exports have a fixed batch size of 1.
2. The output is the patch embeddings, host to host, in every cell: every export gives them.
3. One driver holds the GPU lock. Each worker runs in its own subprocess, with the time budget of `bench.py` (`WARM_UP`: 3 batches and 2 s; `TIMED`: 30 batches and 10 s). Exports run with the compute units CPU_AND_GPU, the device of mlx.
4. A cell runs in two runs of 5 rounds, at different times: run 2 starts after run 1 ends, with the same plan. In each round the groups run in a rotated order; ours runs first in odd rounds and last in even rounds. Two exports of the same configuration share the worker of ours in a round.
5. The ratio of one round is our throughput ÷ the throughput of the export (above 1: ours is faster). The cell reports the lower median of each run (ADR 0004), the pooled lower median of the paired ratios of both runs (n = 10), the spread between runs ((largest − smallest run median) ÷ smallest) and the verdict of `pass_verdict`, as the P1 and P2 rows of Step 6: above 1 only if the pooled median and each run median are above 1, with the mark "inside the noise" when the pooled median is within 1 ± 0.03. A round with a failed side drops only its pair. The table also keeps the run-1 row of each cell with its 5 ratios, the spread inside the run and the verdict of `single_verdict`.
6. The parity of an image-input export uses the spec-array input: the image input of the loaded spec becomes a FLOAT32 multiArray, and the stored pixels go in as 0..255 floats without rounding. The package on disk does not change. One report-only row for each image-input export gives the parity with the 8-bit image input, as shipped.

Why:

- **Runs below 1 s are noise on this Mac.** In the Explore, 30 batches (less than 1 s) gave ratios from 0.65 to 1.12 for the same pair; the single probes were 1.4 to 1.8 times off. With the time budget of our harness, the two rounds of each pair agreed within 4 % (Table 4 of `specs/06b-prior-art/explore.html`).
- **The export's own configuration.** No export exists for ViT-H+/16, and the exports have a fixed input resolution. A cell at the export's configuration compares the same work; a cell at P1 or P2 only would leave out most exports.
- **The spec-array input measures the backbone.** Every other backend of this repo gets the stored pixels. With the 8-bit image input the patch 1st percentile drops to 0.9990 (ViT-S/16) and 0.9989 (ViT-B/16), below the fp32 threshold: that is the cost of the interface, not of the port. The spec-array input gave the same speed in the probes (614 and 595 img/s; 77 and 77 img/s).
- **CPU_AND_GPU.** The probes gave the same speed for ALL and CPU_AND_GPU. ALL lets Core ML pick the device; one report-only round with ALL shows if the default moves.
- **One run is not enough (Verify finding V1).** The build first ran one run of 5 rounds. One Verify round then put 4 of 14 ratios outside the build spread + 0.05 and moved the E3 cells from 0.944 and 0.929 to 0.991 and 0.986. The spread inside one run (0.006 to 0.048, without E7 b1) was smaller than the change to a run 40 to 65 min later, so a verdict near 1 rested on one run.
- **Two runs agree.** The two runs of 5 rounds (2026-11-08, 00:10 to 00:35 and 01:43 to 02:09 local time, UTC+1; the result lines give 23:10 to 23:35 and 00:43 to 01:09 UTC) gave the same cell median within 0.016 in each of the 14 cells, and the pooled verdict equals the run-1 verdict in 14 of 14. An unpaired ALL round ran 0.88 to 1.31 times the CPU_AND_GPU speed with bit-equal parity. Only paired rounds with the full time budget, in two runs, give a stable ratio (`results/prior-art-table.md`, `specs/06b-prior-art/verify.html`).

We considered two other options:

- 3 rounds. Shorter, but the median of 3 rests on 2 values when one round fails.
- One run of 5 rounds (the first choice of this ADR). Half the time, but Verify finding V1 showed that the spread inside one run hides the change between runs.

Consequences:

- A prior-art ratio is not a speedup of the mission: the other side is not torch-mps. The tables of Step 6b say "prior-art ratio".
- The 592 px cell of ours (1374 tokens) pairs with the DINOv2 export at 518 px (1374 tokens) on random pixels: throughput only, no parity.
- The Python ports (C1, C2) follow the same protocol in job J5, after the user's yes (ADR 0013). The plan of J5 gives one run (`ports --rounds 5`); with this amendment J5 needs a second run at another time.

Status: accepted on 2026-11-08 by the Step 6b build, from D3 to D6 of `specs/06b-prior-art/explore.html` (the user asked the agent to decide). Amended on 2026-11-08 after Verify finding V1: two runs and the pooled verdict.
