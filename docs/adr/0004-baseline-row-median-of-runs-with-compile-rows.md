# Baseline row: the fastest valid dtype mode, each with the median of its runs, compile rows included

A speedup divides the throughput of `mlx` by the throughput of the baseline. The baseline row of one model size, input resolution, batch size, output and dtype comes from `baseline_rows` in `src/dinov3_apple/bench.py`:

1. Take the result lines of the dtype modes of that dtype: fp32 → fp32; fp16 → fp16-cast and fp16-autocast; bf16 → bf16-cast and bf16-autocast. Compile rows count as their own configuration.
2. Drop the lines with status "skipped", "deferred" or "error", and the lines with `"valid": false`.
3. For each remaining configuration, take the run with the median throughput of the output: the main run, the repeats and the rounds. For an even number of runs, take the lower of the two middle runs.
4. Take the configuration with the highest median throughput. The row names its dtype mode, "+ compile" for a compile row, and its number of runs.
5. If no line is left, the row is empty, with the reason.

Only the priority configurations P1 and P2 have compile rows. So at P1 and P2 the baseline can be compiled torch, and in all other configurations it is eager torch.

Why:

- **Fastest valid dtype mode (D2 of `specs/03-baseline/explore.html`).** A user of torch-mps picks the fastest mode that gives a valid output. A speedup against a slower mode would overstate the gain of `mlx`.
- **Compile rows join the baseline.** On 2026-11-06 the compile rows of P1 and P2 were more than 10 % faster than the median of their eager runs in 10 of 10 cases (P1 +41 to +153 %, P2 +11 to +68 %, on the 30 W adapter of the build). The user decided that they join the baseline, so that a speedup does not count a gain that `torch.compile` gives too. On the 94 W re-measurement, 9 of 10 are above 10 % (P1 +25 to +127 %, P2 +10 to +30 %; P2 fp32 +9.5 %). The rule does not depend on the size of the gain: a compile row joins when it is faster.
- **Median of the runs.** On this Mac the throughput of one configuration changed by up to 2× from one run to the next on the 30 W adapter, and by up to 1.3× on the 94 W adapter (findings F1 and F11 of `specs/03-baseline/verify.html`). A single run, or the fastest of several runs, picks up this noise. The median of three runs is robust to one outlier. `scripts/benchmark.py --rounds N` adds runs, round after round.

We considered two other options:

- The main run only (the first rule of the build). It is simple, but one run can be off by up to 2×.
- The fastest run of each configuration. It favours torch-mps, but it makes the baseline depend on a lucky run, and the same rule on `mlx` would inflate both sides.

Consequences:

- A speedup at P1 and P2 compares with compiled torch. Step 6 must say so in its tables.
- A baseline row with 1 run is one sample. Step 6 measures each speedup again with torch-mps and `mlx` runs in alternation in one session, and reports the median of the paired ratios (finding F1).
- Plan Section 3.7 of `specs/03-baseline/plan.html` said "drop the compile rows". This ADR replaces that rule.

Status: accepted on 2026-11-06. The compile rule is the user's decision; the median rule follows finding F1, which the user asked to fix.
