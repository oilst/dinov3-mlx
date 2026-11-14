# C7: joined projections against separate projections

Step 6e, B1 and B2 of `specs/06e-kernels/plan.html`. Source: `results/speed-joined.jsonl`.
Each run has 5 paired rounds; the forms R, R-J, R-Jqkv and R-Jff rotate in one process.
Ratio: throughput of the form ÷ throughput of R in the same round (frame embedding at P1,
patch embedding at P2). Pooled: the median of the 10 ratios of both runs (ADR 0004).
D6 rule (P2, batch size 8 only): on if pooled ≥ 1.03 and each run median > 1.
Peak: MLX peak of the timed batches of the form − that of R (largest of the runs).

| Priority | Model size | px | Batch | Dtype mode | Form | Run medians | Pooled | Min | Max | First batch | Valid | Peak − R (MB) | D6 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 | ViT-S/16 | 224 | 8 | fp32 | R-J | 0.938 / 0.936 | 0.937 | 0.935 | 0.941 | bit equal | yes | 110.4 | – |
| P1 | ViT-S/16 | 224 | 8 | fp32 | R-Jqkv | 0.938 / 0.937 | 0.938 | 0.923 | 0.942 | bit equal | yes | 110.4 | – |
| P1 | ViT-S/16 | 224 | 8 | fp32 | R-Jff | 0.977 / 0.981 | 0.981 | 0.971 | 0.984 | bit equal | yes | 0.0 | – |
| P1 | ViT-S/16 | 224 | 8 | fp16-cast | R-J | 0.881 / 0.882 | 0.882 | 0.878 | 0.974 | bit equal | yes | 55.3 | – |
| P1 | ViT-S/16 | 224 | 8 | fp16-cast | R-Jqkv | 0.883 / 0.882 | 0.882 | 0.878 | 0.972 | bit equal | yes | 55.3 | – |
| P1 | ViT-S/16 | 224 | 8 | fp16-cast | R-Jff | 0.960 / 0.965 | 0.964 | 0.949 | 1.069 | bit equal | yes | 0.0 | – |
| P1 | ViT-S/16 | 224 | 8 | bf16-cast | R-J | 0.884 / 0.881 | 0.883 | 0.863 | 1.110 | bit equal | yes | 55.3 | – |
| P1 | ViT-S/16 | 224 | 8 | bf16-cast | R-Jqkv | 0.874 / 0.883 | 0.882 | 0.872 | 1.110 | bit equal | yes | 55.3 | – |
| P1 | ViT-S/16 | 224 | 8 | bf16-cast | R-Jff | 0.961 / 0.967 | 0.964 | 0.957 | 1.221 | bit equal | yes | 0.0 | – |
| P2 | ViT-H+/16 | 512 | 8 | fp32 | R-J | 0.968 / 0.947 | 0.954 | 0.928 | 0.987 | bit equal | yes | 546.1 | off |
| P2 | ViT-H+/16 | 512 | 8 | fp32 | R-Jqkv | 0.966 / 0.933 | 0.961 | 0.871 | 1.058 | bit equal | yes | 230.4 | off |
| P2 | ViT-H+/16 | 512 | 8 | fp32 | R-Jff | 0.994 / 0.985 | 0.990 | 0.893 | 1.042 | bit equal | yes | 534.4 | off |
| P2 | ViT-H+/16 | 512 | 8 | fp16-cast | R-J | 0.948 / 0.953 | 0.950 | 0.874 | 0.964 | bit equal | yes | 146.6 | off |
| P2 | ViT-H+/16 | 512 | 8 | fp16-cast | R-Jqkv | 0.964 / 0.947 | 0.959 | 0.852 | 0.974 | bit equal | yes | 9.8 | off |
| P2 | ViT-H+/16 | 512 | 8 | fp16-cast | R-Jff | 0.997 / 0.992 | 0.995 | 0.888 | 1.003 | bit equal | yes | 157.8 | off |
| P2 | ViT-H+/16 | 512 | 8 | bf16-cast | R-J | 0.942 / 0.955 | 0.943 | 0.924 | 0.961 | bit equal | yes | 146.6 | off |
| P2 | ViT-H+/16 | 512 | 8 | bf16-cast | R-Jqkv | 0.955 / 0.967 | 0.957 | 0.935 | 0.975 | bit equal | yes | 73.0 | off |
| P2 | ViT-H+/16 | 512 | 8 | bf16-cast | R-Jff | 0.997 / 0.997 | 0.997 | 0.981 | 1.038 | bit equal | yes | 157.8 | off |
| P2 | ViT-H+/16 | 512 | 8 | fp16-autocast | R-J | 0.953 / 0.954 | 0.954 | 0.941 | 0.973 | bit equal | yes | 125.5 | off |
| P2 | ViT-H+/16 | 512 | 8 | fp16-autocast | R-Jqkv | 0.952 / 0.952 | 0.952 | 0.944 | 0.990 | bit equal | yes | -137.7 | off |
| P2 | ViT-H+/16 | 512 | 8 | fp16-autocast | R-Jff | 0.992 / 0.999 | 0.992 | 0.979 | 1.015 | bit equal | yes | 305.4 | off |
| P2 | ViT-H+/16 | 512 | 8 | bf16-autocast | R-J | 0.942 / 0.936 | 0.940 | 0.916 | 0.960 | bit equal | yes | 125.5 | off |
| P2 | ViT-H+/16 | 512 | 8 | bf16-autocast | R-Jqkv | 0.957 / 0.955 | 0.956 | 0.943 | 0.981 | bit equal | yes | -137.7 | off |
| P2 | ViT-H+/16 | 512 | 8 | bf16-autocast | R-Jff | 0.998 / 0.986 | 0.989 | 0.980 | 1.004 | bit equal | yes | 305.4 | off |
| P2 | ViT-H+/16 | 512 | 1 | fp16-cast | R-J | 0.922 / 0.914 | 0.921 | 0.908 | 0.942 | bit equal | yes | 183.1 | – |
| P2 | ViT-H+/16 | 512 | 1 | fp16-cast | R-Jqkv | 0.956 / 0.950 | 0.953 | 0.926 | 0.979 | bit equal | yes | 189.0 | – |
| P2 | ViT-H+/16 | 512 | 1 | fp16-cast | R-Jff | 0.963 / 0.956 | 0.963 | 0.947 | 0.988 | bit equal | yes | 132.2 | – |
| P2 | ViT-H+/16 | 512 | 1 | bf16-cast | R-J | 0.889 / 0.918 | 0.917 | 0.870 | 0.939 | bit equal | yes | 183.1 | – |
| P2 | ViT-H+/16 | 512 | 1 | bf16-cast | R-Jqkv | 0.948 / 0.946 | 0.947 | 0.904 | 0.989 | bit equal | yes | 189.0 | – |
| P2 | ViT-H+/16 | 512 | 1 | bf16-cast | R-Jff | 0.953 / 0.965 | 0.955 | 0.928 | 0.983 | bit equal | yes | 132.2 | – |
