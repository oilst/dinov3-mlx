# Speedup of mlx against torch-mps

Made by `scripts/speedup_table.py` on 2026-11-07 from the Step 6 result files. Speedup = mlx throughput / torch-mps throughput of a paired run (ADR 0004, ADR 0009). Each side takes the fastest valid row of its dtype and output. A ratio within 1 ± 0.03 gets the mark "inside the noise". For an even number of values the median is the lower of the two middle values.

## 1 P1 and P2: paired runs

| Priority | Dtype | Output | mlx mode | torch-mps mode | Run 1 median | Run 2 median | Pooled median | Spread between runs | Verdict (D21) | Ratios |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | fp32 | frame | fp32 | fp32 (compile) | 1.067 | 1.068 | 1.067 | 0.001 | above 1 | 10 |
| P1 | fp32 | patch | fp32 | fp32 (compile) | 1.068 | 1.068 | 1.068 | 0.000 | above 1 | 10 |
| P1 | fp16 | frame | fp16-cast | fp16-cast (compile) | 0.867 | 0.872 | 0.867 | 0.006 | not above 1 | 10 |
| P1 | fp16 | patch | fp16-cast | fp16-cast (compile) | 0.900 | 0.906 | 0.900 | 0.006 | not above 1 | 10 |
| P1 | bf16 | frame | bf16-cast | bf16-cast (compile) | 0.892 | 0.868 | 0.878 | 0.027 | not above 1 | 10 |
| P1 | bf16 | patch | bf16-cast | bf16-cast (compile) | 0.897 | 0.873 | 0.881 | 0.027 | not above 1 | 10 |
| P2 | fp32 | frame | fp32 | fp32 (compile) | 1.089 | 1.097 | 1.092 | 0.007 | above 1 | 10 |
| P2 | fp32 | patch | fp32 | fp32 (compile) | 1.089 | 1.097 | 1.092 | 0.008 | above 1 | 10 |
| P2 | fp16 | frame | fp16-cast | fp16-cast (compile) | 1.021 | 1.063 | 1.032 | 0.041 | above 1 | 10 |
| P2 | fp16 | patch | fp16-cast | fp16-cast (compile) | 1.022 | 1.063 | 1.033 | 0.041 | above 1 | 10 |
| P2 | bf16 | frame | bf16-autocast | bf16-cast (compile) | 1.013 | 1.018 | 1.013 | 0.004 | above 1 | 10 |
| P2 | bf16 | patch | bf16-autocast | bf16-cast (compile) | 1.013 | 1.017 | 1.013 | 0.004 | above 1 | 10 |

Table 1. The paired speedup runs at P1 and P2 (D7, D21).

## 2 Matrix

| Model size | Input resolution | Batch size | Output | Dtype | mlx mode | mlx (img/s) | torch-mps mode | torch-mps (img/s) | Paired ratio | Verdict | Mark | Reason of an empty cell |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-S/16 | 224 | 1 | frame | fp32 | fp32 | 333.2 | fp32 (compile) | 321.6 | 1.036 | above 1 |  |  |
| ViT-S/16 | 224 | 1 | frame | fp16 | fp16-autocast | 549.0 | fp16-autocast (compile) | 390.2 | 1.407 | above 1 |  |  |
| ViT-S/16 | 224 | 1 | frame | bf16 | bf16-autocast | 545.5 | bf16-autocast (compile) | 389.5 | 1.400 | above 1 |  |  |
| ViT-S/16 | 224 | 1 | patch | fp32 | fp32 | 334.1 | fp32 (compile) | 322.8 | 1.035 | above 1 |  |  |
| ViT-S/16 | 224 | 1 | patch | fp16 | fp16-cast | 573.0 | fp16-autocast (compile) | 391.1 | 1.465 | above 1 |  |  |
| ViT-S/16 | 224 | 1 | patch | bf16 | bf16-cast | 528.8 | bf16-autocast (compile) | 390.3 | 1.355 | above 1 |  |  |
| ViT-S/16 | 224 | 8 | frame | fp32 | fp32 | 905.6 | fp32 (compile) | 760.1 | 1.191 | above 1 |  |  |
| ViT-S/16 | 224 | 8 | frame | fp16 | fp16-cast | 1772.1 | fp16-cast (compile) | 2101.8 | 0.843 | not above 1 |  |  |
| ViT-S/16 | 224 | 8 | frame | bf16 | bf16-cast | 1893.0 | bf16-cast (compile) | 2088.5 | 0.906 | not above 1 |  |  |
| ViT-S/16 | 224 | 8 | patch | fp32 | fp32 | 905.2 | fp32 (compile) | 758.7 | 1.193 | above 1 |  |  |
| ViT-S/16 | 224 | 8 | patch | fp16 | fp16-cast | 1842.3 | fp16-cast (compile) | 2096.3 | 0.879 | not above 1 |  |  |
| ViT-S/16 | 224 | 8 | patch | bf16 | bf16-cast | 1899.5 | bf16-cast (compile) | 2081.1 | 0.913 | not above 1 |  |  |
| ViT-S/16 | 224 | 32 | frame | fp32 | fp32 | 1102.7 | fp32 (compile) | 998.6 | 1.104 | above 1 |  |  |
| ViT-S/16 | 224 | 32 | frame | fp16 | fp16-cast | 1899.8 | fp16-cast (compile) | 2373.1 | 0.801 | not above 1 |  |  |
| ViT-S/16 | 224 | 32 | frame | bf16 | bf16-cast | 1794.4 | bf16-cast (compile) | 2094.6 | 0.857 | not above 1 |  |  |
| ViT-S/16 | 224 | 32 | patch | fp32 | fp32 | 1099.3 | fp32 (compile) | 995.2 | 1.105 | above 1 |  |  |
| ViT-S/16 | 224 | 32 | patch | fp16 | fp16-cast | 1918.1 | fp16-cast (compile) | 2364.2 | 0.811 | not above 1 |  |  |
| ViT-S/16 | 224 | 32 | patch | bf16 | bf16-cast | 1790.2 | bf16-cast (compile) | 2087.7 | 0.857 | not above 1 |  |  |
| ViT-S/16 | 448 | 1 | frame | fp32 | fp32 | 130.3 | fp32 (compile) | 135.9 | 0.959 | not above 1 |  |  |
| ViT-S/16 | 448 | 1 | frame | fp16 | fp16-autocast | 267.1 | fp16-cast (compile) | 315.8 | 0.846 | not above 1 |  |  |
| ViT-S/16 | 448 | 1 | frame | bf16 | bf16-autocast | 271.8 | bf16-cast (compile) | 304.4 | 0.893 | not above 1 |  |  |
| ViT-S/16 | 448 | 1 | patch | fp32 | fp32 | 130.5 | fp32 (compile) | 136.0 | 0.960 | not above 1 |  |  |
| ViT-S/16 | 448 | 1 | patch | fp16 | fp16-cast | 270.9 | fp16-cast (compile) | 316.3 | 0.856 | not above 1 |  |  |
| ViT-S/16 | 448 | 1 | patch | bf16 | bf16-cast | 266.1 | bf16-cast (compile) | 304.8 | 0.873 | not above 1 |  |  |
| ViT-S/16 | 448 | 8 | frame | fp32 | fp32 | 164.7 | fp32 (compile) | 162.2 | 1.015 | above 1, inside the noise |  |  |
| ViT-S/16 | 448 | 8 | frame | fp16 | fp16-cast | 410.7 | fp16-cast (compile) | 474.7 | 0.865 | not above 1 |  |  |
| ViT-S/16 | 448 | 8 | frame | bf16 | bf16-cast | 413.9 | bf16-cast (compile) | 491.6 | 0.842 | not above 1 |  |  |
| ViT-S/16 | 448 | 8 | patch | fp32 | fp32 | 164.4 | fp32 (compile) | 161.8 | 1.016 | above 1, inside the noise |  |  |
| ViT-S/16 | 448 | 8 | patch | fp16 | fp16-cast | 414.5 | fp16-cast (compile) | 473.4 | 0.875 | not above 1 |  |  |
| ViT-S/16 | 448 | 8 | patch | bf16 | bf16-cast | 413.2 | bf16-cast (compile) | 490.1 | 0.843 | not above 1 |  |  |
| ViT-S/16 | 448 | 32 | frame | fp32 | fp32 | 199.8 | fp32 (compile) | 209.9 | 0.952 | not above 1 |  |  |
| ViT-S/16 | 448 | 32 | frame | fp16 | fp16-cast | 411.5 | fp16-cast (compile) | 453.6 | 0.907 | not above 1 |  |  |
| ViT-S/16 | 448 | 32 | frame | bf16 | bf16-cast | 411.0 | bf16-cast (compile) | 460.3 | 0.893 | not above 1 |  |  |
| ViT-S/16 | 448 | 32 | patch | fp32 | fp32 | 198.8 | fp32 (compile) | 209.2 | 0.950 | not above 1 |  |  |
| ViT-S/16 | 448 | 32 | patch | fp16 | fp16-cast | 411.3 | fp16-cast (compile) | 451.8 | 0.910 | not above 1 |  |  |
| ViT-S/16 | 448 | 32 | patch | bf16 | bf16-cast | 409.6 | bf16-cast (compile) | 458.0 | 0.894 | not above 1 |  |  |
| ViT-S/16 | 512 | 1 | frame | fp32 | fp32 | 99.7 | fp32 (compile) | 104.0 | 0.958 | not above 1 |  |  |
| ViT-S/16 | 512 | 1 | frame | fp16 | fp16-cast | 228.1 | fp16-cast (compile) | 263.0 | 0.867 | not above 1 |  |  |
| ViT-S/16 | 512 | 1 | frame | bf16 | bf16-autocast | 228.8 | bf16-cast (compile) | 261.3 | 0.876 | not above 1 |  |  |
| ViT-S/16 | 512 | 1 | patch | fp32 | fp32 | 99.7 | fp32 (compile) | 104.1 | 0.958 | not above 1 |  |  |
| ViT-S/16 | 512 | 1 | patch | fp16 | fp16-cast | 238.4 | fp16-cast (compile) | 263.2 | 0.906 | not above 1 |  |  |
| ViT-S/16 | 512 | 1 | patch | bf16 | bf16-cast | 228.5 | bf16-cast (compile) | 261.2 | 0.875 | not above 1 |  |  |
| ViT-S/16 | 512 | 8 | frame | fp32 | fp32 | 146.5 | fp32 (compile) | 143.1 | 1.024 | above 1, inside the noise |  |  |
| ViT-S/16 | 512 | 8 | frame | fp16 | fp16-cast | 263.2 | fp16-cast (compile) | 305.4 | 0.862 | not above 1 |  |  |
| ViT-S/16 | 512 | 8 | frame | bf16 | bf16-autocast | 250.8 | bf16-cast (compile) | 293.3 | 0.855 | not above 1 |  |  |
| ViT-S/16 | 512 | 8 | patch | fp32 | fp32 | 145.7 | fp32 (compile) | 143.0 | 1.019 | above 1, inside the noise |  |  |
| ViT-S/16 | 512 | 8 | patch | fp16 | fp16-cast | 264.1 | fp16-cast (compile) | 304.3 | 0.868 | not above 1 |  |  |
| ViT-S/16 | 512 | 8 | patch | bf16 | bf16-autocast | 248.6 | bf16-cast (compile) | 292.6 | 0.849 | not above 1 |  |  |
| ViT-S/16 | 512 | 32 | frame | fp32 | fp32 | 142.0 | fp32 (compile) | 112.5 | 1.262 | above 1 |  |  |
| ViT-S/16 | 512 | 32 | frame | fp16 | fp16-cast | 265.3 | fp16-cast (compile) | 288.1 | 0.921 | not above 1 |  |  |
| ViT-S/16 | 512 | 32 | frame | bf16 | bf16-cast | 266.2 | bf16-cast (compile) | 291.8 | 0.912 | not above 1 |  |  |
| ViT-S/16 | 512 | 32 | patch | fp32 | fp32 | 141.1 | fp32 (compile) | 112.2 | 1.258 | above 1 |  |  |
| ViT-S/16 | 512 | 32 | patch | fp16 | fp16-cast | 265.1 | fp16-cast (compile) | 286.8 | 0.924 | not above 1 |  |  |
| ViT-S/16 | 512 | 32 | patch | bf16 | bf16-cast | 265.3 | bf16-cast (compile) | 290.5 | 0.914 | not above 1 |  |  |
| ViT-S+/16 | 224 | 1 | frame | fp32 | fp32 | 283.3 | fp32 (compile) | 283.7 | 0.999 | not above 1, inside the noise |  |  |
| ViT-S+/16 | 224 | 1 | frame | fp16 | fp16-autocast | 519.3 | fp16-cast (compile) | 352.1 | 1.475 | above 1 |  |  |
| ViT-S+/16 | 224 | 1 | frame | bf16 | bf16-cast | 482.5 | bf16-cast (compile) | 354.2 | 1.362 | above 1 |  |  |
| ViT-S+/16 | 224 | 1 | patch | fp32 | fp32 | 284.1 | fp32 (compile) | 285.3 | 0.996 | not above 1, inside the noise |  |  |
| ViT-S+/16 | 224 | 1 | patch | fp16 | fp16-cast | 494.4 | fp16-cast (compile) | 354.3 | 1.395 | above 1 |  |  |
| ViT-S+/16 | 224 | 1 | patch | bf16 | bf16-cast | 486.9 | bf16-cast (compile) | 356.6 | 1.365 | above 1 |  |  |
| ViT-S+/16 | 224 | 8 | frame | fp32 | fp32 | 616.4 | fp32 (compile) | 565.9 | 1.089 | above 1 |  |  |
| ViT-S+/16 | 224 | 8 | frame | fp16 | fp16-cast | 1433.2 | fp16-cast (compile) | 1681.0 | 0.853 | not above 1 |  |  |
| ViT-S+/16 | 224 | 8 | frame | bf16 | bf16-cast | 1425.1 | bf16-cast (compile) | 1653.7 | 0.862 | not above 1 |  |  |
| ViT-S+/16 | 224 | 8 | patch | fp32 | fp32 | 616.3 | fp32 (compile) | 565.6 | 1.090 | above 1 |  |  |
| ViT-S+/16 | 224 | 8 | patch | fp16 | fp16-cast | 1481.0 | fp16-cast (compile) | 1678.4 | 0.882 | not above 1 |  |  |
| ViT-S+/16 | 224 | 8 | patch | bf16 | bf16-cast | 1424.6 | bf16-cast (compile) | 1651.2 | 0.863 | not above 1 |  |  |
| ViT-S+/16 | 224 | 32 | frame | fp32 | fp32 | 609.9 | fp32 (compile) | 561.3 | 1.087 | above 1 |  |  |
| ViT-S+/16 | 224 | 32 | frame | fp16 | fp16-cast | 1493.0 | fp16-cast (compile) | 1681.6 | 0.888 | not above 1 |  |  |
| ViT-S+/16 | 224 | 32 | frame | bf16 | bf16-cast | 1449.6 | bf16-cast (compile) | 1608.9 | 0.901 | not above 1 |  |  |
| ViT-S+/16 | 224 | 32 | patch | fp32 | fp32 | 607.6 | fp32 (compile) | 560.0 | 1.085 | above 1 |  |  |
| ViT-S+/16 | 224 | 32 | patch | fp16 | fp16-cast | 1503.5 | fp16-cast (compile) | 1678.7 | 0.896 | not above 1 |  |  |
| ViT-S+/16 | 224 | 32 | patch | bf16 | bf16-cast | 1446.8 | bf16-cast (compile) | 1605.9 | 0.901 | not above 1 |  |  |
| ViT-S+/16 | 448 | 1 | frame | fp32 | fp32 | 114.3 | fp32 (compile) | 115.7 | 0.989 | not above 1, inside the noise |  |  |
| ViT-S+/16 | 448 | 1 | frame | fp16 | fp16-autocast | 251.8 | fp16-cast (compile) | 275.8 | 0.913 | not above 1 |  |  |
| ViT-S+/16 | 448 | 1 | frame | bf16 | bf16-cast | 254.0 | bf16-cast (compile) | 277.8 | 0.914 | not above 1 |  |  |
| ViT-S+/16 | 448 | 1 | patch | fp32 | fp32 | 114.5 | fp32 (compile) | 115.8 | 0.989 | not above 1, inside the noise |  |  |
| ViT-S+/16 | 448 | 1 | patch | fp16 | fp16-cast | 260.3 | fp16-cast (compile) | 276.7 | 0.941 | not above 1 |  |  |
| ViT-S+/16 | 448 | 1 | patch | bf16 | bf16-cast | 254.0 | bf16-cast (compile) | 278.4 | 0.912 | not above 1 |  |  |
| ViT-S+/16 | 448 | 8 | frame | fp32 | fp32 | 136.6 | fp32 (compile) | 135.0 | 1.012 | above 1, inside the noise |  |  |
| ViT-S+/16 | 448 | 8 | frame | fp16 | fp16-cast | 360.5 | fp16-cast (compile) | 403.6 | 0.893 | not above 1 |  |  |
| ViT-S+/16 | 448 | 8 | frame | bf16 | bf16-cast | 359.1 | bf16-cast (compile) | 397.0 | 0.904 | not above 1 |  |  |
| ViT-S+/16 | 448 | 8 | patch | fp32 | fp32 | 136.1 | fp32 (compile) | 134.8 | 1.010 | above 1, inside the noise |  |  |
| ViT-S+/16 | 448 | 8 | patch | fp16 | fp16-cast | 363.2 | fp16-cast (compile) | 402.6 | 0.902 | not above 1 |  |  |
| ViT-S+/16 | 448 | 8 | patch | bf16 | bf16-cast | 358.5 | bf16-cast (compile) | 396.1 | 0.905 | not above 1 |  |  |
| ViT-S+/16 | 448 | 32 | frame | fp32 | fp32 | 136.2 | fp32 (compile) | 129.8 | 1.049 | above 1 |  |  |
| ViT-S+/16 | 448 | 32 | frame | fp16 | fp16-cast | 356.5 | fp16-cast (compile) | 389.4 | 0.916 | not above 1 |  |  |
| ViT-S+/16 | 448 | 32 | frame | bf16 | bf16-cast | 359.3 | bf16-cast (compile) | 396.4 | 0.906 | not above 1 |  |  |
| ViT-S+/16 | 448 | 32 | patch | fp32 | fp32 | 135.6 | fp32 (compile) | 129.3 | 1.049 | above 1 |  |  |
| ViT-S+/16 | 448 | 32 | patch | fp16 | fp16-cast | 356.2 | fp16-cast (compile) | 388.0 | 0.918 | not above 1 |  |  |
| ViT-S+/16 | 448 | 32 | patch | bf16 | bf16-cast | 358.1 | bf16-cast (compile) | 394.9 | 0.907 | not above 1 |  |  |
| ViT-S+/16 | 512 | 1 | frame | fp32 | fp32 | 86.8 | fp32 (compile) | 87.6 | 0.991 | not above 1, inside the noise |  |  |
| ViT-S+/16 | 512 | 1 | frame | fp16 | fp16-cast | 209.5 | fp16-cast (compile) | 230.1 | 0.910 | not above 1 |  |  |
| ViT-S+/16 | 512 | 1 | frame | bf16 | bf16-cast | 211.9 | bf16-cast (compile) | 228.3 | 0.928 | not above 1 |  |  |
| ViT-S+/16 | 512 | 1 | patch | fp32 | fp32 | 86.9 | fp32 (compile) | 87.8 | 0.990 | not above 1, inside the noise |  |  |
| ViT-S+/16 | 512 | 1 | patch | fp16 | fp16-cast | 218.5 | fp16-cast (compile) | 230.3 | 0.949 | not above 1 |  |  |
| ViT-S+/16 | 512 | 1 | patch | bf16 | bf16-cast | 212.0 | bf16-cast (compile) | 228.4 | 0.928 | not above 1 |  |  |
| ViT-S+/16 | 512 | 8 | frame | fp32 | fp32 | 99.1 | fp32 (compile) | 96.3 | 1.030 | above 1, inside the noise |  |  |
| ViT-S+/16 | 512 | 8 | frame | fp16 | fp16-cast | 267.7 | fp16-cast (compile) | 292.0 | 0.917 | not above 1 |  |  |
| ViT-S+/16 | 512 | 8 | frame | bf16 | bf16-cast | 269.0 | bf16-cast (compile) | 293.2 | 0.918 | not above 1 |  |  |
| ViT-S+/16 | 512 | 8 | patch | fp32 | fp32 | 98.9 | fp32 (compile) | 96.1 | 1.029 | above 1, inside the noise |  |  |
| ViT-S+/16 | 512 | 8 | patch | fp16 | fp16-cast | 268.9 | fp16-cast (compile) | 291.6 | 0.922 | not above 1 |  |  |
| ViT-S+/16 | 512 | 8 | patch | bf16 | bf16-cast | 268.3 | bf16-cast (compile) | 292.8 | 0.916 | not above 1 |  |  |
| ViT-S+/16 | 512 | 32 | frame | fp32 | fp32 | 98.6 | fp32 (compile) | 79.3 | 1.244 | above 1 |  |  |
| ViT-S+/16 | 512 | 32 | frame | fp16 | fp16-cast | 262.1 | fp16-cast (compile) | 279.3 | 0.938 | not above 1 |  |  |
| ViT-S+/16 | 512 | 32 | frame | bf16 | bf16-cast | 260.1 | bf16-cast (compile) | 283.7 | 0.917 | not above 1 |  |  |
| ViT-S+/16 | 512 | 32 | patch | fp32 | fp32 | 98.4 | fp32 (compile) | 79.0 | 1.245 | above 1 |  |  |
| ViT-S+/16 | 512 | 32 | patch | fp16 | fp16-cast | 261.9 | fp16-cast (compile) | 278.2 | 0.941 | not above 1 |  |  |
| ViT-S+/16 | 512 | 32 | patch | bf16 | bf16-cast | 259.3 | bf16-cast (compile) | 282.5 | 0.918 | not above 1 |  |  |
| ViT-B/16 | 224 | 1 | frame | fp32 | fp32 | 147.1 | fp32 (compile) | 149.7 | 0.982 | not above 1, inside the noise |  |  |
| ViT-B/16 | 224 | 1 | frame | fp16 | fp16-autocast | 324.4 | fp16-cast (compile) | 345.5 | 0.939 | not above 1 |  |  |
| ViT-B/16 | 224 | 1 | frame | bf16 | bf16-autocast | 327.2 | bf16-cast (compile) | 346.4 | 0.944 | not above 1 |  |  |
| ViT-B/16 | 224 | 1 | patch | fp32 | fp32 | 147.5 | fp32 (compile) | 150.1 | 0.982 | not above 1, inside the noise |  |  |
| ViT-B/16 | 224 | 1 | patch | fp16 | fp16-cast | 336.2 | fp16-cast (compile) | 350.2 | 0.960 | not above 1 |  |  |
| ViT-B/16 | 224 | 1 | patch | bf16 | bf16-cast | 319.1 | bf16-cast (compile) | 352.0 | 0.906 | not above 1 |  |  |
| ViT-B/16 | 224 | 8 | frame | fp32 | fp32 | 234.6 | fp32 (compile) | 236.3 | 0.993 | not above 1, inside the noise |  |  |
| ViT-B/16 | 224 | 8 | frame | fp16 | fp16-cast | 639.4 | fp16-cast (compile) | 682.0 | 0.938 | not above 1 |  |  |
| ViT-B/16 | 224 | 8 | frame | bf16 | bf16-cast | 646.5 | bf16-cast (compile) | 674.5 | 0.959 | not above 1 |  |  |
| ViT-B/16 | 224 | 8 | patch | fp32 | fp32 | 234.4 | fp32 (compile) | 236.2 | 0.992 | not above 1, inside the noise |  |  |
| ViT-B/16 | 224 | 8 | patch | fp16 | fp16-cast | 649.1 | fp16-cast (compile) | 680.8 | 0.953 | not above 1 |  |  |
| ViT-B/16 | 224 | 8 | patch | bf16 | bf16-cast | 645.8 | bf16-cast (compile) | 673.3 | 0.959 | not above 1 |  |  |
| ViT-B/16 | 224 | 32 | frame | fp32 | fp32 | 237.9 | fp32 (compile) | 226.9 | 1.048 | above 1 |  |  |
| ViT-B/16 | 224 | 32 | frame | fp16 | fp16-cast | 686.1 | fp16-cast (compile) | 705.1 | 0.973 | not above 1, inside the noise |  |  |
| ViT-B/16 | 224 | 32 | frame | bf16 | bf16-cast | 694.0 | bf16-cast (compile) | 698.0 | 0.994 | not above 1, inside the noise |  |  |
| ViT-B/16 | 224 | 32 | patch | fp32 | fp32 | 237.4 | fp32 (compile) | 226.3 | 1.049 | above 1 |  |  |
| ViT-B/16 | 224 | 32 | patch | fp16 | fp16-cast | 687.4 | fp16-cast (compile) | 702.0 | 0.979 | not above 1, inside the noise |  |  |
| ViT-B/16 | 224 | 32 | patch | bf16 | bf16-cast | 692.5 | bf16-cast (compile) | 695.0 | 0.996 | not above 1, inside the noise |  |  |
| ViT-B/16 | 448 | 1 | frame | fp32 | fp32 | 46.8 | fp32 (compile) | 46.1 | 1.014 | above 1, inside the noise |  |  |
| ViT-B/16 | 448 | 1 | frame | fp16 | fp16-autocast | 126.1 | fp16-cast (compile) | 135.0 | 0.934 | not above 1 |  |  |
| ViT-B/16 | 448 | 1 | frame | bf16 | bf16-cast | 129.9 | bf16-cast (compile) | 124.4 | 1.045 | above 1 |  |  |
| ViT-B/16 | 448 | 1 | patch | fp32 | fp32 | 46.8 | fp32 (compile) | 46.1 | 1.014 | above 1, inside the noise |  |  |
| ViT-B/16 | 448 | 1 | patch | fp16 | fp16-cast | 127.8 | fp16-cast (compile) | 135.5 | 0.943 | not above 1 |  |  |
| ViT-B/16 | 448 | 1 | patch | bf16 | bf16-cast | 130.1 | bf16-cast (compile) | 125.0 | 1.041 | above 1 |  |  |
| ViT-B/16 | 448 | 8 | frame | fp32 | fp32 | 49.5 | fp32 (compile) | 48.8 | 1.015 | above 1, inside the noise |  |  |
| ViT-B/16 | 448 | 8 | frame | fp16 | fp16-cast | 149.0 | fp16-cast (compile) | 157.3 | 0.947 | not above 1 |  |  |
| ViT-B/16 | 448 | 8 | frame | bf16 | bf16-cast | 150.2 | bf16-cast (compile) | 149.8 | 1.003 | above 1, inside the noise |  |  |
| ViT-B/16 | 448 | 8 | patch | fp32 | fp32 | 49.4 | fp32 (compile) | 48.6 | 1.016 | above 1, inside the noise |  |  |
| ViT-B/16 | 448 | 8 | patch | fp16 | fp16-cast | 149.1 | fp16-cast (compile) | 156.6 | 0.952 | not above 1 |  |  |
| ViT-B/16 | 448 | 8 | patch | bf16 | bf16-cast | 149.9 | bf16-cast (compile) | 149.1 | 1.005 | above 1, inside the noise |  |  |
| ViT-B/16 | 448 | 32 | frame | fp32 | fp32 | 50.1 | fp32 (compile) | 44.4 | 1.128 | above 1 |  |  |
| ViT-B/16 | 448 | 32 | frame | fp16 | fp16-cast | 143.2 | fp16-cast (compile) | 138.4 | 1.034 | above 1 |  |  |
| ViT-B/16 | 448 | 32 | frame | bf16 | bf16-cast | 145.1 | bf16-cast (compile) | 140.4 | 1.034 | above 1 |  |  |
| ViT-B/16 | 448 | 32 | patch | fp32 | fp32 | 49.9 | fp32 (compile) | 44.3 | 1.127 | above 1 |  |  |
| ViT-B/16 | 448 | 32 | patch | fp16 | fp16-cast | 142.8 | fp16-cast (compile) | 137.8 | 1.037 | above 1 |  |  |
| ViT-B/16 | 448 | 32 | patch | bf16 | bf16-cast | 144.7 | bf16-cast (compile) | 139.7 | 1.036 | above 1 |  |  |
| ViT-B/16 | 512 | 1 | frame | fp32 | fp32 | 35.3 | fp32 (compile) | 35.3 | 1.002 | above 1, inside the noise |  |  |
| ViT-B/16 | 512 | 1 | frame | fp16 | fp16-cast | 102.6 | fp16-cast (compile) | 107.2 | 0.957 | not above 1 |  |  |
| ViT-B/16 | 512 | 1 | frame | bf16 | bf16-cast | 102.4 | bf16-cast (compile) | 105.6 | 0.970 | not above 1, inside the noise |  |  |
| ViT-B/16 | 512 | 1 | patch | fp32 | fp32 | 35.3 | fp32 (compile) | 35.3 | 1.001 | above 1, inside the noise |  |  |
| ViT-B/16 | 512 | 1 | patch | fp16 | fp16-cast | 104.7 | fp16-cast (compile) | 107.4 | 0.974 | not above 1, inside the noise |  |  |
| ViT-B/16 | 512 | 1 | patch | bf16 | bf16-cast | 102.6 | bf16-cast (compile) | 105.8 | 0.970 | not above 1 |  |  |
| ViT-B/16 | 512 | 8 | frame | fp32 | fp32 | 39.3 | fp32 (compile) | 37.5 | 1.048 | above 1 |  |  |
| ViT-B/16 | 512 | 8 | frame | fp16 | fp16-cast | 120.3 | fp16-cast (compile) | 119.8 | 1.004 | above 1, inside the noise |  |  |
| ViT-B/16 | 512 | 8 | frame | bf16 | bf16-cast | 113.6 | bf16-cast (compile) | 120.2 | 0.946 | not above 1 |  |  |
| ViT-B/16 | 512 | 8 | patch | fp32 | fp32 | 39.2 | fp32 (compile) | 37.4 | 1.046 | above 1 |  |  |
| ViT-B/16 | 512 | 8 | patch | fp16 | fp16-cast | 120.4 | fp16-cast (compile) | 119.4 | 1.009 | above 1, inside the noise |  |  |
| ViT-B/16 | 512 | 8 | patch | bf16 | bf16-cast | 113.4 | bf16-cast (compile) | 119.8 | 0.947 | not above 1 |  |  |
| ViT-B/16 | 512 | 32 | frame | fp32 | fp32 | 39.2 | fp32 (compile) | 31.9 | 1.226 | above 1 |  |  |
| ViT-B/16 | 512 | 32 | frame | fp16 | fp16-cast | 114.4 | fp16-cast (compile) | 108.6 | 1.054 | above 1 |  |  |
| ViT-B/16 | 512 | 32 | frame | bf16 | bf16-cast | 113.8 | bf16-cast (compile) | 107.9 | 1.054 | above 1 |  |  |
| ViT-B/16 | 512 | 32 | patch | fp32 | fp32 | 39.0 | fp32 (compile) | 31.8 | 1.225 | above 1 |  |  |
| ViT-B/16 | 512 | 32 | patch | fp16 | fp16-cast | 114.1 | fp16-cast (compile) | 108.0 | 1.056 | above 1 |  |  |
| ViT-B/16 | 512 | 32 | patch | bf16 | bf16-cast | 113.4 | bf16-cast (compile) | 107.4 | 1.055 | above 1 |  |  |
| ViT-L/16 | 224 | 1 | frame | fp32 | fp32 | 46.1 | fp32 (compile) | 54.6 | 0.846 | not above 1 |  |  |
| ViT-L/16 | 224 | 1 | frame | fp16 | fp16-autocast | 127.6 | fp16-autocast (compile) | 105.8 | 1.205 | above 1 |  |  |
| ViT-L/16 | 224 | 1 | frame | bf16 | bf16-cast | 129.8 | bf16-cast (compile) | 147.2 | 0.882 | not above 1 |  |  |
| ViT-L/16 | 224 | 1 | patch | fp32 | fp32 | 46.2 | fp32 (compile) | 54.7 | 0.844 | not above 1 |  |  |
| ViT-L/16 | 224 | 1 | patch | fp16 | fp16-autocast | 124.7 | fp16-autocast (compile) | 106.0 | 1.176 | above 1 |  |  |
| ViT-L/16 | 224 | 1 | patch | bf16 | bf16-cast | 130.0 | bf16-cast (compile) | 147.8 | 0.880 | not above 1 |  |  |
| ViT-L/16 | 224 | 8 | frame | fp32 | fp32 | 70.8 | fp32 (compile) | 75.3 | 0.940 | not above 1 |  |  |
| ViT-L/16 | 224 | 8 | frame | fp16 | fp16-autocast | 201.2 | fp16-autocast (compile) | 212.4 | 0.947 | not above 1 |  |  |
| ViT-L/16 | 224 | 8 | frame | bf16 | bf16-cast | 212.5 | bf16-cast (compile) | 237.5 | 0.895 | not above 1 |  |  |
| ViT-L/16 | 224 | 8 | patch | fp32 | fp32 | 70.7 | fp32 (compile) | 75.3 | 0.938 | not above 1 |  |  |
| ViT-L/16 | 224 | 8 | patch | fp16 | fp16-autocast | 200.2 | fp16-autocast (compile) | 212.2 | 0.944 | not above 1 |  |  |
| ViT-L/16 | 224 | 8 | patch | bf16 | bf16-cast | 212.4 | bf16-cast (compile) | 237.6 | 0.894 | not above 1 |  |  |
| ViT-L/16 | 224 | 32 | frame | fp32 | fp32 | 71.9 | fp32 (compile) | 69.9 | 1.029 | above 1, inside the noise |  |  |
| ViT-L/16 | 224 | 32 | frame | fp16 | fp16-autocast | 205.5 | fp16-autocast (compile) | 219.4 | 0.937 | not above 1 |  |  |
| ViT-L/16 | 224 | 32 | frame | bf16 | bf16-cast | 220.4 | bf16-cast (compile) | 234.4 | 0.940 | not above 1 |  |  |
| ViT-L/16 | 224 | 32 | patch | fp32 | fp32 | 71.8 | fp32 (compile) | 69.8 | 1.028 | above 1, inside the noise |  |  |
| ViT-L/16 | 224 | 32 | patch | fp16 | fp16-autocast | 205.0 | fp16-autocast (compile) | 218.6 | 0.938 | not above 1 |  |  |
| ViT-L/16 | 224 | 32 | patch | bf16 | bf16-cast | 220.2 | bf16-cast (compile) | 234.0 | 0.941 | not above 1 |  |  |
| ViT-L/16 | 448 | 1 | frame | fp32 | fp32 | 15.6 | fp32 (compile) | 16.8 | 0.931 | not above 1 |  |  |
| ViT-L/16 | 448 | 1 | frame | fp16 | fp16-autocast | 44.4 | fp16-autocast (compile) | 43.8 | 1.015 | above 1, inside the noise |  |  |
| ViT-L/16 | 448 | 1 | frame | bf16 | bf16-cast | 44.8 | bf16-cast (compile) | 51.8 | 0.865 | not above 1 |  |  |
| ViT-L/16 | 448 | 1 | patch | fp32 | fp32 | 15.6 | fp32 (compile) | 16.8 | 0.930 | not above 1 |  |  |
| ViT-L/16 | 448 | 1 | patch | fp16 | fp16-autocast | 44.1 | fp16-autocast (compile) | 43.8 | 1.006 | above 1, inside the noise |  |  |
| ViT-L/16 | 448 | 1 | patch | bf16 | bf16-cast | 44.9 | bf16-cast (compile) | 52.0 | 0.864 | not above 1 |  |  |
| ViT-L/16 | 448 | 8 | frame | fp32 | fp32 | 16.6 | fp32 (compile) | 16.5 | 1.004 | above 1, inside the noise |  |  |
| ViT-L/16 | 448 | 8 | frame | fp16 | fp16-autocast | 49.3 | fp16-autocast (compile) | 52.1 | 0.945 | not above 1 |  |  |
| ViT-L/16 | 448 | 8 | frame | bf16 | bf16-cast | 52.0 | bf16-cast (compile) | 55.0 | 0.945 | not above 1 |  |  |
| ViT-L/16 | 448 | 8 | patch | fp32 | fp32 | 16.5 | fp32 (compile) | 16.5 | 1.002 | above 1, inside the noise |  |  |
| ViT-L/16 | 448 | 8 | patch | fp16 | fp16-autocast | 49.1 | fp16-autocast (compile) | 51.9 | 0.946 | not above 1 |  |  |
| ViT-L/16 | 448 | 8 | patch | bf16 | bf16-cast | 52.0 | bf16-cast (compile) | 54.9 | 0.946 | not above 1 |  |  |
| ViT-L/16 | 448 | 32 | frame | fp32 | fp32 | 16.0 | fp32 (compile) | 14.4 | 1.109 | above 1 |  |  |
| ViT-L/16 | 448 | 32 | frame | fp16 | fp16-autocast | 47.4 | fp16-autocast (compile) | 46.2 | 1.026 | above 1, inside the noise |  |  |
| ViT-L/16 | 448 | 32 | frame | bf16 | bf16-cast | 50.9 | bf16-cast (compile) | 48.8 | 1.044 | above 1 |  |  |
| ViT-L/16 | 448 | 32 | patch | fp32 | fp32 | 16.0 | fp32 (compile) | 14.4 | 1.108 | above 1 |  |  |
| ViT-L/16 | 448 | 32 | patch | fp16 | fp16-autocast | 47.4 | fp16-autocast (compile) | 46.0 | 1.029 | above 1, inside the noise |  |  |
| ViT-L/16 | 448 | 32 | patch | bf16 | bf16-cast | 50.9 | bf16-cast (compile) | 48.7 | 1.045 | above 1 |  |  |
| ViT-L/16 | 512 | 1 | frame | fp32 | fp32 | 11.6 | fp32 (compile) | 12.3 | 0.943 | not above 1 |  |  |
| ViT-L/16 | 512 | 1 | frame | fp16 | fp16-autocast | 33.8 | fp16-autocast (compile) | 34.0 | 0.994 | not above 1, inside the noise |  |  |
| ViT-L/16 | 512 | 1 | frame | bf16 | bf16-cast | 34.5 | bf16-cast (compile) | 38.9 | 0.887 | not above 1 |  |  |
| ViT-L/16 | 512 | 1 | patch | fp32 | fp32 | 11.6 | fp32 (compile) | 12.3 | 0.941 | not above 1 |  |  |
| ViT-L/16 | 512 | 1 | patch | fp16 | fp16-autocast | 33.5 | fp16-autocast (compile) | 34.0 | 0.986 | not above 1, inside the noise |  |  |
| ViT-L/16 | 512 | 1 | patch | bf16 | bf16-cast | 34.5 | bf16-cast (compile) | 38.9 | 0.886 | not above 1 |  |  |
| ViT-L/16 | 512 | 8 | frame | fp32 | fp32 | 12.2 | fp32 (compile) | 11.7 | 1.041 | above 1 |  |  |
| ViT-L/16 | 512 | 8 | frame | fp16 | fp16-autocast | 35.9 | fp16-autocast (compile) | 38.1 | 0.940 | not above 1 |  |  |
| ViT-L/16 | 512 | 8 | frame | bf16 | bf16-cast | 38.6 | bf16-cast (compile) | 40.4 | 0.954 | not above 1 |  |  |
| ViT-L/16 | 512 | 8 | patch | fp32 | fp32 | 12.2 | fp32 (compile) | 11.7 | 1.041 | above 1 |  |  |
| ViT-L/16 | 512 | 8 | patch | fp16 | fp16-autocast | 35.8 | fp16-autocast (compile) | 38.0 | 0.941 | not above 1 |  |  |
| ViT-L/16 | 512 | 8 | patch | bf16 | bf16-cast | 38.5 | bf16-cast (compile) | 40.4 | 0.954 | not above 1 |  |  |
| ViT-L/16 | 512 | 32 | frame | fp32 | fp32 | 12.0 | fp32 (compile) | 9.9 | 1.205 | above 1 |  |  |
| ViT-L/16 | 512 | 32 | frame | fp16 | fp16-autocast | 34.4 | fp16-autocast (compile) | 32.5 | 1.060 | above 1 |  |  |
| ViT-L/16 | 512 | 32 | frame | bf16 | bf16-cast | 36.7 | bf16-autocast (compile) | 32.0 | 1.144 | above 1 |  |  |
| ViT-L/16 | 512 | 32 | patch | fp32 | fp32 | 11.9 | fp32 (compile) | 9.9 | 1.204 | above 1 |  |  |
| ViT-L/16 | 512 | 32 | patch | fp16 | fp16-autocast | 34.3 | fp16-autocast (compile) | 32.3 | 1.062 | above 1 |  |  |
| ViT-L/16 | 512 | 32 | patch | bf16 | bf16-cast | 36.6 | bf16-autocast (compile) | 31.9 | 1.147 | above 1 |  |  |
| ViT-H+/16 | 224 | 1 | frame | fp32 | fp32 | 18.9 | fp32 (compile) | 19.7 | 0.956 | not above 1 |  |  |
| ViT-H+/16 | 224 | 1 | frame | fp16 | fp16-autocast | 53.2 | fp16-autocast (compile) | 38.0 | 1.402 | above 1 |  |  |
| ViT-H+/16 | 224 | 1 | frame | bf16 | bf16-cast | 53.6 | bf16-cast (compile) | 67.2 | 0.797 | not above 1 |  |  |
| ViT-H+/16 | 224 | 1 | patch | fp32 | fp32 | 18.9 | fp32 (compile) | 19.8 | 0.956 | not above 1 |  |  |
| ViT-H+/16 | 224 | 1 | patch | fp16 | fp16-autocast | 52.7 | fp16-autocast (compile) | 38.0 | 1.387 | above 1 |  |  |
| ViT-H+/16 | 224 | 1 | patch | bf16 | bf16-cast | 53.7 | bf16-cast (compile) | 67.4 | 0.796 | not above 1 |  |  |
| ViT-H+/16 | 224 | 8 | frame | fp32 | fp32 | 25.1 | fp32 (compile) | 24.6 | 1.023 | above 1, inside the noise |  |  |
| ViT-H+/16 | 224 | 8 | frame | fp16 | fp16-autocast | 78.4 | fp16-autocast (compile) | 71.7 | 1.093 | above 1 |  |  |
| ViT-H+/16 | 224 | 8 | frame | bf16 | bf16-cast | 81.3 | bf16-cast (compile) | 95.8 | 0.848 | not above 1 |  |  |
| ViT-H+/16 | 224 | 8 | patch | fp32 | fp32 | 25.1 | fp32 (compile) | 24.5 | 1.023 | above 1, inside the noise |  |  |
| ViT-H+/16 | 224 | 8 | patch | fp16 | fp16-autocast | 78.2 | fp16-autocast (compile) | 71.6 | 1.093 | above 1 |  |  |
| ViT-H+/16 | 224 | 8 | patch | bf16 | bf16-cast | 81.3 | bf16-cast (compile) | 95.8 | 0.848 | not above 1 |  |  |
| ViT-H+/16 | 224 | 32 | frame | fp32 | fp32 | 26.3 | fp32 (compile) | 24.2 | 1.086 | above 1 |  |  |
| ViT-H+/16 | 224 | 32 | frame | fp16 | fp16-autocast | 78.1 | fp16-autocast (compile) | 71.9 | 1.085 | above 1 |  |  |
| ViT-H+/16 | 224 | 32 | frame | bf16 | bf16-autocast | 85.9 | bf16-cast (compile) | 79.5 | 1.080 | above 1 |  |  |
| ViT-H+/16 | 224 | 32 | patch | fp32 | fp32 | 26.3 | fp32 (compile) | 24.2 | 1.086 | above 1 |  |  |
| ViT-H+/16 | 224 | 32 | patch | fp16 | fp16-autocast | 78.0 | fp16-autocast (compile) | 71.8 | 1.086 | above 1 |  |  |
| ViT-H+/16 | 224 | 32 | patch | bf16 | bf16-autocast | 85.8 | bf16-cast (compile) | 79.4 | 1.080 | above 1 |  |  |
| ViT-H+/16 | 448 | 1 | frame | fp32 | fp32 | 6.6 | fp32 (compile) | 6.2 | 1.051 | above 1 |  |  |
| ViT-H+/16 | 448 | 1 | frame | fp16 | fp16-autocast | 21.8 | fp16-cast (compile) | 22.1 | 0.985 | not above 1, inside the noise |  |  |
| ViT-H+/16 | 448 | 1 | frame | bf16 | bf16-autocast | 22.0 | bf16-cast (compile) | 22.5 | 0.975 | not above 1, inside the noise |  |  |
| ViT-H+/16 | 448 | 1 | patch | fp32 | fp32 | 6.5 | fp32 (compile) | 6.2 | 1.050 | above 1 |  |  |
| ViT-H+/16 | 448 | 1 | patch | fp16 | fp16-autocast | 21.7 | fp16-cast (compile) | 22.1 | 0.980 | not above 1, inside the noise |  |  |
| ViT-H+/16 | 448 | 1 | patch | bf16 | bf16-autocast | 21.9 | bf16-cast (compile) | 22.6 | 0.970 | not above 1, inside the noise |  |  |
| ViT-H+/16 | 448 | 8 | frame | fp32 | fp32 | 7.1 | fp32 (compile) | 6.8 | 1.043 | above 1 |  |  |
| ViT-H+/16 | 448 | 8 | frame | fp16 | fp16-cast | 22.6 | fp16-cast (compile) | 22.2 | 1.020 | above 1, inside the noise |  |  |
| ViT-H+/16 | 448 | 8 | frame | bf16 | bf16-cast | 22.6 | bf16-cast (compile) | 22.1 | 1.024 | above 1, inside the noise |  |  |
| ViT-H+/16 | 448 | 8 | patch | fp32 | fp32 | 7.1 | fp32 (compile) | 6.8 | 1.042 | above 1 |  |  |
| ViT-H+/16 | 448 | 8 | patch | fp16 | fp16-cast | 22.6 | fp16-cast (compile) | 22.2 | 1.020 | above 1, inside the noise |  |  |
| ViT-H+/16 | 448 | 8 | patch | bf16 | bf16-cast | 22.6 | bf16-cast (compile) | 22.0 | 1.024 | above 1, inside the noise |  |  |
| ViT-H+/16 | 448 | 32 | frame | fp32 | fp32 | 6.5 | fp32 (compile) | 6.1 | 1.067 | above 1 |  |  |
| ViT-H+/16 | 448 | 32 | frame | fp16 | fp16-autocast | 19.9 | fp16-autocast (compile) | 17.4 | 1.148 | above 1 |  |  |
| ViT-H+/16 | 448 | 32 | frame | bf16 | bf16-autocast | 20.0 | bf16-autocast (compile) | 17.7 | 1.130 | above 1 |  |  |
| ViT-H+/16 | 448 | 32 | patch | fp32 | fp32 | 6.5 | fp32 (compile) | 6.0 | 1.067 | above 1 |  |  |
| ViT-H+/16 | 448 | 32 | patch | fp16 | fp16-autocast | 19.9 | fp16-autocast (compile) | 17.3 | 1.148 | above 1 |  |  |
| ViT-H+/16 | 448 | 32 | patch | bf16 | bf16-autocast | 19.9 | bf16-autocast (compile) | 17.6 | 1.128 | above 1 |  |  |
| ViT-H+/16 | 512 | 1 | frame | fp32 | fp32 | 4.6 | fp32 (compile) | 5.1 | 0.910 | not above 1 |  |  |
| ViT-H+/16 | 512 | 1 | frame | fp16 | fp16-autocast | 14.5 | fp16-cast (compile) | 16.7 | 0.870 | not above 1 |  |  |
| ViT-H+/16 | 512 | 1 | frame | bf16 | bf16-autocast | 15.1 | bf16-cast (compile) | 16.6 | 0.909 | not above 1 |  |  |
| ViT-H+/16 | 512 | 1 | patch | fp32 | fp32 | 4.6 | fp32 (compile) | 5.1 | 0.909 | not above 1 |  |  |
| ViT-H+/16 | 512 | 1 | patch | fp16 | fp16-autocast | 14.5 | fp16-cast (compile) | 16.7 | 0.867 | not above 1 |  |  |
| ViT-H+/16 | 512 | 1 | patch | bf16 | bf16-autocast | 15.0 | bf16-cast (compile) | 16.6 | 0.906 | not above 1 |  |  |
| ViT-H+/16 | 512 | 8 | frame | fp32 | fp32 | 5.2 | fp32 (compile) | 4.4 | 1.168 | above 1 |  |  |
| ViT-H+/16 | 512 | 8 | frame | fp16 | fp16-cast | 16.3 | fp16-cast (compile) | 15.9 | 1.026 | above 1, inside the noise |  |  |
| ViT-H+/16 | 512 | 8 | frame | bf16 | bf16-autocast | 14.6 | bf16-cast (compile) | 14.6 | 1.003 | above 1, inside the noise |  |  |
| ViT-H+/16 | 512 | 8 | patch | fp32 | fp32 | 5.2 | fp32 (compile) | 4.4 | 1.168 | above 1 |  |  |
| ViT-H+/16 | 512 | 8 | patch | fp16 | fp16-cast | 16.3 | fp16-cast (compile) | 15.9 | 1.027 | above 1, inside the noise |  |  |
| ViT-H+/16 | 512 | 8 | patch | bf16 | bf16-autocast | 14.6 | bf16-cast (compile) | 14.6 | 1.003 | above 1, inside the noise |  |  |
| ViT-H+/16 | 512 | 32 | frame | fp32 | fp32 | 4.7 | fp32 (compile) | 3.6 | 1.282 | above 1 |  |  |
| ViT-H+/16 | 512 | 32 | frame | fp16 | fp16-cast | 14.4 | fp16-cast (compile) | 13.0 | 1.109 | above 1 |  |  |
| ViT-H+/16 | 512 | 32 | frame | bf16 | bf16-cast | 13.3 | bf16-cast (compile) | 12.1 | 1.095 | above 1 |  |  |
| ViT-H+/16 | 512 | 32 | patch | fp32 | fp32 | 4.7 | fp32 (compile) | 3.6 | 1.282 | above 1 |  |  |
| ViT-H+/16 | 512 | 32 | patch | fp16 | fp16-cast | 14.4 | fp16-cast (compile) | 12.9 | 1.109 | above 1 |  |  |
| ViT-H+/16 | 512 | 32 | patch | bf16 | bf16-cast | 13.3 | bf16-cast (compile) | 12.1 | 1.093 | above 1 |  |  |
| ViT-7B/16 | 224 | 1 | frame | fp32 | – | – | – | – | – | – |  | ADR 0008: ViT-7B/16 never loads its fp32 weights |
| ViT-7B/16 | 224 | 1 | frame | fp16 | fp16-autocast | 9.0 | fp16-cast (compile) | 9.9 | 0.900 | not above 1 |  |  |
| ViT-7B/16 | 224 | 1 | frame | bf16 | bf16-cast | 9.1 | bf16-cast (compile) | 10.7 | 0.849 | not above 1 |  |  |
| ViT-7B/16 | 224 | 1 | patch | fp32 | – | – | – | – | – | – |  | ADR 0008: ViT-7B/16 never loads its fp32 weights |
| ViT-7B/16 | 224 | 1 | patch | fp16 | fp16-autocast | 8.9 | fp16-cast (compile) | 10.0 | 0.898 | not above 1 |  |  |
| ViT-7B/16 | 224 | 1 | patch | bf16 | bf16-cast | 9.1 | bf16-cast (compile) | 10.7 | 0.849 | not above 1 |  |  |
| ViT-7B/16 | 224 | 8 | frame | fp32 | – | – | – | – | – | – |  | ADR 0008: ViT-7B/16 never loads its fp32 weights |
| ViT-7B/16 | 224 | 8 | frame | fp16 | fp16-autocast | 11.8 | fp16-cast (compile) | 12.5 | 0.941 | not above 1 |  |  |
| ViT-7B/16 | 224 | 8 | frame | bf16 | bf16-cast | 12.1 | bf16-cast (compile) | 12.9 | 0.940 | not above 1 |  |  |
| ViT-7B/16 | 224 | 8 | patch | fp32 | – | – | – | – | – | – |  | ADR 0008: ViT-7B/16 never loads its fp32 weights |
| ViT-7B/16 | 224 | 8 | patch | fp16 | fp16-autocast | 11.8 | fp16-cast (compile) | 12.5 | 0.941 | not above 1 |  |  |
| ViT-7B/16 | 224 | 8 | patch | bf16 | bf16-cast | 12.1 | bf16-cast (compile) | 12.9 | 0.940 | not above 1 |  |  |
| ViT-7B/16 | 224 | 32 | frame | fp32 | – | – | – | – | – | – |  | ADR 0008: ViT-7B/16 never loads its fp32 weights |
| ViT-7B/16 | 224 | 32 | frame | fp16 | fp16-cast | 10.0 | fp16-cast (compile) | 9.1 | 1.103 | above 1 |  |  |
| ViT-7B/16 | 224 | 32 | frame | bf16 | bf16-cast | 10.3 | bf16-cast (compile) | 8.9 | 1.154 | above 1 |  |  |
| ViT-7B/16 | 224 | 32 | patch | fp32 | – | – | – | – | – | – |  | ADR 0008: ViT-7B/16 never loads its fp32 weights |
| ViT-7B/16 | 224 | 32 | patch | fp16 | fp16-cast | 10.0 | fp16-cast (compile) | 9.1 | 1.104 | above 1 |  |  |
| ViT-7B/16 | 224 | 32 | patch | bf16 | bf16-cast | 10.3 | bf16-cast (compile) | 8.9 | 1.154 | above 1 |  |  |
| ViT-7B/16 | 448 | 1 | frame | fp32 | – | – | – | – | – | – |  | ADR 0008: ViT-7B/16 never loads its fp32 weights |
| ViT-7B/16 | 448 | 1 | frame | fp16 | fp16-autocast | 2.5 | fp16-cast (compile) | 2.6 | 0.958 | not above 1 |  |  |
| ViT-7B/16 | 448 | 1 | frame | bf16 | bf16-cast | 2.5 | bf16-cast (compile) | 2.7 | 0.938 | not above 1 |  |  |
| ViT-7B/16 | 448 | 1 | patch | fp32 | – | – | – | – | – | – |  | ADR 0008: ViT-7B/16 never loads its fp32 weights |
| ViT-7B/16 | 448 | 1 | patch | fp16 | fp16-autocast | 2.5 | fp16-cast (compile) | 2.6 | 0.957 | not above 1 |  |  |
| ViT-7B/16 | 448 | 1 | patch | bf16 | bf16-cast | 2.5 | bf16-cast (compile) | 2.7 | 0.938 | not above 1 |  |  |
| ViT-7B/16 | 448 | 8 | frame | fp32 | – | – | – | – | – | – |  | ADR 0008: ViT-7B/16 never loads its fp32 weights |
| ViT-7B/16 | 448 | 8 | frame | fp16 | fp16-cast | 2.8 | fp16-cast (compile) | 2.6 | 1.096 | above 1 |  |  |
| ViT-7B/16 | 448 | 8 | frame | bf16 | bf16-cast | 2.9 | bf16-cast (compile) | 2.5 | 1.137 | above 1 |  |  |
| ViT-7B/16 | 448 | 8 | patch | fp32 | – | – | – | – | – | – |  | ADR 0008: ViT-7B/16 never loads its fp32 weights |
| ViT-7B/16 | 448 | 8 | patch | fp16 | fp16-cast | 2.8 | fp16-cast (compile) | 2.6 | 1.095 | above 1 |  |  |
| ViT-7B/16 | 448 | 8 | patch | bf16 | bf16-cast | 2.9 | bf16-cast (compile) | 2.5 | 1.137 | above 1 |  |  |
| ViT-7B/16 | 448 | 32 | frame | fp32 | – | – | – | – | – | – |  | ADR 0008: ViT-7B/16 never loads its fp32 weights |
| ViT-7B/16 | 448 | 32 | frame | fp16 | fp16-cast | 2.6 | fp16-cast (compile) | 2.3 | 1.144 | above 1 |  |  |
| ViT-7B/16 | 448 | 32 | frame | bf16 | bf16-cast | 2.7 | bf16-cast (compile) | 2.4 | 1.161 | above 1 |  |  |
| ViT-7B/16 | 448 | 32 | patch | fp32 | – | – | – | – | – | – |  | ADR 0008: ViT-7B/16 never loads its fp32 weights |
| ViT-7B/16 | 448 | 32 | patch | fp16 | fp16-cast | 2.6 | fp16-cast (compile) | 2.3 | 1.144 | above 1 |  |  |
| ViT-7B/16 | 448 | 32 | patch | bf16 | bf16-cast | 2.7 | bf16-cast (compile) | 2.4 | 1.161 | above 1 |  |  |
| ViT-7B/16 | 512 | 1 | frame | fp32 | – | – | – | – | – | – |  | ADR 0008: ViT-7B/16 never loads its fp32 weights |
| ViT-7B/16 | 512 | 1 | frame | fp16 | fp16-autocast | 2.1 | fp16-cast (compile) | 2.4 | 0.876 | not above 1 |  |  |
| ViT-7B/16 | 512 | 1 | frame | bf16 | bf16-cast | 2.1 | bf16-cast (compile) | 2.3 | 0.907 | not above 1 |  |  |
| ViT-7B/16 | 512 | 1 | patch | fp32 | – | – | – | – | – | – |  | ADR 0008: ViT-7B/16 never loads its fp32 weights |
| ViT-7B/16 | 512 | 1 | patch | fp16 | fp16-autocast | 2.1 | fp16-cast (compile) | 2.4 | 0.876 | not above 1 |  |  |
| ViT-7B/16 | 512 | 1 | patch | bf16 | bf16-cast | 2.1 | bf16-cast (compile) | 2.3 | 0.907 | not above 1 |  |  |
| ViT-7B/16 | 512 | 8 | frame | fp32 | – | – | – | – | – | – |  | ADR 0008: ViT-7B/16 never loads its fp32 weights |
| ViT-7B/16 | 512 | 8 | frame | fp16 | fp16-cast | 2.1 | fp16-cast (compile) | 1.9 | 1.114 | above 1 |  |  |
| ViT-7B/16 | 512 | 8 | frame | bf16 | bf16-cast | 2.1 | bf16-cast (compile) | 1.9 | 1.102 | above 1 |  |  |
| ViT-7B/16 | 512 | 8 | patch | fp32 | – | – | – | – | – | – |  | ADR 0008: ViT-7B/16 never loads its fp32 weights |
| ViT-7B/16 | 512 | 8 | patch | fp16 | fp16-cast | 2.1 | fp16-cast (compile) | 1.9 | 1.114 | above 1 |  |  |
| ViT-7B/16 | 512 | 8 | patch | bf16 | bf16-cast | 2.1 | bf16-cast (compile) | 1.9 | 1.103 | above 1 |  |  |
| ViT-7B/16 | 512 | 32 | frame | fp32 | – | – | – | – | – | – |  | ADR 0008: ViT-7B/16 never loads its fp32 weights |
| ViT-7B/16 | 512 | 32 | frame | fp16 | fp16-cast | 2.0 | fp16-cast (compile) | 1.7 | 1.166 | above 1 |  |  |
| ViT-7B/16 | 512 | 32 | frame | bf16 | bf16-cast | 2.1 | bf16-cast (compile) | 1.8 | 1.154 | above 1 |  |  |
| ViT-7B/16 | 512 | 32 | patch | fp32 | – | – | – | – | – | – |  | ADR 0008: ViT-7B/16 never loads its fp32 weights |
| ViT-7B/16 | 512 | 32 | patch | fp16 | fp16-cast | 2.0 | fp16-cast (compile) | 1.7 | 1.166 | above 1 |  |  |
| ViT-7B/16 | 512 | 32 | patch | bf16 | bf16-cast | 2.1 | bf16-cast (compile) | 1.8 | 1.153 | above 1 |  |  |

Table 2. The 324 cells of the benchmark matrix, one paired round (D20).

## 3 Quantized rows

| Model size | Input resolution | Batch size | Output | Dtype mode | Bits | Valid | Frame | Patch mean | Patch 1st percentile | mlx (img/s) | torch-mps (img/s) | Paired ratio | Ratio to unquantized | Verdict | Reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-H+/16 | 224 | 1 | frame | fp16-autocast | 8 | yes | 0.999975 | 0.999867 | 0.998939 | 46.0 | 38.0 | 1.212 | 0.865 | above 1 |  |
| ViT-H+/16 | 224 | 1 | frame | fp16-autocast | 4 | no | 0.993273 | 0.978011 | 0.919585 | 49.7 | 38.0 | 1.309 | 0.934 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.993273, patch mean 0.978011, patch 1st percentile 0.919585 |
| ViT-H+/16 | 224 | 1 | frame | bf16-autocast | 8 | yes | 0.999927 | 0.999479 | 0.993799 | 46.1 | 67.2 | 0.686 | 0.862 | not above 1 |  |
| ViT-H+/16 | 224 | 1 | frame | bf16-autocast | 4 | no | 0.993764 | 0.979564 | 0.943436 | 49.6 | 67.2 | 0.737 | 0.926 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.993764, patch mean 0.979564, patch 1st percentile 0.943436 |
| ViT-H+/16 | 224 | 1 | patch | fp16-autocast | 8 | yes | 0.999975 | 0.999867 | 0.998939 | 45.7 | 38.0 | 1.201 | 0.866 | above 1 |  |
| ViT-H+/16 | 224 | 1 | patch | fp16-autocast | 4 | no | 0.993273 | 0.978011 | 0.919585 | 49.3 | 38.0 | 1.297 | 0.935 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.993273, patch mean 0.978011, patch 1st percentile 0.919585 |
| ViT-H+/16 | 224 | 1 | patch | bf16-autocast | 8 | yes | 0.999927 | 0.999479 | 0.993799 | 45.8 | 67.4 | 0.680 | 0.863 | not above 1 |  |
| ViT-H+/16 | 224 | 1 | patch | bf16-autocast | 4 | no | 0.993764 | 0.979564 | 0.943436 | 49.2 | 67.4 | 0.730 | 0.927 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.993764, patch mean 0.979564, patch 1st percentile 0.943436 |
| ViT-H+/16 | 224 | 8 | frame | fp16-autocast | 8 | yes | 0.999975 | 0.999867 | 0.998939 | 69.1 | 71.7 | 0.963 | 0.881 | not above 1 |  |
| ViT-H+/16 | 224 | 8 | frame | fp16-autocast | 4 | no | 0.993273 | 0.978011 | 0.919585 | 72.4 | 71.7 | 1.009 | 0.923 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.993273, patch mean 0.978011, patch 1st percentile 0.919585 |
| ViT-H+/16 | 224 | 8 | frame | bf16-autocast | 8 | yes | 0.999927 | 0.999479 | 0.993799 | 68.9 | 95.8 | 0.718 | 0.883 | not above 1 |  |
| ViT-H+/16 | 224 | 8 | frame | bf16-autocast | 4 | no | 0.993764 | 0.979564 | 0.943436 | 71.8 | 95.8 | 0.749 | 0.920 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.993764, patch mean 0.979564, patch 1st percentile 0.943436 |
| ViT-H+/16 | 224 | 8 | patch | fp16-autocast | 8 | yes | 0.999975 | 0.999867 | 0.998939 | 68.9 | 71.6 | 0.963 | 0.881 | not above 1 |  |
| ViT-H+/16 | 224 | 8 | patch | fp16-autocast | 4 | no | 0.993273 | 0.978011 | 0.919585 | 72.2 | 71.6 | 1.009 | 0.923 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.993273, patch mean 0.978011, patch 1st percentile 0.919585 |
| ViT-H+/16 | 224 | 8 | patch | bf16-autocast | 8 | yes | 0.999927 | 0.999479 | 0.993799 | 68.7 | 95.8 | 0.717 | 0.883 | not above 1 |  |
| ViT-H+/16 | 224 | 8 | patch | bf16-autocast | 4 | no | 0.993764 | 0.979564 | 0.943436 | 71.6 | 95.8 | 0.748 | 0.921 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.993764, patch mean 0.979564, patch 1st percentile 0.943436 |
| ViT-H+/16 | 224 | 32 | frame | fp16-autocast | 8 | yes | 0.999975 | 0.999867 | 0.998939 | 72.4 | 71.9 | 1.007 | 0.928 | above 1, inside the noise |  |
| ViT-H+/16 | 224 | 32 | frame | fp16-autocast | 4 | no | 0.993273 | 0.978011 | 0.919585 | 75.9 | 71.9 | 1.056 | 0.973 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.993273, patch mean 0.978011, patch 1st percentile 0.919585 |
| ViT-H+/16 | 224 | 32 | frame | bf16-autocast | 8 | yes | 0.999927 | 0.999479 | 0.993799 | 72.3 | 79.5 | 0.910 | 0.842 | not above 1 |  |
| ViT-H+/16 | 224 | 32 | frame | bf16-autocast | 4 | no | 0.993764 | 0.979564 | 0.943436 | 75.3 | 79.5 | 0.948 | 0.877 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.993764, patch mean 0.979564, patch 1st percentile 0.943436 |
| ViT-H+/16 | 224 | 32 | patch | fp16-autocast | 8 | yes | 0.999975 | 0.999867 | 0.998939 | 72.4 | 71.8 | 1.008 | 0.928 | above 1, inside the noise |  |
| ViT-H+/16 | 224 | 32 | patch | fp16-autocast | 4 | no | 0.993273 | 0.978011 | 0.919585 | 75.9 | 71.8 | 1.056 | 0.973 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.993273, patch mean 0.978011, patch 1st percentile 0.919585 |
| ViT-H+/16 | 224 | 32 | patch | bf16-autocast | 8 | yes | 0.999927 | 0.999479 | 0.993799 | 72.3 | 79.4 | 0.910 | 0.842 | not above 1 |  |
| ViT-H+/16 | 224 | 32 | patch | bf16-autocast | 4 | no | 0.993764 | 0.979564 | 0.943436 | 75.2 | 79.4 | 0.947 | 0.877 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.993764, patch mean 0.979564, patch 1st percentile 0.943436 |
| ViT-H+/16 | 448 | 1 | frame | fp16-autocast | 8 | yes | 0.999974 | 0.999955 | 0.999914 | 19.6 | 22.1 | 0.886 | 0.900 | not above 1 |  |
| ViT-H+/16 | 448 | 1 | frame | fp16-autocast | 4 | no | 0.992514 | 0.987321 | 0.977246 | 18.9 | 22.1 | 0.853 | 0.866 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.992514, patch mean 0.987321, patch 1st percentile 0.977246 |
| ViT-H+/16 | 448 | 1 | frame | bf16-autocast | 8 | yes | 0.999941 | 0.999882 | 0.999530 | 17.5 | 22.5 | 0.778 | 0.798 | not above 1 |  |
| ViT-H+/16 | 448 | 1 | frame | bf16-autocast | 4 | no | 0.992747 | 0.987387 | 0.977394 | 19.0 | 22.5 | 0.843 | 0.864 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.992747, patch mean 0.987387, patch 1st percentile 0.977394 |
| ViT-H+/16 | 448 | 1 | patch | fp16-autocast | 8 | yes | 0.999974 | 0.999955 | 0.999914 | 19.5 | 22.1 | 0.882 | 0.900 | not above 1 |  |
| ViT-H+/16 | 448 | 1 | patch | fp16-autocast | 4 | no | 0.992514 | 0.987321 | 0.977246 | 18.8 | 22.1 | 0.849 | 0.867 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.992514, patch mean 0.987321, patch 1st percentile 0.977246 |
| ViT-H+/16 | 448 | 1 | patch | bf16-autocast | 8 | yes | 0.999941 | 0.999882 | 0.999530 | 17.5 | 22.6 | 0.775 | 0.799 | not above 1 |  |
| ViT-H+/16 | 448 | 1 | patch | bf16-autocast | 4 | no | 0.992747 | 0.987387 | 0.977394 | 18.9 | 22.6 | 0.839 | 0.865 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.992747, patch mean 0.987387, patch 1st percentile 0.977394 |
| ViT-H+/16 | 448 | 8 | frame | fp16-autocast | 8 | yes | 0.999974 | 0.999955 | 0.999914 | 19.9 | 22.2 | 0.897 | 0.909 | not above 1 |  |
| ViT-H+/16 | 448 | 8 | frame | fp16-autocast | 4 | no | 0.992514 | 0.987321 | 0.977246 | 21.3 | 22.2 | 0.959 | 0.973 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.992514, patch mean 0.987321, patch 1st percentile 0.977246 |
| ViT-H+/16 | 448 | 8 | frame | bf16-autocast | 8 | yes | 0.999941 | 0.999882 | 0.999530 | 19.9 | 22.1 | 0.902 | 0.887 | not above 1 |  |
| ViT-H+/16 | 448 | 8 | frame | bf16-autocast | 4 | no | 0.992747 | 0.987387 | 0.977394 | 21.1 | 22.1 | 0.955 | 0.938 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.992747, patch mean 0.987387, patch 1st percentile 0.977394 |
| ViT-H+/16 | 448 | 8 | patch | fp16-autocast | 8 | yes | 0.999974 | 0.999955 | 0.999914 | 19.9 | 22.2 | 0.896 | 0.909 | not above 1 |  |
| ViT-H+/16 | 448 | 8 | patch | fp16-autocast | 4 | no | 0.992514 | 0.987321 | 0.977246 | 21.3 | 22.2 | 0.959 | 0.973 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.992514, patch mean 0.987321, patch 1st percentile 0.977246 |
| ViT-H+/16 | 448 | 8 | patch | bf16-autocast | 8 | yes | 0.999941 | 0.999882 | 0.999530 | 19.9 | 22.0 | 0.902 | 0.887 | not above 1 |  |
| ViT-H+/16 | 448 | 8 | patch | bf16-autocast | 4 | no | 0.992747 | 0.987387 | 0.977394 | 21.0 | 22.0 | 0.955 | 0.938 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.992747, patch mean 0.987387, patch 1st percentile 0.977394 |
| ViT-H+/16 | 448 | 32 | frame | fp16-autocast | 8 | yes | 0.999974 | 0.999955 | 0.999914 | 18.0 | 17.4 | 1.038 | 0.904 | above 1 |  |
| ViT-H+/16 | 448 | 32 | frame | fp16-autocast | 4 | no | 0.992514 | 0.987321 | 0.977246 | 18.0 | 17.4 | 1.035 | 0.901 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.992514, patch mean 0.987321, patch 1st percentile 0.977246 |
| ViT-H+/16 | 448 | 32 | frame | bf16-autocast | 8 | yes | 0.999941 | 0.999882 | 0.999530 | 17.4 | 17.7 | 0.987 | 0.874 | not above 1, inside the noise |  |
| ViT-H+/16 | 448 | 32 | frame | bf16-autocast | 4 | no | 0.992747 | 0.987387 | 0.977394 | 20.2 | 17.7 | 1.142 | 1.011 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.992747, patch mean 0.987387, patch 1st percentile 0.977394 |
| ViT-H+/16 | 448 | 32 | patch | fp16-autocast | 8 | yes | 0.999974 | 0.999955 | 0.999914 | 18.0 | 17.3 | 1.038 | 0.904 | above 1 |  |
| ViT-H+/16 | 448 | 32 | patch | fp16-autocast | 4 | no | 0.992514 | 0.987321 | 0.977246 | 18.0 | 17.3 | 1.035 | 0.901 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.992514, patch mean 0.987321, patch 1st percentile 0.977246 |
| ViT-H+/16 | 448 | 32 | patch | bf16-autocast | 8 | yes | 0.999941 | 0.999882 | 0.999530 | 17.4 | 17.6 | 0.988 | 0.875 | not above 1, inside the noise |  |
| ViT-H+/16 | 448 | 32 | patch | bf16-autocast | 4 | no | 0.992747 | 0.987387 | 0.977394 | 20.2 | 17.6 | 1.143 | 1.013 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.992747, patch mean 0.987387, patch 1st percentile 0.977394 |
| ViT-H+/16 | 512 | 1 | frame | fp16-autocast | 8 | yes | 0.999971 | 0.999956 | 0.999926 | 13.8 | 16.7 | 0.829 | 0.953 | not above 1 |  |
| ViT-H+/16 | 512 | 1 | frame | fp16-autocast | 4 | no | 0.992172 | 0.987592 | 0.978729 | 14.5 | 16.7 | 0.870 | 0.999 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.992172, patch mean 0.987592, patch 1st percentile 0.978729 |
| ViT-H+/16 | 512 | 1 | frame | bf16-autocast | 8 | yes | 0.999933 | 0.999895 | 0.999749 | 13.9 | 16.6 | 0.838 | 0.922 | not above 1 |  |
| ViT-H+/16 | 512 | 1 | frame | bf16-autocast | 4 | no | 0.992410 | 0.987778 | 0.978977 | 14.6 | 16.6 | 0.878 | 0.966 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.99241, patch mean 0.987778, patch 1st percentile 0.978977 |
| ViT-H+/16 | 512 | 1 | patch | fp16-autocast | 8 | yes | 0.999971 | 0.999956 | 0.999926 | 13.8 | 16.7 | 0.827 | 0.953 | not above 1 |  |
| ViT-H+/16 | 512 | 1 | patch | fp16-autocast | 4 | no | 0.992172 | 0.987592 | 0.978729 | 14.4 | 16.7 | 0.867 | 0.999 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.992172, patch mean 0.987592, patch 1st percentile 0.978729 |
| ViT-H+/16 | 512 | 1 | patch | bf16-autocast | 8 | yes | 0.999933 | 0.999895 | 0.999749 | 13.9 | 16.6 | 0.835 | 0.922 | not above 1 |  |
| ViT-H+/16 | 512 | 1 | patch | bf16-autocast | 4 | no | 0.992410 | 0.987778 | 0.978977 | 14.5 | 16.6 | 0.875 | 0.966 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.99241, patch mean 0.987778, patch 1st percentile 0.978977 |
| ViT-H+/16 | 512 | 8 | frame | fp16-autocast | 8 | yes | 0.999971 | 0.999956 | 0.999926 | 13.1 | 15.9 | 0.824 | 0.887 | not above 1 |  |
| ViT-H+/16 | 512 | 8 | frame | fp16-autocast | 4 | no | 0.992172 | 0.987592 | 0.978729 | 13.7 | 15.9 | 0.863 | 0.929 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.992172, patch mean 0.987592, patch 1st percentile 0.978729 |
| ViT-H+/16 | 512 | 8 | frame | bf16-autocast | 8 | yes | 0.999933 | 0.999895 | 0.999749 | 13.2 | 14.6 | 0.901 | 0.899 | not above 1 |  |
| ViT-H+/16 | 512 | 8 | frame | bf16-autocast | 4 | no | 0.992410 | 0.987778 | 0.978977 | 13.7 | 14.6 | 0.941 | 0.938 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.99241, patch mean 0.987778, patch 1st percentile 0.978977 |
| ViT-H+/16 | 512 | 8 | patch | fp16-autocast | 8 | yes | 0.999971 | 0.999956 | 0.999926 | 13.1 | 15.9 | 0.824 | 0.887 | not above 1 |  |
| ViT-H+/16 | 512 | 8 | patch | fp16-autocast | 4 | no | 0.992172 | 0.987592 | 0.978729 | 13.7 | 15.9 | 0.863 | 0.929 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.992172, patch mean 0.987592, patch 1st percentile 0.978729 |
| ViT-H+/16 | 512 | 8 | patch | bf16-autocast | 8 | yes | 0.999933 | 0.999895 | 0.999749 | 13.1 | 14.6 | 0.901 | 0.899 | not above 1 |  |
| ViT-H+/16 | 512 | 8 | patch | bf16-autocast | 4 | no | 0.992410 | 0.987778 | 0.978977 | 13.7 | 14.6 | 0.941 | 0.938 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.99241, patch mean 0.987778, patch 1st percentile 0.978977 |
| ViT-H+/16 | 512 | 32 | frame | fp16-autocast | 8 | yes | 0.999971 | 0.999956 | 0.999926 | 11.8 | 13.0 | 0.909 | 0.929 | not above 1 |  |
| ViT-H+/16 | 512 | 32 | frame | fp16-autocast | 4 | no | 0.992172 | 0.987592 | 0.978729 | 12.2 | 13.0 | 0.946 | 0.967 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.992172, patch mean 0.987592, patch 1st percentile 0.978729 |
| ViT-H+/16 | 512 | 32 | frame | bf16-autocast | 8 | yes | 0.999933 | 0.999895 | 0.999749 | 11.8 | 12.1 | 0.971 | 0.943 | not above 1, inside the noise |  |
| ViT-H+/16 | 512 | 32 | frame | bf16-autocast | 4 | no | 0.992410 | 0.987778 | 0.978977 | 12.4 | 12.1 | 1.017 | 0.988 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.99241, patch mean 0.987778, patch 1st percentile 0.978977 |
| ViT-H+/16 | 512 | 32 | patch | fp16-autocast | 8 | yes | 0.999971 | 0.999956 | 0.999926 | 11.8 | 12.9 | 0.908 | 0.931 | not above 1 |  |
| ViT-H+/16 | 512 | 32 | patch | fp16-autocast | 4 | no | 0.992172 | 0.987592 | 0.978729 | 12.2 | 12.9 | 0.945 | 0.969 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.992172, patch mean 0.987592, patch 1st percentile 0.978729 |
| ViT-H+/16 | 512 | 32 | patch | bf16-autocast | 8 | yes | 0.999933 | 0.999895 | 0.999749 | 11.8 | 12.1 | 0.970 | 0.944 | not above 1 |  |
| ViT-H+/16 | 512 | 32 | patch | bf16-autocast | 4 | no | 0.992410 | 0.987778 | 0.978977 | 12.3 | 12.1 | 1.017 | 0.989 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.99241, patch mean 0.987778, patch 1st percentile 0.978977 |
| ViT-7B/16 | 224 | 1 | frame | fp16-autocast | 8 | yes | 0.999967 | 0.999949 | 0.999488 | 7.7 | 9.9 | 0.775 | 0.861 | not above 1 |  |
| ViT-7B/16 | 224 | 1 | frame | fp16-autocast | 4 | no | 0.993452 | 0.990519 | 0.933926 | 8.5 | 9.9 | 0.853 | 0.947 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.993452, patch mean 0.990519, patch 1st percentile 0.933926 |
| ViT-7B/16 | 224 | 1 | frame | bf16-autocast | 8 | yes | 0.999893 | 0.999316 | 0.983754 | 7.8 | 10.7 | 0.731 | 0.931 | not above 1 |  |
| ViT-7B/16 | 224 | 1 | frame | bf16-autocast | 4 | no | 0.993112 | 0.990340 | 0.950481 | 8.5 | 10.7 | 0.796 | 1.013 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.993112, patch mean 0.99034, patch 1st percentile 0.950481 |
| ViT-7B/16 | 224 | 1 | patch | fp16-autocast | 8 | yes | 0.999967 | 0.999949 | 0.999488 | 7.7 | 10.0 | 0.773 | 0.861 | not above 1 |  |
| ViT-7B/16 | 224 | 1 | patch | fp16-autocast | 4 | no | 0.993452 | 0.990519 | 0.933926 | 8.5 | 10.0 | 0.851 | 0.948 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.993452, patch mean 0.990519, patch 1st percentile 0.933926 |
| ViT-7B/16 | 224 | 1 | patch | bf16-autocast | 8 | yes | 0.999893 | 0.999316 | 0.983754 | 7.8 | 10.7 | 0.730 | 0.931 | not above 1 |  |
| ViT-7B/16 | 224 | 1 | patch | bf16-autocast | 4 | no | 0.993112 | 0.990340 | 0.950481 | 8.5 | 10.7 | 0.794 | 1.013 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.993112, patch mean 0.99034, patch 1st percentile 0.950481 |
| ViT-7B/16 | 224 | 8 | frame | fp16-autocast | 8 | yes | 0.999967 | 0.999949 | 0.999488 | 10.7 | 12.5 | 0.857 | 0.911 | not above 1 |  |
| ViT-7B/16 | 224 | 8 | frame | fp16-autocast | 4 | no | 0.993452 | 0.990519 | 0.933926 | 10.7 | 12.5 | 0.856 | 0.909 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.993452, patch mean 0.990519, patch 1st percentile 0.933926 |
| ViT-7B/16 | 224 | 8 | frame | bf16-autocast | 8 | yes | 0.999893 | 0.999316 | 0.983754 | 10.5 | 12.9 | 0.815 | 0.891 | not above 1 |  |
| ViT-7B/16 | 224 | 8 | frame | bf16-autocast | 4 | no | 0.993112 | 0.990340 | 0.950481 | 10.6 | 12.9 | 0.823 | 0.900 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.993112, patch mean 0.99034, patch 1st percentile 0.950481 |
| ViT-7B/16 | 224 | 8 | patch | fp16-autocast | 8 | yes | 0.999967 | 0.999949 | 0.999488 | 10.7 | 12.5 | 0.857 | 0.911 | not above 1 |  |
| ViT-7B/16 | 224 | 8 | patch | fp16-autocast | 4 | no | 0.993452 | 0.990519 | 0.933926 | 10.7 | 12.5 | 0.856 | 0.909 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.993452, patch mean 0.990519, patch 1st percentile 0.933926 |
| ViT-7B/16 | 224 | 8 | patch | bf16-autocast | 8 | yes | 0.999893 | 0.999316 | 0.983754 | 10.5 | 12.9 | 0.815 | 0.891 | not above 1 |  |
| ViT-7B/16 | 224 | 8 | patch | bf16-autocast | 4 | no | 0.993112 | 0.990340 | 0.950481 | 10.6 | 12.9 | 0.823 | 0.900 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.993112, patch mean 0.99034, patch 1st percentile 0.950481 |
| ViT-7B/16 | 224 | 32 | frame | fp16-autocast | 8 | yes | 0.999967 | 0.999949 | 0.999488 | 8.6 | 9.1 | 0.951 | 0.888 | not above 1 |  |
| ViT-7B/16 | 224 | 32 | frame | fp16-autocast | 4 | no | 0.993452 | 0.990519 | 0.933926 | 10.6 | 9.1 | 1.174 | 1.096 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.993452, patch mean 0.990519, patch 1st percentile 0.933926 |
| ViT-7B/16 | 224 | 32 | frame | bf16-autocast | 8 | yes | 0.999893 | 0.999316 | 0.983754 | 10.5 | 8.9 | 1.176 | 1.105 | above 1 |  |
| ViT-7B/16 | 224 | 32 | frame | bf16-autocast | 4 | no | 0.993112 | 0.990340 | 0.950481 | 10.5 | 8.9 | 1.177 | 1.105 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.993112, patch mean 0.99034, patch 1st percentile 0.950481 |
| ViT-7B/16 | 224 | 32 | patch | fp16-autocast | 8 | yes | 0.999967 | 0.999949 | 0.999488 | 8.6 | 9.1 | 0.951 | 0.888 | not above 1 |  |
| ViT-7B/16 | 224 | 32 | patch | fp16-autocast | 4 | no | 0.993452 | 0.990519 | 0.933926 | 10.6 | 9.1 | 1.174 | 1.096 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.993452, patch mean 0.990519, patch 1st percentile 0.933926 |
| ViT-7B/16 | 224 | 32 | patch | bf16-autocast | 8 | yes | 0.999893 | 0.999316 | 0.983754 | 10.5 | 8.9 | 1.176 | 1.105 | above 1 |  |
| ViT-7B/16 | 224 | 32 | patch | bf16-autocast | 4 | no | 0.993112 | 0.990340 | 0.950481 | 10.5 | 8.9 | 1.177 | 1.105 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.993112, patch mean 0.99034, patch 1st percentile 0.950481 |
| ViT-7B/16 | 448 | 1 | frame | fp16-autocast | 8 | yes | 0.999975 | 0.999983 | 0.999925 | 2.5 | 2.6 | 0.959 | 1.001 | not above 1 |  |
| ViT-7B/16 | 448 | 1 | frame | fp16-autocast | 4 | no | 0.993235 | 0.995998 | 0.987901 | 2.2 | 2.6 | 0.850 | 0.887 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.993235, patch mean 0.995998, patch 1st percentile 0.987901 |
| ViT-7B/16 | 448 | 1 | frame | bf16-autocast | 8 | yes | 0.999937 | 0.999902 | 0.999628 | 2.5 | 2.7 | 0.922 | 1.041 | not above 1 |  |
| ViT-7B/16 | 448 | 1 | frame | bf16-autocast | 4 | no | 0.993409 | 0.995968 | 0.987344 | 2.2 | 2.7 | 0.824 | 0.930 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.993409, patch mean 0.995968, patch 1st percentile 0.987344 |
| ViT-7B/16 | 448 | 1 | patch | fp16-autocast | 8 | yes | 0.999975 | 0.999983 | 0.999925 | 2.5 | 2.6 | 0.959 | 1.001 | not above 1 |  |
| ViT-7B/16 | 448 | 1 | patch | fp16-autocast | 4 | no | 0.993235 | 0.995998 | 0.987901 | 2.2 | 2.6 | 0.849 | 0.887 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.993235, patch mean 0.995998, patch 1st percentile 0.987901 |
| ViT-7B/16 | 448 | 1 | patch | bf16-autocast | 8 | yes | 0.999937 | 0.999902 | 0.999628 | 2.5 | 2.7 | 0.922 | 1.041 | not above 1 |  |
| ViT-7B/16 | 448 | 1 | patch | bf16-autocast | 4 | no | 0.993409 | 0.995968 | 0.987344 | 2.2 | 2.7 | 0.823 | 0.930 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.993409, patch mean 0.995968, patch 1st percentile 0.987344 |
| ViT-7B/16 | 448 | 8 | frame | fp16-autocast | 8 | yes | 0.999975 | 0.999983 | 0.999925 | 2.5 | 2.6 | 0.962 | 0.904 | not above 1 |  |
| ViT-7B/16 | 448 | 8 | frame | fp16-autocast | 4 | no | 0.993235 | 0.995998 | 0.987901 | 2.6 | 2.6 | 1.009 | 0.948 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.993235, patch mean 0.995998, patch 1st percentile 0.987901 |
| ViT-7B/16 | 448 | 8 | frame | bf16-autocast | 8 | yes | 0.999937 | 0.999902 | 0.999628 | 2.5 | 2.5 | 0.983 | 0.915 | not above 1, inside the noise |  |
| ViT-7B/16 | 448 | 8 | frame | bf16-autocast | 4 | no | 0.993409 | 0.995968 | 0.987344 | 2.7 | 2.5 | 1.074 | 0.999 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.993409, patch mean 0.995968, patch 1st percentile 0.987344 |
| ViT-7B/16 | 448 | 8 | patch | fp16-autocast | 8 | yes | 0.999975 | 0.999983 | 0.999925 | 2.5 | 2.6 | 0.962 | 0.904 | not above 1 |  |
| ViT-7B/16 | 448 | 8 | patch | fp16-autocast | 4 | no | 0.993235 | 0.995998 | 0.987901 | 2.6 | 2.6 | 1.009 | 0.948 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.993235, patch mean 0.995998, patch 1st percentile 0.987901 |
| ViT-7B/16 | 448 | 8 | patch | bf16-autocast | 8 | yes | 0.999937 | 0.999902 | 0.999628 | 2.5 | 2.5 | 0.983 | 0.915 | not above 1, inside the noise |  |
| ViT-7B/16 | 448 | 8 | patch | bf16-autocast | 4 | no | 0.993409 | 0.995968 | 0.987344 | 2.7 | 2.5 | 1.074 | 1.000 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.993409, patch mean 0.995968, patch 1st percentile 0.987344 |
| ViT-7B/16 | 448 | 32 | frame | fp16-autocast | 8 | yes | 0.999975 | 0.999983 | 0.999925 | 2.5 | 2.3 | 1.081 | 0.945 | above 1 |  |
| ViT-7B/16 | 448 | 32 | frame | fp16-autocast | 4 | no | 0.993235 | 0.995998 | 0.987901 | 2.6 | 2.3 | 1.143 | 1.000 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.993235, patch mean 0.995998, patch 1st percentile 0.987901 |
| ViT-7B/16 | 448 | 32 | frame | bf16-autocast | 8 | yes | 0.999937 | 0.999902 | 0.999628 | 2.5 | 2.4 | 1.062 | 0.945 | above 1 |  |
| ViT-7B/16 | 448 | 32 | frame | bf16-autocast | 4 | no | 0.993409 | 0.995968 | 0.987344 | 2.7 | 2.4 | 1.126 | 1.002 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.993409, patch mean 0.995968, patch 1st percentile 0.987344 |
| ViT-7B/16 | 448 | 32 | patch | fp16-autocast | 8 | yes | 0.999975 | 0.999983 | 0.999925 | 2.5 | 2.3 | 1.081 | 0.946 | above 1 |  |
| ViT-7B/16 | 448 | 32 | patch | fp16-autocast | 4 | no | 0.993235 | 0.995998 | 0.987901 | 2.6 | 2.3 | 1.143 | 1.000 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.993235, patch mean 0.995998, patch 1st percentile 0.987901 |
| ViT-7B/16 | 448 | 32 | patch | bf16-autocast | 8 | yes | 0.999937 | 0.999902 | 0.999628 | 2.5 | 2.4 | 1.062 | 0.945 | above 1 |  |
| ViT-7B/16 | 448 | 32 | patch | bf16-autocast | 4 | no | 0.993409 | 0.995968 | 0.987344 | 2.7 | 2.4 | 1.126 | 1.003 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.993409, patch mean 0.995968, patch 1st percentile 0.987344 |
| ViT-7B/16 | 512 | 1 | frame | fp16-autocast | 8 | yes | 0.999982 | 0.999988 | 0.999975 | 1.9 | 2.4 | 0.821 | 0.937 | not above 1 |  |
| ViT-7B/16 | 512 | 1 | frame | fp16-autocast | 4 | no | 0.994277 | 0.996500 | 0.990894 | 2.0 | 2.4 | 0.863 | 0.984 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.994277, patch mean 0.9965, patch 1st percentile 0.990894 |
| ViT-7B/16 | 512 | 1 | frame | bf16-autocast | 8 | yes | 0.999957 | 0.999958 | 0.999812 | 1.9 | 2.3 | 0.831 | 0.927 | not above 1 |  |
| ViT-7B/16 | 512 | 1 | frame | bf16-autocast | 4 | no | 0.994344 | 0.996476 | 0.990915 | 2.0 | 2.3 | 0.872 | 0.974 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.994344, patch mean 0.996476, patch 1st percentile 0.990915 |
| ViT-7B/16 | 512 | 1 | patch | fp16-autocast | 8 | yes | 0.999982 | 0.999988 | 0.999975 | 1.9 | 2.4 | 0.821 | 0.937 | not above 1 |  |
| ViT-7B/16 | 512 | 1 | patch | fp16-autocast | 4 | no | 0.994277 | 0.996500 | 0.990894 | 2.0 | 2.4 | 0.862 | 0.984 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.994277, patch mean 0.9965, patch 1st percentile 0.990894 |
| ViT-7B/16 | 512 | 1 | patch | bf16-autocast | 8 | yes | 0.999957 | 0.999958 | 0.999812 | 1.9 | 2.3 | 0.830 | 0.927 | not above 1 |  |
| ViT-7B/16 | 512 | 1 | patch | bf16-autocast | 4 | no | 0.994344 | 0.996476 | 0.990915 | 2.0 | 2.3 | 0.872 | 0.973 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.994344, patch mean 0.996476, patch 1st percentile 0.990915 |
| ViT-7B/16 | 512 | 8 | frame | fp16-autocast | 8 | yes | 0.999982 | 0.999988 | 0.999975 | 1.9 | 1.9 | 0.990 | 0.946 | not above 1, inside the noise |  |
| ViT-7B/16 | 512 | 8 | frame | fp16-autocast | 4 | no | 0.994277 | 0.996500 | 0.990894 | 1.9 | 1.9 | 1.033 | 0.987 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.994277, patch mean 0.9965, patch 1st percentile 0.990894 |
| ViT-7B/16 | 512 | 8 | frame | bf16-autocast | 8 | yes | 0.999957 | 0.999958 | 0.999812 | 1.9 | 1.9 | 0.975 | 0.942 | not above 1, inside the noise |  |
| ViT-7B/16 | 512 | 8 | frame | bf16-autocast | 4 | no | 0.994344 | 0.996476 | 0.990915 | 2.0 | 1.9 | 1.040 | 1.004 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.994344, patch mean 0.996476, patch 1st percentile 0.990915 |
| ViT-7B/16 | 512 | 8 | patch | fp16-autocast | 8 | yes | 0.999982 | 0.999988 | 0.999975 | 1.9 | 1.9 | 0.990 | 0.946 | not above 1, inside the noise |  |
| ViT-7B/16 | 512 | 8 | patch | fp16-autocast | 4 | no | 0.994277 | 0.996500 | 0.990894 | 1.9 | 1.9 | 1.033 | 0.987 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.994277, patch mean 0.9965, patch 1st percentile 0.990894 |
| ViT-7B/16 | 512 | 8 | patch | bf16-autocast | 8 | yes | 0.999957 | 0.999958 | 0.999812 | 1.9 | 1.9 | 0.976 | 0.942 | not above 1, inside the noise |  |
| ViT-7B/16 | 512 | 8 | patch | bf16-autocast | 4 | no | 0.994344 | 0.996476 | 0.990915 | 2.0 | 1.9 | 1.040 | 1.004 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.994344, patch mean 0.996476, patch 1st percentile 0.990915 |
| ViT-7B/16 | 512 | 32 | frame | fp16-autocast | 8 | yes | 0.999982 | 0.999988 | 0.999975 | 1.9 | 1.7 | 1.096 | 0.945 | above 1 |  |
| ViT-7B/16 | 512 | 32 | frame | fp16-autocast | 4 | no | 0.994277 | 0.996500 | 0.990894 | 2.0 | 1.7 | 1.152 | 0.994 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.994277, patch mean 0.9965, patch 1st percentile 0.990894 |
| ViT-7B/16 | 512 | 32 | frame | bf16-autocast | 8 | yes | 0.999957 | 0.999958 | 0.999812 | 1.9 | 1.8 | 1.067 | 0.943 | above 1 |  |
| ViT-7B/16 | 512 | 32 | frame | bf16-autocast | 4 | no | 0.994344 | 0.996476 | 0.990915 | 2.0 | 1.8 | 1.125 | 0.994 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.994344, patch mean 0.996476, patch 1st percentile 0.990915 |
| ViT-7B/16 | 512 | 32 | patch | fp16-autocast | 8 | yes | 0.999982 | 0.999988 | 0.999975 | 1.9 | 1.7 | 1.096 | 0.946 | above 1 |  |
| ViT-7B/16 | 512 | 32 | patch | fp16-autocast | 4 | no | 0.994277 | 0.996500 | 0.990894 | 2.0 | 1.7 | 1.152 | 0.994 | not valid | fails THRESHOLDS[fp16-autocast]: frame 0.994277, patch mean 0.9965, patch 1st percentile 0.990894 |
| ViT-7B/16 | 512 | 32 | patch | bf16-autocast | 8 | yes | 0.999957 | 0.999958 | 0.999812 | 1.9 | 1.8 | 1.067 | 0.943 | above 1 |  |
| ViT-7B/16 | 512 | 32 | patch | bf16-autocast | 4 | no | 0.994344 | 0.996476 | 0.990915 | 2.0 | 1.8 | 1.125 | 0.994 | not valid | fails THRESHOLDS[bf16-autocast]: frame 0.994344, patch mean 0.996476, patch 1st percentile 0.990915 |

Table 3. The quantized rows (D18, D22, ADR 0011): valid only if image 0 passes THRESHOLDS[mode].

## 4 Change ratios

| Change | Name | Priority | Dtype mode | Forms | Output | Median [min–max] | Bit equality |
| --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | Compiled forward pass (E → A) | P1 | fp32 | E → A | frame | 1.051 [1.048–1.056] | yes |
| C1 | Compiled forward pass (E → A) | P1 | fp16-cast | E → A | frame | 1.305 [1.302–1.343] | yes |
| C1 | Compiled forward pass (E → A) | P1 | bf16-cast | E → A | frame | 1.300 [1.289–1.310] | yes |
| C1 | Compiled forward pass (E → A) | P1 | fp16-autocast | E → A | frame | 1.293 [1.278–1.317] | yes |
| C1 | Compiled forward pass (E → A) | P1 | bf16-autocast | E → A | frame | 1.293 [1.292–1.311] | yes |
| C1 | Compiled forward pass (E → A) | P2 | fp32 | E → A | patch | 1.037 [1.001–1.040] | yes |
| C1 | Compiled forward pass (E → A) | P2 | fp16-cast | E → A | patch | 1.272 [1.236–1.276] | yes |
| C1 | Compiled forward pass (E → A) | P2 | bf16-cast | E → A | patch | 1.258 [1.256–1.275] | yes |
| C1 | Compiled forward pass (E → A) | P2 | fp16-autocast | E → A | patch | 1.282 [1.210–1.302] | yes |
| C1 | Compiled forward pass (E → A) | P2 | bf16-autocast | E → A | patch | 1.250 [1.237–1.287] | yes |
| C2 | Concat-free RoPE (A → R) | P1 | fp32 | A → R | frame | 1.012 [0.967–1.038] | yes |
| C2 | Concat-free RoPE (A → R) | P1 | fp16-cast | A → R | frame | 1.060 [1.008–1.067] | yes |
| C2 | Concat-free RoPE (A → R) | P1 | bf16-cast | A → R | frame | 1.068 [1.015–1.082] | yes |
| C2 | Concat-free RoPE (A → R) | P1 | fp16-autocast | A → R | frame | 1.089 [1.059–1.106] | yes |
| C2 | Concat-free RoPE (A → R) | P1 | bf16-autocast | A → R | frame | 1.079 [1.032–1.099] | yes |
| C2 | Concat-free RoPE (A → R) | P2 | fp32 | A → R | patch | 1.010 [0.985–1.021] | yes |
| C2 | Concat-free RoPE (A → R) | P2 | fp16-cast | A → R | patch | 1.040 [1.013–1.059] | yes |
| C2 | Concat-free RoPE (A → R) | P2 | bf16-cast | A → R | patch | 1.047 [1.005–1.080] | yes |
| C2 | Concat-free RoPE (A → R) | P2 | fp16-autocast | A → R | patch | 1.056 [1.041–1.067] | yes |
| C2 | Concat-free RoPE (A → R) | P2 | bf16-autocast | A → R | patch | 1.042 [1.018–1.059] | yes |
| C4 | fp16 patch embeddings (R-fp32 → R-fp16) | P1 | fp16-cast | R-fp32 → R-fp16 | frame | 0.993 [0.953–1.011] | frame yes |
| C4 | fp16 patch embeddings (R-fp32 → R-fp16) | P1 | bf16-cast | R-fp32 → R-fp16 | frame | 1.007 [0.968–1.033] | frame yes |
| C4 | fp16 patch embeddings (R-fp32 → R-fp16) | P1 | fp16-autocast | R-fp32 → R-fp16 | frame | 0.991 [0.968–1.043] | frame yes |
| C4 | fp16 patch embeddings (R-fp32 → R-fp16) | P1 | bf16-autocast | R-fp32 → R-fp16 | frame | 0.997 [0.955–1.028] | frame yes |
| C4 | fp16 patch embeddings (R-fp32 → R-fp16) | P2 | fp16-cast | R-fp32 → R-fp16 | patch | 1.004 [0.958–1.046] | frame yes |
| C4 | fp16 patch embeddings (R-fp32 → R-fp16) | P2 | bf16-cast | R-fp32 → R-fp16 | patch | 1.042 [0.998–1.070] | frame yes |
| C4 | fp16 patch embeddings (R-fp32 → R-fp16) | P2 | fp16-autocast | R-fp32 → R-fp16 | patch | 1.004 [0.989–1.031] | frame yes |
| C4 | fp16 patch embeddings (R-fp32 → R-fp16) | P2 | bf16-autocast | R-fp32 → R-fp16 | patch | 1.005 [0.983–1.013] | frame yes |
| C3 | Autocast / cast after compile | P1 | bf16 | bf16-cast → bf16-autocast | frame | 1.034 [0.981–1.051] | no (two configurations) |
| C3 | Autocast / cast after compile | P1 | fp16 | fp16-cast → fp16-autocast | frame | 0.999 [0.980–1.124] | no (two configurations) |
| C3 | Autocast / cast after compile | P2 | bf16 | bf16-cast → bf16-autocast | patch | 0.939 [0.921–0.958] | no (two configurations) |
| C3 | Autocast / cast after compile | P2 | fp16 | fp16-cast → fp16-autocast | patch | 0.954 [0.653–0.960] | no (two configurations) |
| C5 | Quantized weights / unquantized | P2 | bf16-autocast | unquantized → 8-bit | patch | 0.874 [0.872–0.943] | no (two configurations) |
| C5 | Quantized weights / unquantized | P2 | bf16-autocast | unquantized → 4-bit | patch | 0.924 [0.879–0.968] | no (two configurations) |
| C5 | Quantized weights / unquantized | P2 | fp16-autocast | unquantized → 8-bit | patch | 0.884 [0.878–0.893] | no (two configurations) |
| C5 | Quantized weights / unquantized | P2 | fp16-autocast | unquantized → 4-bit | patch | 0.935 [0.909–0.988] | no (two configurations) |
| C6 | Preprocessing in MLX / CPU path | 224 px | fp32 | CPU 1.74 ms → MLX 0.93 ms per frame | pixels | 1.900 [1.737–2.201] | T52 covers the values |
| C6 | Preprocessing in MLX / CPU path | 512 px | fp32 | CPU 2.01 ms → MLX 1.15 ms per frame | pixels | 1.778 [1.403–1.959] | T52 covers the values |

Table 4. The change ratios (Table 23 of the plan), 5 rounds each.

## 5 Best batch size

| Model size | Input resolution | Dtype mode | Best batch size | Frame: throughput, median / p90 latency |
| --- | --- | --- | --- | --- |
| ViT-S/16 | 224 | fp32 | 8 | b1: 333.2 img/s, 2.9 / 3.4 ms; b8: 905.6 img/s, 8.9 / 9.3 ms; b32: 721.6 img/s, 44.1 / 45.1 ms |
| ViT-S/16 | 224 | fp16-cast | 32 | b1: 528.4 img/s, 1.8 / 2.2 ms; b8: 1772.1 img/s, 4.5 / 4.9 ms; b32: 1899.8 img/s, 16.9 / 17.2 ms |
| ViT-S/16 | 224 | bf16-cast | 8 | b1: 524.5 img/s, 1.8 / 2.2 ms; b8: 1893.0 img/s, 4.2 / 4.5 ms; b32: 1794.4 img/s, 17.9 / 18.2 ms |
| ViT-S/16 | 224 | fp16-autocast | 8 | b1: 549.0 img/s, 1.8 / 2.1 ms; b8: 1693.6 img/s, 4.7 / 4.9 ms; b32: 1574.6 img/s, 20.3 / 20.8 ms |
| ViT-S/16 | 224 | bf16-autocast | 8 | b1: 545.5 img/s, 1.8 / 2.1 ms; b8: 1651.0 img/s, 4.9 / 5.1 ms; b32: 1549.7 img/s, 20.7 / 21.1 ms |
| ViT-S/16 | 448 | fp32 | 8 | b1: 130.3 img/s, 7.7 / 7.9 ms; b8: 164.7 img/s, 48.4 / 49.0 ms; b32: 163.2 img/s, 195.0 / 197.4 ms |
| ViT-S/16 | 448 | fp16-cast | 32 | b1: 257.1 img/s, 3.9 / 4.2 ms; b8: 410.7 img/s, 19.6 / 19.9 ms; b32: 411.5 img/s, 77.1 / 78.9 ms |
| ViT-S/16 | 448 | bf16-cast | 8 | b1: 265.6 img/s, 3.8 / 4.0 ms; b8: 413.9 img/s, 19.4 / 19.7 ms; b32: 411.0 img/s, 77.2 / 79.7 ms |
| ViT-S/16 | 448 | fp16-autocast | 32 | b1: 267.1 img/s, 3.8 / 4.0 ms; b8: 375.5 img/s, 21.3 / 21.9 ms; b32: 378.7 img/s, 84.3 / 85.9 ms |
| ViT-S/16 | 448 | bf16-autocast | 32 | b1: 271.8 img/s, 3.7 / 3.9 ms; b8: 376.0 img/s, 21.3 / 21.8 ms; b32: 379.6 img/s, 83.9 / 85.7 ms |
| ViT-S/16 | 512 | fp32 | 32 | b1: 99.7 img/s, 10.1 / 10.3 ms; b8: 102.0 img/s, 75.5 / 91.5 ms; b32: 102.6 img/s, 305.5 / 335.1 ms |
| ViT-S/16 | 512 | fp16-cast | 32 | b1: 228.1 img/s, 4.4 / 4.6 ms; b8: 263.2 img/s, 29.8 / 32.8 ms; b32: 265.3 img/s, 118.9 / 125.4 ms |
| ViT-S/16 | 512 | bf16-cast | 32 | b1: 227.9 img/s, 4.4 / 4.7 ms; b8: 249.3 img/s, 31.8 / 33.4 ms; b32: 266.2 img/s, 117.6 / 125.5 ms |
| ViT-S/16 | 512 | fp16-autocast | 32 | b1: 227.1 img/s, 4.4 / 4.6 ms; b8: 243.0 img/s, 31.9 / 37.0 ms; b32: 248.1 img/s, 128.4 / 133.4 ms |
| ViT-S/16 | 512 | bf16-autocast | 8 | b1: 228.8 img/s, 4.4 / 4.6 ms; b8: 250.8 img/s, 30.8 / 35.0 ms; b32: 239.4 img/s, 131.3 / 136.7 ms |
| ViT-S+/16 | 224 | fp32 | 8 | b1: 283.3 img/s, 3.5 / 3.9 ms; b8: 616.4 img/s, 12.9 / 13.3 ms; b32: 609.9 img/s, 52.2 / 53.6 ms |
| ViT-S+/16 | 224 | fp16-cast | 32 | b1: 456.0 img/s, 2.2 / 2.4 ms; b8: 1433.2 img/s, 5.6 / 6.0 ms; b32: 1493.0 img/s, 21.3 / 22.0 ms |
| ViT-S+/16 | 224 | bf16-cast | 32 | b1: 482.5 img/s, 2.0 / 2.4 ms; b8: 1425.1 img/s, 5.6 / 6.0 ms; b32: 1449.6 img/s, 21.8 / 22.8 ms |
| ViT-S+/16 | 224 | fp16-autocast | 8 | b1: 519.3 img/s, 1.9 / 2.0 ms; b8: 1354.6 img/s, 5.9 / 6.3 ms; b32: 1317.2 img/s, 24.1 / 25.4 ms |
| ViT-S+/16 | 224 | bf16-autocast | 32 | b1: 465.7 img/s, 2.1 / 2.4 ms; b8: 1331.6 img/s, 6.0 / 6.4 ms; b32: 1393.3 img/s, 22.7 / 23.8 ms |
| ViT-S+/16 | 448 | fp32 | 8 | b1: 114.3 img/s, 8.8 / 9.0 ms; b8: 136.6 img/s, 57.9 / 59.9 ms; b32: 136.2 img/s, 234.3 / 238.7 ms |
| ViT-S+/16 | 448 | fp16-cast | 8 | b1: 248.6 img/s, 4.1 / 4.3 ms; b8: 360.5 img/s, 22.2 / 22.5 ms; b32: 356.5 img/s, 89.4 / 90.9 ms |
| ViT-S+/16 | 448 | bf16-cast | 32 | b1: 254.0 img/s, 4.0 / 4.2 ms; b8: 359.1 img/s, 22.3 / 22.9 ms; b32: 359.3 img/s, 88.8 / 90.3 ms |
| ViT-S+/16 | 448 | fp16-autocast | 8 | b1: 251.8 img/s, 4.0 / 4.2 ms; b8: 340.3 img/s, 23.5 / 24.0 ms; b32: 333.1 img/s, 95.5 / 99.1 ms |
| ViT-S+/16 | 448 | bf16-autocast | 8 | b1: 252.0 img/s, 4.0 / 4.2 ms; b8: 342.1 img/s, 23.4 / 23.8 ms; b32: 333.4 img/s, 95.5 / 97.3 ms |
| ViT-S+/16 | 512 | fp32 | 8 | b1: 86.8 img/s, 11.5 / 11.8 ms; b8: 99.1 img/s, 80.3 / 81.5 ms; b32: 98.6 img/s, 322.9 / 325.3 ms |
| ViT-S+/16 | 512 | fp16-cast | 8 | b1: 209.5 img/s, 4.8 / 5.0 ms; b8: 267.7 img/s, 29.8 / 30.3 ms; b32: 262.1 img/s, 121.9 / 123.2 ms |
| ViT-S+/16 | 512 | bf16-cast | 8 | b1: 211.9 img/s, 4.7 / 5.0 ms; b8: 269.0 img/s, 29.7 / 30.2 ms; b32: 260.1 img/s, 122.1 / 126.5 ms |
| ViT-S+/16 | 512 | fp16-autocast | 8 | b1: 209.4 img/s, 4.8 / 5.0 ms; b8: 247.7 img/s, 32.2 / 32.7 ms; b32: 243.5 img/s, 131.0 / 133.7 ms |
| ViT-S+/16 | 512 | bf16-autocast | 8 | b1: 208.9 img/s, 4.8 / 5.0 ms; b8: 248.3 img/s, 32.1 / 32.8 ms; b32: 242.7 img/s, 131.1 / 136.4 ms |
| ViT-B/16 | 224 | fp32 | 32 | b1: 147.1 img/s, 6.8 / 7.0 ms; b8: 234.6 img/s, 34.1 / 34.6 ms; b32: 237.9 img/s, 134.4 / 135.5 ms |
| ViT-B/16 | 224 | fp16-cast | 32 | b1: 316.8 img/s, 3.2 / 3.3 ms; b8: 639.4 img/s, 12.5 / 12.8 ms; b32: 686.1 img/s, 46.2 / 47.7 ms |
| ViT-B/16 | 224 | bf16-cast | 32 | b1: 318.4 img/s, 3.2 / 3.3 ms; b8: 646.5 img/s, 12.4 / 12.6 ms; b32: 694.0 img/s, 45.7 / 47.1 ms |
| ViT-B/16 | 224 | fp16-autocast | 8 | b1: 324.4 img/s, 3.1 / 3.3 ms; b8: 613.8 img/s, 13.0 / 13.5 ms; b32: 576.7 img/s, 53.4 / 65.8 ms |
| ViT-B/16 | 224 | bf16-autocast | 8 | b1: 327.2 img/s, 3.1 / 3.2 ms; b8: 616.6 img/s, 12.9 / 13.8 ms; b32: 588.6 img/s, 53.5 / 56.4 ms |
| ViT-B/16 | 448 | fp32 | 32 | b1: 46.8 img/s, 21.2 / 21.9 ms; b8: 49.5 img/s, 154.1 / 192.5 ms; b32: 50.1 img/s, 632.9 / 675.4 ms |
| ViT-B/16 | 448 | fp16-cast | 8 | b1: 124.7 img/s, 7.9 / 8.8 ms; b8: 149.0 img/s, 52.7 / 55.0 ms; b32: 143.2 img/s, 219.1 / 232.2 ms |
| ViT-B/16 | 448 | bf16-cast | 8 | b1: 129.9 img/s, 7.6 / 8.1 ms; b8: 150.2 img/s, 52.2 / 54.7 ms; b32: 145.1 img/s, 217.8 / 223.6 ms |
| ViT-B/16 | 448 | fp16-autocast | 8 | b1: 126.1 img/s, 7.9 / 8.4 ms; b8: 141.4 img/s, 56.1 / 58.1 ms; b32: 132.3 img/s, 238.3 / 260.9 ms |
| ViT-B/16 | 448 | bf16-autocast | 8 | b1: 128.5 img/s, 7.7 / 8.1 ms; b8: 141.5 img/s, 55.8 / 58.1 ms; b32: 133.8 img/s, 236.9 / 247.2 ms |
| ViT-B/16 | 512 | fp32 | 8 | b1: 35.3 img/s, 27.8 / 30.2 ms; b8: 39.3 img/s, 202.5 / 206.4 ms; b32: 39.2 img/s, 813.2 / 831.8 ms |
| ViT-B/16 | 512 | fp16-cast | 8 | b1: 102.6 img/s, 9.7 / 10.0 ms; b8: 120.3 img/s, 66.3 / 67.4 ms; b32: 114.4 img/s, 277.6 / 285.2 ms |
| ViT-B/16 | 512 | bf16-cast | 32 | b1: 102.4 img/s, 9.7 / 10.1 ms; b8: 113.6 img/s, 69.3 / 70.7 ms; b32: 113.8 img/s, 278.7 / 288.9 ms |
| ViT-B/16 | 512 | fp16-autocast | 32 | b1: 99.5 img/s, 10.0 / 10.4 ms; b8: 105.8 img/s, 73.5 / 84.2 ms; b32: 106.3 img/s, 299.4 / 305.7 ms |
| ViT-B/16 | 512 | bf16-autocast | 8 | b1: 99.9 img/s, 10.0 / 10.3 ms; b8: 108.6 img/s, 72.8 / 76.1 ms; b32: 105.8 img/s, 299.9 / 306.3 ms |
| ViT-L/16 | 224 | fp32 | 32 | b1: 46.1 img/s, 21.8 / 22.0 ms; b8: 70.8 img/s, 112.9 / 113.6 ms; b32: 71.9 img/s, 443.2 / 446.6 ms |
| ViT-L/16 | 224 | fp16-cast | 32 | b1: 130.9 img/s, 7.7 / 7.9 ms; b8: 217.1 img/s, 36.8 / 37.3 ms; b32: 221.8 img/s, 143.1 / 146.1 ms |
| ViT-L/16 | 224 | bf16-cast | 32 | b1: 129.8 img/s, 7.8 / 8.0 ms; b8: 212.5 img/s, 37.5 / 38.1 ms; b32: 220.4 img/s, 143.9 / 145.7 ms |
| ViT-L/16 | 224 | fp16-autocast | 32 | b1: 127.6 img/s, 7.9 / 8.1 ms; b8: 201.2 img/s, 39.7 / 40.1 ms; b32: 205.5 img/s, 154.9 / 159.1 ms |
| ViT-L/16 | 224 | bf16-autocast | 32 | b1: 128.5 img/s, 7.8 / 8.1 ms; b8: 201.4 img/s, 39.7 / 40.4 ms; b32: 205.0 img/s, 156.0 / 157.3 ms |
| ViT-L/16 | 448 | fp32 | 8 | b1: 15.6 img/s, 64.0 / 64.6 ms; b8: 16.6 img/s, 481.2 / 484.3 ms; b32: 16.0 img/s, 1989.6 / 2049.8 ms |
| ViT-L/16 | 448 | fp16-cast | 8 | b1: 46.4 img/s, 21.6 / 21.9 ms; b8: 52.8 img/s, 151.2 / 154.3 ms; b32: 51.2 img/s, 621.5 / 631.0 ms |
| ViT-L/16 | 448 | bf16-cast | 8 | b1: 44.8 img/s, 22.2 / 22.8 ms; b8: 52.0 img/s, 151.5 / 154.5 ms; b32: 50.9 img/s, 625.5 / 638.4 ms |
| ViT-L/16 | 448 | fp16-autocast | 8 | b1: 44.4 img/s, 22.6 / 22.9 ms; b8: 49.3 img/s, 161.7 / 163.2 ms; b32: 47.4 img/s, 673.0 / 676.9 ms |
| ViT-L/16 | 448 | bf16-autocast | 8 | b1: 44.0 img/s, 22.7 / 23.1 ms; b8: 48.6 img/s, 161.9 / 164.5 ms; b32: 46.4 img/s, 688.2 / 695.2 ms |
| ViT-L/16 | 512 | fp32 | 8 | b1: 11.6 img/s, 86.2 / 86.8 ms; b8: 12.2 img/s, 653.9 / 657.4 ms; b32: 12.0 img/s, 2666.5 / 2709.7 ms |
| ViT-L/16 | 512 | fp16-cast | 8 | b1: 34.8 img/s, 28.6 / 29.2 ms; b8: 38.8 img/s, 205.7 / 209.7 ms; b32: 32.8 img/s, 974.8 / 1044.1 ms |
| ViT-L/16 | 512 | bf16-cast | 8 | b1: 34.5 img/s, 28.9 / 29.6 ms; b8: 38.6 img/s, 206.6 / 209.5 ms; b32: 36.7 img/s, 872.1 / 877.3 ms |
| ViT-L/16 | 512 | fp16-autocast | 8 | b1: 33.8 img/s, 29.6 / 30.0 ms; b8: 35.9 img/s, 221.2 / 225.4 ms; b32: 34.4 img/s, 926.4 / 939.0 ms |
| ViT-L/16 | 512 | bf16-autocast | 8 | b1: 33.9 img/s, 29.4 / 29.9 ms; b8: 35.7 img/s, 222.1 / 224.8 ms; b32: 33.9 img/s, 940.0 / 952.4 ms |
| ViT-H+/16 | 224 | fp32 | 32 | b1: 18.9 img/s, 52.5 / 53.3 ms; b8: 25.1 img/s, 310.8 / 342.3 ms; b32: 26.3 img/s, 1214.2 / 1258.8 ms |
| ViT-H+/16 | 224 | fp16-cast | 32 | b1: 56.2 img/s, 17.8 / 18.2 ms; b8: 82.9 img/s, 95.3 / 96.7 ms; b32: 83.3 img/s, 380.3 / 400.9 ms |
| ViT-H+/16 | 224 | bf16-cast | 8 | b1: 53.6 img/s, 18.6 / 19.3 ms; b8: 81.3 img/s, 96.5 / 105.0 ms; b32: 80.1 img/s, 391.7 / 418.1 ms |
| ViT-H+/16 | 224 | fp16-autocast | 8 | b1: 53.2 img/s, 18.6 / 19.7 ms; b8: 78.4 img/s, 100.6 / 103.1 ms; b32: 78.1 img/s, 403.2 / 426.1 ms |
| ViT-H+/16 | 224 | bf16-autocast | 32 | b1: 53.5 img/s, 18.4 / 19.4 ms; b8: 78.0 img/s, 100.0 / 113.5 ms; b32: 85.9 img/s, 369.0 / 394.1 ms |
| ViT-H+/16 | 448 | fp32 | 8 | b1: 6.6 img/s, 151.8 / 152.4 ms; b8: 7.1 img/s, 1127.6 / 1138.2 ms; b32: 6.5 img/s, 4975.2 / 5023.3 ms |
| ViT-H+/16 | 448 | fp16-cast | 8 | b1: 21.3 img/s, 46.8 / 47.4 ms; b8: 22.6 img/s, 352.3 / 356.0 ms; b32: 18.9 img/s, 1687.0 / 1717.3 ms |
| ViT-H+/16 | 448 | bf16-cast | 8 | b1: 21.6 img/s, 46.2 / 46.6 ms; b8: 22.6 img/s, 351.0 / 363.5 ms; b32: 19.6 img/s, 1629.8 / 1652.3 ms |
| ViT-H+/16 | 448 | fp16-autocast | 8 | b1: 21.8 img/s, 45.9 / 46.6 ms; b8: 21.9 img/s, 361.8 / 366.4 ms; b32: 19.9 img/s, 1600.2 / 1634.8 ms |
| ViT-H+/16 | 448 | bf16-autocast | 8 | b1: 22.0 img/s, 45.3 / 45.7 ms; b8: 22.5 img/s, 354.4 / 358.5 ms; b32: 20.0 img/s, 1594.3 / 1636.4 ms |
| ViT-H+/16 | 512 | fp32 | 8 | b1: 4.6 img/s, 215.3 / 216.7 ms; b8: 5.2 img/s, 1545.9 / 1573.1 ms; b32: 4.7 img/s, 6827.8 / 6892.8 ms |
| ViT-H+/16 | 512 | fp16-cast | 8 | b1: 14.4 img/s, 68.5 / 70.3 ms; b8: 16.3 img/s, 488.3 / 492.1 ms; b32: 14.4 img/s, 2218.0 / 2278.3 ms |
| ViT-H+/16 | 512 | bf16-cast | 1 | b1: 14.6 img/s, 68.4 / 69.3 ms; b8: 14.3 img/s, 508.1 / 692.3 ms; b32: 13.3 img/s, 2399.8 / 2493.0 ms |
| ViT-H+/16 | 512 | fp16-autocast | 8 | b1: 14.5 img/s, 68.5 / 69.5 ms; b8: 14.7 img/s, 540.3 / 548.0 ms; b32: 12.7 img/s, 2510.6 / 2708.7 ms |
| ViT-H+/16 | 512 | bf16-autocast | 1 | b1: 15.1 img/s, 65.9 / 66.7 ms; b8: 14.6 img/s, 540.7 / 552.0 ms; b32: 12.5 img/s, 2554.6 / 2621.7 ms |
| ViT-7B/16 | 224 | fp16-cast | 8 | b1: 8.9 img/s, 110.5 / 115.8 ms; b8: 11.6 img/s, 683.7 / 693.0 ms; b32: 10.0 img/s, 3188.5 / 3401.9 ms |
| ViT-7B/16 | 224 | bf16-cast | 8 | b1: 9.1 img/s, 110.1 / 112.1 ms; b8: 12.1 img/s, 660.6 / 672.2 ms; b32: 10.3 img/s, 3093.8 / 3308.4 ms |
| ViT-7B/16 | 224 | fp16-autocast | 8 | b1: 9.0 img/s, 111.3 / 112.8 ms; b8: 11.8 img/s, 673.7 / 682.5 ms; b32: 9.7 img/s, 3262.4 / 3517.8 ms |
| ViT-7B/16 | 224 | bf16-autocast | 8 | b1: 8.4 img/s, 118.8 / 122.4 ms; b8: 11.8 img/s, 677.4 / 682.2 ms; b32: 9.5 img/s, 3394.6 / 3570.6 ms |
| ViT-7B/16 | 448 | fp16-cast | 8 | b1: 2.4 img/s, 416.6 / 460.4 ms; b8: 2.8 img/s, 2822.0 / 2913.9 ms; b32: 2.6 img/s, 12119.0 / 12234.6 ms |
| ViT-7B/16 | 448 | bf16-cast | 8 | b1: 2.5 img/s, 396.9 / 422.4 ms; b8: 2.9 img/s, 2798.8 / 2853.8 ms; b32: 2.7 img/s, 11658.1 / 11718.9 ms |
| ViT-7B/16 | 448 | fp16-autocast | 8 | b1: 2.5 img/s, 403.5 / 426.3 ms; b8: 2.7 img/s, 2921.4 / 2971.8 ms; b32: 2.6 img/s, 12144.7 / 12305.7 ms |
| ViT-7B/16 | 448 | bf16-autocast | 8 | b1: 2.4 img/s, 408.3 / 481.8 ms; b8: 2.7 img/s, 2956.1 / 3053.1 ms; b32: 2.7 img/s, 12062.3 / 12295.6 ms |
| ViT-7B/16 | 512 | fp16-cast | 8 | b1: 2.0 img/s, 486.5 / 491.3 ms; b8: 2.1 img/s, 3799.5 / 3882.5 ms; b32: 2.0 img/s, 15979.4 / 16112.9 ms |
| ViT-7B/16 | 512 | bf16-cast | 1 | b1: 2.1 img/s, 472.4 / 479.4 ms; b8: 2.1 img/s, 3814.3 / 3878.0 ms; b32: 2.1 img/s, 15443.5 / 15606.6 ms |
| ViT-7B/16 | 512 | fp16-autocast | 1 | b1: 2.1 img/s, 476.8 / 488.2 ms; b8: 2.0 img/s, 4036.5 / 4125.6 ms; b32: 2.0 img/s, 16073.2 / 16228.9 ms |
| ViT-7B/16 | 512 | bf16-autocast | 1 | b1: 2.1 img/s, 475.4 / 490.7 ms; b8: 2.0 img/s, 4082.2 / 4101.1 ms; b32: 2.0 img/s, 15770.7 / 15944.8 ms |

Table 5. The best batch size of the compiled mlx rows of the matrix (D10 option a).

| Model size | Input resolution | Batch size | Output | Rounds | Throughput median [min–max] | Median latency (ms) | p90 latency (ms) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ViT-S/16 | 224 | 1 | frame | 2 | 532.995 [520.478–545.513] | 1.84 | 1.99 |
| ViT-S/16 | 224 | 8 | frame | 2 | 1749.038 [1473.055–2025.020] | 4.70 | 4.84 |
| ViT-S/16 | 224 | 32 | frame | 2 | 2054.812 [1712.639–2396.985] | 16.07 | 16.36 |
| ViT-S/16 | 224 | 64 | frame | 2 | 2091.739 [1727.670–2455.808] | 31.56 | 32.16 |
| ViT-S/16 | 224 | 128 | frame | 2 | 2073.484 [1706.490–2440.479] | 63.53 | 64.72 |
| ViT-S/16 | 224 | 256 | frame | 2 | 2271.031 [2128.634–2413.429] | 113.22 | 114.25 |
| ViT-H+/16 | 512 | 1 | patch | 2 | 18.007 [16.948–19.066] | 55.49 | 56.52 |
| ViT-H+/16 | 512 | 2 | patch | 2 | 19.299 [17.908–20.690] | 104.07 | 105.06 |
| ViT-H+/16 | 512 | 4 | patch | 2 | 19.043 [17.397–20.690] | 211.70 | 214.51 |
| ViT-H+/16 | 512 | 8 | patch | 2 | 18.204 [15.990–20.418] | 445.41 | 452.44 |
| ViT-H+/16 | 512 | 16 | patch | 2 | 17.078 [14.451–19.706] | 961.47 | 982.89 |

Table 6. The batch-size sweep, compiled mlx in bf16-cast (D10 option c).

## 6 Autocast / cast

| Priority | Dtype | Output | Autocast / cast (mlx, compiled) |
| --- | --- | --- | --- |
| P1 | bf16 | frame | 0.951 |
| P1 | bf16 | patch | 0.919 |
| P1 | fp16 | frame | 0.960 |
| P1 | fp16 | patch | 0.901 |
| P2 | bf16 | frame | 0.968 |
| P2 | bf16 | patch | 0.968 |
| P2 | fp16 | frame | 0.969 |
| P2 | fp16 | patch | 0.968 |

Table 7. Autocast / cast on mlx after compile, median over the rounds of the paired runs (C3, R30).

## 7 Compile check, marks and the P1 profile

Groups where the compiled torch-mps row is below the Step 3 eager median of its mode:

- ViT-S/16 224 px b32 fp32
- ViT-S/16 448 px b32 fp32
- ViT-S/16 512 px b8 fp32
- ViT-S/16 512 px b32 fp32

Eager fallbacks and other marks:

- none

The P1 profile (Table 12 of the plan), ViT-S/16, 224 px, batch size 8:

- fp16-cast: compiled call 0.135 ms, call to the end of mx.eval 3.828 ms, gap 0.002 ms, host share 0.035; kernels: compiled 637, eager 841; batch size 8: 2089.1 img/s, batch size 32: 2768.6 img/s; block 1 compiled 0.493 ms.
- bf16-cast: compiled call 0.124 ms, call to the end of mx.eval 3.752 ms, gap 0.001 ms, host share 0.033; kernels: compiled 637, eager 841; batch size 8: 2126.2 img/s, batch size 32: 2779.4 img/s; block 1 compiled 0.491 ms.

| Part | mlx fp16-cast (ms) | mlx fp16-cast queued (ms) | mlx bf16-cast (ms) | mlx bf16-cast queued (ms) | torch-mps fp16-cast eager, Step 3 (ms) |
| --- | --- | --- | --- | --- | --- |
| norm1 | 0.1833 | 0.0210 | 0.1744 | 0.0209 | 0.0104 |
| qkv_proj | 0.2285 | 0.0620 | 0.2249 | 0.0470 | 0.0837 |
| rope | 0.2026 | 0.0599 | 0.2062 | 0.0668 | 0.1757 |
| sdpa | 0.1768 | 0.0227 | 0.1797 | 0.0227 | 0.0294 |
| o_proj | 0.1804 | 0.0328 | 0.1725 | 0.0216 | 0.0402 |
| layer_scale1 | 0.1740 | 0.0122 | 0.1753 | 0.0139 | 0.0336 |
| norm2 | 0.1776 | 0.0212 | 0.1784 | 0.0221 | 0.0103 |
| ff_in | 0.2229 | 0.0865 | 0.2155 | 0.0884 | 0.0734 |
| activation | 0.1948 | 0.0558 | 0.1828 | 0.0324 | 0.0150 |
| ff_out | 0.2090 | 0.0433 | 0.2284 | 0.0463 | 0.0703 |
| layer_scale2 | 0.1625 | 0.0115 | 0.1940 | 0.0143 | 0.0335 |

Table 8. The parts of block 1 at P1, each alone. mlx: compiled, 3 warm-up and 20 calls with mx.eval after each; queued: 20 calls and one mx.eval (report only). torch-mps: eager, Step 3.

## 8 Learn data

| Model size | Dtype mode | Input resolution | Batch size | Status | Finite | Throughput or value |
| --- | --- | --- | --- | --- | --- | --- |
| ViT-S/16 | fp32 | 768 | 1 | done | True | 36.0 |
| ViT-S/16 | fp32 | 768 | 8 | done | True | 36.9 |
| ViT-S/16 | bf16-cast | 768 | 1 | done | True | 99.0 |
| ViT-S/16 | bf16-cast | 768 | 8 | done | True | 107.3 |
| ViT-H+/16 | bf16-cast | 768 | 1 | done | True | 6.2 |
| ViT-H+/16 | bf16-cast | 768 | 8 | done | True | 6.0 |
| ViT-S/16 | fp32 | 1024 | 1 | done | True | 16.2 |
| ViT-S/16 | fp32 | 1024 | 8 | done | True | 15.7 |
| ViT-S/16 | bf16-cast | 1024 | 1 | done | True | 48.0 |
| ViT-S/16 | bf16-cast | 1024 | 8 | done | True | 49.5 |
| ViT-H+/16 | bf16-cast | 1024 | 1 | done | True | 3.2 |
| ViT-H+/16 | bf16-cast | 1024 | 8 | done | True | 3.0 |
| ViT-H+/16 | bf16-autocast | 768 | 1 | done | True | largest value 52389.4 |
| ViT-H+/16 | bf16-autocast | 1024 | 1 | done | True | largest value 52784.1 |
| ViT-S/16 | bf16-cast NAX off | 224 | 8 | done | True | 817.4 |
| ViT-S/16 | bf16-cast NAX on | 224 | 8 | done | True | 1536.2 |
| ViT-S/16 | bf16-cast NAX on | 224 | 8 | done | True | 1538.0 |
| ViT-S/16 | bf16-cast NAX off | 224 | 8 | done | True | 824.4 |
| ViT-H+/16 | bf16-cast NAX off | 512 | 8 | done | True | 5.8 |
| ViT-H+/16 | bf16-cast NAX on | 512 | 8 | done | True | 16.4 |
| ViT-H+/16 | bf16-cast NAX on | 512 | 8 | done | True | 16.4 |
| ViT-H+/16 | bf16-cast NAX off | 512 | 8 | done | True | 5.9 |

Table 9. The Learn data above 512 px and with the Neural Accelerators off (D11, D19).
