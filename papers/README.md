# Interesting papers

Papers that are relevant to dinov3-mlx. Each row gives the paper, the PDF in this folder and the reason that it is relevant.

| Paper | Authors | Date | PDF | Relevance |
| --- | --- | --- | --- | --- |
| [OmniStream: Mastering Perception, Reconstruction and Action in Continuous Streams](https://arxiv.org/abs/2603.12265) (arXiv 2603.12265) | Yibin Yan, Jilan Xu, Shangzhe Di, Haoning Wu, Weidi Xie | 2026-03-12 | [2603.12265-omnistream.pdf](2603.12265-omnistream.pdf) | Starts from DINOv3 ViT-L and makes it a causal streaming backbone: causal spatiotemporal attention with a persistent KV-cache, and 3D-RoPE that extends the 2D RoPE of DINOv3. The frozen features stay close to DINOv3 on image tasks and are better on video tasks (SSv2: 68.5 % against 54.0 %). Relevant for frame-by-frame video embedding. Code: [Go2Heart/OmniStream](https://github.com/Go2Heart/OmniStream). |

## Add a paper

1. Copy the PDF into this folder. Name it `<arxiv-id>-<short-name>.pdf`.
2. Add one row to the table above.
