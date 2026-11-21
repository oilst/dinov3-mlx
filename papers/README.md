# Interesting papers

Papers that are relevant to dinov3-mlx. Each row gives the paper, the PDF in this folder and the reason that it is relevant.

| Paper | Authors | Date | PDF | Relevance |
| --- | --- | --- | --- | --- |
| [OmniStream: Mastering Perception, Reconstruction and Action in Continuous Streams](https://arxiv.org/abs/2603.12265) (arXiv 2603.12265) | Yibin Yan, Jilan Xu, Shangzhe Di, Haoning Wu, Weidi Xie | 2026-03-12 | [2603.12265-omnistream.pdf](2603.12265-omnistream.pdf) | Starts from DINOv3 ViT-L and makes it a causal streaming backbone: causal spatiotemporal attention with a persistent KV-cache, and 3D-RoPE that extends the 2D RoPE of DINOv3. The frozen features stay close to DINOv3 on image tasks and are better on video tasks (SSv2: 68.5 % against 54.0 %). Relevant for frame-by-frame video embedding. Code: [Go2Heart/OmniStream](https://github.com/Go2Heart/OmniStream). |
| [Who Needs Labels? Adapting Vision Foundation Models With the Metadata You Already Have](https://arxiv.org/abs/2606.05107) (arXiv 2606.05107, NeurIPS 2026) | Elouan Gardès, Seung Eun Yi, Kartik Ahuja, Théo Moutakanni, Huy V. Vo, Piotr Bojanowski, Wolfgang M. Pernice, Loïc Landrieu, Camille Couprie | 2026-06 (v2 2026-11-09) | [2606.05107-fino.pdf](2606.05107-fino.pdf) | FINO adapts DINOv3 ViT-L to a new domain without task labels. It adds metadata losses to the DINO and iBOT objective: a prototype bank for discrete metadata and a regressor for continuous metadata. Gradient reversal suppresses spurious metadata (plate, time of day). A frozen backbone with a 2.7M-parameter probe then beats full supervised fine-tuning on HPA microscopy, iWildCam, FMoW and MIMIC-CXR (iWildCam OOD F1: 52.9 against 43.4). Relevant as a label-free fine-tuning recipe; video metadata (camera, site, timestamp) fits it directly. Code: [facebookresearch/dinov3, branch FINO](https://github.com/facebookresearch/dinov3/tree/FINO). |

## Add a paper

1. Copy the PDF into this folder. Name it `<arxiv-id>-<short-name>.pdf`.
2. Add one row to the table above.
