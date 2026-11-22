# Challenges

Open or recent machine learning challenges that can be an example task for dinov3-mlx. Each entry gives the task, the status, the best known methods and what we must build.

## 0. Overview

### 0.1 What we look for

An example task for this repo has these properties:

- The input is video (or a sequence of images). Medical data is a bonus.
- DINOv3 frame embeddings or patch embeddings help.
- The data fits on a Mac (about 100 GB or less).
- The challenge is open, or its evaluation server still accepts submissions.
- Bonus: metadata (site, scanner, device) and unlabelled data, for FINO-style adaptation (Section 1.3.1).

### 0.2 Candidates (status on 2026-11-22)

| Challenge | Domain | Video | Status | Section |
|---|---|---|---|---|
| TrackRAD2025 | Tumor tracking in cine-MRI | yes | Main challenge finished. Post-challenge phase open until 2026-12-31. | 1 |
| AMD Global Physical AI Behavioral Safety (Zindi) | Robot safety from RGB-D frames | yes | Phase 1 open until 2027-02-15 | 2 |
| Cell Tracking Challenge | Cell segmentation and tracking in time-lapse microscopy | yes | Ongoing, evaluated each month | 3 |
| Others | See Section 4 | | | 4 |

What the search showed:

- Kaggle had no open video or medical vision competition (featured or community) in November 2026.
- Most medical video challenges (EndoVis, SurgVU, MedVidU, UUSIVC, EchoRisk) run for MICCAI or ECCV and close in the autumn. The ISBI 2027 and MICCAI 2027 lists come out in early 2027.
- Long-running evaluation servers (Grand Challenge post-challenge phases, Cell Tracking Challenge) are the best source of open medical tasks.

## 1. TrackRAD2025: real-time tumor tracking in cine-MRI

### 1.1 Facts

| Item | Value |
|---|---|
| Web site | [trackrad2025.grand-challenge.org](https://trackrad2025.grand-challenge.org/) |
| Venue | MICCAI 2025 |
| Task | Segment the tumor in every frame of a sagittal 2D cine-MRI sequence. A doctor gives the tumor mask on the first frame only. |
| Input | 2D cine-MRI at 1 to 8 frames per second, from MRI-linacs at 0.35 T and 1.5 T |
| Data | 6 hospitals (3 Dutch, 1 German, 1 Australian, 1 Chinese). 477 unlabelled cases and 108 labelled cases (a mask on every frame). |
| Test | Private: 8 cases (preliminary phase), 50 cases (final phase) |
| Metrics | Dice, HD95, mean surface distance, 2D centre error, dose accuracy (relative D98) under simulated beam tracking, inference speed |
| Submission | A Docker container. Grand Challenge runs it on a Linux machine. |
| Status | The main challenge is finished. The post-challenge phase accepts submissions until 2026-12-31 00:00, "as long as the computational budget allows". |
| Papers | [Dataset paper](https://arxiv.org/abs/2503.19119), [challenge report](https://pubmed.ncbi.nlm.nih.gov/42191418/) |

### 1.2 Best known methods

The challenge report says: fine-tuned promptable foundation models are the current best method. The top 5 teams reach Dice > 0.87 and centre error < 2.1 mm. This is about the agreement between two doctors.

| Rank | Team | Method | Result |
|---|---|---|---|
| 1 | Track'n'Treat | MedSAM2, fine-tuned. The first-frame mask is the prompt. Rank-weighted average of 3 checkpoints. [Code and weights](https://zenodo.org/records/22110707) | Dice 0.891, 0.04 s per frame |
| 2 | Hkini | SAM2 (tiny backbone), 3-stage semi-supervised fine-tuning that also uses the unlabelled cases. [Code](https://github.com/togetherhkl/TrackRAD2025) | Dice 0.886 |
| 3 | felixknispelrwth (CoTrackRAD) | CoTracker3 point tracker | not found |
| 4 | Cit | not found | not found |
| 5 | Reg'n'Track | Hierarchical SAM2 | not found |
| 6 | (paper) | SAM2.1, fine-tuned on the labelled cases only. [arXiv 2510.25990](https://arxiv.org/abs/2510.25990) | Dice 0.879 |

SAM2 works like this: an image encoder, a memory bank with the features and masks of earlier frames, memory attention, and a mask decoder. Our plan copies this structure, with DINOv3 as the image encoder.

### 1.3 What we must build

We need four parts:

1. **Adaptation (FINO style).** Adapt DINOv3 to cine-MRI without labels.
2. **Segmentation.** Make a pixel mask from the patch embeddings.
3. **Tracking.** Carry the mask from frame to frame.
4. **Evaluation and submission.** Score locally, then submit a Docker container.

#### 1.3.1 Adaptation (FINO style)

Cine-MRI is grayscale and looks very different from the web images of LVD-1689M. Frozen patch embeddings can be weak. FINO ([arXiv 2606.05107](papers/2606.05107-fino.pdf)) adapts the backbone without task labels.

- **Data:** the frames of the 477 unlabelled cases. The frames of the 108 labelled cases can also be used, without their masks.
- **Losses:** the DINO loss on the CLS token, the iBOT loss on the patch tokens, and the SIGReg regularizer, as in FINO. Roadmap Step 8 needs the same losses.
- **Metadata to suppress (gradient reversal):** hospital and field strength. These cause the domain shift.
- **Metadata to encourage:** possibly the anatomy site of the tumor (lung, liver, pancreas, and others), if the data gives it. Check the metadata files.
- **Optional temporal signal:** two frames of the same sequence form a positive pair. Then motion does not change the embedding of the same tissue.
- **Check:** compare the propagation baseline (1.3.3) with frozen and with adapted weights. The adaptation must give a better Dice on a held-out hospital.

#### 1.3.2 Segmentation: how we get the pixel mask

The main problem is resolution. DINOv3 gives one patch embedding for each 16 × 16 pixel patch. A tumor in cine-MRI is small, so at native resolution it covers only a few patches. A mask on the patch grid gives a poor Dice at the tumor border.

We use three measures together:

1. **Larger input resolution.** Upsample each frame (for example 2× to 4×) before the backbone. Then one patch covers 4 to 8 original pixels, and the tumor covers many patches. Step 6 measured the cost of large input resolutions. RoPE makes the backbone work at any input resolution.
2. **A patch-level mask prior from propagation (no training).** For each patch of the new frame, find the most similar patches in the memory frames (cosine similarity of patch embeddings, top-k, softmax). Take the weighted mean of their mask values. This gives a soft mask on the patch grid. This is the label propagation method of the DINO papers for video object segmentation.
3. **A small learned mask decoder.** It turns the coarse prior into a pixel mask. Train it on the 108 labelled cases.
   - **Inputs:** the patch embeddings of the new frame, the soft mask prior of step 2, and the raw frame at full resolution (a skip path, because the patch grid has no fine border detail).
   - **Structure:** 2 to 3 upsampling stages (each ×2, a 3 × 3 convolution and a norm), then a 1 × 1 convolution to one logit for each pixel.
   - **Loss:** Dice loss plus binary cross-entropy.
   - **Size:** small (well under 1M parameters), so it trains fast in MLX and runs fast.

Simplest version: steps 1 and 2 only, with a bilinear upsampling of the soft mask and a threshold at 0.5. This is the baseline. The decoder must beat it.

Other options for the future: a learned feature upsampler for the patch embeddings, or a SAM2-style mask decoder with the first-frame mask as a prompt.

#### 1.3.3 Tracking

- **Memory:** the first frame (its mask comes from the doctor, so it is always correct) plus the last N predicted frames (for example N = 5).
- **Spatial window:** search for matches only near the last tumor position. Breathing motion is mostly a few centimetres along the head-foot axis. This removes false matches in far tissue.
- **Temporal smoothing:** limit the jump of the mask centre between two frames.
- **Real time:** each frame is used once. The backbone runs once for each frame, and the memory keeps the patch embeddings of earlier frames.

#### 1.3.4 Evaluation and submission

- **Local evaluation:** split the 108 labelled cases by hospital. Leave one hospital out, to measure the domain shift. Compute Dice, HD95, centre error and milliseconds per frame. Use the official evaluation code of the [challenge repository](https://github.com/LMUK-RADONC-PHYS-RES/trackrad2025).
- **Submission:** the container runs on Linux. MLX does not run there. We need a PyTorch inference path with the same weights (the Step 2 reference code and the weight converter can help). Speed on Grand Challenge is then not the speed on the Mac.
- **Comparison:** the Dice of the winners (0.891) is the target. The Mac speed in milliseconds per frame is our own result.

### 1.4 OmniStream as the backbone

[OmniStream](papers/2603.12265-omnistream.pdf) (arXiv 2603.12265) makes DINOv3 ViT-L a causal streaming backbone.

#### 1.4.1 What OmniStream changes

- **Causal spatiotemporal attention.** All tokens of frame t attend to all tokens of frames ≤ t, never to later frames. Inside one frame, attention stays full (all patches see all patches).
- **KV-cache.** When frame t arrives, the model computes the queries, keys and values of frame t only. It reuses the cached keys and values of frames ≤ t − 1. The output is equal to full causal attention over the whole stream.
- **3D-RoPE.** The head dimension is split 2:3:3 over (t, y, x). Every 4th RoPE index (i ≡ 3 mod 4) encodes time; the other indices keep the 2D RoPE of DINOv3.
- **Tokens for each frame:** 1 CLS token, 4 register tokens, an optional camera token, then the patch tokens.
- **Training:** initialized from DINOv3 ViT-L. Losses: DINO, iBOT, KoLeo and Gram anchoring (the DINOv3 losses), plus depth, ray, point and camera losses, plus captioning through a small LLM (Qwen3-0.6B). 200M frames from 29 datasets, 64 H200 GPUs, 60K steps at 224 px and 120K steps at 512 px, clips of T = 16 frames.

#### 1.4.2 Results that matter for us

| Benchmark | OmniStream | DINOv3 ViT-L |
|---|---|---|
| ADE20K segmentation (mIoU) | 49.1 | 51.5 |
| DAVIS'17 video object segmentation (J&F) | 71.6 | 73.2 |
| SSv2 action recognition (accuracy) | 68.5 | 54.0 |

On DAVIS'17, the task closest to TrackRAD (mask propagation from the first frame), OmniStream is slightly **worse** than DINOv3 ViT-L. Its gain is in motion understanding (SSv2), not in dense masks.

#### 1.4.3 Does the KV-cache reduce the FLOPs?

**Not against plain DINOv3.** The KV-cache does not skip image regions that did not change.

- Each new frame runs through all blocks, for all of its patch tokens. The patch embedding, the projections, the feed-forward networks and the attention inside the frame cost the same as one DINOv3 forward pass.
- Attention over the cache is extra work. It grows linearly with the number of cached frames.
- The cache saves work only against the alternative of a **bidirectional** model that computes the whole clip again for each new frame.

The paper's own numbers (ViT-L, 224 px, batch size 1, one H800 GPU, Table 9):

| Context length T | 16 | 32 | 64 | 128 | 512 |
|---|---|---|---|---|---|
| Recompute the full clip (s per frame) | 0.125 | 0.329 | 0.998 | out of memory | out of memory |
| OmniStream with KV-cache (s per frame) | 0.042 | 0.057 | 0.067 | 0.115 | 0.414 |
| OmniStream memory (GB) | 3.6 | 4.8 | 7.3 | 12.1 | 41.5 |

The "15× faster" claim compares with full recompute. The latency for each frame grows with T. For TrackRAD we would keep a short cache window (the first frame plus the last few frames).

#### 1.4.4 Skip the regions that did not change

The idea "compute only the tokens of regions that changed" is a different method. It is called token reuse or delta computation. One known example is [Eventful Transformers](papers/2308.13494-eventful-transformers.pdf) (Dutson et al., ICCV 2023): each block computes only the tokens whose input changed more than a threshold, and reuses the cached output of the other tokens. This gives approximate outputs, and the FLOPs fall with the share of changed tokens.

For TrackRAD it is a weak fit:

- Breathing moves most of the anatomy in the frame, not only the tumor.
- MRI noise changes every pixel from frame to frame, so a threshold is necessary, and it costs accuracy.
- Attention mixes all tokens of a frame, so a change in one region changes the output of other tokens. Reuse is only approximate.

Facts from the paper: Eventful Transformers keep, for each token, a reference of its last computed input and output. A gate selects the tokens with the largest change (top-r, or a threshold). Only these tokens go through the projections and the feed-forward network; the attention is also updated only for them. An existing ViT often converts without new training. The paper reports 2 to 4 times less compute with small accuracy losses (ViTDet on ImageNet VID, EPIC-Kitchens 100). The number r gives control over the compute at run time.

Where it fits better: video with a static camera and a small moving part, for example the camera of the AMD challenge in a quiet scene, or time-lapse microscopy with a fixed field of view (Section 3). Roadmap Step 9 builds it.

A simpler way to save FLOPs for TrackRAD: **a region of interest.** Run the backbone only on a window around the last tumor position (plus a margin for breathing motion). This cuts the tokens for each frame by a large factor, and it also allows a higher input resolution inside the window. The first frame can still be embedded once in full.

#### 1.4.5 What we can use, for inference and for training

| Use | Value for TrackRAD | Cost |
|---|---|---|
| OmniStream weights as a frozen backbone, with our mask decoder | Temporal context inside the patch embeddings. But DAVIS'17 is lower than DINOv3. | ViT-L only, about 0.04 s per frame on an H800 at 224 px; slower on a Mac and at higher input resolution |
| The causal KV-cache design for our own memory | Same output as the memory bank in 1.3.3, but inside the backbone | We must add causal attention and 3D-RoPE to our MLX model |
| The causal mask in training | One clip of T frames gives supervision for every context length from 1 to T. This is the main training-efficiency gain. | Training on clips needs more memory than training on single frames, because gradients flow through all frames |
| The DINOv3 losses (DINO, iBOT, KoLeo, Gram anchoring) | The same losses that FINO and our Step 8 need | Shared work |

Recommendation:

1. Build the DINOv3 baseline first (1.3.2 and 1.3.3), with a region of interest.
2. Then compare DINOv3 ViT-L and OmniStream as the frozen backbone, with the same decoder and the same cache window.
3. Consider streaming FINO training (causal clips of cine-MRI with the FINO metadata losses) only if step 2 shows a clear Dice gain for OmniStream. Roadmap Step 10 builds our own streaming backbone in MLX.

### 1.5 Open questions

1. What is the frame size in pixels, and how large is the tumor in pixels? This decides the upsampling factor.
2. Which metadata does the data give (hospital, field strength, anatomy site, patient)?
3. What is the data license? Can we put small examples into this public repo?
4. Does the post-challenge phase stay open after 2026-12-31?
5. Which input resolution and region of interest give the best trade-off between Dice and milliseconds per frame on the Mac?

### 1.6 Roadmap steps for TrackRAD

| Step | Content |
|---|---|
| 9 Eventful frame embedding | Token reuse for video (Section 1.4.4) |
| 10 Streaming backbone | Causal attention, KV-cache and 3D-RoPE on top of DINOv3, like OmniStream (Section 1.4) |
| 11 Mask tracking | Mask propagation, mask decoder, memory, region of interest, evaluation, PyTorch path for the submission (Sections 1.3.2 to 1.3.4) |
| 12 FINO | Metadata-guided adaptation without labels (Section 1.3.1) |

## 2. AMD Global Physical AI Behavioral Safety Challenge

### 2.1 Facts

| Item | Value |
|---|---|
| Web site | [zindi.world/competitions/amd-global-physical-ai-safety-challenge](https://zindi.world/competitions/amd-global-physical-ai-safety-challenge) |
| Task | A Spot robot moves in a furnished Habitat scene with a moving person and obstacles. Predict the risk of a safety event (contact, or clearance below 0.20 m) at 1, 2 and 3 s ahead, for 3 commands: continue, slow to 35 %, stop. 9 values for each episode. |
| Input for each episode | 9 RGB frames (192 × 256 px) and 9 depth frames at 10 Hz (the last 0.8 s), camera intrinsics and poses, robot telemetry (position, heading, speed command, goal), the pending yaw-rate command |
| Encounters | 12 encounter families: crossing, following, delayed crossing, hesitation, change of direction and others |
| Target | The fraction of 64 simulated futures with a safety event |
| Metric | 1 − the mean RMSE of the 9 columns |
| Train data | Labels for 864 reference episodes. The RGB-D frames must be generated with the supplied Habitat generator; more episodes come from new seeds. Train metadata: scene, avatar, encounter. |
| Test data | 2,160 episodes, 2.3 GB, for inference only. Train and Test share the buildings and the avatars. |
| Timeline | Phase 1 from 2026-11-09 to 2027-02-15. The top 40 go to Phase 2 (2027-02-14 to 2027-03-14): code that controls the robot in new simulated encounters. |
| Prizes | USD 10,000, for Phase 2 only. Winners must join the AMD AI Developer Program, pass the ROCm certificate and use AMD hardware or software in the code. |
| Leaderboard (2026-11-22) | Best public score 0.900; organizer benchmark 0.740; about 146 active teams |

### 2.2 Known approaches

A forum post: a DETR detector finds the person, then the depth, the telemetry and the change rates go into LightGBM. This gives about 0.88 to 0.90.

### 2.3 What we would build

- **Patch embeddings** of each RGB frame locate the person (in place of the DETR detector). Depth at those patches gives the distance.
- **Frame embeddings** of the 9 frames feed a small temporal head, together with the telemetry.
- **Eventful frame embedding** (Step 9) fits well: the camera moves slowly, and the frames are small.
- **FINO** fits only moderately: the encounter family can be informative metadata, but Train and Test share the buildings and the avatars.

### 2.4 Blockers

1. The generator is tested only on Linux x86-64 with Habitat-Sim 0.3.3 and EGL rendering. We need a Linux machine to generate the training data.
2. The prize rules need AMD tools, which do not fit an MLX repo. Entry without a prize is possible.
3. The upstream 3D assets keep their own licenses. Check them before any frame goes into this repo.

## 3. Cell Tracking Challenge

| Item | Value |
|---|---|
| Web site | [celltrackingchallenge.net](https://celltrackingchallenge.net/) |
| Task | Segment and track cells and nuclei in 2D and 3D time-lapse microscopy videos. Three benchmarks: tracking, segmentation only, linking only. |
| Data | Real and synthetic datasets from many microscopes and cell types, free to download after registration |
| Status | Ongoing. Submissions are evaluated each month. The Cell Linking Benchmark is open since 2026-10. |

Why it fits:

- It is a biomedical video task with no end date, so it stays a stable example task.
- The camera is fixed and most of the frame changes little: a good fit for eventful frame embedding (Step 9).
- The datasets differ in microscope and cell type: good metadata for FINO (Step 12).
- Mask tracking (Step 11) applies, but many cells must be tracked at the same time, and cells divide.

Not checked yet: the data size, the label format, and the license.

## 4. Other candidates

### 4.1 Open (status on 2026-11-22)

| Challenge | Domain | Status | Note |
|---|---|---|---|
| [Light My Cells 2](https://lightmycells2.grand-challenge.org/) (ISBI 2027) | Fluorescence from transmitted-light microscopy | Final phase 2026-12-01 to 2026-12-13 | A bonus for metadata-aware training; good FINO fit |
| [SELMA3D 2026](https://selma3d2026.grand-challenge.org/) | Self-supervised 3D light-sheet segmentation | Accepting submissions | Large unlabelled set |
| Cell Segmentation in Multi-modality Microscopy (NeurIPS 2022, Grand Challenge) | Cell segmentation | Long-term leaderboard | Many modalities, unlabelled images |
| crossMoDA 2022 (Grand Challenge) | MRI domain adaptation | Long-term leaderboard | Domain shift without target labels |
| [MEGC2022](https://megc2022.grand-challenge.org/) | Facial micro-expression spotting in long videos | Live leaderboard | Video; data needs a signed license form |
| [MedVidBench leaderboard](https://huggingface.co/spaces/UII-AI/MedVidBench-Leaderboard) | Surgical and nursing video | Live | Built for video-language models |
| [Impostor Avatar Challenge](https://dpai-cvpr2027.github.io/) (CVPR 2027) | Talking-head video forensics | Starts 2026-12-15 | Video |
| [DrivenData Conser-vision](https://www.drivendata.org/competitions/) | Camera-trap images with site metadata | Practice, open about 1 year | Close to iWildCam in the FINO paper |
| [Solafune Tree Canopy Detection](https://community.solafune.com/) | Satellite images | Late submissions | DINOv3 has a satellite model |

### 4.2 Closed, with public data (useful as benchmarks)

| Challenge | Domain | Note |
|---|---|---|
| SAR-RARP50 (EndoVis 2022) | Robotic prostatectomy video: action recognition and tool segmentation | Data public |
| PhaKIR 2024 (EndoVis) | Cholecystectomy video: phase, keypoints, tools | Data on request |
| Capsule Vision 2024 | Capsule endoscopy frames | Data on Figshare |
| EchoRisk (MICCAI 2026) | Echocardiography video, ejection fraction | A DINOv2 paper uses it |
| DFL Bundesliga Data Shootout (Kaggle 2022) | Football event detection in video | Kaggle data |
| SoccerNet ball action spotting | Football broadcast video | Yearly challenge |

