# DINOv3 on Apple silicon

The language of this repo: the DINOv3 model family, its outputs and training method, the backends we compare, the adapters we fine-tune and the spec workflow.

## Model

**DINOv3**:
Meta's family of self-supervised vision backbones, described in arXiv 2508.10104 (August 2025).
_Avoid_: DINO v3, Dino3

**Backbone**:
A pretrained DINOv3 network that maps an image to tokens. It has no task head.
_Avoid_: encoder, feature extractor

**Model size**:
One member of the DINOv3 ViT family: ViT-S/16, ViT-S+/16, ViT-B/16, ViT-L/16, ViT-H+/16 or ViT-7B/16. The "/16" is the patch size in pixels.
_Avoid_: variant, flavor, version

**Teacher**:
The ViT-7B/16 backbone. The smaller model sizes learn from it by distillation.
_Avoid_: base model

**Student**:
A model size smaller than ViT-7B/16 that learned from the teacher by distillation.
_Avoid_: distilled variant

**LVD-1689M**:
The training dataset of DINOv3: 1,689 million curated web images.

**Patch**:
A 16 × 16 pixel square of the input image.
_Avoid_: tile, cell

**Token**:
One vector in the sequence that the blocks process.

**CLS token**:
The learned token at position 0 of the sequence. Its final value describes the whole image.
_Avoid_: class token, pooled token

**Register token**:
One of the 4 learned tokens after the CLS token. Register tokens hold global information inside the backbone. Frame embeddings and patch embeddings do not include them.
_Avoid_: storage token, register (alone)

**Patch token**:
The token of one patch.

**Prefix tokens**:
The CLS token and the 4 register tokens, at positions 0 to 4 of the sequence. RoPE does not rotate them.
_Avoid_: special tokens

**Block**:
One transformer layer: an attention branch and a feed-forward branch, each with a pre-norm, a LayerScale and a residual connection.
_Avoid_: layer (alone)

**RoPE**:
The rotary position embedding of DINOv3. It rotates the queries and keys of the patch tokens by angles that come from the 2D patch coordinates in a [-1, 1] box.
_Avoid_: positional encoding, position embedding (alone)

**LayerScale**:
A learned factor for each channel on the output of each branch of a block.

**Feed-forward network**:
The per-token network in each block. ViT-S/16, ViT-B/16 and ViT-L/16 use an MLP with GELU. ViT-S+/16, ViT-H+/16 and ViT-7B/16 use SwiGLU.
_Avoid_: FFN (in prose), MLP (for the general case)

**SwiGLU**:
The gated feed-forward network of ViT-S+/16, ViT-H+/16 and ViT-7B/16: `down(silu(gate(x)) * up(x))`. Its hidden width is 2/3 of the width that the Meta ratio gives.

**Embedding dimension**:
The length of each token vector of a model size: 384, 768, 1024, 1280 or 4096.
_Avoid_: hidden size, width, d_model

## Weights

**Port**:
A public implementation of DINOv3 in another framework, for example MLX. `specs/01-scope/explore.html` compares the known ports.
_Avoid_: re-implementation, clone

**Gated repo**:
A Hugging Face repo that gives its files only to accounts that the owner approved. All six DINOv3 model repos are gated.

**Hugging Face cache**:
The folder where `huggingface_hub` stores downloaded files, `~/.cache/huggingface/hub` by default. The weights live there, outside the repo.

**Meta checkpoint**:
A `.pth` weight file of the official DINOv3 code, with the Meta weight names, for example `blocks.0.attn.qkv.weight`.
_Avoid_: original weights, raw weights

**Converted weights**:
A Hugging Face model folder that `scripts/convert_meta_checkpoint.py` made from a Meta checkpoint, in `~/.cache/dinov3-apple/converted/`. We use them only while a gated repo is not available.

## Outputs

**Frame embedding**:
The CLS token after the final layer norm. One vector for each input image.
_Avoid_: image embedding, global embedding, CLS embedding, pooler output

**Patch embeddings**:
The patch tokens after the final layer norm, as a grid of (H/16) × (W/16) vectors. The CLS token and the register tokens are not part of it.
_Avoid_: dense features, feature map, patch features

**Input resolution**:
The height and width in pixels of the image that enters the backbone. Both are multiples of 16.
_Avoid_: image size, crop size

## Training method

**Iteration**:
One optimizer update during training.
_Avoid_: step (in this sense)

**Online network**:
The network that self-supervised training updates by gradient descent.
_Avoid_: student (in this sense)

**EMA network**:
A copy of the online network whose weights are a moving average of the online weights. It gives the training targets.
_Avoid_: teacher (in this sense)

**DINO loss**:
The image-level loss: the online network's output for one crop must match the EMA network's output for another crop of the same image.

**iBOT loss**:
The patch-level loss: the online network sees an image with masked patches and must match the EMA network's outputs for those patches.

**KoLeo regularizer**:
A loss term that spreads the CLS outputs of a batch uniformly in the feature space.

**Gram matrix**:
The matrix of all pairwise dot products of the L2-normalized patch embeddings of one image.

**Gram teacher**:
An earlier checkpoint of the EMA network that gives the target Gram matrix.

**Gram anchoring**:
A loss that keeps the Gram matrix of the online network close to the Gram matrix of the Gram teacher. It stops the degradation of patch embeddings in long training.

**High-resolution adaptation**:
10k extra training iterations with large crops (512 and 768 pixels) for high input resolutions.

**Distillation**:
Training a student to reproduce the outputs of the teacher.

**RoPE-box jittering**:
A training augmentation that scales the RoPE coordinate box from [-1, 1] to [-s, s], with a random s between 0.5 and 2.

## Backends and measurement

**Backend**:
The software that runs the backbone: `torch-cpu`, `torch-mps` or `mlx`.
_Avoid_: engine, runtime, framework (in this sense)

**Reference output**:
The output of the Hugging Face backbone on `torch-cpu` in fp32. All parity tests compare with it.
_Avoid_: ground truth, golden output

**Baseline**:
The speed and the memory of the Hugging Face backbone on `torch-mps`. All speedups are relative to it.
_Avoid_: reference (for speed)

**Parity**:
The agreement of an output with the reference output, measured as cosine similarity.
_Avoid_: accuracy, correctness, fidelity

**Configuration**:
One combination of model size, backend, dtype, input resolution, batch size and output.
_Avoid_: setting, case, run (in this sense)

**Benchmark matrix**:
The set of configurations that the benchmarks measure.

**Throughput**:
Images per second in steady state, after warm-up.
_Avoid_: speed (as a number), FPS

**Peak memory**:
The largest memory use during one benchmark run, as the backend reports it.

**Speedup**:
The throughput of `mlx` divided by the throughput of the baseline, for the same configuration.

**Unified memory**:
The one memory pool that the CPU and the GPU share on Apple silicon.

**Neural Accelerators**:
The matrix-multiply units in each GPU core of the M5 chip.

## Fine-tuning

**Adapter**:
The small set of trainable weights that fine-tuning adds to the frozen backbone.
_Avoid_: delta, plugin, PEFT module

**LoRA**:
An adapter that adds a low-rank product B·A, scaled by alpha / r, to a frozen weight matrix.

**DoRA**:
An adapter that splits each frozen weight matrix into a magnitude vector and a direction matrix. It applies a LoRA update to the direction and learns the magnitude separately.

**Target modules**:
The weight matrices that get an adapter.

**Reference fine-tuning setup**:
The LoRA settings of our production video classifier. Step 8a reproduces it in MLX.

**Task quality**:
The metric of the downstream task on its evaluation data, for example the accuracy.
_Avoid_: performance (for quality)

## Workflow

**Step**:
One row of `roadmap.md`, with its own spec folder.
_Avoid_: phase, milestone, stage, task

**Spec loop**:
The six stages of one step: explore, plan, build, verify, learn and close.

**Spec**:
The HTML files of one step: `explore.html`, `plan.html` and `verify.html`.
_Avoid_: PRD, design doc

**Steering files**:
The Markdown files that every session reads: `mission.md`, `roadmap.md`, `tech-stack.md`, `GLOSSARY.md`, `CLAUDE.md` and `CLAUDE.local.md`.

**ADR**:
An architecture decision record: one short file in `docs/adr/` for a hard-to-reverse decision.

**Explainer**:
An HTML page in `docs/learn/` that explains one part of DINOv3 with diagrams and annotated code.
_Avoid_: tutorial, write-up

**Acceptance criterion**:
A numbered, measurable condition in `plan.html` that `verify.html` checks.
