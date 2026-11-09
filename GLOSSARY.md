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

**RoPE tables**:
The cos and sin arrays of one patch grid, with one row for each token and one column for each value of a head. The prefix tokens get identity rows (cos 1, sin 0).

**Concat-free rotation**:
The RoPE rotation x · cos2 + reversed(x) · sin2 on a (…, 2, head_dim / 2) view, with the sign of `rotate_half` in the sin table. It gives one fused kernel.

**RoPE cache**:
The RoPE tables of each patch grid of one backbone, in the layout of the concat-free rotation (`RopeCache`). It is outside the module tree.

**SDPA**:
Scaled dot-product attention: softmax(q · kᵀ / √head_dim) · v for each head. On `mlx` one fused kernel computes it (`mx.fast.scaled_dot_product_attention`).
_Avoid_: attention kernel (alone)

**LayerScale**:
A learned factor for each channel on the output of each branch of a block.

**Feed-forward network**:
The per-token network in each block. ViT-S/16, ViT-B/16 and ViT-L/16 use an MLP with GELU. ViT-S+/16, ViT-H+/16 and ViT-7B/16 use SwiGLU.
_Avoid_: FFN (in prose), MLP (for the general case)

**SwiGLU**:
The gated feed-forward network of ViT-S+/16, ViT-H+/16 and ViT-7B/16: `down(silu(gate(x)) * up(x))`. Its hidden width is 2/3 of the width that the Meta ratio gives.

**Residual stream**:
The token sequence that passes from block to block. Each branch of a block adds its output to it.

**Branch output**:
The output of the attention branch or the feed-forward branch of a block, before LayerScale.

**Feed-forward hidden values**:
The input of `down_proj` in one block: the GELU output of an MLP, or `silu(gate_proj(x)) · up_proj(x)` of SwiGLU.

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
A Hugging Face model folder that `scripts/convert_meta_checkpoint.py` made from a Meta checkpoint, in `~/.cache/dinov3-mlx/converted/`. We use them only while a gated repo is not available.

**MLX weights**:
The weights of one model size as MLX arrays, with the names and layouts of the MLX backbone. The key map makes them at load time from the Hugging Face safetensors.

**Key map**:
The map from a Hugging Face tensor name to the name of the MLX parameter (`map_key`). It drops the mask token and reshapes the patch embedding weight.

**Shard**:
One safetensors file of a model size whose weights are in more than one file. `model.safetensors.index.json` maps each tensor name to its shard.

**Outer parts**:
The parts of the backbone outside the blocks: the embeddings, the RoPE tables and the final layer norm (`OuterParts`).

**Placeholder skeleton**:
The backbone before a load, with a zero-stride placeholder for each parameter. An evaluation of it allocates one value for each parameter.

**Quantized weights**:
The weights of an `nn.Linear` module stored as 8-bit or 4-bit integers, with one scale and one offset for each group of weights (affine format). The scales and the offsets have the matmul dtype. A quantized row keeps the name of its dtype mode and adds the field `bits`.

**Group size**:
The number of weights of one row that share one scale and one offset. Step 6 uses 64.

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

**Preprocessing**:
The steps from an RGB image to the normalized fp32 pixels: divide by 255, resize (bilinear, antialiased), normalize with the ImageNet mean and standard deviation.

**Separable resize**:
A resize as two matmuls with fixed weight matrices, one for the rows and one for the columns.

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

**TF32**:
A matmul mode that rounds the fp32 inputs to a 10-bit mantissa. MLX turns it on by default; the fp32 dtype mode of `mlx` turns it off (`MLX_ENABLE_TF32=0`).

**TF32 probe**:
A check of the real TF32 state: one fp32 matmul on the GPU whose exact result needs more than a 10-bit mantissa. The API runs `tf32_active` (the 64 × 64 identity times values 1 + k · 2⁻²⁰) once in the constructor of `Embedder`. The test `test_mlx_tf32_probe` runs one matmul with `MLX_ENABLE_TF32=0` and `=1` against NumPy float64.

**Reference output**:
The output of the Hugging Face backbone on `torch-cpu` in fp32. All parity tests compare with it.
_Avoid_: ground truth, golden output

**Streamed reference**:
A reference output computed on torch-cpu in fp32 with the weights of one block in memory at a time.

**Apple Neural Engine**:
The matrix accelerator of Apple silicon, apart from the GPU. Core ML runs a model on it.
_Avoid_: ANE (in prose), NPU

**Prior-art comparison**:
The measurement of a related public repo with our benchmark harness and our parity tests.

**Prior-art ratio**:
Our throughput divided by the throughput of a port or a Core ML export in the same round. Above 1: ours is faster.

**Core ML export**:
A Core ML model package (`.mlpackage`) of a DINOv3 or DINOv2 backbone that another party made and published.

**Compute units**:
The Core ML setting that selects the devices of a model: ALL, CPU_AND_GPU, CPU_AND_NE or CPU_ONLY.

**Spec-array input**:
The image input of a Core ML export, changed in the loaded spec to a FLOAT32 multiArray, so the stored pixels go in without 8-bit rounding.

**Port change**:
One difference between a port and the Hugging Face backbone, for example bf16 RoPE tables.

**Port-change variant**:
Our backbone with one port change, to measure the cost of that change in parity and speed.

**Transfer share**:
1 − patch throughput ÷ device throughput of one worker of `mlx`: the part of the batch time that moves the outputs to the host.
_Avoid_: host share (that term is for the profile of a compiled call)

**Baseline**:
The speed and the memory of the Hugging Face backbone on `torch-mps`. All speedups are relative to it.
_Avoid_: reference (for speed)

**Parity**:
The agreement of an output with the reference output, measured as cosine similarity.
_Avoid_: accuracy, correctness, fidelity

**Expected failure**:
A configuration that a test marks as xfail with a reason, because evidence shows that it cannot pass, for example the fp16 overflow.

**Fault bound**:
A loose parity bound that catches only faults, for example a wrong input cast. It is not a threshold.

**Cross-check**:
The parity of an `mlx` output against a torch output in the same dtype mode, without the reference output. Report only.

**Part test**:
A test of one part of a block in `mlx` against the Hugging Face module on torch-cpu, with a small model.

**Small model**:
A Hugging Face backbone with 2 blocks and random weights (`tests/small_model.py`). It has the values of ViT-S/16 or of ViT-S+/16, or the shape of ViT-7B/16 at a small embedding dimension. The oracle of the part tests. It needs no gated weights.

**Configuration**:
One combination of model size, backend, dtype, input resolution, batch size and output. In a parity test, the test image is also part of it.
_Avoid_: setting, case, run (in this sense)

**Benchmark matrix**:
The set of configurations that the benchmarks measure.

**Test image**:
One of the three public-domain photos in `tests/fixtures/images/`. The parity tests use them at 4 input resolutions each.
_Avoid_: sample image, fixture

**Stored pixels**:
The normalized fp32 pixel tensor of each test image at each input resolution, in `inputs.safetensors`. Every backend reads them, so a parity test measures only the backbone.

**Reference file**:
The safetensors file with the reference output of one model size: all tokens after the final layer norm for each configuration, the hidden states of one configuration and the metadata of the run.
_Avoid_: golden file

**Dtype mode**:
The way a backend uses a dtype: fp32, cast mode or autocast mode. The five dtype modes are fp32, fp16-cast, bf16-cast, fp16-autocast and bf16-autocast.
_Avoid_: precision mode

**Cast mode**:
A dtype mode in which the weights, the inputs and the residual stream are in fp16 or bf16.

**Autocast mode**:
A dtype mode in which the matmuls run in fp16 or bf16, and the residual stream, the layer norms and LayerScale stay in fp32. On torch, the weights stay in fp32 and `torch.autocast` casts them at each matmul. On `mlx`, the load casts the weights of the matmuls once.
_Avoid_: mixed precision (alone)

**Matmul**:
One matrix multiplication of the backbone, for example in an `nn.Linear` module or in the patch embedding.

**Matmul dtype**:
The dtype of the weights and the inputs of the matmuls in one dtype mode of `mlx`.

**Stream dtype**:
The dtype of the residual stream, the layer norms, LayerScale, the CLS token and the register tokens in one dtype mode of `mlx`.

**Eager forward pass**:
The forward pass of the MLX backbone without `mx.compile`. Each call builds the graph again.

**Compiled forward pass**:
The forward pass of the MLX backbone from `compile_backbone`: one `mx.compile` function with the weights as inputs (`inputs=model.state`) and the RoPE tables as arguments. It has one trace for each input shape.

**Trace**:
The record of the operations of one call of a compiled function for one set of input shapes and dtypes. `mx.compile` makes it at the first call and replays it at later calls.

**Fused kernel**:
One GPU kernel that `mx.compile` makes from a chain of element-wise operations.

**Fusion barrier**:
An operation that `mx.compile` does not fuse with its neighbours, for example `mx.fast.layer_norm`, a matmul, slice and concatenate.

**Token range**:
The largest absolute value in the token sequence before block 1 and after each block, for one configuration.

**Memory check**:
The comparison of the free memory with the need of a model load, before the load. A load that does not fit stops with a recorded reason.

**Memory need**:
The bytes that one model load or one benchmark configuration needs, from the model size, the dtype mode, the bits, the batch size and the token count (`required_bench_bytes`). Since Step 6 it depends on the backend. The memory check compares it with the free memory.

**MLX memory limit**:
The limit of `mx.set_memory_limit` in one process. A benchmark worker of `mlx` loads with the load need as the limit, then raises the limit to the memory need of its configuration (fields `mlx_load_limit_bytes` and `mlx_memory_limit_bytes`).

**MLX cache**:
The freed buffers that MLX keeps for later use (`mx.get_cache_memory`). They are not active memory. `mx.clear_cache()` empties it.

**Load peak**:
The largest MLX active memory (`mx.get_peak_memory`) during one model load, minus the active memory before the load. It does not include the MLX cache. It is not the load footprint peak.

**Load footprint peak**:
The lifetime maximum of the process footprint after a model load (the field `load_footprint_peak_bytes`). It is not the load peak.

**MLX peak**:
`mx.get_peak_memory()` after `mx.reset_peak_memory()`: the largest MLX allocation during one measured phase. It is not the peak memory (the lifetime maximum of the process footprint).

**Load report**:
The memory record of one `mlx` load (`LoadReport`): the need of the memory check, the load peak, the MLX cache after the load and the process footprint after the load.

**Run rule**:
One rule of the section "Run rules" of `CLAUDE.md`, for example: ViT-7B/16 never loads with fp32 weights.

**Heavy job**:
A job that loads a model, runs a benchmark or trains. One heavy job runs at a time (`CLAUDE.md`, run rules).

**GPU lock**:
The exclusive file lock that each heavy job and each light GPU job takes before it starts (`heavy_job`). Its holder also waits for jobs that do not use the lock.

**Lock-aware job**:
A job that takes the GPU lock. It writes its process id into the lock registry before it waits.

**Lock registry**:
The folder with one entry for each lock-aware process. When the holder of the GPU lock looks for other heavy jobs (`other_heavy_jobs`), it leaves out each registered process, its ancestors and its children.

**Lock log**:
The log of the GPU lock: one line for the start of the wait, one for the start of the hold and one for the end of each job.

**Benchmark worker**:
The subprocess that measures one configuration on one backend: one memory check, one model load, the warm-up and the timed batches. It writes one result line with the frame time and the patch time.

**Result line**:
One JSON line in a file of `results/`: one configuration, its measured values, the versions and the machine state.

**Probe**:
A short measurement with throw-away code during the Explore stage. Its values are evidence for a decision, not a result of the step.

**Throughput**:
Images per second in steady state, after warm-up.
_Avoid_: speed (as a number), FPS

**Reached TFLOP/s**:
The FLOPs of one batch divided by its time: the compute rate that one configuration reaches. A low value shows a fixed cost, for example at batch size 1.

**Peak memory**:
The largest memory use during one benchmark run: the lifetime maximum of the process footprint of the benchmark worker. The same measure serves torch-mps and mlx.

**Process footprint**:
The memory that macOS charges to one process, from `proc_pid_rusage`. On unified memory it includes the GPU memory of torch-mps and mlx. Its lifetime maximum is the peak memory of a benchmark worker.
_Avoid_: RSS, resident memory

**Time budget**:
The rule that ends the warm-up and the timed batches of one benchmark configuration: a minimum number of batches, a minimum time and a time limit.

**Priority configuration**:
One of the two configurations that the benchmarks measure in two paired runs of five rounds, and where each change gets its change ratio: P1 (ViT-S/16, 224 px, batch size 8, frame embedding) and P2 (ViT-H+/16, 512 px, batch size 8, patch embeddings).

**Compile row**:
A benchmark result with the backbone wrapped by `torch.compile` on `torch-mps` or by `mx.compile` on `mlx`. Since Step 6, every configuration has compile rows (ADR 0009).

**Compile check**:
The comparison of a compiled `torch-mps` row of the matrix run with the median of the Step 3 eager rows of the same configuration and dtype mode, for the frame and the patch output (`compile_check`). A compiled row below that median gets a follow-up group.

**Matrix run**:
One run of the benchmark matrix in paired groups, one round for each paired group (`results/speed-matrix.jsonl`).

**Follow-up group**:
A paired group that runs again after the matrix run for a cell that the compile check lists. It holds the eager and the compiled `torch-mps` row and the fastest valid `mlx` row of the dtype, so the speedup table can compare them in one group.

**Eager fallback**:
The eager `torch-mps` row that the driver runs in the same position when a compiled `torch-mps` worker of the matrix run ends with an error. The speedup table uses it as the baseline row of that cell, with the mark "eager torch: `torch.compile` failed".

**Gate**:
A short job that runs before a large job and decides how the large job runs. Step 6: the gate of ViT-7B/16 (job J13, phase "gate") decides whether its matrix rows on `torch-mps` run with `torch.compile` or eager.

**Batch-size sweep**:
A benchmark run over more batch sizes than the benchmark matrix has, for some model sizes and input resolutions (`results/speed-sweep.jsonl`). It shows how the throughput changes with the batch size. The best batch size of the speedup table comes from the matrix run, not from the sweep.

**Host share**:
The median time of a compiled call before `mx.eval` divided by the median time to the end of `mx.eval` (`run_profile_worker`). A low value shows that the GPU, not Python, sets the time.

**Learn data**:
Benchmark lines that only the explainers use, for example the input resolutions 768 and 1024 px and the pairs with the Neural Accelerators on and off (`results/speed-learn.jsonl`). The speedup table does not use them.

**Best batch size**:
The batch size with the highest throughput for one model size, input resolution and dtype mode.

**Round**:
One run of each configuration of a benchmark command, in order. A second round runs the whole list again. The baseline row takes the median of the runs of each configuration.

**Block profile**:
The time of each part of block 1, with each part run alone in a loop, next to its FLOP count.

**Speedup**:
The throughput of `mlx` divided by the throughput of the baseline row, from a paired run (ADR 0009). At P1 and P2 the median of the paired ratios of two runs.

**Speedup table**:
The table with the speedup of each configuration of the benchmark matrix (`results/speedup-table.md`).

**Paired ratio**:
The throughput of `mlx` divided by the throughput of torch-mps in the same round of one benchmark run, for the same configuration. The median of the paired ratios over the rounds is robust to a change of the speed of the Mac between rounds.

**Paired group**:
The workers of one model size, input resolution and batch size in one matrix run: the `torch-mps` workers and the `mlx` workers, one after the other, in a rotated order.

**Change ratio**:
The throughput after a change divided by the throughput before it, both forms in one process, in alternation and in a rotated order. The median over the rounds. A change that compares two configurations (C3, C5) runs each configuration in its own subprocess, in alternation.

**Pass rule**:
The rule that gives the verdict of a speedup. At P1 and P2: "above 1" only if the pooled median of the paired ratios and the median of each run are above 1. Outside P1 and P2: "above 1" if the one paired ratio is above 1. Else "not above 1". A quantized row that fails the thresholds of its dtype mode is "not valid".

**Inside the noise**:
The mark of a paired ratio within 1 ± 0.03, the spread inside one arm in the probe paired-noise.

**Spread between runs**:
(largest run median − smallest run median) / smallest run median, over two or more runs of the same configuration at different times.

**Unified memory**:
The one memory pool that the CPU and the GPU share on Apple silicon.

**Neural Accelerators**:
The matrix-multiply units in each GPU core of the M5 chip.

**NAX kernel**:
An MLX Metal kernel that uses the Neural Accelerators. Its name contains "nax", for example `steel_attention_nax`.

**Neural Accelerators on / off**:
The state with and without the NAX kernels. `MLX_METAL_GPU_ARCH=applegpu_g16s` turns the NAX kernels off.

**SDPA fingerprint**:
The share of the fp16 and bf16 SDPA outputs that differ from the float64 result rounded once, at a fixed shape. A high share shows the NAX attention kernel with relaxed precision. The report-only test `test_mlx_sdpa_nax_fingerprint` measures it in each default session.

**GPU matmul limit**:
The highest compute rate that the matmuls of one shape and dtype reach on the GPU. Where both backends reach it, the speedup stays near 1.

**Relaxed precision**:
The option of the NAX matrix operation that lets the hardware compute with less precision. MLX 0.32.4 turns it on in the NAX matmul and attention kernels.

## Public API

**Public API**:
The functions and classes that the README documents: `Embedder` and its errors. A change to them needs a new version.

**Embedder**:
The object of the public API. Its constructor loads and compiles one backbone; `embed_frames`, `embed_patches` and `embed` give the outputs from uint8 RGB frames.

**Padded batch**:
The last batch of an `Embedder` call, filled with zero frames to the batch size, so each input resolution needs one trace only. The outputs drop the padded rows.

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

**Deviation**:
A change of the approved plan that the build makes, for example a changed target. The build hand-over lists each deviation, and the user confirms or rejects it.

**Mutation probe**:
A deliberate fault in a copy of the code, which a test must catch. The Verify stage uses it to check that a test goes red for its red case. It is not a probe of the Explore stage.

**Default session**:
A pytest session without `DINOV3_SLOW`.

**Slow session**:
A pytest session with `DINOV3_SLOW=1`.

**Release check**:
The full default session with the weights, the slow session for the API parity and the fingerprint test on this Mac, before each tag.

**Gate**:
A step of the release that only the user can do or approve: G1 the legal check, G2 the push to the private repo, G3 the repo goes public.

**CPU stream**:
The CPU device of MLX (`mx.cpu`). Some tests run the MLX backbone there as a second device; it rounds differently from the GPU.

**Core ML package**:
One converted backbone for one model size, input resolution and batch size (`.mlpackage`, with its compiled `.mlmodelc`), in `~/.cache/dinov3-mlx/coreml/`. Its name holds every setting.

**Compute plan**:
The device and the estimated cost of each op of a compiled Core ML package, from `MLComputePlan`.

**Neural Engine rewrite**:
The exact changes to the Hugging Face backbone before the Core ML conversion that keep every value inside fp16 and the Neural Engine output right: score scale, residual-stream scale, layer norm ε / r² and the constant prefix. Its version is part of the package name (`ne3`).

**Concat fault**:
The Neural Engine writes a wrong token concat when the concat has two or more constant inputs. `constant_concats` finds such a concat in a converted graph.

**Constant prefix**:
The CLS token and the 4 register tokens as one constant of 5 rows, in front of the patch tokens (`ne_embeddings`).

**Score scale**:
The power of two (256) that divides the query before q·kᵀ. The softmax gets (z − max z) · 256.

**Residual-stream scale**:
The power of two r that divides the embeddings and the weights and bias of `o_proj` and `down_proj`. The layer norms remove it; their ε becomes ε / r².

**Batch API**:
One Core ML `predict` call with a list of batch-size-1 inputs.

**Two-stream run**:
Core ML on the Neural Engine and compiled mlx on the GPU at the same time, in two processes with a common start time.

**Stream ratio**:
The summed throughput of both sides of a two-stream run divided by the throughput of mlx alone in the same round.
