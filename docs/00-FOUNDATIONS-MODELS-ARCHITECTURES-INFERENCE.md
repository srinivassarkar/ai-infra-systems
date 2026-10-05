# 00 — Foundations: Models, Architectures, Attention & Inference Systems

A complete, first-principles systems engineering taxonomy of large language models, silicon interactions, memory economics, and serving runtimes.

---

## Table of Contents
1. [Core Transformer Architectures (The Physical Structure)](#1-core-transformer-architectures)
2. [Attention Mechanisms & KV Cache Economics](#2-attention-mechanisms--kv-cache-economics)
3. [Model Modalities & Ingress Types](#3-model-modalities--ingress-types)
4. [The Token Lifecycle: Text to Silicon Numbers](#4-the-token-lifecycle)
5. [Logits, Probabilities & Sampling Controls](#5-logits-probabilities--sampling-controls)
6. [Model Serialization & Weight Storage Formats](#6-model-serialization--weight-storage-formats)
7. [Positional Encodings & Context Extension (RoPE)](#7-positional-encodings--context-extension)
8. [Anatomy of a Single Transformer Layer](#8-anatomy-of-a-single-transformer-layer)
9. [Advanced Serving Optimizations](#9-advanced-serving-optimizations)
10. [The Roofline Model & Arithmetic Intensity](#10-the-roofline-model--arithmetic-intensity)
11. [The Complete End-to-End Execution Trace](#11-the-complete-end-to-end-execution-trace)
12. [The Production Engineering Decision Matrix](#12-the-production-engineering-decision-matrix)
13. [CUDA Execution Realities: Streams, Allocators & Timing Traps](#13-cuda-execution-realities)
14. [Quantization Mechanics: Scale, Zero-Point & Activation Outliers](#14-quantization-mechanics)
15. [Distributed Multi-GPU Serving: TP All-Reduce vs PP Bubbles](#15-distributed-multi-gpu-serving)
16. [Structured Outputs & Constrained Decoding (Grammar Masking)](#16-structured-outputs--constrained-decoding)
17. [Model Cold Starts, mmap & Linux Page Cache Dynamics](#17-model-cold-starts-mmap--linux-page-cache)
18. [Ingress Traps: HTTP/2 Multiplexing & Client Abort Propagation](#18-ingress-traps)
19. [KV Cache Capacity Planning, Preemption & Swapping Dynamics](#19-kv-cache-capacity-planning-preemption--swapping-dynamics)
20. [Disaggregated Prefill-Decode Architecture (Split-Phase Serving)](#20-disaggregated-prefill-decode-architecture-split-phase-serving)
21. [Speculative Decoding Mechanics & Verification Economics](#21-speculative-decoding-mechanics--verification-economics)
22. [Multi-LoRA Serving Runtimes & Dynamic Adapter Paging](#22-multi-lora-serving-runtimes--dynamic-adapter-paging)
23. [Observability, DCGM & True Hardware Saturation Signals](#23-observability-dcgm--true-hardware-saturation-signals)

---

## 1. Core Transformer Architectures

Inference behavior is fundamentally dictated by how the model's layers are structurally connected:

```
[ 1. Encoder-Only ]     ──▶ Bidirectional. 1 forward pass. ZERO KV cache. (Embeddings / BERT)
[ 2. Decoder-Only ]     ──▶ Causal. Prefill + Autoregressive Decode. KV cache grows. (Llama / Qwen)
[ 3. Encoder-Decoder ]  ──▶ Input encoded once, decoded autoregressively. (Whisper / T5)
```

### A. Decoder-Only (95% of What You Will Ever Serve)
* **Examples:** Llama 3/3.1/3.2, Qwen 2.5, DeepSeek, Mistral, GPT-4.
* **Mechanism:** Uses **causal masking**—each token can only attend to past tokens, never future tokens.
* **Inference Reality:**
  * Executes in two distinct phases: **Prefill** (compute-bound matrix multiplication) $\rightarrow$ **Decode** (memory-bandwidth-bound autoregression).
  * Requires a **Key-Value (KV) Cache** that dynamically expands in silicon memory with every newly generated token.
  * All capacity planning, batch scheduling, and OOM prevention exist because of this architecture.

### B. Encoder-Only (Representation & Search)
* **Examples:** BERT, RoBERTa, BGE, E5, ColBERT.
* **Mechanism:** **Bidirectional attention**—every token attends to every other token simultaneously.
* **Inference Reality:**
  * **There is NO autoregressive decode loop.**
  * The model executes **exactly ONE forward pass** across the input text and outputs a dense vector of floating-point numbers (e.g., 768 or 1,536 dimensions).
  * **Zero KV cache is allocated.** Memory footprint is 100% static and predictable.
  * Exceptionally easy to scale: you can batch 64–128 requests together to fully saturate GPU Tensor Cores without worrying about memory bloat or streaming proxies.

### C. Encoder-Decoder (Translation & Audio)
* **Examples:** T5, Whisper (Speech-to-Text), original Vaswani 2017 Transformer.
* **Mechanism:** An Encoder processes the input context (audio spectrogram or source text), and a Decoder generates output tokens autoregressively.
* **Inference Reality:** Requires **two separate KV caches**: the Decoder self-attention cache plus a *Cross-Attention cache* referencing the Encoder's hidden representations.

---

### D. Dense vs Mixture-of-Experts (MoE)

| Dimension | Dense Models (e.g. Llama 3 8B, Qwen 7B) | Mixture-of-Experts (e.g. Mixtral 8x7B, DeepSeek V3) |
| :--- | :--- | :--- |
| **Active Parameters** | **100% of parameters** activate for every token (8B params = 8B math ops per token). | A router routes each token to only **$K$ out of $N$ experts** (e.g. 2 of 8 experts = ~13B active params out of 47B total). |
| **VRAM Footprint** | Fits base weights (~16 GB in FP16 for an 8B model). | **Must store ALL experts in VRAM.** A 47B MoE requires ~90 GB+ VRAM in FP16, even though only 13B activate per step. |
| **Decode Latency** | Generates tokens at the speed of an 8B model. | Generates tokens at the speed of a 13B model (low latency), but requires the memory pool of a 47B model. |
| **Systems Verdict** | Standard compute-to-memory ratio. | **Memory-capacity heavy, bandwidth-efficient.** Ideal for high-memory unified pools (e.g. 256GB Mac Studio M3 Ultra). |

---

## 2. Attention Mechanisms & KV Cache Economics

The attention mechanism determines how many Megabytes of KV cache each token consumes:

```
MHA (1:1 Ratio)           GQA (Grouped - 4:1 or 8:1)          MQA (All-to-1)
Q1 Q2 Q3 Q4               [Q1 Q2]  [Q3 Q4]                    Q1 Q2 Q3 Q4
 │  │  │  │                  \  /     \  /                       \  │  /  /
K1 K2 K3 K4                   K1       K2                            K1
(Massive KV Cache)        (4x-8x Smaller Cache)               (Tiny Cache)
```

1. **MHA (Multi-Head Attention - Legacy):**
   * Number of Query heads ($H_Q$) = Number of Key/Value heads ($H_{KV}$).
   * **Result:** Massive KV cache footprint. Running long contexts (32k) causes instant OOM crashes.
2. **GQA (Grouped-Query Attention - Modern Standard):**
   * Multiple Query heads share a single Key/Value head (e.g., 32 Query heads share 8 KV heads $\rightarrow$ 4:1 ratio).
   * Used in Llama 3, Qwen 2.5, Mistral.
   * **Result:** Cuts KV cache memory by **4x to 8x** compared to MHA with negligible quality degradation.
3. **MLA (Multi-Head Latent Attention - DeepSeek Innovation):**
   * Compresses Key and Value matrices into a low-dimensional latent space using low-rank projection *before* caching.
   * **Result:** Slashes KV cache memory by up to **90%**, unlocking massive concurrency at 64k+ context lengths.
4. **FlashAttention (SRAM Tiling vs HBM Round-Trips):**
   * Standard attention computes an $N \times N$ attention matrix in GPU High-Bandwidth Memory (HBM), resulting in $O(N^2)$ memory reads and writes.
   * FlashAttention computes softmax incrementally in tiny, ultra-fast **on-chip SRAM** using mathematical tiling (Online Softmax).
   * **Result:** The full $N \times N$ matrix is never materialized in VRAM. Cuts memory traffic by 5x–10x and dramatically speeds up prefill.

---

## 3. Model Modalities & Ingress Types

As a platform engineer, you classify models by **what data types cross the network and memory bus**:

1. **LLMs (Text $\rightarrow$ Text):**
   * Input: Text tokens. Output: Text tokens.
   * Bottlenecks: Prefill queueing under concurrency, memory-bandwidth decode ceiling.
2. **VLMs / Multimodal (Image/Video + Text $\rightarrow$ Text):**
   * Examples: Qwen2-VL, Llama 3.2-Vision, BluEye ONNX.
   * **Systems Reality:** Images are passed through a Vision Transformer (ViT) encoder and projected into **500 to 2,500 discrete "image tokens"**.
   * A single image upload triggers a massive prefill computation spike. A fleet serving VLMs is heavily **prefill-bound** and requires aggressive chunking.
3. **Embedding & Reranking Models (Text $\rightarrow$ Floats):**
   * Input: Text chunks. Output: 1D vectors (`[0.12, -0.45, ...]`).
   * Bottlenecks: Pure throughput (tokens/sec). Can batch 64–128 requests together. Zero streaming or KV cache overhead.
4. **Diffusion Models (Text $\rightarrow$ Image/Video):**
   * Examples: Stable Diffusion, FLUX.
   * **Systems Reality:** Not autoregressive. Executes **iterative denoising loops** (20 to 50 complete forward passes over a 2D/3D tensor grid). Compute-heavy; no KV cache.

---

## 4. The Token Lifecycle

Models do not see characters, words, or ASCII strings. They execute linear algebra on integer token IDs:

```
"Hello world" ──▶ [ Tokenizer (CPU) ] ──▶ [ 9906, 1917 ] ──▶ [ Embedding Lookup (GPU) ] ──▶ [ Vectors (d_model) ]
```

### A. Tokenization (The BPE Algorithm)
* Text is partitioned into sub-word chunks and mapped to integer IDs using a pre-compiled **Vocabulary**.
* **Vocabulary Size ($V$):** Llama 2 had ~32,000 tokens; Llama 3 has **128,256 tokens**.
* **Operational Gotchas:**
  1. **CPU Bottleneck:** Tokenization runs on the **host CPU** before requests hit the GPU. Single-threaded Python tokenizers leave GPUs idle with gaps between kernel launches (CPU launch-bound).
  2. **Token Inflation:** English averages ~1.3 tokens per word. Indic languages (Hindi, Telugu) average **3 to 5 tokens per word**. A translation model (Sarvam) uses 3x more compute and KV cache per sentence than an English model.

### B. Input Embedding vs The LM Head
* **Input Embedding:** Token ID `9906` indexes a table to produce a dense vector of size $d_{\text{model}}$ (e.g. 4,096 floats).
* **The LM Head (Un-embedding):** At the final layer, the hidden state vector is projected back against the vocabulary matrix ($d_{\text{model}} \times V$), outputting a list of **Logits** of size $V$ (e.g. 128,256 unnormalized scores).

---

## 5. Logits, Probabilities & Sampling Controls

At every single decode iteration, the model outputs raw, unnormalized scores (**Logits**):

```
Hidden Vector ──▶ [ LM Head ] ──▶ Logits (128k floats) ──▶ [ Softmax ] ──▶ Probabilities (0.0 to 1.0) ──▶ [ Sampler ] ──▶ Next Token ID
```

### The API Parameters You Configure in Production

1. **Temperature:** Controls the sharpness of probabilities: $\text{Logit}' = \frac{\text{Logit}}{T}$.
   * **$T = 0.0$ (Greedy Search):** Picks the single highest-probability token. Deterministic. Used for code, SQL, and math.
   * **$T > 0.7$:** Flattens the distribution, giving lower-ranked tokens a chance to be selected.
2. **Top-P (Nucleus Sampling):** Retains only the smallest set of tokens whose cumulative probability reaches $P$ (e.g., top 90%), cutting off the improbable tail.
3. **Top-K:** Retains only the top $K$ most likely tokens (e.g., top 50), regardless of probability.
4. **Repetition Penalty:** Artificially discounts the logits of recently generated tokens to eliminate infinite repeating loops.
5. **Stop Sequences & End-of-Sequence (`EOS`) Tokens:**
   * Special tokens (e.g., `<|eot_id|>`). When emitted, the serving engine terminates the request and frees the slot.
   * **The Infra Danger:** If a prompt fails to emit an EOS token, it generates until `max_tokens` (e.g. 4,096 tokens), holding its KV cache slot and consuming GPU memory on a zombie stream.

---

## 6. Model Serialization & Weight Storage Formats

How weights live on disk dictates startup time, I/O bandwidth, and memory allocation:

| Format | Extension | What It Is | Systems Trade-off |
| :--- | :--- | :--- | :--- |
| **PyTorch / Pickle** | `.bin`, `.pt` | Legacy Python object serialization. | **High security risk.** Executes arbitrary Python bytecode on load; slow disk parsing. Outdated. |
| **SafeTensors** | `.safetensors` | The modern open standard. | **Pure binary tensors.** Zero executable code. Designed for zero-copy memory mapping (`mmap`). |
| **GGUF** | `.gguf` | Single-file format for `llama.cpp` and Ollama. | **Self-contained.** Stores model metadata, tokenizer vocabulary, and quantized weights in **one binary file**. Loads in seconds via `mmap`. |
| **AWQ / GPTQ** | `.safetensors` | Quantized for NVIDIA Tensor Cores. | Requires specialized CUDA dequantization kernels; maximizes throughput on discrete GPUs. |

---

## 7. Positional Encodings & Context Extension (RoPE)

Matrix multiplications possess no inherent order: to attention, *"server crashed container"* looks identical to *"container crashed server"*.

* **RoPE (Rotary Position Embedding):**
  * Modern standard across Llama, Qwen, and Mistral.
  * Rotates Query and Key vectors in 2D coordinate pairs based on their absolute token position in the sequence.
* **Why Platform Engineers Care:**
  * Models claiming "expanded 128k context" use **RoPE Frequency Scaling** (YaRN).
  * Long contexts scale the KV cache linearly: at 128k context, KV memory dwarfs model weight memory, requiring aggressive paging (PagedAttention).

---

## 8. Anatomy of a Single Transformer Layer

Every transformer model is a stack of identical blocks repeated $N$ times (e.g., 28 to 32 layers):

```
Input Vector
    │
    ├─────────────────────────────┐ (Residual Connection)
    ▼                             │
[ RMSNorm / LayerNorm ]           │
    ▼                             │
[ Self-Attention (Q, K, V) ] ─────┴──▶ [+]
    │
    ├─────────────────────────────┐ (Residual Connection)
    ▼                             │
[ RMSNorm / LayerNorm ]           │
    ▼                             │
[ Feed-Forward Network (MLP) ] ───┴──▶ [+]
    │
    ▼
Output Vector (Passed to Next Layer)
```

1. **Attention Sub-Layer (Communication):** Tokens exchange information across time by reading from the past KV cache.
2. **MLP / Feed-Forward Sub-Layer (Computation):** Each token is processed independently through dense projection matrices (projecting from dimension 4,096 to 14,336 and back down). This is where factual parameters reside.
3. **Residual Connections ($x + f(x)$):** Sum the input of each block directly to its output, preventing signal degradation across 32+ deep layers.

When an 8B model generates a single token, that token passes through **32 Attention blocks** and **32 MLP blocks** sequentially.

---

## 9. Advanced Serving Optimizations

### A. Continuous Batching (Iteration-Level Scheduling)
* **The Problem:** Static batching bundles requests together and stalls until the *slowest* request finishes, leaving compute units idle.
* **The Solution:** The scheduler re-forms the batch at every single decode step. Finished sequences exit immediately, and newly arrived requests join the next forward pass.

### B. PagedAttention (Virtual Memory for KV Caches)
* **The Problem:** Traditional runtimes allocate contiguous VRAM buffers for `max_context_length`, wasting 60–80% of memory to internal and external fragmentation.
* **The Solution:** vLLM partitions the KV cache into fixed-size physical memory blocks (e.g., 16 tokens). Block tables map logical token sequences to non-contiguous physical blocks, exactly like OS virtual memory paging.

### C. Chunked Prefill (Eliminating Decode Jitter)
* **The Problem:** If a 10,000-token prompt arrives while 8 users are generating tokens, computing the 10,000-token prefill takes 500 ms, freezing active users' token delivery.
* **The Solution:** The serving engine slices the prefill into chunks (e.g., 512 tokens). Each forward pass processes 512 prefill tokens alongside the active decode tokens, keeping token cadence smooth.

### D. Prefix Caching & Radix Trees
* **The Problem:** Repetitive system prompts (e.g., 2,000 tokens of agent instructions) force the engine to recompute the exact same KV cache on every single request.
* **The Solution:** The engine stores KV blocks in a Radix tree keyed by token prefix. Subsequent requests match the prefix and reuse existing KV blocks, dropping TTFT from 500 ms to **10 ms**.

---

## 10. The Roofline Model & Arithmetic Intensity

The Roofline Model determines whether your inference system is bottlenecked by **Compute (FLOPs)** or **Memory Bandwidth (GB/s)**:

$$\text{Arithmetic Intensity} = \frac{\text{Floating Point Operations (FLOPs)}}{\text{Memory Bytes Transferred (Bytes)}}$$

```
Performance (FLOPs/s)
        ▲
        │                    COMPUTE-BOUND REGION (Prefill & Large Batches)
Peak    ├──────────────────────────────────────────────
FLOPs   │                                 /
        │                                /
        │                               /   MEMORY-BANDWIDTH BOUND REGION
        │                              /    (Decode at Small Batches)
        │                             /
        │                            /
        └───────────────────────────┴──────────────────▶
                                 Ridge Point          Arithmetic Intensity
```

* **Decode Phase (Batch Size = 1):** Low arithmetic intensity. The GPU reads billions of weight bytes across the bus to do a handful of math operations per token. **Strictly Memory-Bandwidth Bound.**
* **Prefill Phase & Large Batches (Batch Size $\ge$ 32):** High arithmetic intensity. Thousands of tokens share the same loaded weights. **Compute-Bound** (saturates Tensor Cores).

---

## 11. The Complete End-to-End Execution Trace

Here is the entire physical lifecycle of a request from client to silicon:

```
1. Client sends POST /v1/chat/completions (stream: true)
2. Edge Ingress (Nginx): proxy_buffering off; flushes HTTP chunks immediately
3. Platform Gateway: checks queue depth and token budget; admits request
4. Host CPU: Tokenizer encodes prompt string into Token IDs [ 9906, 1917, ... ]
5. PREFILL PHASE (GPU / Silicon):
   - Computes Q, K, V across all layers in parallel
   - Allocates PagedAttention blocks in VRAM for the KV Cache
   - Predicts first token -> First chunk emitted -> TTFT recorded
6. DECODE PHASE (Autoregressive Loop):
   - Loads last token vector
   - Reads historical KV cache blocks from memory bus
   - Executes Layer 1 through Layer N (Attention + MLP)
   - LM Head outputs Logits (128k floats)
   - Sampler applies Temperature and Top-P to pick 1 Token ID
   - Token ID appended to KV cache block
   - Token streamed to client via SSE chunk
7. Check Termination:
   - If Token ID == EOS (<|eot_id|>) OR length == max_tokens -> Terminate
   - Otherwise -> Repeat Step 6
8. Cleanup: PagedAttention blocks returned to free pool; connection closed
```

---

## 12. The Production Engineering Decision Matrix

| Observed Production Failure | Root Cause | Immediate Systems Fix |
| :--- | :--- | :--- |
| **TTFT explodes under load; ITL is normal.** | Queue wait saturation; prefill queues blocked. | Enable admission control with HTTP 429 shedding; implement prefix caching; cap `max_tokens`. |
| **Tokens arrive in stuttering bursts.** | Reverse proxy buffer bloat. | Configure `proxy_buffering off;`, `X-Accel-Buffering: no;`, disable gzip on event streams. |
| **Decode speed collapses mid-traffic.** | Layers offloaded across PCIe bus; PCIe bottleneck. | Increase GPU layers (`-ngl`), quantize weights (FP16 $\rightarrow$ Q4), eliminate CPU layer offloading. |
| **OOM crash occurs during generation.** | Dynamic KV cache bloat exceeds free VRAM. | Enable PagedAttention; switch to GQA models; enable FP8 KV cache quantization; enforce KV-aware admission limits. |
| **GPU utilization is 95%, but tokens/sec is terrible.** | Single tiny kernel running continuously; memory bandwidth saturated. | Check memory bandwidth utilization (DCGM); increase batch size; check for CPU-offloaded stalls. |
| **Active streams freeze when large prompts arrive.** | Prefill interference stalls running decodes. | Enable **Chunked Prefill**; route long-context prompts to a separate dedicated prefill pool. |
| **Client disconnects, but GPU remains 100% busy.** | Client cancellation not propagated upstream. | Configure reverse proxy to close upstream sockets on client abort; enable engine `abort-on-disconnect`. |

---

## 13. CUDA Execution Realities: Streams, Allocators & Timing Traps

When deploying Python-based inference engines (PyTorch, TensorRT-LLM, vLLM), naive assumptions about hardware execution lead to severe performance and telemetry bugs.

### A. Asynchronous Kernel Execution & Timing Traps
In PyTorch, CUDA operations are **asynchronous**. When you invoke `output = model(input_ids)`, the CPU does not wait for the GPU to perform matrix multiplication. Instead, the CPU enqueues a pointer to the kernel in a hardware FIFO queue (a **CUDA Stream**) and immediately executes the next Python bytecode instruction.

```python
# FATAL BENCHMARKING BUG:
t0 = time.time()
output = model(input_ids)  # CPU enqueues kernel in ~15 microseconds
t1 = time.time()
print(f"Latency: {(t1 - t0) * 1000} ms")  # Measures CPU enqueue time, NOT GPU execution!
```

If the GPU takes 25 ms to compute the forward pass, the script above prints `Latency: 0.015 ms`. To measure true silicon execution, you must force host-device synchronization or use CUDA Events:

```python
# CORRECT TIMING METHOD:
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
output = model(input_ids)
end_event.record()

# Wait for GPU execution to finish
torch.cuda.synchronize()
print(f"True Silicon Latency: {start_event.elapsed_time(end_event)} ms")
```

### B. The PyTorch Caching Allocator
Invoking the OS driver to allocate GPU memory (`cudaMalloc`) is an extremely expensive kernel syscall, costing hundreds of microseconds to milliseconds. To prevent latency jitter, PyTorch implements a **caching allocator**:

1. **`torch.cuda.memory_allocated()`:** Memory currently holding active tensor data.
2. **`torch.cuda.memory_reserved()`:** Memory claimed by PyTorch from the NVIDIA driver pool.
3. **The `nvidia-smi` Trap:** `nvidia-smi` reports total reserved memory. If PyTorch reserves 15 GB of a 16 GB card to avoid repeated syscalls, `nvidia-smi` shows 94% VRAM utilization even if active tensor data occupies only 4 GB!

```
Total GPU VRAM (16 GB)
├── Active Tensors (4 GB)      ──▶ torch.cuda.memory_allocated()
├── Caching Memory Pool (11 GB)──▶ torch.cuda.memory_reserved() - allocated()
└── Free to Driver (1 GB)      ──▶ Remaining physical VRAM
```

* **Memory Fragmentation OOM:** If PyTorch holds 11 GB in its reserved pool, but the free memory is fragmented into non-contiguous blocks of 256 MB, requesting a contiguous 1 GB tensor triggers a `CUDA Out of Memory` error despite 11 GB of available pool memory.
* **Systems Mitigation:** Set environment variable `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to allow PyTorch to stitch virtual address spaces using driver virtual memory management.

### C. CUDA Context Baseline Overhead
Merely importing PyTorch and calling `torch.cuda.init()` allocates between **500 MB and 1.2 GB of VRAM** for the CUDA Driver Context (internal dispatch tables, cublas handles, and kernel symbol maps). On edge GPUs with tight budgets (such as an NVIDIA GTX 1050 Ti with 4 GB VRAM), the driver context alone consumes 25% of the entire hardware capacity before a single weight is loaded.

### D. Pinned Host Memory & DMA
Standard CPU RAM allocated by Python is **pageable**—the OS kernel can move it to swap or change its physical page address at any time. When copying tensors from CPU to GPU (`tensor.to('cuda')`), the CUDA driver must first copy data from pageable RAM to an internal page-locked (pinned) staging buffer, then execute a Direct Memory Access (DMA) transfer across PCIe.
* By passing `pin_memory=True` in your dataloaders or using `tensor.pin_memory()`, you allocate page-locked memory directly, enabling zero-copy DMA transfers at full PCIe 3.0/4.0/5.0 bus saturation without CPU intervention.

---

## 14. Quantization Mechanics: Scale, Zero-Point & Activation Outliers

Quantization compresses floating-point representations into lower-bit integers to reduce memory footprint and memory bus traffic.

### A. Numerical Formats
* **FP32 (Single Precision):** 1 sign bit, 8 exponent bits, 23 mantissa bits (4 bytes).
* **FP16 (Half Precision):** 1 sign bit, 5 exponent bits, 10 mantissa bits (2 bytes). Max representable value: 65,504. Vulnerable to underflow/overflow during long-context prefill.
* **BF16 (Bfloat16):** 1 sign bit, 8 exponent bits, 7 mantissa bits (2 bytes). Preserves the identical dynamic range of FP32 while halving memory footprint. Industry standard for modern LLM training and serving.
* **FP8:**
  * **E4M3 (4 exponent, 3 mantissa):** Higher numerical accuracy; standard for weights and forward-pass activations.
  * **E5M2 (5 exponent, 2 mantissa):** Wider dynamic range; standard for gradients and attention scores.

### B. Quantization Mathematics
To map a continuous floating-point range $[X_{\min}, X_{\max}]$ to an integer range $[-2^{b-1}, 2^{b-1}-1]$ (e.g. $[-128, 127]$ for INT8):

$$\text{Scale } (S) = \frac{\max(|X_{\min}|, |X_{\max}|)}{2^{b-1} - 1}$$

$$q = \text{clamp}\left(\left\lfloor \frac{X}{S} \right\rceil, -2^{b-1}, 2^{b-1}-1\right)$$

$$\hat{X} = q \times S$$

* **Uniform Symmetric Quantization:** Zero point is pinned to 0 ($Z = 0$). Eliminates integer offset additions during GEMM multiplication.
* **Uniform Asymmetric Quantization:** Uses a non-zero integer offset ($Z$) to preserve asymmetric distributions (e.g., post-GELU activations where values are strictly $\ge 0$).

### C. Weight-Only vs Weight-and-Activation Quantization
* **Weight-Only (W4A16, W8A16 - AWQ / GPTQ):** Weights are stored in INT4 or INT8. Before matrix multiplication, GPU registers dynamically dequantize weights back to FP16. Activations remain FP16.
  * *Why this works:* In the autoregressive decode phase (batch size 1–4), the GPU is strictly **memory-bandwidth bound**. Loading an INT4 weight uses 4x fewer bus bytes. The compute overhead of on-the-fly dequantization in SRAM is virtually free compared to memory bus wait states.
* **Weight-and-Activation (W8A8 / W4A4 - SmoothQuant):** Both weights and activations are quantized to INT8/INT4. The GPU executes INT8 Tensor Core GEMMs, providing a 2x theoretical math throughput boost.

### D. The Emergent Outlier Feature Trap (>6.7B Parameters)
In models larger than 6.7B parameters, attention layers spontaneously develop **emergent outlier channels** (Dettmers et al.):
* In 0.1% of hidden dimensions, activation magnitudes reach values up to 100x larger than the remaining 99.9% of channels (e.g. hidden values jumping from $1.2$ to $150.0$).
* If you apply uniform per-tensor activation quantization, the scale $S$ is forced to expand to accommodate $150.0$. Consequently, the remaining 99.9% of activations round to 0 or 1, completely destroying model generation capability.

```
Hidden Channels Across Sequence:
Channel 0..4094:  [ 0.12, -0.45,  0.88, -0.15,  0.34 ]  ──▶ Quantizes cleanly into INT8
Channel 4095:     [ 98.4, 112.1, 105.7, 120.3, 115.0 ]  ──▶ OUTLIER! Destroys INT8 dynamic range
```

* **SmoothQuant Solution:** Mathematically migrates the outlier difficulty from activations to weights. Because weights do not have extreme dynamic variance across tokens, an inverse diagonal scaling matrix $s$ divides activation channels and multiplies weight rows:

$$Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W)$$

This smooths out activation peaks, enabling stable INT8/INT8 matrix multiplication on Tensor Cores.

### E. KV Cache Quantization
Storing KV cache entries in FP8 or INT8 reduces per-token memory footprint from 2 bytes to 1 byte (or 0.5 bytes).
* Because KV cache memory dictates maximum batch concurrency, enabling FP8 KV cache immediately **doubles the maximum concurrent user capacity** on the same GPU without degrading token perplexity.

---

## 15. Distributed Multi-GPU Serving: TP All-Reduce vs PP Bubbles

When a model's parameters exceed the VRAM of a single physical accelerator (e.g., Llama 70B requiring 140 GB in FP16), the workload must be partitioned across multiple GPUs.

```
Distributed Serving Primitives:
├── Tensor Parallelism (TP)   ──▶ Splits layers horizontally/vertically. 2 All-Reduces per layer. NVLink MANDATORY.
├── Pipeline Parallelism (PP) ──▶ Slices layers sequentially across GPUs. Suffers from Pipeline Bubbles.
└── Data Parallelism (DP)     ──▶ Replicates full model across GPUs. Serves independent requests. Zero communication.
```

### A. Tensor Parallelism (Megatron-LM Style)
Tensor Parallelism splits individual weight matrices inside each transformer layer across $N$ GPUs.

1. **Column Parallel Linear (MLP Up-Projection / QKV Projection):**
   * Weight matrix $W$ is split column-wise: $W = [W_1 \mid W_2]$.
   * GPU 0 computes $Y_1 = X W_1$; GPU 1 computes $Y_2 = X W_2$.
   * No inter-GPU communication is required during this step!
2. **Row Parallel Linear (MLP Down-Projection / Attention Output Projection):**
   * Weight matrix $W$ is split row-wise: $W = \begin{bmatrix} W_1 \\ W_2 \end{bmatrix}$.
   * GPU 0 computes $Z_1 = Y_1 W_1$; GPU 1 computes $Z_2 = Y_2 W_2$.
   * Mathematical identity: $Z = X W = Y_1 W_1 + Y_2 W_2 = Z_1 + Z_2$.
   * To produce the final output $Z$, GPU 0 and GPU 1 must sum their local results together.
   * **The Communication Primitive:** An **All-Reduce SUM** operation across all participating GPUs.

```
GPU 0: [ Z_1 ] ──┐
                 ├──▶ [ All-Reduce SUM (NCCL) ] ──▶ GPU 0: [ Z_1 + Z_2 ]
GPU 1: [ Z_2 ] ──┘                                GPU 1: [ Z_1 + Z_2 ]
```

### B. The 2 All-Reduces Per Layer Reality
Every single transformer layer contains:
* 1 All-Reduce after the Attention block.
* 1 All-Reduce after the MLP block.
* **Total:** 2 All-Reduces per layer per token!
* For an 80-layer model (Llama 70B), generating **one single token** requires **160 sequential All-Reduce network operations**.
* At 40 tokens per second, the cluster executes **6,400 All-Reduces per second**.

### C. The Interconnect Latency Ceiling (Why TP Requires NVLink)
* Over **NVIDIA NVLink** (900 GB/s bandwidth, sub-microsecond latency): A 160-operation All-Reduce loop adds $< 1.5$ ms total overhead per token.
* Over **PCIe Gen4 x16** (32 GB/s, higher kernel launch latency): All-Reduce latency explodes to 15–30 ms per token, reducing generation speed by 75%.
* Over **Standard 10GbE / 25GbE Ethernet**: All-Reduce latency exceeds 150 ms per token, making real-time interactive generation physically impossible.
* **Golden Rule of AI Infrastructure:** **Tensor Parallelism must NEVER cross node boundaries** unless connected via 400 Gbps InfiniBand RDMA / RoCE with specialized GPUDirect NCCL configurations.

### D. Pipeline Parallelism (PP) & The Pipeline Bubble
Pipeline Parallelism partitions layers sequentially across GPUs: GPU 0 hosts layers 1–16; GPU 1 hosts layers 17–32.
* **Communication Cost:** Low. GPUs exchange only boundary activation tensors ($B \times S \times d_{\text{model}}$ floats) via Point-to-Point (P2P) `Send` and `Recv`. Can run across standard networks.
* **The Pipeline Bubble Problem:** GPU 1 sits 100% idle while GPU 0 processes the first stage. GPU 0 sits 100% idle while GPU 1 processes the second stage.

$$\text{Bubble Fraction } (F_{\text{bubble}}) = \frac{P - 1}{P + M - 1}$$

Where $P$ is the number of pipeline stages and $M$ is the number of micro-batches.
* During **single-request interactive decode** ($M = 1$), the bubble fraction is $(P - 1) / P$. On a 4-stage pipeline, **75% of your total cluster FLOPs are completely wasted to idle bubble stalls**!
* Pipeline Parallelism is viable for high-throughput offline batch processing, but structurally unsuitable for low-latency interactive streaming.

---

## 16. Structured Outputs & Constrained Decoding (Grammar Masking)

Modern platform APIs frequently require guaranteed schemas: JSON objects, SQL statements, or strict enum strings. Prompting alone ("Please return valid JSON") fails non-deterministically.

### A. How Constrained Decoding Works
Engines like Outlines, Guidance, XGrammar, and llama.cpp enforce schemas directly at the **Logit Sampling stage**:

```
Prompt ──▶ Forward Pass ──▶ Logits (128k floats)
                                │
                                ▼
         [ Context-Free Grammar / FSM Validator ]
                                │
                                ▼
         Mask invalid tokens: Logits[invalid] = -inf
                                │
                                ▼
                         [ Softmax ]
                                │
                                ▼
                  Only valid tokens have P > 0
```

1. The user supplies a JSON Schema or Regular Expression.
2. The runtime compiles the schema into a **Deterministic Finite Automaton (DFA)** or **Pushdown Automaton (PDA)**.
3. At decode step $t$, the DFA's current state determines which characters/tokens are grammatically permissible.
4. For all tokens in the 128,256 vocabulary that violate the grammar state transition, the engine sets their raw logit value to $-\infty$:

$$\text{Logit}_i = -\infty \quad \forall i \notin \mathcal{V}_{\text{valid}}$$

5. When Softmax is computed, $e^{-\infty} = 0.0$. It is mathematically impossible for the sampler to select a syntax-violating token.

### B. The CPU Logit Masking Trap
Evaluating grammar state transitions across a 128,256-token vocabulary in Python on the host CPU at every decode iteration introduces severe latency overhead:
* If GPU forward pass takes **8 ms**, but Python DFA validation takes **25 ms**, generation speed collapses from 125 tokens/sec to 30 tokens/sec. The inference engine becomes **CPU grammar-bound**.
* **Modern Fix:** High-performance runtimes pre-compile token transition tables into index arrays and execute logit masking via CUDA bitmask kernels directly in VRAM before the sampling kernel fires.

---

## 17. Model Cold Starts, mmap & Linux Page Cache Dynamics

Model weight serialization dictates disk read throughput, memory allocation latency, and container boot times.

### A. Serialization Formats: SafeTensors vs Pickle
* **PyTorch Pickle (`.pt` / `.bin`):** Serializes arbitrary Python object graphs. When loading, Python instantiates classes, copies memory blocks, and executes unpickling bytecode.
  * *Security Danger:* Loading untrusted `.bin` files allows arbitrary shell execution (`os.system("rm -rf /")`).
  * *I/O Performance:* Forces full heap allocation and double-buffering.
* **SafeTensors (`.safetensors`):** Developed by Hugging Face. Contains an 8-byte header size, a JSON metadata header detailing tensor shapes and byte offsets, followed by raw, uncompressed binary tensor buffers.
  * Zero executable code. Zero security vulnerability.
  * Aligned for direct memory mapping (`mmap`).

### B. `mmap()` (Memory Mapping) Under the Hood
When an inference runtime loads a SafeTensors or GGUF file via `mmap()`:
1. The process calls:
   ```c
   int fd = open("model.safetensors", O_RDONLY);
   void *ptr = mmap(NULL, file_size, PROT_READ, MAP_SHARED, fd, 0);
   ```
2. The Linux kernel allocates virtual memory addresses for `ptr`, but **zero bytes of weight data are loaded into physical RAM**.
3. When the GPU worker attempts to copy weights into VRAM (`cudaMemcpy(d_weights, ptr, size)`), the CPU encounters **Page Faults** (Major Faults).
4. The OS kernel's Virtual Memory Subsystem reads 4 KB pages directly from the NVMe disk into the **Linux Page Cache** (unreserved physical RAM), which are then transferred over the PCIe bus to GPU VRAM via Direct Memory Access (DMA).

### C. Cold Starts vs Warm Restarts
* **Cold Start (Clean OS Boot):** The model file is not in RAM. Loading a 16 GB model is bounded strictly by NVMe sequential read speed:
  $$\text{Load Time} = \frac{16 \text{ GB}}{3.5 \text{ GB/s (PCIe 3.0 NVMe)}} \approx 4.6 \text{ seconds}$$
* **Warm Restart (Process Restart):** If the inference container crashes or restarts, the 16 GB file **already resides in the Linux Page Cache**. `mmap()` executes in $< 10$ milliseconds, and DMA transfers to GPU VRAM operate at RAM bus bandwidth (50–100 GB/s on x86, ~800 GB/s on Mac Studio M3 Ultra), achieving full engine readiness in $< 300$ milliseconds.

### D. Linux Page Cache Eviction Traps
Under high memory pressure, the Linux kernel reclaims page cache memory to satisfy anonymous heap allocations:
* If background processes (log aggregators, backup cron jobs, ETL workers) allocate physical RAM, the kernel evicts the cached model pages.
* The next model load or memory access triggers massive disk I/O, causing unexpected multi-second latency spikes.
* **Production Tuning:**
  ```bash
  # Discourage kernel from aggressively evicting filesystem cache
  sysctl -w vm.vfs_cache_pressure=50
  
  # Prevent swapping anonymous memory
  sysctl -w vm.swappiness=10
  ```

---

## 18. Ingress Traps: HTTP/2 Multiplexing & Client Abort Propagation

Streaming tokens back to clients via Server-Sent Events (SSE) introduces unique networking failure modes.

### A. Reverse Proxy Buffer Bloat
Standard web servers (Nginx, Traefik, AWS ALB) default to buffering responses to optimize TCP packet utilization.
* **The Symptom:** An LLM generates tokens smoothly at 40 tokens/sec, but the user sees nothing for 6 seconds. Suddenly, the entire 240-token response dumps onto the screen at once.
* **The Root Cause:** Nginx holds upstream response bytes in its internal 4 KB / 8 KB buffer before sending an MTU packet down the client TCP socket.
* **The Fix:**
  ```nginx
  # Mandatory Nginx SSE Configuration:
  location /v1/chat/completions {
      proxy_pass http://vllm_upstream;
      proxy_http_version 1.1;
      proxy_set_header Connection '';
      proxy_buffering off;
      proxy_cache off;
      chunked_transfer_encoding off;
  }
  ```
  The inference application must also emit the header: `X-Accel-Buffering: no`.

### B. HTTP/2 Stream Exhaustion
HTTP/2 multiplexes hundreds of logical requests over a single shared TCP connection to eliminate TCP handshake latency.
* **The Trap:** Nginx has a default directive: `http2_max_concurrent_streams 128;`.
* In standard REST APIs (where requests complete in 15 ms), 128 concurrent streams handle tens of thousands of requests per second.
* In LLM streaming (where requests remain open for 15–45 seconds), 128 users will immediately saturate the connection pool. The 129th user receives an immediate `HTTP 429 Too Many Requests` or connection reset error.
* **The Fix:** Increase `http2_max_concurrent_streams 1024;` and ensure edge gateways configure sufficient client worker pools.

### C. Zombie Token Generation & Client Disconnect Propagation
In interactive applications, users routinely click "Stop Generating", navigate away, or close the browser tab.

```
Client Clicks 'Stop' ──▶ Browser sends TCP FIN / RST
                               │
                               ▼
                        [ Edge Nginx ]
                               │
    DEFAULT BEHAVIOR: Ignores disconnect! Upstream socket stays open!
                               │
                               ▼
                         [ vLLM Engine ]
    KEEPS GENERATING 2,000 TOKENS! BURNS GPU FLOPS! LOCKS KV CACHE!
```

* **The Disaster:** If the client disconnect is not propagated upstream, the GPU continues generating tokens for up to 4,096 iterations. The request retains its KV cache allocation in VRAM, starving incoming users and inflating cloud compute bills.
* **The Solution Architecture:**
  1. In Nginx: Enable client abort termination:
     ```nginx
     proxy_ignore_client_abort off;
     ```
  2. In FastAPI / ASGI: Listen for client disconnection in the streaming loop:
     ```python
     @app.post("/v1/chat/completions")
     async def chat_completions(request: Request):
         async def event_generator():
             async for token in engine.generate(prompt):
                 if await request.is_disconnected():
                     await engine.abort(request_id)
                     break
                 yield f"data: {token}\n\n"
         return StreamingResponse(event_generator(), media_type="text/event-stream")
     ```
  3. In the Serving Engine: The engine immediately deallocates the request's PagedAttention KV blocks and returns them to the free block pool.

---

## 19. KV Cache Capacity Planning, Preemption & Swapping Dynamics

The Key-Value (KV) cache is the primary constraint limiting concurrency and context length in modern LLM serving.

### A. The KV Cache Sizing Formula
For every token stored in the KV cache across all layers:

$$\text{Bytes Per Token} = 2 \times L \times H_{KV} \times D \times P_{\text{bytes}}$$

Where:
* $2$: Factor for storing both Key and Value tensors.
* $L$: Number of transformer layers.
* $H_{KV}$: Number of Key-Value attention heads (dictated by MHA or GQA).
* $D$: Head dimension ($d_{\text{model}} / H_Q$).
* $P_{\text{bytes}}$: Bytes per numerical precision (2 bytes for FP16/BF16, 1 byte for FP8, 0.5 bytes for INT4).

#### Concrete Derivation: Llama 3 8B (GQA: 32 Query Heads, 8 KV Heads, 32 Layers, Dim 4096)
* Head dimension $D = 4096 / 32 = 128$.
* In **FP16** ($P_{\text{bytes}} = 2$):
  $$\text{Bytes/Token} = 2 \times 32 \times 8 \times 128 \times 2 = 131,072 \text{ bytes} = 128 \text{ KB per token}$$
* For a **4,096-token sequence**:
  $$4,096 \text{ tokens} \times 128 \text{ KB} = 524,288 \text{ KB} = 512 \text{ MB per concurrent request!}$$
* For **32 concurrent users** generating at 4,096 context length:
  $$32 \times 512 \text{ MB} = 16 \text{ GB of VRAM strictly dedicated to KV cache!}$$

### B. What Happens When KV Cache Exhausts Under Load?
Unlike traditional web servers that return HTTP 503 or queue in RAM, an LLM generating tokens cannot pause without retaining its existing KV state.
* **Naive Engines:** Encounter a CUDA Out-of-Memory exception $\rightarrow$ the Python process crashes, dropping all active user connections.
* **PagedAttention Engines (vLLM / TGI):** Trigger **Preemption**.

### C. The Two Preemption Strategies
When physical KV cache blocks are 100% saturated and an active request requires a new block to emit its next token, the scheduler must preempt an existing request:

1. **Swapping (PCIe Offload):**
   * The scheduler suspends the lowest-priority request and transfers its allocated KV cache blocks across PCIe to host CPU RAM (`block_manager.swap_out()`).
   * When VRAM frees up, the blocks are transferred back (`block_manager.swap_in()`).
   * *Systems Bottleneck:* PCIe transfer bandwidth (32 GB/s) adds dozens of milliseconds of latency jitter to the swapped sequence.
2. **Recomputation:**
   * The scheduler terminates the request's decode state and completely frees its VRAM blocks.
   * When capacity frees up, the engine re-runs the entire prompt + generated prefix through the **Prefill phase** from scratch.
   * *Systems Bottleneck:* High compute cost, but avoids PCIe bandwidth contention.

### D. The Production Preemption Cliff
When request arrival rates exceed cluster KV capacity:
* A tiny 5% increase in traffic triggers cascading preemptions.
* Preempted requests recompute, consuming massive prefill compute and blocking decode queues.
* **P99 latency explodes from 250 ms to 20+ seconds.**
* **SRE Golden Rule:** Alert on `vllm:num_preemptions_total > 0` as a P0 capacity breach. Configure aggressive token bucket admission control before requests reach the engine.

---

## 20. Disaggregated Prefill-Decode Architecture (Split-Phase Serving)

Colocating the Prefill and Decode phases on the same physical accelerator is the single largest structural inefficiency in modern AI infrastructure.

```
Hardware Mismatch:
├── PREFILL PHASE ──▶ Compute-Bound. High arithmetic intensity. Saturates Tensor Cores. (100+ TFLOPs).
└── DECODE PHASE  ──▶ Memory-Bandwidth-Bound. Low arithmetic intensity. Idles Tensor Cores. Bounded by HBM bus.
```

### A. The Prefill Interference Problem
When an LLM server is decoding responses for 16 active users at 40 tokens/sec, an incoming request with an 8,000-token prompt arrives.
* Computing the 8k prefill requires a massive matrix multiplication that occupies the GPU's streaming multiprocessors for 400–600 ms.
* During these 600 ms, the 16 active decode streams **cannot execute their decode step**.
* The end-user experiences severe Inter-Token Latency (ITL) stuttering, violating tight SLAs.

### B. The Disaggregated Serving Solution (DistServe / Splitwise / Mooncake)
Disaggregated serving physically separates the cluster into two specialized GPU pools:

```
User Prompt ──▶ [ Edge Gateway ]
                      │
                      ▼
          [ PREFILL WORKER POOL ]  (NVIDIA H100 SXM5 - Compute Dense)
          - Computes initial prompt Q, K, V
          - Emits Token 0 (TTFT < 150 ms)
                      │
                      ▼ (High-Speed RDMA / InfiniBand Network Transfer)
          [ DECODE WORKER POOL ]   (NVIDIA L40S / Apple Silicon - Bandwidth Dense)
          - Receives serialized KV Cache
          - Executes autoregressive token loop at high ITL cadence
          - Streams tokens back to client
```

### C. Network Bandwidth Requirements for KV Transfer
For disaggregation to succeed, transferring the generated KV cache over the internal network must take significantly less time than computing the prefill:
* An 8,000-token prompt on Llama 3 8B generates:
  $$8,000 \text{ tokens} \times 128 \text{ KB} = 1.024 \text{ GB of KV Cache}$$
* Over standard **10GbE Network**: Transfer time $= (1.024 \times 8) / 10 = 820 \text{ ms}$ (Unusable; slower than local computation!).
* Over **100 Gbps InfiniBand**: Transfer time $= (1.024 \times 8) / 100 = 82 \text{ ms}$ (Viable).
* Over **400 Gbps InfiniBand RDMA (GPUDirect)**: Transfer time $= (1.024 \times 8) / 400 = 20.5 \text{ ms}$ (Near-instantaneous handoff).

---

## 21. Speculative Decoding Mechanics & Verification Economics

Autoregressive decoding is bounded by memory bandwidth: to emit a single token, a 70B model must stream 140 GB of weights from HBM to registers. If a response generates 100 tokens, the memory bus reads 14,000 GB of data sequentially.

### A. The Speculative Decoding Principle
Can we generate multiple tokens per memory read pass? **Speculative Decoding** solves this by pairing two models:

1. **The Draft Model (Small & Fast):** e.g., Llama 3 1B. Because its weights are only 2 GB, it generates $K$ candidate tokens (e.g. $K=4$) rapidly using autoregressive decode.
2. **The Target Model (Large & Slow):** e.g., Llama 3 70B. It receives the prompt plus the $K$ proposed candidate tokens.
3. **Parallel Verification:** Instead of running $K$ sequential steps, the Target Model evaluates all $K$ candidate tokens simultaneously in **ONE single forward pass** (which executes with the high arithmetic intensity of prefill mode!).

```
Step 1: Draft Model (1B) proposes 4 tokens:   [ "Kubernetes", "is", "a", "container" ]
Step 2: Target Model (70B) runs 1 forward pass: [ Valid,      Valid, Valid, Invalid ]
Step 3: Engine accepts 3 tokens + generates 1 correction token from Target Model.
Result: 4 tokens produced in the time of 1 Target Model forward pass!
```

### B. Mathematical Acceptance Criterion (Rejection Sampling)
To guarantee that the output text distribution is **100% mathematically identical** to running the 70B target model alone, candidate tokens are accepted or rejected via modified rejection sampling:

Let $q(x)$ be the probability assigned by the Draft Model, and $p(x)$ be the probability assigned by the Target Model.
* If $p(x) \ge q(x)$: The token is **accepted** with probability 1.
* If $p(x) < q(x)$: The token is **accepted with probability** $\frac{p(x)}{q(x)}$.
* If rejected: The target model samples a replacement token from the normalized residual distribution:
  $$p'(x) = \frac{\max(0, p(x) - q(x))}{\sum_y \max(0, p(y) - q(y))}$$

### C. Speedup Economics
The effective wall-clock acceleration is governed by the **Acceptance Rate** ($\alpha$):

$$\text{Speedup} \approx \frac{1}{(1 - \alpha) + \frac{\alpha}{K}}$$

* In predictable domains (code generation, structured JSON, standard grammar), $\alpha \approx 0.75 - 0.85$, yielding a **2.0x to 2.8x speedup** in tokens per second without any loss in model capability or reasoning depth.

---

## 22. Multi-LoRA Serving Runtimes & Dynamic Adapter Paging

In multi-tenant SaaS environments, serving 50 distinct fine-tuned customer models by deploying 50 separate model instances requires astronomical infrastructure spend ($50 \times 16 \text{ GB} = 800 \text{ GB VRAM}$).

### A. LoRA (Low-Rank Adaptation) Theory
LoRA freezes the base model weights $W_0 \in \mathbb{R}^{d \times k}$ and introduces low-rank decomposition matrices:

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} (B \times A)$$

Where $A \in \mathbb{R}^{r \times k}$ and $B \in \mathbb{R}^{d \times r}$ with rank $r \ll d$ (e.g. $r=8$ or $16$).
* The base model weights remain static (16 GB).
* The customer-specific adapter consists of only **10 MB to 50 MB** of floating-point parameters.

### B. Segmented Batched GEMM (SGMV) & Adapter Paging
Modern inference runtimes (vLLM Multi-LoRA, S-LoRA, Punica) serve hundreds of dynamic LoRA adapters concurrently on a single base model instance:

```
Batch at Iteration t:
├── Token from Request 1 ──▶ Uses Adapter A (Legal Domain, r=8)
├── Token from Request 2 ──▶ Uses Adapter B (Medical Domain, r=16)
└── Token from Request 3 ──▶ Uses Base Model (No Adapter)
```

1. **Base Computation:** The engine batches all tokens together and executes the large base model GEMM ($X W_0$) across all inputs simultaneously.
2. **Segmented Adapter Computation:** Specialized CUDA kernels (SGMV) multiply each token segment by its corresponding low-rank adapter matrices $A$ and $B$, adding the result back to the hidden state.
3. **LoRA Memory Paging:** Adapters reside in host RAM or NVMe storage and are paged into a dedicated VRAM adapter cache pool on-demand, enabling thousands of custom models to be served from a single GPU.

---

## 23. Observability, DCGM & True Hardware Saturation Signals

Traditional systems monitoring tools (`top`, `iostat`, standard Prometheus node exporters) are blind to silicon accelerators. Worse, standard GPU tools frequently produce misleading signals.

### A. The Deception of `nvidia-smi`
`nvidia-smi` reports `GPU-Util` as the percentage of time over the last sampling interval (e.g. 1 second) during which **at least one kernel was active on the GPU**.
* If a single-threaded Python script launches a kernel that uses **1 out of 108 Streaming Multiprocessors** to copy a single float, `nvidia-smi` reports **100% GPU-Util**!
* It provides zero insight into whether Tensor Cores are active, whether memory bandwidth is saturated, or whether threads are stalled waiting on memory fetches.

### B. True Hardware Telemetry via NVIDIA DCGM (Data Center GPU Manager)
In high-performance inference clusters, platform engineers monitor low-level hardware counters via DCGM:

| DCGM Metric Field | Hardware Subsystem Measured | Interpretation for AI Infrastructure |
| :--- | :--- | :--- |
| `DCGM_FI_PROF_SM_ACTIVE` | Streaming Multiprocessors | Percentage of SMs with active warps. If $< 50\%$, engine is launch-bound or batch size is too small. |
| `DCGM_FI_PROF_TENSOR_ACTIVE` | Tensor Core Math Units | Cycles where Tensor Cores compute matrix multiplications. High during Prefill; low during small-batch Decode. |
| `DCGM_FI_PROF_DRAM_ACTIVE` | High-Bandwidth Memory (HBM) | Memory bus saturation. Saturated ($> 80\%$) during healthy, high-throughput Decode phases. |
| `DCGM_FI_PROF_PCIE_TX_BYTES` | PCIe Bus Transmit | Bandwidth leaving GPU. Spikes during KV cache CPU swapping. |
| `DCGM_FI_PROF_PCIE_RX_BYTES` | PCIe Bus Receive | Bandwidth entering GPU. Spikes during initial model weight loading. |

### C. The 5 Golden Metrics of LLM Serving (Application Plane)
Exposed by modern serving engines via `/metrics` (Prometheus format):

1. **`vllm:time_to_first_token_seconds` (Histogram):**
   * Measures Prefill latency + Queue Wait time.
   * If TTFT spikes while ITL remains flat: Ingress queues are backed up or prefix caching missed.
2. **`vllm:time_per_output_token_seconds` (Histogram):**
   * Measures Inter-Token Latency (ITL) during the Decode phase.
   * Direct proxy for user-perceived streaming cadence.
3. **`vllm:num_requests_waiting` (Gauge):**
   * Number of admitted requests waiting in the engine queue because VRAM KV cache blocks are exhausted.
4. **`vllm:gpu_cache_usage_factor` (Gauge, 0.0 to 1.0):**
   * Percentage of total PagedAttention KV cache blocks currently allocated.
   * If this exceeds $0.85$, the cluster is within seconds of preemption or request shedding.
5. **`vllm:num_preemptions_total` (Counter):**
   * Cumulative count of running requests evicted from VRAM. Must be **strictly zero** in a healthy production environment.
