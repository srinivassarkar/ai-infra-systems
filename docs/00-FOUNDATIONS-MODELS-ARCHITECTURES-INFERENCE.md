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
