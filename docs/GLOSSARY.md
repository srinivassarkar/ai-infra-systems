# Systems Engineering & LLM Inference Glossary

A comprehensive, production-grade reference mapping Artificial Intelligence and Machine Learning primitives directly to physical systems, memory architectures, and networking realities.

---

## 1. The Core Data Units

| AI / ML Concept | What It Actually Is (Systems Translation) | Why Infrastructure Engineers Care |
| :--- | :--- | :--- |
| **Token** | A discrete chunk of text (~4 characters or 0.75 words). For example, `DevOps` is 2 tokens (`Dev` + `Ops`). | All throughput, pricing, context budgeting, and latency metrics are measured in tokens (Tokens/Sec, TTFT). |
| **Context Window** | The maximum memory buffer (Prompt + Generated Output) an LLM can hold during a single request (e.g. 4k, 8k, 32k, 128k tokens). | Direct linear driver of **KV Cache VRAM consumption**. Larger context limits risk sudden CUDA Out-of-Memory crashes. |
| **Parameters (e.g. 7B, 70B)** | The number of frozen floating-point numbers (weights) that make up the neural network. (7B = 7 Billion floats). | Determines the **minimum static RAM/VRAM floor** needed to load the model into hardware. |
| **Weights** | The static arrays of numbers saved inside binary files (`.safetensors`, `.gguf`, or `.onnx`). | The actual static build artifact you download, copy via `rsync`, and mount into inference containers. |

---

## 2. The AI Lifecycle: Where Platform Engineers Live

```text
[ Pre-Training ] ──► [ Fine-Tuning (LoRA/SFT) ] ──► [ Inference / Serving ]
 10,000+ GPUs           Specializing on data             Deploying the frozen
 (OpenAI, Meta)         (Data Science / ML)               weights in production
      [OFF-LIMITS]             [OFF-LIMITS]                   [YOUR ARENA (100%)]
```

* **Pre-Training:** Training a foundation model from scratch. Requires massive clusters, high-speed InfiniBand networking, and months of compute. Handled by a handful of frontier research labs.
* **Fine-Tuning:** Taking an existing frozen base model and adjusting a small subset of weights on domain-specific datasets (e.g., medical or code data). Handled by internal ML researchers.
* **Inference (Serving):** Loading the finished, frozen model into memory to answer live user requests via HTTP/gRPC APIs in real time. **This is 100% Platform Engineering, Systems Operations, and SRE.**

---

## 3. The Two Phases of LLM Inference

Large Language Model generation is fundamentally divided into two distinct computational phases:

```text
[ Input Prompt: "Explain Linux kernel memory paging in detail..." ]
                             │
                             ▼
  ┌───────────────────────────────────────────────────────┐
  │ 1. PREFILL PHASE (Prompt Processing)                  │
  │  - Processes all prompt tokens in parallel            │
  │  - Highly COMPUTE-BOUND (Matrix Multiplication / GEMM)│
  │  - Saturates GPU Tensor Cores / Compute Units         │
  │  - Populates initial Key-Value (KV) cache entries     │
  └───────────────────────────────────────────────────────┘
                             │
                             ▼ (First token emitted -> TTFT recorded)
  ┌───────────────────────────────────────────────────────┐
  │ 2. DECODE PHASE (Autoregressive Token Generation)     │
  │  - Emits tokens sequentially, one by one              │
  │  - Highly MEMORY-BANDWIDTH BOUND                      │
  │  - Must load ALL model weights from memory for EVERY  │
  │    single output token emitted                        │
  │  - Grows the KV cache by 1 entry per token per layer  │
  └───────────────────────────────────────────────────────┘
```

### Why This Distinction Matters
* **Prefill:** Constrained by FLOPS (Floating Point Operations per Second). Fast computation hardware makes prefill faster.
* **Decode:** Constrained by memory bandwidth (GB/s). How fast the GPU or unified memory can shovel billions of parameters into the compute cores determines token generation speed.

---

## 4. Key Latency & Performance Metrics

* **TTFT (Time-To-First-Token):**
  * The duration between the client sending the request and receiving the very first streaming token chunk.
  * **Composed of:** Network transit time + Queue wait time + Prefill computation time.
  * **Impact:** Dictates perceived user responsiveness (UI responsiveness).
* **TPOT (Time-Per-Output-Token):**
  * The average time required to generate each subsequent token during the decode phase:
    $$\text{TPOT} = \frac{\text{Total Generation Time} - \text{TTFT}}{\text{Number of Output Tokens} - 1}$$
  * Also referred to as **Inter-Token Latency (ITL)**.
  * **Impact:** Dictates perceived reading speed (typewriter effect). Human reading speed is approx. 5–8 tokens/sec (125–200 ms TPOT).
* **Throughput (Tokens per Second):**
  * Total tokens (input + output) generated across all concurrent users per second:
    $$\text{System Throughput} = \frac{\sum \text{Tokens Generated}}{\text{Total Elapsed Seconds}}$$
  * Under concurrency, system throughput increases while individual user TPOT degrades due to queueing and memory contention.

---

## 5. Precision, Bit-Width & Quantization

Quantization is the process of compressing model weights from high-precision floating-point representations to lower-bit integers.

| Format / Precision | Bytes per Weight | Memory Required (7B Model) | Operational Trade-Off |
| :--- | :--- | :--- | :--- |
| **FP32 (Single Precision)** | 4.0 bytes | ~28 GB | Maximum numerical precision; completely wasteful for inference. |
| **FP16 / BF16 (Half Precision)** | 2.0 bytes | ~14 GB | Industry standard baseline for datacenter inference (A100/H100/L4). |
| **INT8 (8-bit Quantized)** | 1.0 byte | ~7.0 GB | 50% memory reduction with negligible perceptual accuracy loss. |
| **INT4 / Q4_K_M (4-bit)** | ~0.55 bytes | ~3.9 GB | 75% memory reduction; enables running 7B models on 4GB consumer GPUs or Apple Silicon. |

### Common Weight Formats
* **GGUF:** Unified single-file binary format (weights + architecture metadata) optimized for fast CPU/Metal/consumer GPU inference via `llama.cpp`.
* **AWQ / GPTQ:** 4-bit quantization formats optimized specifically for NVIDIA Tensor Cores in production engines like `vLLM`.
* **Safetensors:** Fast, memory-mapped (`mmap`) format developed by Hugging Face that eliminates arbitrary code execution risks found in legacy Python pickle files.

---

## 6. The 3 Buckets of VRAM / Silicon Memory

When an LLM runs, silicon memory (VRAM or Unified RAM) is consumed by three distinct pools:

$$\text{Total Memory} = \text{Model Weights} + \text{KV Cache Memory} + \text{Activation Scratchpad}$$

1. **Model Weights (Static):**
   * Constant memory required to hold the model parameters:
     $$\text{Weights (Bytes)} = \text{Parameters} \times \text{Bytes per Parameter}$$
     * FP16 / BF16: 2 bytes per parameter (e.g., 7B model $\approx$ 14 GB).
     * INT8: 1 byte per parameter (e.g., 7B model $\approx$ 7 GB).
     * INT4: 0.5 bytes per parameter (e.g., 7B model $\approx$ 3.5 GB).
2. **KV Cache (Dynamic & Expanding):**
   * Stores the past Key and Value attention matrices for each token to avoid recomputing attention history.
   * Scales linearly with: Batch size ($B$), Context sequence length ($S$), Number of layers ($L$), Number of attention heads ($H$), and Head dimension ($D$):
     $$\text{KV Cache Size (Bytes)} = 2 \times B \times S \times L \times H_{KV} \times D \times \text{Bytes per Element}$$
   * Under long contexts (e.g., 32K tokens) or high concurrency, the KV cache can exceed the size of the model weights!
3. **Activation & Scratchpad Memory (Transient):**
   * Ephemeral memory allocated by CUDA or Metal to hold intermediate activation tensors during matrix multiplications. Typically 400 MB to 1.5 GB.

---

## 7. Concurrency & Serving Optimizations

* **Static Batching:** Waits for a fixed number of requests to arrive, processes them together, and waits until the *longest* request finishes before accepting new requests. Causes severe Head-of-Line (HoL) blocking.
* **Continuous Batching (Iteration-Level Scheduling):** Dynamically injects newly arriving requests into the decode loop at each token generation step. As soon as a request finishes, its memory is freed and a new request is admitted without waiting for others.
* **PagedAttention:** A memory allocation algorithm that partitions the KV cache into fixed-size physical memory blocks (similar to virtual memory paging in Linux), eliminating internal and external memory fragmentation.
* **Prefix Caching (Prompt Caching):** Reuses precomputed KV cache blocks for shared prompt prefixes (e.g., system instructions or few-shot examples), reducing prefill TTFT from seconds to milliseconds.

---

## 8. Sampling & Decoding Parameters

Runtime parameters passed in HTTP request payloads that control how the model selects its next token:

* **Temperature (0.0 to 1.0):**
  * `0.0 – 0.2`: Sharpens probability distribution. Picks the mathematically highest probability token. Essential for coding, JSON output, and factual extraction.
  * `0.7 – 1.0`: Flattens probability distribution. Allows lower-probability tokens to be picked, generating varied and creative responses.
* **Top-P (Nucleus Sampling):** Selects candidate tokens only from the top cumulative probability mass (e.g. `0.90`), dynamically trimming unlikely tail tokens.
* **Top-K:** Restricts candidate selection to the top $K$ most likely tokens (e.g. top 40).

---

## 9. The Rosetta Stone: AI Terms ⟶ Systems Equivalents

| AI / ML Concept | Classical Systems / DevOps Equivalent |
| :--- | :--- |
| **Model Weights** | Static binary artifact / executable image |
| **KV Cache** | In-memory session state / dynamic cache buffer |
| **PagedAttention** | Linux Virtual Memory Paging (4KB physical blocks) |
| **Continuous Batching** | Dynamic OS Process Preemptive Scheduling |
| **Quantization** | Binary compression / reduced data bit-widths |
| **Embedding** | Hash / high-dimensional indexing key |
| **RAG** | Database query + template rendering |
| **TTFT (Time To First Token)** | Request Queue Wait Time + Prefill Handshake Latency |
| **TPS (Tokens Per Second)** | Streaming I/O Throughput |
| **CUDA OOM** | Linux Kernel OOMKill (Exit Code 137) |
