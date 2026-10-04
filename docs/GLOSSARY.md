# Systems Engineering & LLM Inference Glossary

Foundational mental models, metrics, and definitions required to understand AI systems engineering.

---

## 1. The Two Phases of Inference

Large Language Model generation is fundamentally divided into two distinct computational phases:

```
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

## 2. Key Latency & Performance Metrics

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

## 3. The 3 Buckets of VRAM / Silicon Memory

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

## 4. Concurrency & Queueing Mechanics

* **Static Batching:** Waits for a fixed number of requests to arrive, processes them together, and waits until the *longest* request finishes before accepting new requests. Causes severe Head-of-Line (HoL) blocking.
* **Continuous Batching (Iteration-Level Scheduling):** Dynamically injects newly arriving requests into the decode loop at each token generation step. As soon as a request finishes, its memory is freed and a new request is admitted without waiting for others.
* **PagedAttention:** A memory allocation algorithm that partitions the KV cache into fixed-size physical memory blocks (similar to virtual memory paging in Linux), eliminating internal and external memory fragmentation.
