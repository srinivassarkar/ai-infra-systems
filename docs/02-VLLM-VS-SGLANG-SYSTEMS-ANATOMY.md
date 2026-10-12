# Deep Systems Anatomy: vLLM vs. SGLang Inference Engines

This document synthesizes the high-throughput inference engineering architectures powering modern LLM serving engines (**vLLM** and **SGLang**), based on first-principles systems analysis. It compares memory allocation strategies, batch scheduling lifecycles, speculative decoding economics, and cache-aware fleet routing.

---

## 1. High-Level Architectural Comparison

| Dimension | vLLM (Berkeley / Open Source) | SGLang (LMSYS / Stanford / Berkeley) | Systems Impact |
| :--- | :--- | :--- | :--- |
| **KV Cache Data Structure** | **PagedAttention Block Pool:** Physical blocks (e.g. 16 tokens/block) mapped via page tables. Hash-based Automatic Prefix Caching (APC). | **Radix Tree (RadixAttention):** Hierarchical prefix tree where edges store token chunks; shared prefixes across arbitrary branches are reused natively. | SGLang has zero overhead for multi-turn conversations, few-shot prompts, and agent tree-search; vLLM requires exact block-hash alignment. |
| **Scheduler Design** | **Iteration-Level Continuous Batching:** Synchronous step scheduler within token budget (`max_num_seqs`, `max_num_batched_tokens`). | **Overlap Scheduler:** Two-stage asynchronous pipeline. The CPU schedules step $N+1$ while the GPU executes step $N$ (via CUDA graphs). | SGLang eliminates CPU host-side scheduling bubbles between decode steps, achieving higher token throughput at small-to-medium batch sizes. |
| **Structured Outputs** | External schema parsing engine (Outlines / regex guidance) applied at logit masking layer. | **First-Class Frontend Runtime:** SGLang interpreter with fast regex/CFG parsers (`xgrammar`, `llguidance`) + KV cache reuse across `fork()`. | SGLang evaluates structured JSON schemas and parallel multi-branch LLM programs orders of magnitude faster. |
| **Attention Backend** | FlashAttention 2/3, FlashInfer, Triton PagedAttention. | FlashInfer (default for fast Paged/Radix kernels), FlashAttention 3, Triton. | FlashInfer specializes in ragged tensor batching and prefill/decode sharing. |
| **Prefill / Decode Disaggregation** | KV connectors protocol (**LMCache**, NIXL). | PD Disaggregation engine (**Mooncake** / KVCache store over RDMA/PCIe). | Decouples compute-bound prompt prefill from memory-bandwidth-bound decode. |
| **Fleet Routing** | vLLM Production Stack with Prefix-Aware Router. | Cache-Aware Router (`sglang.router`). | Prevents cache cold-starts by hashing prefix signatures to consistent worker pods in Kubernetes. |

---

## 2. Memory Accounting: PagedAttention vs. RadixAttention

### 2.1 The Traditional Problem (Pre-2023)
Standard HuggingFace / naive serving allocated contiguous VRAM buffers for the maximum sequence length (`max_model_len` = 4096 or 8192):
* **Internal Fragmentation:** A sequence requesting 100 tokens reserves 4096 tokens of KV memory.
* **External Fragmentation:** Dynamic memory allocators leave gaps unusable by larger incoming requests.
* **Result:** 60%–80% of GPU VRAM wasted on unallocated or pre-reserved memory slots.

### 2.2 vLLM PagedAttention: Operating System Virtual Memory for Tensors
* **Physical Memory:** Divided into non-contiguous physical blocks (typically 16 or 32 tokens).
* **Logical Blocks:** The sequence views KV cache as contiguous virtual memory.
* **Block Table:** Translates logical block index $\rightarrow$ physical block frame index.
* **Prefix Caching (APC):** Hashes the tokens of a completed block $H(\text{tokens}_{0..15})$. If an incoming prompt shares this exact block hash, the physical block is referenced rather than recomputed.
* **Limitation:** Sharing only occurs on discrete block boundaries (modulo 16). If a prefix is 31 tokens, the second block cannot be shared until token 32 is reached.

### 2.3 SGLang RadixAttention: Tree-Based Adaptive Prefix Caching
Instead of a flat block table, SGLang models all sequences resident in GPU VRAM as a **Radix Tree** (a compressed trie):
* **Nodes & Edges:** Edges store sequences of token IDs; nodes represent branch points.
* **Lookup:** An incoming prompt traverses the tree from the root. The matched path represents the longest common prefix.
* **Zero Boundary Constraints:** Prefixes match regardless of block alignment.
* **Eviction Policy:** When VRAM is under pressure, SGLang runs **LRU eviction** on the leaf nodes of the radix tree. Pinned root nodes (such as the system prompt) persist across thousands of requests.
* **Multi-Turn Workload Efficiency:** In multi-turn chat and agentic workflows (e.g. Tree of Thoughts, Reflexion), RadixAttention eliminates up to 90% of prefill computations.

```
       Root (System Prompt: "You are a senior SRE...")
         /                                    \
  User A (Incident 1)                  User B (Incident 2)
     /             \                         /
Turn 2 (OOM)    Turn 2 (Disk)          Turn 2 (DNS)
```

---

## 3. Scheduler Internals: Iteration-Level vs. Overlap Scheduling

### 3.1 vLLM Continuous Batching (Iteration-Level)
Traditional LLM inference used request-level batching: incoming requests waited for the slowest sequence in the batch to finish generating before returning.
vLLM introduced **iteration-level scheduling**:
1. At every decode step, the scheduler inspects the pool of waiting requests and running requests.
2. Finished sequences (emitting `<EOS>`) are immediately evicted, returning their physical blocks to the free pool.
3. Free block capacity is evaluated against `max_num_batched_tokens` and `max_num_seqs`.
4. If free capacity exists, new requests are admitted and run in **chunked prefill** mode (prefill tokens are broken into chunks and executed alongside ongoing decode tokens).

### 3.2 SGLang Overlap Scheduler (Zero-Overhead Scheduling)
In standard iteration-level scheduling, the CPU host sits idle while the GPU runs the forward pass; then the GPU sits idle while the CPU parses logits, samples tokens, updates the block table, and prepares the next batch tensors.
SGLang implements an **asynchronous overlap scheduler**:
* **Phase 1 (GPU Forward):** The GPU executes the current decode step $N$ using static **CUDA Graphs**.
* **Phase 2 (CPU Scheduling):** Concurrently on the host CPU, the scheduler inspects step $N-1$ outputs, samples step $N$ tokens, admits new arrivals, and builds input tensors for step $N+1$.
* **Result:** Near-zero inter-token gap on high-speed hardware, achieving up to 1.5x–2x higher token throughput on decode-heavy workloads.

---

## 4. Speculative Decoding Economics & The Acceptance Ceiling

Speculative decoding pairs a small, fast **Draft Model** ($M_D$) with a large, authoritative **Target Model** ($M_T$).

### 4.1 The Step Mechanics
1. $M_D$ autoregressively generates $K$ draft tokens in $K$ forward passes: $[t_1, t_2, \dots, t_K]$.
2. $M_T$ verifies all $K$ tokens simultaneously in **one single forward pass** (compute-bound prefill regime).
3. The target model accepts tokens up to the first rejection point $i \le K$ using modified rejection sampling:
   $$\text{Acceptance Condition: } U < \min\left(1, \frac{P_T(t_j)}{P_D(t_j)}\right)$$
4. If token $i$ is rejected, $M_T$ samples a replacement token, and tokens $[t_{i+1}, \dots, t_K]$ are discarded.

### 4.2 The Mathematical Speedup Condition
Let:
* $t_D$ = execution time of one draft token step.
* $t_T$ = execution time of one target model forward pass.
* $\alpha$ = average token acceptance rate ($0 \le \alpha \le 1$).
* $K$ = speculative draft length.

The expected number of accepted tokens per verification round is:
$$\mathbb{E}[\text{Accepted}] = 1 + \alpha + \alpha^2 + \dots + \alpha^K = \frac{1 - \alpha^{K+1}}{1 - \alpha}$$

The latency per verification round is:
$$\text{Latency}_{\text{round}} = K \cdot t_D + t_T$$

**The Speculative Speedup Ratio ($S$):**
$$S = \frac{\mathbb{E}[\text{Accepted}] \cdot t_T}{K \cdot t_D + t_T}$$

**When Speculative Decoding Destroys Performance ($S < 1$):**
1. **Low Acceptance Rate ($\alpha < 0.5$):** For reasoning-heavy, code-dense, or math tasks, the draft model guesses incorrectly. The system pays the draft latency $K \cdot t_D$ for zero net accepted tokens.
2. **High Batch Concurrency ($B \gg 16$):** At high batch sizes, the target model's decode pass is already compute-bound. Adding draft verification causes compute contention, dropping overall throughput.
3. **Speculative Decoding is for low-concurrency, latency-critical interactive chat ($B=1$), NOT high-throughput batching.**

---

## 5. Prefill-Decode (PD) Disaggregation

### 5.1 The Root Cause of ITL Spikes
* **Prefill:** Matrix multiplication ($GEMM$), compute-bound, high arithmetic intensity ($AI \gg 1$). Uses maximum SM capacity.
* **Decode:** Matrix-vector multiplication ($GEMV$), memory-bandwidth bound, low arithmetic intensity ($AI \approx 1$). Constantly waits on memory bus.
* **Colocation Conflict:** When a user submits a 10,000-token prompt to an engine actively serving 20 ongoing decode streams, the engine must either:
  1. Stall all 20 decode streams for 500ms to run the full prefill (creating massive P99 **Inter-Token Latency (ITL)** spikes), or
  2. Use **Chunked Prefill**, interleaving 512 prefill tokens with each decode step (reducing ITL spikes but lengthening **Time to First Token (TTFT)**).

### 5.2 Disaggregated Architecture
Separate the fleet into dedicated pools:
1. **Prefill Nodes:** Optimized for raw compute (high FP16/FP8 TFLOPs). Receive user prompts, run prefill, compute the full KV cache tensors.
2. **Decode Nodes:** Optimized for memory bandwidth (high HBM/UMA bandwidth). Receive the completed KV cache and run continuous decode streams.
3. **Interconnect (Mooncake / LMCache / NIXL):** Transfers KV cache blocks from Prefill Nodes to Decode Nodes over PCIe Gen5 or RDMA (RoCEv2/InfiniBand) in sub-millisecond windows.

```
Incoming Request
       │
       ▼
[ Prefill Node Pool ] ──(RDMA / Mooncake KV Transfer)──▶ [ Decode Node Pool ]
 (High TFLOPs/Compute)                                    (High Memory Bandwidth)
   Computes KV Cache                                        Generates Output Tokens
```

---

## 6. Fleet Architecture: Cache-Aware Routing on Kubernetes

### 6.1 The Load Balancer Trap
Traditional Kubernetes Ingress / Envoy / AWS ALB routes requests using:
* Round Robin
* Least Connections

In LLM serving, this causes catastrophic cache thrashing. If User A submits a multi-turn prompt:
* Turn 1 hits Replica Pod 1 (KV cache computed and stored in Pod 1's Radix tree).
* Turn 2 hits Replica Pod 2 (Cold cache! Pod 2 must re-run prefill for Turn 1 + Turn 2 from scratch).
* Turn 3 hits Replica Pod 3 (Cold cache again!).
* **Result:** Cache hit rate drops to 0%, GPU load triples, TTFT scales quadratically with conversation turns.

### 6.2 Cache-Aware Routing Architecture
A dedicated L7 inference router sits in front of the Kubernetes worker pods:
1. **Prefix Hashing:** The router hashes the static system prompt and session identifiers.
2. **Locality Table:** Tracks which worker pods hold which prefix hashes in their active KV block pool.
3. **Consistent Hashing with Virtual Nodes:** Directs requests sharing the same prefix to the warm pod, maximizing RadixAttention and PagedAttention reuse.
4. **Fallback / Eviction Sensing:** If the warm pod's KV pool utilization exceeds 90%, the router sheds traffic to the least-loaded replica.

---

## 7. Systems Metrics & Verification Harness

A production inference benchmark harness must measure four distinct dimensions simultaneously:

| Metric | Measurement Point | Target (Interactive SLA) | Root Cause if Degraded |
| :--- | :--- | :--- | :--- |
| **TTFT (Time to First Token)** | Gateway arrival to first chunk emitted | $< 800\text{ ms}$ (P99) | Queue wait time, cold KV cache, un-chunked prefill stall. |
| **TPOT (Time Per Output Token)** | Total generation time / generated tokens | $< 35\text{ ms}$ (P99) ($\approx 28\text{ tok/s}$) | Batch size past knee, memory bus saturation, deep KV cache read. |
| **ITL (Inter-Token Latency)** | Variance between consecutive token timestamps | Low jitter ($P99/P50 < 1.5$) | Heavy prefill requests interleaved into running decode batch. |
| **Token Goodput** | Tokens delivered strictly within SLA threshold | $> 98\%$ of total traffic | Capacity planning error, lack of HTTP 429 admission shedding. |

---
*Synthesized for `ai-infra-systems` engineering reference. Zero fluff. First principles.*
