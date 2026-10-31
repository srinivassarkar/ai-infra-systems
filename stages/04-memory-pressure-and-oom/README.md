# Stage 04: Memory Pressure, KV Cache Bloat & OOM Prevention

## 1. Objective
Mathematically model and experimentally verify memory consumption under deep context sequences (1K up to 128K tokens) and design production admission control mechanisms to eliminate both GPU VRAM memory exhaustion and Host RAM kernel OOM crashes.

---

## 2. Host RAM OOM (Exit Code 137) vs. GPU VRAM Out-of-Memory

A common point of confusion in AI infrastructure is conflating Host RAM exhaustion with GPU VRAM exhaustion:

| Failure Type | Trigger Mechanism | Process Behavior | Error / Exit Code |
| :--- | :--- | :--- | :--- |
| **Host System OOM** | Host physical RAM or cgroups container memory limit exceeded. | Linux kernel `oom-killer` sends an uncatchable `SIGKILL` (signal 9) to the process. | **Exit Code 137** ($128 + 9$). Container terminated. |
| **GPU VRAM OOM** | GPU GDDR6 / HBM capacity exceeded during tensor allocation or context pre-allocation. | CUDA runtime returns `cudaErrorMemoryAllocation` inside process memory. | **In-process crash / abort** (e.g. `ggml_cuda_init: failed to allocate buffer`). |

---

## 3. Allocation Paradigms: `llama.cpp` vs. `vLLM`

Production inference engines handle KV cache memory using fundamentally different paradigms:

1. **Static Pre-Allocation (`llama.cpp` / `llama-server`):**
   * Allocates the full context buffer upfront at process startup based on `-c` and `--parallel`.
   * **Failure Mode:** It *cannot* trigger a dynamic runtime VRAM OOM mid-request. If `-c` is too large for physical VRAM, it fails immediately at boot. If an incoming sequence exceeds `-c`, the server rejects or truncates the sequence.
2. **Dynamic Paged Allocation (`vLLM` / `SGLang`):**
   * Allocates memory dynamically in non-contiguous 16-token physical blocks (PagedAttention).
   * **Failure Mode:** Subject to dynamic runtime VRAM exhaustion. When GPU physical blocks are exhausted, the server must either shed requests, preempt active streams (swapping KV blocks to CPU RAM), or abort.

*Note: The Admission Controller and PagedAttention simulator in this stage model **vLLM-style dynamic systems**, establishing token reservation budgets before requests enter the execution queue.*

---

## 4. First-Principles KV Cache Memory Formulation

### The Mathematical Formula
For Multi-Head Attention (MHA) and Grouped-Query Attention (GQA):

$$\text{KV Cache Bytes per Token} = 2 \times L \times H_{\mathrm{kv}} \times D_{\mathrm{head}} \times B_{\mathrm{element}}$$

Where:
* Factor of $2$ accounts for separate Key ($K$) and Value ($V$) tensors.
* $L$: number of Transformer layers (`layers`).
* $H_{\mathrm{kv}}$: number of Key-Value attention heads (`kv_heads`). For GQA, $H_{\mathrm{kv}} = \frac{H_q}{G}$ where $G$ is the group ratio (e.g. Qwen 2.5 7B has 28 Q heads, 4 KV heads $\rightarrow 7\times$ reduction).
* $D_{\mathrm{head}}$: dimension per head (`head_dim`).
* $B_{\mathrm{element}}$: bytes per tensor element ($B_{\mathrm{element}} = 2$ for FP16/BF16, $1$ for INT8/Q8_0, and $0.5$ for FP4/Q4_0).

$$\text{Total KV Cache Memory (GiB)} = \frac{2 \times L \times H_{\mathrm{kv}} \times D_{\mathrm{head}} \times C \times N \times B_{\mathrm{element}}}{1024^3}$$

Where $C$ is the sequence context length and $N$ is the concurrent active user count.

---

## 5. Architectural Attention Comparison (MHA vs. GQA vs. MLA)

Under deep context ($128\mathrm{K}$ tokens, $N = 1$, FP16 precision):

| Architecture | Model Archetype | Layers | KV Heads | Head Dim | KV Bytes / Token | 8K Context (1 User) | 128K Context (1 User) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MHA** (Multi-Head) | GPT-3 (175B) | 96 | 96 | 128 | 4,718,592 B (4.50 MiB) | 35.16 GiB | **562.50 GiB** *(Physically Impossible)* |
| **GQA** (Grouped-Query) | Llama 3.1 (8B) | 32 | 8 | 128 | 131,072 B (128 KiB) | 0.98 GiB | **15.63 GiB** |
| **GQA** (Grouped-Query) | Qwen 2.5 (7B) | 28 | 4 | 128 | 57,344 B (56 KiB) | 0.43 GiB | **6.84 GiB** |
| **GQA** (Grouped-Query) | Qwen 2.5 Coder (1.5B)| 28 | 2 | 64 | 14,336 B (14 KiB) | 0.11 GiB | **1.71 GiB** |
| **MLA** (Latent Attention)| DeepSeek-V3 (671B)| 61 | 1 (latent)| 576 | 70,272 B (68.6 KiB) | 0.54 GiB | **8.58 GiB** *(98.2% reduction vs MHA)* |

---

## 6. The Context Memory Crossover Point

In autoregressive decode, model weights are read once per step.
* For `Qwen2.5-Coder-1.5B-Q4_K_M`, static weights require reading **$986\ \mathrm{MB}$** per step.
* At $14,336\ \mathrm{bytes/token}$, when context length reaches:
  $$C_{\mathrm{crossover}} = \frac{986 \times 1024^2\ \mathrm{bytes}}{14,336\ \mathrm{bytes/token}} \approx \mathbf{72,120\ \mathrm{tokens}}$$
* Beyond 72,000 tokens (or at 2,048 context across 36 concurrent users), **memory bandwidth spent streaming the KV cache exceeds memory bandwidth spent streaming model weights**. Decode speed degrades non-linearly even when VRAM has not run out.

---

## 7. Reproduction Commands

Run these directly on any machine:

```bash
cd ~/ai-infra-systems/stages/04-memory-pressure-and-oom

# 1. Run the PagedAttention Block Allocation Simulator:
python3 paged_kv_cache_simulator.py

# 2. Run the VRAM Budget Estimator for 8 users at 32K context:
python3 vram_calculator.py --params 1.5 --precision q4_k_m --context 32768 --concurrency 8

# 3. Run the Token Admission Controller:
python3 kv_cache_admission_controller.py --matrix
```

---

## 8. Verified Fleet Memory Footprint & Paging Results

Here are the real outputs produced by our tools on the physical hardware:

### A. VRAM Budget Calculation (8 Concurrent Users @ 32,768 Context Limit)
```text
============================================================
VRAM BUDGET ESTIMATOR: 1.5B Model (Q4_K_M)
   Context: 32768 tokens | Concurrency: 8 users
============================================================
1. Model Weights:      0.81 GB
2. KV Cache (Dynamic): 28.00 GB
3. CUDA/Scratch Base:  0.35 GB
------------------------------------------------------------
Total VRAM Needed:     29.16 GB (29860 MB)
============================================================
```

### What This Means Across Our 3 Hardware Nodes:
- **Node 1: GTX 1050 Ti (4 GB VRAM):** Instant crash. Total demand (29.16 GB) exceeds physical VRAM by $7.3\times$.
- **Node 3: NVIDIA L4 (24 GB VRAM):** Also crashes! Even a $24\ \mathrm{GB}$ enterprise GPU cannot hold 8 users at full 32K context ($29.16\ \mathrm{GB} > 24\ \mathrm{GB}$). To fit on an L4, context must be capped to 16K or quantized to INT8/Q4.
- **Node 2: Mac Studio M3 Ultra (256 GB Unified Memory):** Passes with ease. $29.16\ \mathrm{GB}$ uses only **11.4%** of the 256 GB pool, leaving over **226 GB of headroom** free.

### B. PagedAttention & Prefix Sharing Verification
Running `paged_kv_cache_simulator.py` proved:
1. **Zero External Fragmentation:** Logical token sequences are broken into 4-token blocks and mapped to physical block tables without requiring contiguous VRAM allocation.
2. **Prefix Compute Skip:** When Request 2 arrived sharing an 8-token prompt prefix with Request 1, it reused physical blocks `[0, 1]` via reference counting (`ref_count = 2`), completely skipping prompt prefill computation.

For the full architectural deep dive, read the [Stage 04 Full Report](REPORT.md).
