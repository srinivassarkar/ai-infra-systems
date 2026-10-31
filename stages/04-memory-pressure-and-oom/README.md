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
* $H_{\mathrm{kv}}$: number of Key-Value attention heads (`kv_heads`). For GQA, $H_{\mathrm{kv}} = \frac{H_q}{\text{group\_ratio}}$ (e.g. Qwen 2.5 7B has 28 Q heads, 4 KV heads $\rightarrow 7\times$ reduction).
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

> [!NOTE] DeepSeek-V3 Multi-Head Latent Attention (MLA)
> Under plain MHA at 128K context, DeepSeek-V3 (61 layers, 128 heads, 128 head dim) would demand **$488.0\ \mathrm{GiB}$ of VRAM for a single sequence**. By compressing $K$ and $V$ into a shared low-dimensional latent vector ($d_c = 512, d_r = 64$), DeepSeek cuts single-sequence memory to **$8.58\ \mathrm{GiB}$** (a **$98.2\%$ reduction**), making 128K concurrency economically viable.

---

## 6. The Context Memory Crossover Point

In autoregressive decode, model weights are read once per step.
* For `Qwen2.5-Coder-1.5B-Q4_K_M`, static weights require reading **$986\ \mathrm{MB}$** per step.
* At $14,336\ \mathrm{bytes/token}$, when context length reaches:
  $$C_{\mathrm{crossover}} = \frac{986 \times 1024^2\ \mathrm{bytes}}{14,336\ \mathrm{bytes/token}} \approx \mathbf{72,120\ \mathrm{tokens}}$$
* Beyond 72,000 tokens (or at 2,048 context across 36 concurrent users), **memory bandwidth spent streaming the KV cache exceeds memory bandwidth spent streaming model weights**. Decode speed degrades non-linearly even when VRAM has not run out.

---

## 7. Tooling & Simulation

1. **Fleet Capacity Planner:** `python3 kv_cache_admission_controller.py --matrix`
2. **vLLM PagedAttention & Prefix Caching Simulator:** `python3 paged_kv_cache_simulator.py`
3. **VRAM Footprint Estimator:** `python3 vram_calculator.py --params 1.5 --precision q4_k_m --context 32768`
