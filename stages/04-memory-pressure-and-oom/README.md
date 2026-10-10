# Stage 04: Memory Pressure, KV Cache Bloat & OOM Prevention

## 1. Objective
Mathematically model and experimentally verify memory consumption under deep context sequences (1K up to 128K tokens) and design production admission control mechanisms to eliminate CUDA Out-of-Memory (OOM) fatal process crashes.

---

## 2. The Fundamental Law of KV Cache

In autoregressive language models, generating token $N+1$ requires attention over all preceding $N$ tokens. 

- **Naive Generation (No Cache):** The model recomputes $Q, K, V$ for the entire sequence at every step. Compute complexity scales quadratically ($O(N^2)$). Generating 500 tokens on top of a 100-token prompt requires 500 trillion FLOPs.
- **Cached Generation (KV Cache):** Key ($K$) and Value ($V$) tensors are computed once and stored in GPU memory (HBM/SRAM). New decode iterations only compute $Q, K, V$ for the single newest token.
- **The Core Trade-off:** The KV cache eliminates trillions of FLOPs, but **trades a compute problem for a memory-bandwidth and capacity problem**. Decode is pushed into the memory-bandwidth-bound regime ($AI \approx 1$ FLOP/byte).

```
[ Input Tokens ] ──▶ Compute K, V ──▶ Stored in KV Cache RAM (Grows linearly O(N))
                           ▲
[ Newest Token ] ──▶ Compute Q only ──▶ Attention against full KV Cache ──▶ Sample
```

---

## 3. Mathematical Formulation of KV Cache Memory

### The General Formula
For standard Multi-Head Attention (MHA) and Grouped-Query Attention (GQA):

$$\text{KV Cache Bytes per Token} = 2 \times \text{layers} \times \text{kv\_heads} \times \text{head\_dim} \times \text{bytes\_per\_element}$$

Where:
- Factor of $2$ accounts for separate Key ($K$) and Value ($V$) tensors.
- $\text{bytes\_per\_element} = 2$ for FP16/BF16, $1$ for INT8/FP8.
- $\text{kv\_heads} = \text{q\_heads}$ for MHA.
- $\text{kv\_heads} = \frac{\text{q\_heads}}{\text{group\_ratio}}$ for GQA (e.g., Llama 3.1 has 32 Q heads, 8 KV heads $\rightarrow 4\times$ reduction).
- $\text{kv\_heads} = 1$ for MQA (Multi-Query Attention).

### Total Memory Footprint
For a sequence length $S$ and batch size $B$:

$$\text{Total KV Cache Memory (Bytes)} = \text{Bytes per Token} \times S \times B$$

$$\text{Total KV Cache Memory (GiB)} = \frac{2 \cdot \text{layers} \cdot B \cdot \text{kv\_heads} \cdot \text{head\_dim} \cdot S \cdot \text{bytes\_per\_element}}{1024^3}$$

---

## 4. Attention Architecture Comparison

| Architecture | Model Example | Layers | KV Heads | Head Dim | Bytes/Token (FP16) | 8K Context (1 User) | 128K Context (1 User) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MHA** (Multi-Head) | GPT-3 (175B) | 96 | 96 | 128 | 4,718,592 B (4.50 MB) | 36.0 GB | 576.0 GB *(Impossible)* |
| **GQA** (Grouped-Query) | Llama 3.1 (8B) | 32 | 8 | 128 | 131,072 B (128 KB) | 1.00 GB | 16.0 GB |
| **GQA** (Grouped-Query) | Qwen 2.5 (7B) | 28 | 4 | 128 | 57,344 B (56 KB) | 0.44 GB | 7.0 GB |
| **MLA** (Latent Attention) | DeepSeek-V3 (671B) | 61 | 1 (latent) | 576 | 70,272 B (68.6 KB) | 0.54 GB | 8.58 GB *(93% reduction)* |

> [!NOTE] Why DeepSeek-V3 Uses MLA
> Under standard MHA at 128K context, DeepSeek-V3 would demand **478 GB of VRAM for a single user sequence**. By compressing $K$ and $V$ into a low-dimensional latent space ($d_c = 512, d_r = 64$), DeepSeek cuts single-sequence KV memory to **8.58 GB**, enabling high concurrency.

---

## 5. Capacity Equation & The Safe KV Pool

A production GPU node must partition memory into three strict zones:

$$\text{Total Physical Memory} = \text{Model Weights} + \text{Runtime Headroom} + \text{KV Cache Pool}$$

$$\text{Available KV Pool} = \text{VRAM}_{\text{total}} - \text{VRAM}_{\text{weights}} - \text{Headroom}_{\text{CUDA}}$$

Where:
- $\text{VRAM}_{\text{weights}} \approx \text{Params} \times \text{bytes\_per\_param}$ (e.g. Q4 = $\sim 0.55$ B/param).
- $\text{Headroom}_{\text{CUDA}}$: CUDA context, activation buffers, NCCL communication buffers, and OS graphics buffers (allocate 1.5 GB to 4.0 GB).

### Maximum Concurrency Formula:
$$\text{Max Concurrent Users} = \left\lfloor \frac{\text{Available KV Pool}}{\text{Bytes per Token} \times \text{Max Context Length}} \right\rfloor$$

---

## 6. The Production Admission Controller

Serving engines that accept requests without checking token budgets crash with **CUDA Out of Memory** (Exit Code 137). 

In enterprise architectures, an **Admission Controller** intercepts requests at the gateway:
1. Calculates $\text{Worst-Case Token Reservation} = P_{\text{prompt}} + G_{\text{max\_tokens}}$.
2. Computes required memory: $\text{Req RAM} = \text{Tokens} \times \text{Bytes per Token}$.
3. If $\text{Available KV Pool} \ge \text{Req RAM}$, the request is **Admitted (200 OK)** and token slots are locked.
4. If $\text{Available KV Pool} < \text{Req RAM}$, the request is **Shed (429 Rate Limit / 503 Overload)** or queued with backpressure.

---

## 7. Tooling: `kv_cache_admission_controller.py`

This repository includes a production CLI capacity planner and admission controller simulator.

### Run Fleet Capacity Matrix
```bash
python kv_cache_admission_controller.py --matrix
```

### Evaluate Single Request Admission
```bash
python kv_cache_admission_controller.py \
  --hw mac-studio-m3-ultra \
  --model llama-3.1-70b \
  --quant q4 \
  --prompt 8192 \
  --max-gen 2048
```
