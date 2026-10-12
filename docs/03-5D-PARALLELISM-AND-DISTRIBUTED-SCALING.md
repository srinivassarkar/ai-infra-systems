# The 5D Parallelism Blueprint: Scaling LLMs Across Distributed GPUs

This blueprint synthesizes the distributed systems architecture taught in top-tier GPU engineering curriculums (including the MIT/Vizuara 5D Parallelism program and the Hugging Face Ultra Scale Playbook). It covers the five mathematical dimensions of distributed model execution: **Data**, **Tensor**, **Pipeline**, **Sequence**, and **Context Parallelism**.

---

## 1. The Physical Constraint: Why Single-GPU Serving Fails

A model's memory footprint during execution consists of three distinct components:
$$\text{Total Memory} = M_{\text{weights}} + M_{\text{KV\_cache}} + M_{\text{activations}}$$

For a 70B parameter model in 16-bit precision ($\text{BF16}$):
* **Model Weights:** $70 \times 10^9 \times 2\text{ bytes} = 140\text{ GB}$.
* A single NVIDIA H100 (80 GB) or A100 (80 GB) cannot physically load the static weights, let alone allocate a single KV cache block.
* Even on hardware where weights fit (e.g., Apple M3 Ultra with 256 GB Unified Memory), the arithmetic intensity of single-stream decode ($AI \approx 1$) bounds execution strictly to memory bandwidth (~800 GB/s).

To scale training and high-throughput serving beyond single-device memory and bandwidth walls, we shard computation across a multi-dimensional GPU grid:
$$\text{World Size } (N) = \text{DP} \times \text{TP} \times \text{PP} \times \text{SP} \times \text{CP}$$

---

## 2. Dimension 1: Data Parallelism (DP & ZeRO / FSDP)

### 2.1 Classical Distributed Data Parallel (DDP)
* **Architecture:** Every GPU holds a complete replica of the model weights and optimizer states.
* **Mechanism:** The global batch is split across GPUs. Each GPU computes forward/backward passes on its local slice. Gradients are synchronized via an `AllReduce` operation over NCCL before weight update.
* **Limitation:** Redundant memory consumption. If the model exceeds single-GPU VRAM, classical DDP cannot run.

### 2.2 ZeRO (Zero Redundancy Optimizer) & PyTorch FSDP
ZeRO shards redundant states across the data-parallel ranks without changing the underlying mathematical model:

| Stage | What is Sharded Across Ranks | Memory Reduction | Communication Cost |
| :--- | :--- | :--- | :--- |
| **ZeRO-1** | **Optimizer States:** Each GPU stores only $1/\text{DP}$ of the Adam states (fp32 momentum and variance). | $4\times$ memory savings | Zero extra communication ($1\times \text{AllReduce}$ on gradients). |
| **ZeRO-2** | **Optimizer States + Gradients:** Each GPU stores only its assigned slice of gradients. | $8\times$ memory savings | Zero extra communication (converts `AllReduce` to `ReduceScatter`). |
| **ZeRO-3 (FSDP)** | **Optimizer States + Gradients + Model Weights:** No GPU holds full weights. Weights are fetched on-the-fly right before the forward/backward layer pass via `AllGather`, then immediately discarded. | Scales linearly with GPU count: $\approx 1/\text{DP}$ | **$1.5\times$ extra communication overhead** across the interconnect. |

---

## 3. Dimension 2: Tensor Parallelism (TP / Megatron-LM)

When a single transformer layer does not fit on one GPU, Tensor Parallelism shards the linear weight matrices inside the Multi-Head Attention (MHA) and Feed-Forward Network (FFN) blocks.

### 3.1 Megatron Slicing Strategy
A standard Transformer layer consists of Attention and MLP (two matrix multiplications each):
$$\text{Attention: } Y = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d}}\right) V \cdot W_O$$
$$\text{MLP: } Z = \text{GELU}(X \cdot W_1) \cdot W_2$$

To minimize communication, Megatron-LM alternates between **Column-Parallel** and **Row-Parallel** slicing:

```
Multi-Layer Perceptron (FFN) Sharding:
Input X
   ├──▶ GPU 0: [X] × [W1_col0] ──▶ GELU ──▶ [H0] × [W2_row0] ──┐
   └──▶ GPU 1: [X] × [W1_col1] ──▶ GELU ──▶ [H1] × [W2_row1] ──┴──▶ AllReduce ──▶ Output Z
```

1. **First Linear Layer ($W_1$): Column Parallel.**
   * Weight $W_1$ is split vertically: $W_1 = [W_{1,0} \mid W_{1,1}]$.
   * Each GPU computes: $H_i = \text{GELU}(X \cdot W_{1,i})$.
   * **Communication:** Zero. No communication needed between column operations.
2. **Second Linear Layer ($W_2$): Row Parallel.**
   * Weight $W_2$ is split horizontally: $W_2 = \begin{bmatrix} W_{2,0} \\ W_{2,1} \end{bmatrix}$.
   * Each GPU computes: $Z_i = H_i \cdot W_{2,i}$.
   * **Communication:** One single `AllReduce` sum combines the outputs: $Z = Z_0 + Z_1$.
3. **Communication Budget:** Exactly **two `AllReduce` operations per transformer layer** (one for Attention output, one for MLP output).
4. **Hardware Requirement:** TP requires ultra-high bandwidth (NVLink $\ge 600\text{ GB/s}$). Running TP over standard PCIe or Ethernet introduces severe communication latency bubbles.

---

## 4. Dimension 3: Pipeline Parallelism (PP / 1F1B)

When a model is too deep (e.g. 80 layers) to fit on one node, Pipeline Parallelism splits layers sequentially across stages:
* **GPU 0:** Layers 1–20.
* **GPU 1:** Layers 21–40.
* **GPU 2:** Layers 41–60.
* **GPU 3:** Layers 61–80.

### 4.1 The Pipeline Bubble Problem
Naive sequential execution leaves downstream GPUs idle waiting for activations from upstream GPUs.
* **GPipe:** Divides a batch into $m$ micro-batches.
* **The Bubble Fraction ($F$):**
  $$F = \frac{p - 1}{m + p - 1}$$
  where $p$ is the number of pipeline stages and $m$ is micro-batches. As $m \gg p$, idle bubble time shrinks, but activation memory swells linearly.

### 4.2 The 1F1B Schedule (One Forward, One Backward)
In the 1F1B steady state, each GPU alternates between executing one forward micro-batch and one backward micro-batch.
* Activations are freed immediately after their corresponding backward step runs.
* Caps peak activation memory to $O(p)$ instead of $O(m)$, eliminating OOM risks on deep models.

---

## 5. Dimension 4: Sequence Parallelism (SP)

In Megatron-LM Tensor Parallelism, the Attention and MLP matrices are sharded, but the **LayerNorm and Dropout** operations remain duplicated across all TP ranks.
At sequence lengths $\ge 8\text{K}$, intermediate activation tensors stored for LayerNorm dominate VRAM.

* **Megatron Sequence Parallelism:** Replaces the duplicate LayerNorm with a sequence-split operation.
* The activation tensor of shape $[S, B, H]$ is sharded along sequence dimension $S$ across the TP group:
  $$\text{Shape per GPU: } \left[\frac{S}{\text{TP}}, B, H\right]$$
* Converts the `AllReduce` communication into an `AllGather` before column-parallel GEMM and a `ReduceScatter` after row-parallel GEMM.
* **Communication Volume:** Zero net increase in bytes transferred over NVLink, but **activation memory drops by exactly $\text{TP}$ times**.

---

## 6. Dimension 5: Context Parallelism (CP / Ring Attention)

When sequence lengths scale to 128K, 512K, or 1M+ tokens, attention memory scales quadratically ($O(S^2)$). Sequence Parallelism alone cannot prevent OOM because computing attention requires comparing token $i$ with all prior tokens $0 \dots i$.

### 6.1 Ring Attention Mechanics
Context Parallelism shards the sequence across a logical ring of GPUs:
1. Each GPU holds a local chunk of Query ($Q_i$), Key ($K_i$), and Value ($V_i$) of size $S / \text{CP}$.
2. In step 0, each GPU computes local self-attention between its local $Q_i$ and local $K_i, V_i$.
3. Concurrently, using non-blocking point-to-point communication (`P2P` / ring shift), GPU $i$ sends its $K_i, V_i$ blocks to GPU $i+1$, and receives $K_{i-1}, V_{i-1}$ from GPU $i-1$.
4. GPU $i$ computes cross-attention between its stationary $Q_i$ and the incoming key/value blocks, accumulating the online softmax statistics.
5. After $\text{CP}$ ring shifts, the full attention matrix is computed.
* **Maximum Sequence Length:** Scales linearly with the number of GPUs ($\text{CP} \times \text{VRAM}$), enabling **1-million token context windows without a single out-of-memory crash**.

---

## 7. The Unified Scaling Playbook: How to Partition Any Fleet

When sizing a production distributed cluster for large models, apply the hierarchy from fastest interconnect to slowest:

```
Hardware Hierarchy:
1. Intra-Socket (SRAM / HBM): Single GPU
2. Intra-Node (NVLink / NVSwitch): ~900 GB/s ──▶ Apply Tensor Parallelism (TP) & Sequence Parallelism (SP)
3. Intra-Rack (Infiniband / RoCE RDMA): ~400 Gbps ──▶ Apply Pipeline Parallelism (PP) & Context Parallelism (CP)
4. Inter-Rack / Cluster (Standard Fabric): ~100 Gbps ──▶ Apply Data Parallelism (DP / ZeRO / FSDP)
```

| Decision Rule | Recommended Strategy |
| :--- | :--- |
| **Model fits on 1 node, need high throughput:** | Set $\text{TP} = 1$, use Data Parallelism with FSDP/ZeRO-3. |
| **Model exceeds 1 GPU, fits inside 1 node (e.g. 70B on 8x H100):** | Set $\text{TP} = 8$ across NVLink. Keep $\text{PP} = 1$. |
| **Model exceeds 1 node (e.g. 405B model across 32x H100):** | Set $\text{TP} = 8$ (intra-node), $\text{PP} = 4$ (inter-node), $\text{DP} = 1$. |
| **Long Context Serving ($> 64\text{K}$ tokens):** | Enable Context Parallelism ($\text{CP} = 4$ or $8$) with Ring Attention. |

---
*Synthesized for `ai-infra-systems` engineering reference. Zero fluff. First principles.*
