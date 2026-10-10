# Systems Guide: The Roofline Model & Arithmetic Intensity in LLM Serving

```
   Achieved TFLOPs/s
         ▲
         │                        Compute Ceiling (Flat)
Compute  │                    ┌────────────────────────────
Ceiling  │                   /
         │                  /  
         │                 / ◄── Ridge Point = Peak Compute / Bandwidth
         │                /
Bandwidth│               /
Ceiling  │              /
(Slope)  │             /  Memory-Bandwidth Bound (AI < Ridge)
         └────────────┴──────────────────────────────────────►
         0            Ridge Point              Arithmetic Intensity (FLOPs/Byte)
```

---

## 1. What is Arithmetic Intensity?

**Arithmetic Intensity (AI)** measures the ratio of computation performed to memory transferred between physical memory (HBM/DRAM) and processing registers (SRAM/ALU):

$$\text{Arithmetic Intensity (AI)} = \frac{\text{Total FLOPs Performed}}{\text{Bytes Moved Across Memory Bus}}$$

- **Unit:** $\text{FLOPs per Byte}$.
- **Significance:** Arithmetic Intensity does not measure execution time; it dictates **which physical hardware bottleneck caps system performance**.

---

## 2. The Roofline Model & The Ridge Point

The Roofline sits two hardware physical limits on a log-log plot:

1. **Memory Bandwidth Ceiling (Slope):**
   $$\text{Attainable Performance} \le \text{AI} \times \text{Memory Bandwidth (TB/s)}$$
   When $\text{AI}$ is low, the compute cores are starved of data. Performance scales strictly with bus bandwidth.
2. **Compute Ceiling (Flat):**
   $$\text{Attainable Performance} \le \text{Peak Hardware Compute (TFLOPs/s)}$$
   When $\text{AI}$ is high, the memory bus can easily feed the cores. Performance is capped by ALU/Tensor Core clock cycles.

### The Ridge Point Formula
The intersection where a kernel transitions from memory-bound to compute-bound is the **Ridge Point**:

$$\text{Ridge Point (FLOPs/Byte)} = \frac{\text{Peak Compute Capacity (TFLOPs/s)}}{\text{Memory Bandwidth (TB/s)}}$$

### Hardware Ridge Points

| Physical Hardware | Precision | Peak Compute | Memory Bandwidth | Ridge Point (FLOPs/Byte) | Regime |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NVIDIA H100 SXM5** | FP16 Tensor | 989 TFLOPs | 3.35 TB/s | **295.2** | High compute ceiling |
| **NVIDIA H100 SXM5** | FP8 Tensor | 1,979 TFLOPs | 3.35 TB/s | **590.7** | Extreme compute ceiling |
| **NVIDIA A10G (AWS g5)** | FP16 Tensor | 125 TFLOPs | 0.60 TB/s | **208.3** | Moderate |
| **Mac Studio M3 Ultra** | FP16 Metal | ~56 TFLOPs | 0.80 TB/s | **70.0** | High bandwidth relative to compute |
| **GTX 1050 Ti (Edge)** | FP32 CUDA | 2.1 TFLOPs | 0.112 TB/s | **18.7** | Constrained |

> [!IMPORTANT] Precision Pushes the Ridge Right
> Moving from FP16 to FP8 doubles the compute ceiling, pushing the Ridge Point from $295$ to $591$ FLOPs/byte. Lower precision demands higher batch sizes or deeper fusion to saturate the Tensor Cores.

---

## 3. The Asymmetry: Prefill vs Decode

Why does an LLM feel fast on the first token but slow during streaming? The Roofline explains the physical reason:

### A. Prefill Phase (Prompt Evaluation)
- Shape: Matrix multiply $X \cdot W$, where $X$ has shape $(N, d)$ and $N \gg 1$ (the prompt length).
- Math:
  $$\text{FLOPs} = 2 \cdot N \cdot d^2$$
  $$\text{Bytes} \approx 2d^2 + 2Nd \approx 2d^2 \quad (\text{dominated by weight matrix } W)$$
  $$\text{AI}_{\text{prefill}} \approx \frac{2Nd^2}{2d^2} \approx N \text{ FLOPs/Byte}$$
- **Verdict:** For a 512-token prompt, $\text{AI} \approx 512$. This exceeds the H100 Ridge Point ($295$).
- **Prefill is Compute-Bound.** Tensor cores run at high saturation ($\ge 60\%$). Prefill dictates **Time-To-First-Token (TTFT)**.

### B. Decode Phase (Token Generation)
- Shape: Matrix-vector multiply $X \cdot W$, where $X$ has shape $(1, d)$ (only the single newly generated token).
- Math:
  $$\text{FLOPs} = 2 \cdot 1 \cdot d^2 = 2d^2$$
  $$\text{Bytes} \approx 2d^2 \quad (\text{the entire weight matrix must be loaded from HBM})$$
  $$\text{AI}_{\text{decode}} \approx \frac{2d^2}{2d^2} \approx 1 \text{ FLOP/Byte}$$
- **Verdict:** For unbatched decode, $\text{AI} \approx 1$. On an H100, the maximum attainable throughput is $1 \times 3.35 \text{ TB/s} = 3.35 \text{ TFLOPs/s}$.
- Out of a potential 989 TFLOPs, **an unbatched decode step utilizes only 0.34% of the GPU.**
- **The dark secret of LLM serving:** Over 99% of the GPU hardware sits idle, starved for memory bandwidth. Decode dictates **Inter-Token Latency (ITL / TPOT)**.

---

## 4. The Three Production Levers to Increase Arithmetic Intensity

As an AI Infrastructure Engineer, your entire job in runtime serving is pushing $\text{AI}$ toward the Ridge Point:

### Lever 1: Continuous Batching (Amortizing Weight Reads)
In batch decode with batch size $B$, the weight matrix $W$ is loaded across the memory bus once, but multiplies $B$ token vectors simultaneously:

$$\text{AI}_{\text{batched decode}} \approx B \text{ FLOPs/Byte}$$

- Going from $B=1$ to $B=64$ increases Arithmetic Intensity from $1$ to $64$, climbing the bandwidth slope linearly and multiplying GPU throughput by $\sim 40\times$ without proportional memory bus inflation.

### Lever 2: Operator Fusion (Eliminating Intermediate HBM I/O)
- Standard attention reads inputs from HBM, writes intermediate $N \times N$ attention matrices to HBM, and reads them back for softmax ($\text{AI} \ll 1$).
- **FlashAttention** tiles the computation into on-chip SRAM, computing softmax in registers without writing $N \times N$ matrices back to HBM. This eliminates $\sim 80\%$ of memory bus traffic, doubling effective AI.

### Lever 3: KV Cache Compression (GQA, MLA, Quantization)
- **Quantization (FP16 $\rightarrow$ INT4 / FP8):** Halves the bytes loaded per weight across the memory bus, doubling Arithmetic Intensity.
- **Grouped-Query Attention (GQA) & Multi-Head Latent Attention (MLA):** Shrinks the KV cache size by $4\times$ to $14\times$, drastically reducing HBM round-trips during decode attention.
