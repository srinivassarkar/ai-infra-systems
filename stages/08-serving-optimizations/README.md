# Stage 08: Serving Engine Optimizations & Empirical Unit Economics

## Objective
Benchmark modern serving optimizations on enterprise silicon: comparing static batching against continuous iteration-level scheduling (vLLM / PagedAttention) and measuring the latency, memory footprint, and empirical cost-per-token impact of weight quantization (FP16 vs AWQ / INT4).

## Target Silicon
* **NVIDIA L4 (24 GB GDDR6, Ada Lovelace, Compute Capability 8.9):** Dedicated cloud benchmark run (RunPod / GCP, ~$10-$15 total run cost).
* **Apple Mac Studio M3 Ultra (256 GB Unified Memory):** High-memory bandwidth comparison.

## Key Questions
1. How does PagedAttention eliminate memory fragmentation in the KV cache under variable sequence lengths?
2. How does continuous batching boost system throughput without inflating individual user TPOT?
3. What is the measured speedup and VRAM reduction when quantizing weights from BF16 to AWQ/INT4 on actual hardware?
4. What is the empirical cost per 1M tokens ($/1M tokens) derived from measured throughput and instance hourly rates:
   $$\text{Cost per 1M Tokens} = \left(\frac{\text{Instance Hourly Cost}}{\text{Measured Tokens per Hour}}\right) \times 1,000,000$$
