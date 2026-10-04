# Stage 08: Serving Engine Optimizations (Batching & Quantization)

## Objective
Benchmark modern serving optimizations: comparing static batching against continuous iteration-level scheduling (vLLM / PagedAttention) and measuring the latency vs memory footprint impact of weight quantization (FP16 vs AWQ / INT4).

## Key Questions
1. How does PagedAttention eliminate memory fragmentation in the KV cache under variable sequence lengths?
2. How does continuous batching boost system throughput without inflating individual user TPOT?
3. What is the measured speedup and VRAM reduction when quantizing weights from BF16 to AWQ/INT4 on actual hardware?
