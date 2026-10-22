# Stage 08: Serving Engine Optimizations & Continuous Batching

## Objective
Benchmark modern serving optimizations: comparing static batching against continuous iteration-level scheduling (via `llama.cpp` and `mlx-lm`) and measuring the latency, memory footprint, and throughput impact of weight quantization (FP16 vs GGUF Q4_K_M / INT4).

## Target Silicon
* **Node 2: Apple Mac Studio M3 Ultra (256 GB Unified Memory):** High-memory bandwidth baseline (~800 GB/s) running large-scale continuous batching.
* **Node 1: Custom Linux Box (GTX 1050 Ti 4 GB):** Constrained edge batching limits.

## Key Questions
1. How does continuous iteration-level batching eliminate head-of-line blocking under variable sequence lengths?
2. How does continuous batching boost system throughput without inflating individual user TPOT?
3. What is the measured speedup and VRAM reduction when quantizing weights from FP16 to 4-bit integer formats on actual hardware?
