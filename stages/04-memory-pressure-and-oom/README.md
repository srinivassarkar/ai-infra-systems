# Stage 04: Memory Pressure, KV Cache Bloat & OOM Prevention

## Objective
Mathematically model and experimentally verify memory consumption under deep context sequences (1K up to 32K tokens) and induce controlled Out-of-Memory (OOM) states to design prevention gates.

## Key Questions
1. How does context sequence length linearly inflate the KV cache memory footprint?
2. What happens when VRAM is exhausted on discrete GPUs (CUDA OOM vs PCIe memory spilling)?
3. How can an admission controller calculate token memory budgets before admitting requests?
