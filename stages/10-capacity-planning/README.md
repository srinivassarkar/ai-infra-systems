# Stage 10: Capacity Planning, Token Economics & Hardware Sizing

## Objective
Develop a production capacity planning framework to calculate exactly how many concurrent users and tokens per second a given hardware configuration can support, and model cost-per-token economics.

## Key Questions
1. How do we calculate the maximum concurrent streams a GPU can serve before KV cache memory exhausts VRAM?
2. What is the financial cost per 1M tokens comparing on-prem / bare-metal hardware (M3 Ultra, edge Linux) versus public cloud APIs?
3. How do we design an automated queue-aware scaling policy based on KV cache saturation rather than CPU/GPU percentage?
