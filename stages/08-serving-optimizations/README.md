# Stage 08: Serving Engine Optimizations & Ablation Matrix

## 1. Executive Summary & Objective

Rather than treating inference performance as a monolithic black box, production inference engineers isolate and benchmark the individual knobs of the serving engine.

Stage 08 systematically evaluates the **Engine Ablation Matrix**:
1. **FlashAttention (`-fa 1` vs `-fa 0`):** Tiled online softmax attention versus standard attention, evaluating prefill speedup and VRAM efficiency.
2. **Chunked Prefill (`-ub` micro-batch chunking):** Mitigating inter-token latency (ITL) stutter on ongoing streams during long prompt burst arrivals.
3. **KV Cache Quantization (`-ctk q8_0 -ctv q8_0` vs FP16 default):** Halving KV memory footprint to double concurrency within identical VRAM budgets.
4. **Prompt / Prefix KV Caching:** Reusing cached attention state across multi-turn sessions to collapse TTFT from hundreds of milliseconds down to sub-10ms.

---

## 2. Theoretical Ablation Matrix & Predictions

| Optimization Knob | CLI Parameter | Architectural Mechanism | Predicted Impact | Trade-Off / Constraint |
| :--- | :--- | :--- | :--- | :--- |
| **FlashAttention** | `-fa 1` | Fused GPU kernel tiling in SRAM; avoids writing $N \times N$ attention matrix to HBM. | $1.5\times - 3\times$ prefill speedup; reduced intermediate VRAM. | Requires CUDA Compute Capability $\ge 7.0$ (Volta/Turing+). |
| **KV Cache Quant** | `-ctk q8_0 -ctv q8_0` | Compresses FP16 Key/Value tensors into 8-bit symmetric blocks. | **$50\%$ reduction in KV VRAM**; doubles maximum context/concurrency. | Negligible perplexity penalty ($\Delta \text{PPL} < 0.05$). |
| **Chunked Prefill** | `-ub 128` (vs 2048) | Slices incoming prompt GEMMs into small chunks, interleaving decode GEMVs between chunks. | Suppresses ITL stutter spikes on active streams during intruder prompt arrivals. | Slightly higher aggregate prefill duration for the intruder request. |
| **Prompt Caching** | `--cache-prompt` | Reuses existing KV activations for identical prefix tokens. | **$10\times - 20\times$ TTFT reduction** on shared system instructions. | Requires session affinity or deterministic routing at the ingress proxy. |

---

## 3. Step-by-Step Reproduction Guide

### Run the Automated Ablation Benchmark Suite:
```bash
cd ~/ai-infra-systems/stages/08-serving-optimizations

python3 ablation_benchmarks.py \
  --bin /root/llama-bin/llama-b11963/llama-server \
  --model /root/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf
```

The harness will:
1. Benchmark **FlashAttention ON vs OFF** under identical 450-token prompts.
2. Benchmark **KV Cache Memory Footprint (FP16 vs Q8_0)** under 32,768 context scale.
3. Measure **Prompt Prefix Cache hit acceleration** across multi-turn prompts.
