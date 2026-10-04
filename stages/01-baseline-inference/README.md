# Stage 01: Baseline Inference Profiling

## Objective
Measure and isolate the fundamental performance characteristics of large language model serving before introducing concurrency, proxies, or optimization techniques.

Specifically:
1. **Deconstruct TTFT (Time-To-First-Token):** Measure prompt prefill latency as a function of prompt token length.
2. **Deconstruct TPOT (Time-Per-Output-Token):** Measure autoregressive decode token generation latency across single-token iterations.
3. **Establish Baseline Throughput:** Measure raw generation speed in tokens per second under single-tenant (concurrency = 1) conditions.

---

## Architecture

```
[ benchmark_baseline.py ]
         │
         │ HTTP/1.1 POST /v1/chat/completions (stream: true)
         ▼
[ Serving Engine Daemon (Port 11434 / 8000 / 8089) ]
         │
         ▼
[ Physical Silicon (GPU / UMA Memory Bus) ]
```

---

## Metrics Formulas

1. **TTFT (Time-To-First-Token):**
   $$\text{TTFT} = t_{\text{first\_token}} - t_{\text{request\_sent}}$$
2. **TPOT (Time-Per-Output-Token / Inter-Token Latency):**
   $$\text{TPOT} = \frac{t_{\text{stream\_end}} - t_{\text{first\_token}}}{N_{\text{output\_tokens}} - 1}$$
3. **System Output Throughput:**
   $$\text{Throughput} = \frac{N_{\text{output\_tokens}}}{t_{\text{stream\_end}} - t_{\text{request\_sent}}}$$

---

## Execution Guide

### Prerequisites
Ensure your target serving engine is running (e.g., Ollama on Linux/macOS, vLLM on port 8000, or `mlx_lm.server` on port 8089).

### Run Baseline Profiling
```bash
python benchmark_baseline.py --endpoint http://localhost:11434/v1/chat/completions --model qwen2.5-coder:1.5b --samples 5
```

The script will output exact millisecond timestamps, per-token interval distributions, and generate a standardized markdown table for inclusion in `REPORT.md`.
