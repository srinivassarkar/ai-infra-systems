# Stage 01: Baseline Inference Profiling Report

## 1. System Context & Hardware Target
* Target Fleet Node: [Specify target node: e.g., Node 1 (NVIDIA GTX 1050 Ti) or Node 2 (Apple Mac Studio M3 Ultra)]
* Serving Daemon: [e.g., Ollama / vLLM / mlx_lm.server]
* Model Identifier: [e.g., Qwen2.5-Coder-1.5B / Sarvam-Translate-BF16]

---

## 2. Hypothesis
* In a single-stream (concurrency = 1) scenario with warm model weights:
  * TTFT will be dominated by prompt prefill computation ($O(N_{\text{prompt}})$ GEMM operations).
  * TPOT will remain steady and bounded by hardware memory bandwidth loading the weights once per output token.
  * System throughput will be identical to single-user generation speed.

---

## 3. Benchmark Execution
Command executed:
```bash
python stages/01-baseline-inference/benchmark_baseline.py \
  --endpoint http://localhost:11434/v1/chat/completions \
  --model qwen2.5-coder:1.5b \
  --samples 5
```

---

## 4. Observed Telemetry Data

| Metric | Min | Avg | Max | P90 |
| :--- | :--- | :--- | :--- | :--- |
| **TTFT (ms)** | - | - | - | - |
| **TPOT (ms/token)** | - | - | - | - |
| **Throughput (tok/s)**| - | - | - | - |

---

## 5. Architectural Findings
* Prompt length vs TTFT scaling: [To be recorded upon run]
* Memory bandwidth utilization during decode: [To be recorded upon run]
