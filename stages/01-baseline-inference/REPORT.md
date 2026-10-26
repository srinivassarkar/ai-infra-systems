# Stage 01: Baseline Inference Profiling Report

## 1. System Context & Hardware Target

* **Serving Daemon:** `llama-server` (b11901, Native C++)
* **Model Identifier:** `Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf` (Unified ~0.98 GB / ~1.1 GB in VRAM)
* **Offload Parameters:** `-ngl 99 -c 2048` (Full GPU layer offload, 2k context allocation)

### Fleet Nodes Evaluated:
* **Node 1 (Linux CUDA):** NVIDIA GeForce GTX 1050 Ti (4 GB GDDR5, 112 GB/s theoretical peak bus bandwidth)
* **Node 2 (Apple Metal):** Mac Studio M3 Ultra (256 GB Unified LPDDR5X, 800 GB/s theoretical peak bus bandwidth)

---

## 2. Hypothesis & Architectural Ground Truth

In a single-tenant (concurrency = 1) streaming scenario with warm model weights:
1. **Prefill Phase (TTFT):** Dominated by compute-bound prompt evaluation ($\mathcal{O}(N_{\mathrm{prompt}})$ GEMM operations). Silicon compute capacity (TFLOPS) determines prefill latency.
2. **Decode Phase (TPOT):** Autoregressively memory-bandwidth bound ($\mathcal{O}(1)$ GEMV operations). For every generated token, the entire active weight parameter tensor must be streamed from VRAM into the compute cores:
   $$\mathrm{TPOT}_{\mathrm{theoretical}} \approx \frac{\mathrm{ModelSize}_{\mathrm{bytes}}}{\mathrm{MemoryBandwidth}_{\mathrm{bytes/sec}}}$$
3. **Achieved Memory Bandwidth:** 
   $$\mathrm{BW}_{\mathrm{achieved}} = \mathrm{ModelSize}_{\mathrm{GB}} \times \mathrm{Throughput}_{\mathrm{tok/s}}$$

---

## 3. Benchmark Execution Commands

```bash
# Node 1 execution (GTX 1050 Ti - 112 GB/s peak):
python stages/01-baseline-inference/benchmark_baseline.py \
  --endpoint http://<node-1-ip>:8080/v1/chat/completions \
  --model qwen2.5-coder:1.5b \
  --samples 5 \
  --model-size-gb 1.1 \
  --peak-bandwidth-gbps 112

# Node 2 execution (Mac Studio M3 Ultra - 800 GB/s peak):
python stages/01-baseline-inference/benchmark_baseline.py \
  --endpoint http://<node-2-ip>:8080/v1/chat/completions \
  --model qwen2.5-coder:1.5b \
  --samples 5 \
  --model-size-gb 1.1 \
  --peak-bandwidth-gbps 800
```

---

## 4. Observed Telemetry Data

### Node 1: NVIDIA GeForce GTX 1050 Ti (4 GB GDDR5 - 112 GB/s)

| Metric | Min | Avg | Max | P90 |
| :--- | :--- | :--- | :--- | :--- |
| **TTFT (ms)** | - | - | - | - |
| **TPOT (ms/token)** | - | - | - | - |
| **Throughput (tok/s)**| - | - | - | - |
| **Achieved Bandwidth (GB/s)** | - | - | - | - |
| **Bus Saturation (%)** | - | - | - | - |

### Node 2: Apple Mac Studio M3 Ultra (256 GB UMA - 800 GB/s)

| Metric | Min | Avg | Max | P90 |
| :--- | :--- | :--- | :--- | :--- |
| **TTFT (ms)** | - | - | - | - |
| **TPOT (ms/token)** | - | - | - | - |
| **Throughput (tok/s)**| - | - | - | - |
| **Achieved Bandwidth (GB/s)** | - | - | - | - |
| **Bus Saturation (%)** | - | - | - | - |

---

## 5. Architectural Findings & Roofline Analysis

* **Empirical Ridge Point:** [To be calculated: $I_{\mathrm{ridge}} = \frac{\mathrm{Peak\ TFLOPS}}{\mathrm{Peak\ Bandwidth}}$]
* **Observed Bus Efficiency:** [Analysis of achieved GB/s versus theoretical peak]
* **Prefill vs Decode Variance:** [Empirical observations comparing TTFT variance against steady-state TPOT variance]

