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

**Telemetry Capture:** 2026-10-26 (Release b11901 Metal ARM64, Tailscale endpoint `100.102.40.99:8080`)

| Metric | Min | Avg | Max | P90 |
| :--- | :--- | :--- | :--- | :--- |
| **TTFT (ms)** | 92.1 | 170.4 | 415.0 | 415.0 |
| **TPOT (ms/token)** | 4.2 | 4.8 | 5.6 | 5.6 |
| **Throughput (tok/s)**| 48.9 | 103.4 | 137.1 | 137.1 |
| **Achieved Bandwidth (GB/s)** | 53.79 | 113.71 | 150.81 | 150.81 |
| **Bus Saturation (%)** | 6.7% | 14.2% | 18.9% | 18.9% |

#### Telemetry Trace & Phase Deconstruction:
* **Cold Cache (Iteration 1):** $\mathrm{TTFT} = 415.0\ \mathrm{ms}$. Server evaluated 49 prompt tokens in $213.21\ \mathrm{ms}$ ($4.35\ \mathrm{ms/tok}$, $229.82\ \mathrm{tok/s}$), with $\sim 201.8\ \mathrm{ms}$ attributable to TCP/Tailscale handshake, HTTP framing, and client stream initiation.
* **Warm Prefix Cache (Iterations 2–5):** Prefix caching (`LCP similarity 1.000`) successfully reused 48/49 prompt tokens. Server prompt evaluation dropped to $14.05 - 17.05\ \mathrm{ms}$ ($15.2\times$ prefill acceleration), bringing wire $\mathrm{TTFT}$ down to $92.1 - 106.4\ \mathrm{ms}$.
* **Autoregressive Decode Stability:** $\mathrm{TPOT}$ remained exceptionally deterministic between $4.2\ \mathrm{ms}$ and $5.6\ \mathrm{ms}$ per token ($\sim 187 - 222\ \mathrm{tok/s}$ decode rate).
* **Bus Saturation Analysis:** Achieved $113.71\ \mathrm{GB/s}$ ($14.2\%$ of the $800\ \mathrm{GB/s}$ theoretical bus ceiling). Because $Qwen2.5\text{-}1.5B$ has a footprint of only $\sim 1.1\ \mathrm{GB}$, single-tenant ($\mathrm{concurrency} = 1$) decode is bounded by memory latency and Metal kernel launch overhead rather than bus saturation. Higher concurrency batching or larger parameter weights ($70\mathrm{B}+$) are required to saturate the $800\ \mathrm{GB/s}$ UMA pipe.

---

## 5. Architectural Findings & Roofline Analysis

* **Empirical Ridge Point:** [To be calculated: $I_{\mathrm{ridge}} = \frac{\mathrm{Peak\ TFLOPS}}{\mathrm{Peak\ Bandwidth}}$]
* **Observed Bus Efficiency:** [Analysis of achieved GB/s versus theoretical peak]
* **Prefill vs Decode Variance:** [Empirical observations comparing TTFT variance against steady-state TPOT variance]

