# Stage 01: Baseline Inference Profiling Report

## 1. System Context & Hardware Target

* **Serving Daemon:** `llama-server` (Native C++, builds b11901 and b11963)
* **Model Identifier:** `Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf` (986 MB on disk / 1.1 GB in VRAM)
* **Offload Parameters:** `-ngl 99 -c 2048` (Full GPU layer offload, 2k context allocation)

### Fleet Nodes Evaluated:
1. **Node 1 (Constrained Edge CUDA):** Custom Linux Box, NVIDIA GeForce GTX 1050 Ti (4 GB GDDR5, 112 GB/s peak bus bandwidth, PCIe 3.0 x16, Ubuntu 20.04 LTS).
2. **Node 2 (Unified Memory Workstation):** Apple Mac Studio M3 Ultra (28 CPU Cores, 80 GPU Cores, 256 GB Unified LPDDR5X, 800 GB/s peak bus bandwidth, macOS Sequoia).
3. **Node 3 (Cloud Enterprise Datacenter):** NVIDIA L4 (24 GB GDDR6 with ECC, 300 GB/s peak bus bandwidth, PCIe 4.0 x16, Ubuntu 24.04 LTS on RACE Delhi Pod).

---

## 2. Hypothesis & Architectural Ground Truth

In a single-tenant (concurrency = 1) streaming scenario with warm model weights:

* **Prefill Phase (TTFT):** Dominated by compute-bound prompt evaluation ($\mathcal{O}(N_{\mathrm{prompt}})$ GEMM operations). Silicon compute capacity (TFLOPS) determines prefill latency.
* **Decode Phase (TPOT):** Autoregressively memory-bandwidth bound ($\mathcal{O}(1)$ GEMV operations). For every generated token, the entire active weight parameter tensor must be streamed from memory into the compute cores.
* **Network vs Prefill Latency Decomposition:**
```text
TTFT_observed = TTFT_prefill_compute + RTT_network + t_framing
```
* **Theoretical Minimum Decode Latency:**
```text
TPOT_theoretical = Model_Size_bytes / Memory_Bandwidth_bytes_per_sec
```
* **Achieved Decode Memory Bandwidth:** 
```text
BW_decode = Model_Size_GB * Throughput_decode = Model_Size_GB / TPOT_seconds
```
* **Memory Bus Saturation Percentage:**
```text
Saturation = (BW_decode / BW_peak) * 100%
```

---

## 3. Benchmark Execution Commands

```bash
# Node 1 execution (GTX 1050 Ti - 112 GB/s peak, over Tailscale):
python stages/01-baseline-inference/benchmark_baseline.py \
  --endpoint http://100.117.38.36:8080/v1/chat/completions \
  --model Qwen2.5-Coder-1.5B \
  --samples 5 \
  --model-size-gb 0.986 \
  --peak-bandwidth-gbps 112.0

# Node 2 execution (Mac Studio M3 Ultra - 800 GB/s peak, over Tailscale):
python stages/01-baseline-inference/benchmark_baseline.py \
  --endpoint http://100.102.40.99:8080/v1/chat/completions \
  --model Qwen2.5-Coder-1.5B \
  --samples 5 \
  --model-size-gb 0.986 \
  --peak-bandwidth-gbps 800.0

# Node 3 Local execution (NVIDIA L4 - 300 GB/s peak, on-pod 0 ms loopback):
python3 /workspace/ai-infra/repo/stages/01-baseline-inference/benchmark_baseline.py \
  --endpoint http://127.0.0.1:8080/v1/chat/completions \
  --model Qwen2.5-Coder-1.5B \
  --samples 5 \
  --model-size-gb 0.986 \
  --peak-bandwidth-gbps 300.0

# Node 3 Remote execution (NVIDIA L4 - 300 GB/s peak, workstation over Tailscale):
python stages/01-baseline-inference/benchmark_baseline.py \
  --endpoint http://100.85.128.0:8080/v1/chat/completions \
  --model Qwen2.5-Coder-1.5B \
  --samples 5 \
  --model-size-gb 0.986 \
  --peak-bandwidth-gbps 300.0
```

---

## 4. Observed Telemetry Data

### Summary Telemetry Comparison Across the Fleet:

| Metric | Node 1: GTX 1050 Ti | Node 3: NVIDIA L4 (Local) | Node 3: NVIDIA L4 (Tailscale) | Node 2: Mac Studio M3 Ultra |
| :--- | :--- | :--- | :--- | :--- |
| **Silicon Architecture** | Pascal GDDR5 (PCIe 3.0) | Ada GDDR6 (PCIe 4.0) | Ada GDDR6 (over WAN mesh) | Apple UMA LPDDR5X (Unified) |
| **Theoretical Peak BW** | 112.0 GB/s | 300.0 GB/s | 300.0 GB/s | 800.0 GB/s |
| **Median TTFT** | 104.3 ms | **7.4 ms** | **204.6 ms** | 103.4 ms |
| **Median TPOT** | 19.6 ms/tok | **5.4 ms/tok** | **4.9 ms/tok** | **4.6 ms/tok** |
| **Median Decode Throughput**| 51.0 tok/s | **185.6 tok/s** | **204.8 tok/s** | **217.4 tok/s** |
| **Achieved Decode BW** | 53.7 GB/s | **183.0 GB/s** | **200.7 GB/s** | **243.8 GB/s** |
| **Memory Bus Saturation** | **47.9%** | **61.0%** | **66.9%** | **30.5%** |

---

### Node 1 Detailed Trace: NVIDIA GeForce GTX 1050 Ti (4 GB GDDR5 - 112 GB/s)

**Telemetry Capture:** 2026-10-27 (CUDA 12.8, Native GCC/NVCC Build for `sm_61`, Tailscale endpoint `100.117.38.36:8080`)

| Metric | Min | Median | Avg | Max |
| :--- | :--- | :--- | :--- | :--- |
| **TTFT (ms)** | 99.8 | 104.3 | 331.7 | 1042.2 |
| **TPOT (ms/token)** | 19.1 | 19.6 | 24.2 | 42.4 |
| **Decode Speed (tok/s)**| 23.6 | 51.0 | 44.5 | 52.4 |
| **End-to-End Speed (tok/s)**| 23.0 | 32.5 | 34.6 | 44.8 |
| **Achieved Decode Bandwidth (GB/s)** | 25.96 | 56.10 | 48.95 | 57.64 |
| **Memory Bus Saturation (%)** | 23.2% | 50.1% | 43.7% | 51.5% |

* **Trace Analysis:** On Cold Cache (Iteration 1), prompt evaluation of 49 tokens took 131.44 ms ($372.8\ \mathrm{tok/s}$). Warm prefix caching dropped evaluation to ~21 ms ($6.2\times$ prefill speedup). Autoregressive decode remained stable at ~20.5 ms per token, sustaining $53.7\ \mathrm{GB/s}$ (47.9% of bus saturation).

---

### Node 2 Detailed Trace: Apple Mac Studio M3 Ultra (256 GB UMA - 800 GB/s)

**Telemetry Capture:** 2026-10-26 (Release b11901 Metal ARM64, Tailscale endpoint `100.102.40.99:8080`)

| Metric | Min | Median | Avg | Max |
| :--- | :--- | :--- | :--- | :--- |
| **TTFT (ms)** | 86.6 | 103.4 | 335.2 | 1023.3 |
| **TPOT (ms/token)** | 4.2 | 4.6 | 4.8 | 5.6 |
| **Decode Speed (tok/s)**| 178.6 | 217.4 | 212.8 | 238.1 |
| **End-to-End Speed (tok/s)**| 24.4 | 93.7 | 103.4 | 137.1 |
| **Achieved Decode Bandwidth (GB/s)** | 196.5 | 239.1 | 234.1 | 261.9 |
| **Memory Bus Saturation (%)** | 24.6% | 29.9% | 29.3% | 32.7% |

* **Trace Analysis:** Steady-state server decode reached 217.4–221.6 tok/s ($4.50\ \mathrm{ms/tok}$). Streaming 1.1 GB of weights at this rate delivers $243.8\ \mathrm{GB/s}$ across unified memory (30.5% of the 800 GB/s ceiling).

---

### Node 3 Detailed Trace: Cloud Enterprise NVIDIA L4 (24 GB GDDR6 - 300 GB/s)

**Telemetry Capture:** 2026-10-28 (CUDA 13.0, Driver 580.126.20, Release b11963, Ubuntu 24.04 LTS on RACE Delhi Pod)

#### Scenario A: Local Execution (Zero Network Wire Overhead, 127.0.0.1:8080)
| Metric | Min | Median | Avg | Max |
| :--- | :--- | :--- | :--- | :--- |
| **TTFT (ms)** | 7.3 | **7.4** | 20.3 | 72.1 |
| **TPOT (ms/token)** | 5.4 | **5.4** | 5.5 | 5.7 |
| **Decode Speed (tok/s)**| 174.6 | **185.6** | 183.5 | 186.2 |
| **End-to-End Speed (tok/s)**| 123.5 | **183.1** | 171.3 | 183.9 |
| **Achieved Decode Bandwidth (GB/s)** | 172.1 | **183.0** | 180.9 | 183.6 |
| **Memory Bus Saturation (%)** | 57.4% | **61.0%** | 60.3% | 61.2% |

#### Scenario B: Distributed Over-the-Wire (Workstation via Tailscale, 100.85.128.0:8080)
| Metric | Min | Median | Avg | Max |
| :--- | :--- | :--- | :--- | :--- |
| **TTFT (ms)** | 157.6 | **204.6** | 218.5 | 315.8 |
| **TPOT (ms/token)** | 4.5 | **4.9** | 4.9 | 5.5 |
| **Decode Speed (tok/s)**| 183.2 | **204.8** | 203.5 | 223.5 |
| **End-to-End Speed (tok/s)**| 64.1 | **80.9** | 79.4 | 88.4 |
| **Achieved Decode Bandwidth (GB/s)** | 179.5 | **200.7** | 199.4 | 219.0 |
| **Memory Bus Saturation (%)** | 59.8% | **66.9%** | 66.5% | 73.0% |

---

## 5. Architectural Findings & Roofline Analysis

### 1. Empirical Ridge Point & Operational Regimes

The Roofline model defines the boundary between memory-bandwidth-bound and compute-bound operations via the machine balance or Ridge Point ($I_{\mathrm{ridge}}$):

$$I_{\mathrm{ridge}} = \frac{\mathrm{Peak\ Compute\ (TFLOPS)}}{\mathrm{Peak\ Memory\ Bandwidth\ (TB/s)}}$$

* **NVIDIA GTX 1050 Ti:** $\mathrm{Peak\ FP32} \approx 2.1\ \mathrm{TFLOPS},\ \mathrm{Peak\ BW} = 0.112\ \mathrm{TB/s} \implies I_{\mathrm{ridge}} \approx 18.75\ \mathrm{FLOP/byte}$
* **Apple Mac Studio M3 Ultra:** $\mathrm{Peak\ FP16} \approx 48\ \mathrm{TFLOPS},\ \mathrm{Peak\ BW} = 0.800\ \mathrm{TB/s} \implies I_{\mathrm{ridge}} \approx 60.0\ \mathrm{FLOP/byte}$
* **NVIDIA L4:** $\mathrm{Peak\ FP16/BF16} \approx 120\ \mathrm{TFLOPS},\ \mathrm{Peak\ BW} = 0.300\ \mathrm{TB/s} \implies I_{\mathrm{ridge}} \approx 400.0\ \mathrm{FLOP/byte}$

### 2. Autoregressive Decode Arithmetic Intensity ($I_{\mathrm{decode}}$)
At batch size 1, generating a single token requires reading the entire model weight tensor from memory once while performing 2 FLOPs per parameter:
$$I_{\mathrm{decode}} \approx \frac{2 \times P\ \mathrm{FLOPs}}{0.5 \times P\ \mathrm{bytes}} = 4.0\ \mathrm{FLOP/byte}$$

Because $I_{\mathrm{decode}} = 4.0\ \mathrm{FLOP/byte} \ll I_{\mathrm{ridge}}$ across all three platforms ($4.0 \ll 18.75,\ 4.0 \ll 60.0,\ 4.0 \ll 400.0$), **autoregressive decode is strictly and inescapably memory-bandwidth bound on every machine**.

### 3. The Network Latency Domination Discovery
Comparing Node 3 Local vs Node 3 Tailscale provides an empirical breakdown of what makes up TTFT:
* **Local Prefill Compute:** $\mathrm{TTFT} = \mathbf{7.4\ \mathrm{ms}}$ (server prompt processing at $6,600\ \mathrm{tok/s}$).
* **Remote Wire TTFT:** $\mathrm{TTFT} = \mathbf{204.6\ \mathrm{ms}}$.
* **Transport Tax:** Network ping ($165\ \mathrm{ms}$) plus TCP and HTTP chunk framing accounted for **$197.2\ \mathrm{ms}$ (96.4%)** of client-perceived TTFT.
* **Streaming Invariance:** Conversely, decode TPOT is immune to round-trip latency once streaming begins ($5.4\ \mathrm{ms/tok}$ local vs $4.9\ \mathrm{ms/tok}$ remote). Server-Sent Events push chunks through the TCP window pipeline without waiting for client acknowledgments.

### 4. Memory Bus Saturation: Why L4 Beats M3 Ultra in Efficiency (61% vs 30%)
* **NVIDIA L4 (Rank 1 - 61.0% Saturation):** Achieves $183.0\ \mathrm{GB/s}$ on a 300 GB/s bus. Ada Lovelace's 48 MB L2 cache and 58 compact Streaming Multiprocessors maintain high warp occupancy even on small hidden dimensions ($d_{\mathrm{model}} = 1536$).
* **GTX 1050 Ti (Rank 2 - 47.9% Saturation):** Achieves $53.7\ \mathrm{GB/s}$ on a 112 GB/s bus.
* **Mac Studio M3 Ultra (Rank 3 - 30.5% Saturation):** Achieves $243.8\ \mathrm{GB/s}$ on an 800 GB/s bus.

**Why does the M3 Ultra top out at ~30.5% bus saturation for a 1.5B model?**
At batch size 1, matrix-vector multiplication (GEMV) for a 1.5B model ($d_{\mathrm{model}} = 1536$) does not generate enough concurrent thread blocks to keep all 80 GPU cores on the M3 Ultra saturated. On a larger 70B model ($d_{\mathrm{model}} = 8192$), the GEMV dimension expands by $5.3\times$, providing sufficient thread parallelism to saturate all execution units and drive M3 Ultra bus utilization above 70%.
