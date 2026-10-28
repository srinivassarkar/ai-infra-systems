# Stage 01: Baseline Inference Profiling Report

## 1. System Context & Hardware Target

* **Serving Daemon:** `llama-server` (b11901, Native C++)
* **Model Identifier:** `Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf` (Unified ~0.98 GB / ~1.1 GB in VRAM)
* **Offload Parameters:** `-ngl 99 -c 2048` (Full GPU layer offload, 2k context allocation)

### Fleet Nodes Evaluated:
* **Node 1 (Linux CUDA):** NVIDIA GeForce GTX 1050 Ti (4 GB GDDR5, 112 GB/s theoretical peak bus bandwidth)
* **Node 2 (Apple Metal):** Mac Studio M3 Ultra (256 GB Unified LPDDR5X, 800 GB/s theoretical peak bus bandwidth)
* **Node 3 (Cloud Enterprise Ada CUDA):** NVIDIA L4 (24 GB GDDR6, PCIe 4.0 x16, 300 GB/s theoretical peak bus bandwidth)

---

## 2. Hypothesis & Architectural Ground Truth

In a single-tenant (concurrency = 1) streaming scenario with warm model weights:
1. **Prefill Phase (TTFT):** Dominated by compute-bound prompt evaluation ($\mathcal{O}(N_{\mathrm{prompt}})$ GEMM operations). Silicon compute capacity (TFLOPS) determines prefill latency.
2. **Decode Phase (TPOT):** Autoregressively memory-bandwidth bound ($\mathcal{O}(1)$ GEMV operations). For every generated token, the entire active weight parameter tensor must be streamed from VRAM into the compute cores:
   $$\mathrm{TPOT}_{\mathrm{theoretical}} \approx \frac{\mathrm{ModelSize}_{\mathrm{bytes}}}{\mathrm{MemoryBandwidth}_{\mathrm{bytes/sec}}}$$
3. **Achieved Decode Memory Bandwidth:** 
   Memory bandwidth utilized while generating tokens is governed strictly by the autoregressive decode throughput, decoupled from TTFT and network transfer overhead:
   $$\mathrm{BW}_{\mathrm{decode}} = \mathrm{ModelSize}_{\mathrm{GB}} \times \mathrm{Throughput}_{\mathrm{decode}} = \frac{\mathrm{ModelSize}_{\mathrm{GB}}}{\mathrm{TPOT}_{\mathrm{seconds}}}$$
4. **Memory Bus Saturation Percentage:**
   $$\mathrm{Saturation} = \frac{\mathrm{BW}_{\mathrm{decode}}}{\mathrm{BW}_{\mathrm{peak}}} \times 100\%$$

---

## 3. Benchmark Execution Commands

```bash
# Node 1 execution (GTX 1050 Ti - 112 GB/s peak):
python stages/01-baseline-inference/benchmark_baseline.py \
  --endpoint http://100.117.38.36:8080/v1/chat/completions \
  --model qwen2.5-coder:1.5b \
  --samples 5 \
  --model-size-gb 1.1 \
  --peak-bandwidth-gbps 112

# Node 2 execution (Mac Studio M3 Ultra - 800 GB/s peak):
python stages/01-baseline-inference/benchmark_baseline.py \
  --endpoint http://100.102.40.99:8080/v1/chat/completions \
  --model qwen2.5-coder:1.5b \
  --samples 5 \
  --model-size-gb 1.1 \
  --peak-bandwidth-gbps 800
```

---

## 4. Observed Telemetry Data

### Node 1: NVIDIA GeForce GTX 1050 Ti (4 GB GDDR5 - 112 GB/s)

**Telemetry Capture:** 2026-10-27 (CUDA 12.8, Native GCC/NVCC Build for `sm_61`, Tailscale endpoint `100.117.38.36:8080`)

| Metric | Min | Median | Avg | Max |
| :--- | :--- | :--- | :--- | :--- |
| **TTFT (ms)** | 99.8 | 104.3 | 331.7 | 1042.2 |
| **TPOT (ms/token)** | 19.1 | 19.6 | 24.2 | 42.4 |
| **Decode Speed (tok/s)**| 23.6 | 51.0 | 44.5 | 52.4 |
| **End-to-End Speed (tok/s)**| 23.0 | 32.5 | 34.6 | 44.8 |
| **Achieved Decode Bandwidth (GB/s)** | 25.96 | 56.10 | 48.95 | 57.64 |
| **Memory Bus Saturation (%)** | 23.2% | 50.1% | 43.7% | 51.5% |

#### Telemetry Trace & Phase Deconstruction:
* **Cold Cache (Iteration 1):** $\mathrm{TTFT} = 305.9\ \mathrm{ms}$. Server evaluated 49 prompt tokens in $131.44\ \mathrm{ms}$ ($2.68\ \mathrm{ms/tok}$, $372.81\ \mathrm{tok/s}$), demonstrating tight GEMM kernel execution on Pascal compute cores. Wire/Tailscale overhead accounted for $\sim 174.5\ \mathrm{ms}$.
* **Warm Prefix Cache (Iterations 2–4):** Prefix caching (`LCP similarity 1.000`) dropped server prompt evaluation to $20.89 - 21.23\ \mathrm{ms}$ ($6.2\times$ prefill acceleration), yielding client-side wire $\mathrm{TTFT}$ of $99.8 - 106.3\ \mathrm{ms}$. (Iteration 5 experienced a Tailscale network packet jitter event yielding wire $\mathrm{TTFT} = 1042.2\ \mathrm{ms}$ despite identical server-side prefill of $21.20\ \mathrm{ms}$).
* **Autoregressive Decode Stability:** Server-side token generation remained anchored at $20.48 - 21.64\ \mathrm{ms}$ per token ($\sim 46.2 - 48.8\ \mathrm{tok/s}$ pure on-chip decode throughput).
* **Bus Saturation Analysis:** Based on steady-state server decode throughput of $48.8\ \mathrm{tok/s}$ ($20.48\ \mathrm{ms/tok}$), the GTX 1050 Ti streams $1.1\ \mathrm{GB} \times 48.8 = \mathbf{53.68\ \mathrm{GB/s}}$, sustaining **$47.9\%$** of its theoretical 112 GB/s physical memory bus ceiling.

### Node 2: Apple Mac Studio M3 Ultra (256 GB UMA - 800 GB/s)

**Telemetry Capture:** 2026-10-26 (Release b11901 Metal ARM64, Tailscale endpoint `100.102.40.99:8080`)

| Metric | Min | Median | Avg | Max |
| :--- | :--- | :--- | :--- | :--- |
| **TTFT (ms)** | 86.6 | 103.4 | 335.2 | 1023.3 |
| **TPOT (ms/token)** | 4.2 | 4.6 | 4.8 | 5.6 |
| **Decode Speed (tok/s)**| 178.6 | 217.4 | 212.8 | 238.1 |
| **End-to-End Speed (tok/s)**| 24.4 | 93.7 | 103.4 | 137.1 |
| **Achieved Decode Bandwidth (GB/s)** | 196.5 | 239.1 | 234.1 | 261.9 |
| **Memory Bus Saturation (%)** | 24.6% | 29.9% | 29.3% | 32.7% |

#### Telemetry Trace & Phase Deconstruction:
* **Cold Cache (Iteration 1):** $\mathrm{TTFT} = 415.0\ \mathrm{ms}$. Server evaluated 49 prompt tokens in $213.21\ \mathrm{ms}$ ($4.35\ \mathrm{ms/tok}$, $229.82\ \mathrm{tok/s}$), with $\sim 201.8\ \mathrm{ms}$ attributable to TCP/Tailscale handshake, HTTP framing, and client stream initiation.
* **Warm Prefix Cache (Iterations 2–5):** Prefix caching (`LCP similarity 1.000`) successfully reused 48/49 prompt tokens. Server prompt evaluation dropped to $14.05 - 17.05\ \mathrm{ms}$ ($15.2\times$ prefill acceleration), bringing wire $\mathrm{TTFT}$ down to $86.6 - 106.4\ \mathrm{ms}$.
* **Autoregressive Decode Stability:** Steady-state server-side decode latency was $4.50 - 4.79\ \mathrm{ms}$ per token ($\sim 208 - 222\ \mathrm{tok/s}$ on-chip decode rate).
* **Bus Saturation Analysis:** Based on steady-state server decode throughput of $221.6\ \mathrm{tok/s}$ ($4.50\ \mathrm{ms/tok}$), the M3 Ultra streams $1.1\ \mathrm{GB} \times 221.6 = \mathbf{243.76\ \mathrm{GB/s}}$, sustaining **$30.5\%$** of its theoretical 800 GB/s physical memory bus ceiling.

### Node 3: Cloud Enterprise NVIDIA L4 (24 GB GDDR6 - 300 GB/s)

**Telemetry Capture:** 2026-10-28 (CUDA 13.0 / Driver 580.126.20, Release b11963, Ubuntu 24.04 LTS on RACE Delhi Pod)

#### Scenario A: Local Execution (Zero Network Wire Overhead, localhost:8080)
Measures pure physical silicon prefill and decode execution:

| Metric | Min | Median | Avg | Max |
| :--- | :--- | :--- | :--- | :--- |
| **TTFT (ms)** | 7.3 | 7.4 | 20.3 | 72.1 |
| **TPOT (ms/token)** | 5.4 | 5.4 | 5.5 | 5.7 |
| **Decode Speed (tok/s)**| 174.6 | 185.6 | 183.5 | 186.2 |
| **End-to-End Speed (tok/s)**| 123.5 | 183.1 | 171.3 | 183.9 |
| **Achieved Decode Bandwidth (GB/s)** | 172.1 | 183.0 | 180.9 | 183.6 |
| **Memory Bus Saturation (%)** | 57.4% | 61.0% | 60.3% | 61.2% |

#### Scenario B: Distributed Over-the-Wire (Tailscale Internet Mesh from Workstation, 100.85.128.0:8080)
Measures real-world remote client impact across a 165 ms network RTT hop:

| Metric | Min | Median | Avg | Max |
| :--- | :--- | :--- | :--- | :--- |
| **TTFT (ms)** | 157.6 | 204.6 | 218.5 | 315.8 |
| **TPOT (ms/token)** | 4.5 | 4.9 | 4.9 | 5.5 |
| **Decode Speed (tok/s)**| 183.2 | 204.8 | 203.5 | 223.5 |
| **End-to-End Speed (tok/s)**| 64.1 | 80.9 | 79.4 | 88.4 |
| **Achieved Decode Bandwidth (GB/s)** | 179.5 | 200.7 | 199.4 | 219.0 |
| **Memory Bus Saturation (%)** | 59.8% | 66.9% | 66.5% | 73.0% |

#### Telemetry Trace & Network Deconstruction:
* **True Silicon Prefill:** On local execution, the NVIDIA L4 delivers a median $\mathrm{TTFT} = \mathbf{7.4\ \mathrm{ms}}$, processing 49 prompt tokens at $\sim 6.6\ \mathrm{tok/ms}$ ($6,600\ \mathrm{tok/s}$).
* **Network Latency Domination:** When queried over Tailscale from the remote workstation, $\mathrm{TTFT}$ jumps to **$204.6\ \mathrm{ms}$**. The measured ICMP ping between the workstation and the Delhi pod is $163\text{--}168\ \mathrm{ms}$. Network handshake, TLS, and framing account for $197.2\ \mathrm{ms}$ ($96.4\%$) of observed client-side TTFT, proving empirically that **client-perceived TTFT in distributed LLM architectures is overwhelmingly network-bound rather than compute-bound for short prompts**.
* **Streaming Immunity:** Once the HTTP chunked transfer stream begins, remote TPOT ($4.9\ \mathrm{ms/token}$) mirrors local TPOT ($5.4\ \mathrm{ms/token}$). Server-Sent Events (SSE) stream tokens continuously without stalling on round-trip latency.
* **Bus Saturation Champion:** The NVIDIA L4 achieved **$60.3\%\text{--}66.9\%$** memory bus saturation ($183.0\text{--}200.7\ \mathrm{GB/s}$ of its 300 GB/s peak), outperforming both the GTX 1050 Ti (47.9%) and the M3 Ultra (30.5%) in architectural memory bus efficiency for narrow GEMVs.

---

## 5. Architectural Findings & Roofline Analysis

### 1. Empirical Ridge Point & Operational Regimes

The hardware Roofline model defines the operational boundary between memory-bandwidth-bound and compute-bound regimes via the machine balance or Ridge Point ($I_{\mathrm{ridge}}$):

$$I_{\mathrm{ridge}} = \frac{\mathrm{Peak\ Compute\ (TFLOPS)}}{\mathrm{Peak\ Memory\ Bandwidth\ (TB/s)}}$$

* **NVIDIA GTX 1050 Ti:**
  $$\mathrm{Peak\ FP32} \approx 2.1\ \mathrm{TFLOPS},\quad \mathrm{Peak\ Bandwidth} = 0.112\ \mathrm{TB/s} \implies I_{\mathrm{ridge}} \approx 18.75\ \mathrm{FLOP/byte}$$
* **Apple Mac Studio M3 Ultra:**
  $$\mathrm{Peak\ FP16} \approx 45\text{--}50\ \mathrm{TFLOPS},\quad \mathrm{Peak\ Bandwidth} = 0.800\ \mathrm{TB/s} \implies I_{\mathrm{ridge}} \approx 56\text{--}62\ \mathrm{FLOP/byte}$$

### 2. Autoregressive Decode Arithmetic Intensity ($I_{\mathrm{decode}}$)

In single-token autoregressive generation ($\mathrm{batch\ size} = 1$), each forward step performs $2 \times P$ floating-point operations for a model with $P$ parameters, while transferring the entire weight tensor of size $P \times B_{\mathrm{weight}}$ bytes from VRAM into compute registers. For 4-bit quantization ($B_{\mathrm{weight}} \approx 0.5\ \mathrm{bytes/param}$):

$$I_{\mathrm{decode}} \approx \frac{2 \times P\ \mathrm{FLOPs}}{0.5 \times P\ \mathrm{bytes}} = 4.0\ \mathrm{FLOP/byte}$$

Because $I_{\mathrm{decode}} = 4.0\ \mathrm{FLOP/byte} \ll I_{\mathrm{ridge}}$ on both machines ($4.0 \ll 18.75$ and $4.0 \ll 60.0$), **single-stream autoregressive decode is strictly memory-bandwidth bound on both architectures**. 

### 3. Decode Comparison Across the 3-Tier Silicon Fleet

* **Node 1 (GTX 1050 Ti - Pascal GDDR5):** Steady-state decode is **$20.48\ \mathrm{ms/token}$** ($48.8\ \mathrm{tok/s}$).
* **Node 3 (NVIDIA L4 - Ada GDDR6):** Steady-state decode is **$5.40\ \mathrm{ms/token}$** ($185.6\ \mathrm{tok/s}$).
* **Node 2 (Mac Studio M3 Ultra - Unified LPDDR5X):** Steady-state decode is **$4.50\ \mathrm{ms/token}$** ($221.6\ \mathrm{tok/s}$).

**Empirical Speedups:**
* NVIDIA L4 is **$3.80\times$** faster than the GTX 1050 Ti, directly matching its $2.68\times$ wider memory bus and modern 4th-gen Tensor Core dispatch.
* Mac Studio M3 Ultra is **$1.19\times$** faster than the NVIDIA L4 and **$4.55\times$** faster than the GTX 1050 Ti, propelled by its massive 800 GB/s unified memory pipeline.

### 4. Memory Bus Saturation: The Architectural Efficiency Ranking (60% vs 48% vs 30%)

Decoupling decode speed from end-to-end latency reveals the physical bus utilization efficiency:
* **NVIDIA L4 (Rank 1 - 60.3% Saturation):** Steaming $0.986\ \mathrm{GB}$ weights at $185.6\ \mathrm{tok/s}$ achieves **$183.0\ \mathrm{GB/s}$** on a 300 GB/s bus. Ada Lovelace's optimized memory controllers and high-bandwidth L2 cache hierarchy (48 MB L2 cache) allow narrow GEMVs ($d_{\mathrm{model}} = 1536$) to achieve the highest physical bus saturation in the fleet.
* **GTX 1050 Ti (Rank 2 - 47.9% Saturation):** Streaming $1.1\ \mathrm{GB}$ at $48.8\ \mathrm{tok/s}$ achieves **$53.7\ \mathrm{GB/s}$** on a 112 GB/s bus.
* **Mac Studio M3 Ultra (Rank 3 - 30.5% Saturation):** Streaming $1.1\ \mathrm{GB}$ at $221.6\ \mathrm{tok/s}$ achieves **$243.8\ \mathrm{GB/s}$** on an 800 GB/s bus.

**Why does the M3 Ultra top out at ~30.5% saturation for a 1.5B model?**
The answer lies in **GPU core occupancy and kernel launch granularity**:
1. A 1.5B model has narrow hidden dimensions ($d_{\mathrm{model}} = 1536$). At batch size 1, the matrix-vector multiplication (GEMV) does not yield enough concurrent thread blocks to keep all 80 GPU cores on the M3 Ultra fully saturated with warps.
2. Each generated token incurs non-weight-streaming overhead: Softmax normalization, activation quantization, sampling routines, and Metal command buffer dispatch latency.
3. On a 70B parameter model ($d_{\mathrm{model}} = 8192$), each layer's GEMV is wide enough to saturate all GPU execution units, amortizing dispatch overhead and driving bus saturation above 70%. In contrast, the NVIDIA L4's 58 SMs are more compact and fully utilized even at narrow dimensions.

### 5. Prefix Caching Efficiency

Across both platforms, `llama.cpp`'s Longest Common Prefix (LCP) slot caching reduced prompt evaluation time by **$6.2\times$** on CUDA ($131.4\ \mathrm{ms} \to 20.9\ \mathrm{ms}$) and **$15.2\times$** on Apple Metal ($213.2\ \mathrm{ms} \to 14.1\ \mathrm{ms}$). In real-world multi-turn conversational agents, prompt caching provides the single largest empirical latency reduction before physical silicon limits are reached.

