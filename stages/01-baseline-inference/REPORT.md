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

**Telemetry Capture:** 2026-10-27 (CUDA 12.8, Native GCC/NVCC Build for `sm_61`, Tailscale endpoint `100.117.38.36:8080`)

| Metric | Min | Avg | Max | P90 |
| :--- | :--- | :--- | :--- | :--- |
| **TTFT (ms)** | 99.8 | 331.7 | 1042.2 | 1042.2 |
| **TPOT (ms/token)** | 19.1 | 24.2 | 42.4 | 42.4 |
| **Throughput (tok/s)**| 23.0 | 34.6 | 44.8 | 44.8 |
| **Achieved Bandwidth (GB/s)** | 25.30 | 38.04 | 49.28 | 49.28 |
| **Bus Saturation (%)** | 22.6% | 34.0% | 44.0% | 44.0% |

#### Telemetry Trace & Phase Deconstruction:
* **Cold Cache (Iteration 1):** $\mathrm{TTFT} = 305.9\ \mathrm{ms}$. Server evaluated 49 prompt tokens in $131.44\ \mathrm{ms}$ ($2.68\ \mathrm{ms/tok}$, $372.81\ \mathrm{tok/s}$), demonstrating tight GEMM kernel execution on Pascal compute cores. Wire/Tailscale overhead accounted for $\sim 174.5\ \mathrm{ms}$.
* **Warm Prefix Cache (Iterations 2–4):** Prefix caching (`LCP similarity 1.000`) dropped server prompt evaluation to $20.89 - 21.23\ \mathrm{ms}$ ($6.2\times$ prefill acceleration), yielding client-side wire $\mathrm{TTFT}$ of $99.8 - 106.3\ \mathrm{ms}$. (Iteration 5 experienced a Tailscale network packet jitter event yielding wire $\mathrm{TTFT} = 1042.2\ \mathrm{ms}$ despite identical server-side prefill of $21.20\ \mathrm{ms}$).
* **Autoregressive Decode Stability:** Server-side token generation remained anchored at $20.48 - 21.64\ \mathrm{ms}$ per token ($\sim 46.2 - 48.8\ \mathrm{tok/s}$ pure decode throughput).
* **Bus Saturation Analysis:** Achieved $38.04\ \mathrm{GB/s}$ average bandwidth ($34.0\%$ of the $112\ \mathrm{GB/s}$ theoretical bus ceiling, peaking at $44.0\%$).

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

### 1. Empirical Ridge Point & Operational Regimes

The hardware Roofline model defines the operational boundary between memory-bandwidth-bound and compute-bound regimes via the machine balance or Ridge Point ($I_{\mathrm{ridge}}$):

$$I_{\mathrm{ridge}} = \frac{\mathrm{Peak\ Compute\ (TFLOPS)}}{\mathrm{Peak\ Memory\ Bandwidth\ (TB/s)}}$$

* **NVIDIA GTX 1050 Ti:**
  $$\mathrm{Peak\ FP32} \approx 2.1\ \mathrm{TFLOPS},\quad \mathrm{Peak\ Bandwidth} = 0.112\ \mathrm{TB/s} \implies I_{\mathrm{ridge}} \approx 18.75\ \mathrm{FLOP/byte}$$
* **Apple Mac Studio M3 Ultra:**
  $$\mathrm{Peak\ FP16} \approx 56\ \mathrm{TFLOPS},\quad \mathrm{Peak\ Bandwidth} = 0.800\ \mathrm{TB/s} \implies I_{\mathrm{ridge}} \approx 70.0\ \mathrm{FLOP/byte}$$

### 2. Autoregressive Decode Arithmetic Intensity ($I_{\mathrm{decode}}$)

In single-token autoregressive generation ($\mathrm{batch\ size} = 1$), each forward step performs $2 \times P$ floating-point operations for a model with $P$ parameters, while transferring the entire weight tensor of size $P \times B_{\mathrm{weight}}$ bytes from VRAM into compute registers. For 4-bit quantization ($B_{\mathrm{weight}} \approx 0.5\ \mathrm{bytes/param}$):

$$I_{\mathrm{decode}} \approx \frac{2 \times P\ \mathrm{FLOPs}}{0.5 \times P\ \mathrm{bytes}} = 4.0\ \mathrm{FLOP/byte}$$

Because $I_{\mathrm{decode}} = 4.0\ \mathrm{FLOP/byte} \ll I_{\mathrm{ridge}}$ on both machines ($4.0 \ll 18.75$ and $4.0 \ll 70.0$), **single-stream autoregressive decode is strictly memory-bandwidth bound on both architectures**. 

### 3. Decode Speedup: M3 Ultra vs GTX 1050 Ti

* **Node 1 (GTX 1050 Ti):** Steady-state server decode latency is **$20.48\ \mathrm{ms/token}$** ($48.8\ \mathrm{tok/s}$).
* **Node 2 (Mac Studio M3 Ultra):** Steady-state server decode latency is **$4.50\ \mathrm{ms/token}$** ($221.6\ \mathrm{tok/s}$).
* **Empirical Speedup:**
  $$\mathrm{Speedup} = \frac{20.48\ \mathrm{ms}}{4.50\ \mathrm{ms}} \approx 4.55\times$$
The M3 Ultra decodes $4.55\times$ faster directly due to its $7.14\times$ wider memory bus ($800\ \mathrm{GB/s}$ vs $112\ \mathrm{GB/s}$) and larger on-chip cache hierarchy.

### 4. Memory Bus Saturation Discrepancy (34.0% vs 14.2%)

* On the **GTX 1050 Ti**, the 128-bit GDDR5 bus ($112\ \mathrm{GB/s}$) is narrow enough that streaming $1.1\ \mathrm{GB}$ at $48.8\ \mathrm{tok/s}$ approaches physical bus limits, yielding **$34.0\% - 44.0\%$ bus saturation**.
* On the **Mac Studio M3 Ultra**, the massive $800\ \mathrm{GB/s}$ UMA pipe easily sustains $221.6\ \mathrm{tok/s}$ with only **$14.2\%$ bus saturation**. The remaining $85\%$ of bandwidth remains idle because at $\mathrm{concurrency} = 1$, the memory controller spends clock cycles idling between kernel dispatches.
* **Production Takeaway:** To fully saturate an Apple Silicon M-series memory bus or modern datacenter H100 GPU, continuous batching ($\mathrm{concurrency} > 16$) or large-parameter weights ($70\mathrm{B}+$) are required to amortize kernel dispatch latency and drive arithmetic intensity toward $I_{\mathrm{ridge}}$.

### 5. Prefix Caching Efficiency

Across both platforms, `llama.cpp`'s Longest Common Prefix (LCP) slot caching reduced prompt evaluation time by **$6.2\times$** on CUDA ($131.4\ \mathrm{ms} \to 20.9\ \mathrm{ms}$) and **$15.2\times$** on Apple Metal ($213.2\ \mathrm{ms} \to 14.1\ \mathrm{ms}$). In real-world multi-turn conversational agents, prompt caching provides the single largest empirical latency reduction before physical silicon limits are reached.

