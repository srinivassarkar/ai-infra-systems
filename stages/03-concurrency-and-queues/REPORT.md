# Stage 03: Concurrency, Queuing & Head-of-Line Blocking Report

## 1. Executive Summary & The Core Architectural Laws

In Stage 01 and Stage 02, we established baseline single-stream latency and virtual memory cold starts. Stage 03 investigates the behavior of LLM serving under concurrent multi-user load.

### Key Architectural Findings:
1. **Coordinated Omission in Closed-Loop Testing:** Closed-loop benchmarks ($N$ concurrent workers) hide queuing delays because slow server responses naturally throttle request generation. Only an open-loop Poisson arrival model ($\lambda$ req/s) exposes the true non-linear $W \propto \frac{1}{1 - \rho}$ tail latency explosion.
2. **The Hardware Slot Saturation Cliff:** With `--parallel 8`, TTFT remains virtually flat ($\le 43\ \mathrm{ms}$) from $N = 1$ to $N = 8$. Doubling concurrency to $N = 16$ triggers immediate request-level Head-of-Line queuing: TTFT explodes by **$10\times$** ($428.21\ \mathrm{ms}$ median, $520.11\ \mathrm{ms}$ P99), while throughput gains taper off from $+48\%$ to $+10\%$.
3. **Little's Law Validated with $<2\%$ Error:** Across all 5 closed-loop concurrency tiers, Little's Law ($L = \lambda W$) holds with empirical perfection ($L = 1.00, 2.00, 3.99, 7.92, 15.69$).
4. **The Modern Form of Head-of-Line (HoL) Blocking:** Continuous batching eliminates request-level FIFO HoL blocking across slots, but introduces **iteration-level prefill-decode interference**. Injecting a single 1,500-token prompt caused a **$15.4\times$ Inter-Token Latency (ITL) stutter spike** ($122.16\ \mathrm{ms}$ max vs $7.91\ \mathrm{ms}$ median) on ongoing background streams.

---

## 2. Theoretical Roofline Predictions

In Stage 01, we proved that single-stream decode ($B = 1$) is strictly memory-bandwidth bound at an arithmetic intensity of $I_{\mathrm{decode}} \approx 4.0\ \mathrm{FLOP/byte}$.

As concurrent batch size $B$ increases, the GPU reads active model weights once from memory but performs $B$ matrix-vector multiplications in parallel:
$$I_{\mathrm{decode}}(B) \approx 4.0 \times B\ \mathrm{FLOP/byte}$$

The transition from the memory-bandwidth-bound regime to the compute-bound regime occurs at the hardware Ridge Point ($I_{\mathrm{ridge}}$):

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     THEORETICAL CROSSOVER BATCH SIZES                            │
├─────────────────────────┬──────────────────────────┬─────────────────────────────┤
│ Node 1: GTX 1050 Ti     │ Node 2: Mac Studio       │ Node 3: NVIDIA L4           │
│ (Pascal GDDR5)          │ M3 Ultra (Apple UMA)     │ (Ada Lovelace GDDR6)        │
├─────────────────────────┼──────────────────────────┼─────────────────────────────┤
│ • Peak FP32: 2.1 TFLOPS │ • Peak FP16: ~48 TFLOPS  │ • Peak FP16: 120 TFLOPS     │
│ • Peak BW: 112 GB/s     │ • Peak BW: 800 GB/s      │ • Peak BW: 300 GB/s         │
│ • Ridge Point: 18.75    │ • Ridge Point: ~60.0     │ • Ridge Point: 400.0        │
│                         │                          │                             │
│ Predicted Crossover:    │ Predicted Crossover:     │ Predicted Crossover:        │
│ B_crossover ≈ 4.7 (~5)  │ B_crossover ≈ 15.0 (~15) │ B_crossover ≈ 100.0 (>50)   │
└─────────────────────────┴──────────────────────────┴─────────────────────────────┘
```

* **Prediction for NVIDIA L4:** With $I_{\mathrm{ridge}} = 400\ \mathrm{FLOP/byte}$, the L4 remains deep in the memory-bandwidth-bound regime across all standard concurrency levels ($B \le 16$), scaling decode throughput steadily as batch size grows.

---

## 3. Discrete-Event Simulation: Static vs. Continuous Batching

To model the fundamental mechanism of Head-of-Line blocking, we simulated an identical Poisson arrival trace (100 requests, 4 processing slots, variable output lengths 16–128 tokens) across both scheduling paradigms (`batching_simulator.py`):

```text
==========================================================================
 BATCHING SIMULATOR: STATIC VS. CONTINUOUS ITERATION SCHEDULING
 Requests: 100 | Slots: 4 | Identical Arrival Trace
==========================================================================
 METRIC                      | STATIC BATCHING    | CONTINUOUS BATCHING
-----------------------------+--------------------+--------------------
 Average Queue Wait (steps)  |         493.67     |         100.38
 P50 Queue Wait (steps)      |         496.00     |         112.00
 P90 Queue Wait (steps)      |         931.00     |         163.00
 P99 Queue Wait (steps)      |        1037.00     |         202.00
 Hardware Slot Utilization   |          64.0%     |          96.3%
 Speedup / Wait Reduction    |       BASELINE     |          4.92x faster
==========================================================================
```

### Architectural Insight:
* **The Static Batching Flaw:** In static batching, all 4 slots lock until the slowest request in the batch completes. Slots that finish early sit idle (utilization drops to 64%), while incoming requests accumulate in the queue (P99 wait: 1037 steps).
* **The Continuous Batching Win:** Continuous iteration-level batching backfills empty slots on the very next token step, cutting average queue wait time by **$4.92\times$** and driving slot utilization to **96.3%**.

---

## 4. Empirical Telemetry Matrix (Node 3: NVIDIA L4 Cloud GPU)

*Hardware: 1x NVIDIA L4 (24 GB GDDR6), Driver 580.126.20, CUDA 13.0.*  
*Server Config: `--parallel 8`, context locked to `-c 16384` ($8 \times 2048$ tokens per slot), `--metrics`.*

### Experiment A: Closed-Loop Concurrency Scaling ($N = 1, 2, 4, 8, 16$)

| Concurrency ($N$) | Completed Reqs | Aggregate tok/s | Request Throughput (req/s) | Median TTFT (ms) | P90 TTFT (ms) | P99 TTFT (ms) | Median TPOT (ms/tok) | P99 ITL (ms) | Little's Law ($L$) | SLO Goodput (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1**  | 129 | **167.30** | 6.43  | 15.05 | 15.68  | 16.04  | 5.59  | 6.36  | **1.00**  | 100.0% |
| **2**  | 214 | **267.47** | 10.64 | 23.36 | 24.25  | 31.29  | 6.80  | 17.75 | **2.00**  | 100.0% |
| **4**  | 299 | **372.95** | 14.78 | 35.75 | 43.65  | 50.45  | 9.76  | 24.24 | **3.99**  | 100.0% |
| **8**  | 369 | **457.58** | 18.09 | 43.00 | 58.71  | 69.40  | 16.28 | 37.14 | **7.92**  | 100.0% |
| **16** | 408 | **504.23** | 19.68 | **428.21** | **475.76** | **520.11** | 15.23 | 41.17 | **15.69** | 96.6%  |

```
CONCURRENCY VS TTFT INFLATION (THE SLOT SATURATION CLIFF)
TTFT (ms)
  500 ┤                                                 ● (N=16: 428ms - 10x cliff!)
  400 ┤
  300 ┤
  200 ┤
  100 ┤
   50 ┤          ● (N=2: 23ms)   ● (N=4: 36ms)   ● (N=8: 43ms)
    0 ┼──● (N=1: 15ms)
      └──┬─────────────┬───────────────┬───────────────┬───────────────┬──
        N=1           N=2             N=4             N=8            N=16
```

---

### Experiment B: Open-Loop Poisson Arrival Rate Sweep ($\lambda = 1, 2, 4, 8, 12, 16\ \mathrm{req/s}$)

| Arrival Rate ($\lambda$) | Completed Reqs | Total Tokens | Aggregate tok/s | Request Throughput (req/s) | Median TTFT (ms) | P90 TTFT (ms) | P99 TTFT (ms) | Max TTFT (ms) | P99 ITL (ms) | SLO Goodput (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 req/s**  | 13  | 331  | 17.01  | 0.67  | 15.05 | 23.16  | 39.88  | 42.12  | 8.56  | 100.0% |
| **2 req/s**  | 56  | 1392 | 69.08  | 2.78  | 17.09 | 24.35  | 30.97  | 37.51  | 17.02 | 100.0% |
| **4 req/s**  | 82  | 2033 | 101.09 | 4.08  | 15.76 | 26.69  | 36.84  | 40.79  | 17.75 | 100.0% |
| **8 req/s**  | 155 | 3876 | 191.92 | 7.67  | 24.56 | 33.83  | 43.57  | 46.47  | 23.02 | 100.0% |
| **12 req/s** | 230 | 6007 | 297.74 | 11.40 | 30.11 | 44.35  | **116.91** | **205.53** | 29.01 | 100.0% |
| **16 req/s** | 300 | 7758 | 376.08 | 14.54 | 42.01 | **175.60** | **269.36** | **290.81** | 32.39 | 100.0% |

---

### Experiment C: Mixed-Workload Intruder Test (Prefill-Decode Interference)

*4 background short streams (50 prompt tokens, 64 output tokens) + 1 mid-flight 1,500-token intruder prompt:*

| Metric | Normal Background Value | Intruder Impact (Spike) | Degradation Factor |
| :--- | :--- | :--- | :--- |
| **Median TTFT** | 36.80 ms | - | Baseline |
| **Max TTFT** | 50.60 ms (P99) | **130.15 ms** | **2.57x** TTFT Inflation |
| **Median Inter-Token Latency (ITL)** | 7.91 ms | - | Smooth human reading pace |
| **P99 Inter-Token Latency (ITL)** | 25.81 ms | - | Standard tail jitter |
| **Max Inter-Token Latency (ITL)** | 25.81 ms | **122.16 ms** | **15.44x Stutter Spike** |

**Observation:** While background streams generated tokens at a steady 7.91 ms cadence, the 1,500-token prompt prefill monopolized the GPU tensor cores, forcing running decode steps to stall for **122.16 ms**. This is empirical proof of prefill-decode interference in continuous batching.

---

## 5. Architectural Conclusions & SRE Production Rules

1. **Size Hardware Slots to Peak Load:** Because `--parallel 8` caps physical continuous batching slots, any traffic spike above 8 concurrent streams triggers linear FIFO queue wait times ($10\times$ TTFT explosion). Autoscaling policies (KEDA) must trigger before slot occupancy hits 85%.
2. **Open-Loop Telemetry is Required for True Capacity:** In closed-loop testing at $N = 16$, the system reported an aggregate 504 tok/s. But in reality, individual clients experienced a 10x degradation in response time. Always use open-loop Poisson arrival generators to establish realistic production SLO goodput.
3. **Chunk Prefills to Protect Streaming UX:** Large prompts must be sliced via chunked prefill (`-ub 128` or `-ub 512`) to prevent 15x ITL stutter spikes on active conversational streams.
