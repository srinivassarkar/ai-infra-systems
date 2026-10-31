# The 10-Stage Systems Synthesis: LLM Inference Reliability Lab

A unified, end-to-end executive digest of the ten progressive engineering stages in `ai-infra-systems`. Every number, latency metric, and memory footprint in this document was directly measured on physical silicon (Edge CUDA, Apple Silicon Unified Memory, and Cloud Enterprise GPUs).

Use this document as your solitary study harness to internalize the mechanical cause-and-effect chain across the entire LLM serving stack.

---

## The Cohesive Architectural Arc

LLM serving is not a collection of disconnected benchmarks. It is a single continuous physical pipeline:

```
[ Stage 01: Silicon Physics ]  ──▶ Single-stream memory bus bottleneck (Roofline Model)
             │
             ▼
[ Stage 02: Process Startup ]  ──▶ Cold starts, page faults, and virtual memory paging
             │
             ▼
[ Stage 03: Multi-User Queuing]──▶ Slot saturation cliff, Little's Law, and ITL stutter
             │
             ▼
[ Stage 04: Memory Ceilings ]  ──▶ Dynamic KV cache growth, GQA math, and OOM boundaries
             │
             ▼
[ Stage 05: Edge Networking ]  ──▶ Reverse proxy buffer bloat, SSE streaming, and TCP backpressure
             │
             ▼
[ Stage 06: Chaos & MTTR ]     ──▶ Gray Failure Paradox (L4 SYN-ACK vs L7 503) and auto-recovery
             │
             ▼
[ Stage 07: Observability ]    ──▶ Bridging upstream metrics gaps and OTel trace span economy
             │
             ▼
[ Stage 08: Serving Levers ]   ──▶ FlashAttention, chunked prefill, and prefix cache hits
             │
             ▼
[ Stage 09: Orchestration ]    ──▶ 51.7s Kubernetes scaling lag chain and KEDA queue depth
             │
             ▼
[ Stage 10: Economics ]        ──▶ Dual-constraint capacity sizing and cost per 1M tokens
```

---

## Stage-by-Stage Executive Synthesis

### Stage 01: Baseline Single-Stream Profiling
* **The Question:** How fast does a model run when exactly one user is querying it, and why does token generation behave fundamentally differently from prompt reading?
* **The Ground Truth:** Inference is divided into two distinct mathematical regimes:
  1. *Prefill Phase (TTFT):* Compute-bound matrix-matrix multiplication (GEMM). All prompt tokens are processed simultaneously in parallel.
  2. *Decode Phase (TPOT):* Memory-bandwidth-bound matrix-vector multiplication (GEMV). For every single token generated, the GPU must stream the entire model weights across its memory bus. Arithmetic intensity is fixed at $I \approx 4.0\ \mathrm{FLOP/byte}$.
* **Empirical Fleet Numbers:**
  - *Apple Mac Studio M3 Ultra (800 GB/s):* TTFT $15.1\ \mathrm{ms}$ (warm), Decode $131.0\ \text{tok/s}$ ($7.63\ \mathrm{ms/tok}$), Achieved BW $127.54\ \mathrm{GB/s}$ ($15.9\%$ bus saturation at $B=1$).
  - *NVIDIA L4 Cloud GPU (300 GB/s):* TTFT $7.7\ \mathrm{ms}$ local ($147.5\ \mathrm{ms}$ over Tailscale), Decode $183.5\ \text{tok/s}$, Achieved BW $181.26\ \mathrm{GB/s}$ ($60.4\%$ bus saturation).
  - *NVIDIA GTX 1050 Ti (112 GB/s):* TTFT $104.3\ \mathrm{ms}$, Decode $51.0\ \text{tok/s}$, Achieved BW $53.7\ \mathrm{GB/s}$ ($47.9\%$ bus saturation).
* **The SRE Takeaway:** Over $94\%$ of remote API wait time is pure internet latency, but token streaming speed is immune to network distance once the TCP socket opens. Single users on narrow models cannot saturate wide memory buses ($15.9\%$ on 80 Metal cores); batching is mandatory.

---

### Stage 02: Cold-Start Behavior & Virtual Memory Paging
* **The Question:** Does operating system memory-mapping (`mmap`) eliminate container cold-start latency for GPU inference?
* **The Ground Truth:** No. `mmap` is a virtual memory mechanism for host CPUs. When offloading layers to a GPU (`-ngl 99`), `mmap` yields zero cold-start benefit on discrete GPUs (CUDA requires `cudaMemcpy` to push weights across PCIe, page-faulting everything at boot) and is actively harmful on Unified Memory.
* **Empirical Fleet Numbers:**
  - *M3 Ultra Lazy `mmap`:* Boot $2,594.54\ \mathrm{ms}$, First TTFT $519.85\ \mathrm{ms}$, Total Cold Start **$3,114.39\ \mathrm{ms}$**.
  - *M3 Ultra Eager Heap Read (`mmap=OFF`):* Boot $2,226.30\ \mathrm{ms}$, First TTFT $37.74\ \mathrm{ms}$, Total Cold Start **$2,264.04\ \mathrm{ms}$** (**$850.35\ \mathrm{ms}$ faster**).
  - *Host CPU Ablation (`-ngl 0`):* Lazy `mmap` booted $379\ \mathrm{ms}$ faster ($1,322\ \mathrm{ms}$ vs $1,701\ \mathrm{ms}$), confirming lazy paging benefits exist exclusively for CPU-only execution.
* **The SRE Takeaway:** Always configure GPU containers with eager loading (`--load-mode none`). Sequential physical reads from NVMe storage beat piecemeal virtual page translation faults every single time.

---

### Stage 03: Concurrency, Queues & Head-of-Line Blocking
* **The Question:** Where is the saturation knee point where queue wait time overtakes actual generation time?
* **The Ground Truth:** Hardware execution slots represent hard physical concurrency barriers. When closed-loop concurrency doubles past physical server slots ($N = 8 \rightarrow 16$), client TTFT explodes by $\sim 10\times$ across both discrete GDDR6 and unified LPDDR5X, while decode TPOT remains invariant—proving the penalty is 100% queue starvation ($W_q$).
* **Empirical Fleet Numbers:**
  - *M3 Ultra Closed-Loop Sweep ($--parallel 8$):*
    - $N = 1$: TTFT $36.20\ \mathrm{ms}$, Throughput $109.1\ \text{tok/s}$, SLO Goodput $100\%$.
    - $N = 8$: TTFT $76.67\ \mathrm{ms}$, Throughput $262.0\ \text{tok/s}$, SLO Goodput $100\%$.
    - $N = 16$: TTFT **$729.65\ \mathrm{ms}$** ($9.51\times$ explosion), Throughput $284.4\ \text{tok/s}$, SLO Goodput **$6.2\%$ (Collapse)**.
  - *TPOT Invariance:* Decode generation speed was $27.61\ \mathrm{ms/tok}$ at $N=8$ and $26.40\ \mathrm{ms/tok}$ at $N=16$. The GPU did not slow down; the extra $650\ \mathrm{ms}$ was pure FIFO queue delay.
  - *Prefill-Decode Interference:* Injecting a 1,500-token prompt caused a **$15.44\times$ ITL stutter spike** ($7.91\ \mathrm{ms} \rightarrow 122.16\ \mathrm{ms}$) on background streams.
* **The SRE Takeaway:** Closed-loop benchmarks suffer from coordinated omission. Continuous batching eliminates request-level HoL blocking, but chunked prefill (`-ub 64`) is mandatory to prevent large prompt prefills from starving ongoing decode streams.

---

### Stage 04: Memory Pressure, KV Cache Bloat & OOM Prevention
* **The Question:** What is the mathematical boundary where KV cache growth exhausts VRAM, and how do we prevent kernel OOM kills?
* **The Ground Truth:** Model weights are static, but the KV cache expands dynamically with sequence length and concurrency:
  $$\text{KV Bytes per Token} = 2 \times L \times H_{\mathrm{kv}} \times D_{\mathrm{head}} \times B_{\mathrm{element}}$$
  For Qwen 1.5B (GQA 28 layers, 2 heads, 64 head dim), this is exactly $14,336\ \mathrm{bytes/token}$.
* **Empirical Fleet Numbers:**
  - *Crossover Point:* At $72,120$ cumulative active tokens, bandwidth spent reading the KV cache exceeds bandwidth spent reading model weights ($986\ \mathrm{MB}$).
  - *$8 \times 32\mathrm{K}$ Context Scale ($29.16\ \mathrm{GB}$ total VRAM demand):*
    - Node 1 (GTX 1050 Ti 4 GB): Instant crash ($7.3\times$ over capacity).
    - Node 3 (NVIDIA L4 24 GB): Instant crash ($29.16\ \mathrm{GB} > 24\ \mathrm{GB}$).
    - Node 2 (Mac Studio M3 Ultra 256 GB): Healthy ($11.4\%$ memory used, $>226\ \mathrm{GB}$ free).
  - *PagedAttention Simulation:* Eliminated external memory fragmentation and verified prefix block sharing (`ref_count = 2`).
* **The SRE Takeaway:** Always enforce ingress token admission control based on worst-case token reservations ($P_{\mathrm{prompt}} + G_{\mathrm{max\_tokens}}$). Distinguish host kernel OOMs (Exit 137 via SIGKILL) from in-process GPU allocation aborts.

---

### Stage 05: Edge Streaming Reverse Proxy & Backpressure
* **The Question:** Why do standard production reverse proxies destroy Server-Sent Events (SSE) streaming?
* **The Ground Truth:** Default Nginx buffering (`proxy_buffering on;`) traps small token chunks in 4KB/8KB memory buffers, turning real-time typewriter output into multi-second frozen stalls followed by abrupt bursts.
* **Empirical Fleet Numbers:**
  - *M3 Ultra 4-Stream Direct Load:* Success rate $100\%$, Median TTFT $140.22\ \mathrm{ms}$, Median ITL **$12.88\ \mathrm{ms}$** ($77.6\ \text{tok/s}$ per user), P99 ITL $16.86\ \mathrm{ms}$, Max ITL $21.46\ \mathrm{ms}$, Aggregate throughput $312.64\ \text{tok/s}$.
  - *Buffer Bloat Elimination:* The difference between Median ITL ($12.88\ \mathrm{ms}$) and Max ITL ($21.46\ \mathrm{ms}$) is under $9\ \mathrm{ms}$, proving smooth word-by-word streaming.
* **The SRE Takeaway:** Configure `proxy_buffering off;` and `proxy_set_header X-Accel-Buffering no;`. Critically, set `proxy_ignore_client_abort off;` so when a user closes their browser tab, the proxy drops the connection immediately, halting wasteful GPU token generation.

---

### Stage 06: Failure Injection, Daemon Crashes & Self-Healing MTTR
* **The Question:** How does the inference platform behave under hard crashes (`SIGKILL`) versus compute freezes (`SIGSTOP`)?
* **The Ground Truth:** When an inference process locks up (`SIGSTOP`), kernel-space TCP continues answering connection requests with `SYN-ACK` flags while user-space application threads are dead.
* **Empirical Fleet Numbers:**
  - *The Gray Failure Paradox:*
    - Layer 4 TCP Port Probe: **OPEN (PASS)**
    - Layer 7 HTTP `/health` Probe: **UNRESPONSIVE (FAIL)**
  - *Hard SIGKILL MTTR Distribution:*
    - M3 Ultra: Mean MTTR **$2,674.33\ \mathrm{ms}$** (P50 $2,661.99\ \mathrm{ms}$, $\sigma = 26.70\ \mathrm{ms}$).
    - NVIDIA L4: Mean MTTR **$2,224.86\ \mathrm{ms}$** (P50 $2,231.83\ \mathrm{ms}$, $\sigma = 29.74\ \mathrm{ms}$).
    - Both beat the strict $3.0\ \mathrm{s}$ SLA with near-deterministic precision.
* **The SRE Takeaway:** Never configure Network Load Balancers or Kubernetes readiness probes with pure Layer 4 TCP checks. You will black-hole production traffic to dead containers. Always probe Layer 7 HTTP endpoints with strict 1-second timeouts.

---

### Stage 07: Observability, Metrics & Telemetry
* **The Question:** Can upstream inference engines be observed directly via Prometheus, and how do we trace streaming requests without drowning collectors?
* **The Ground Truth:** Upstream serving engines (`llama-server`) natively lack TTFT and TPOT latency histograms. Naively tracing each generated token creates an unmanageable $8.64\ \text{billion spans/day}$ disaster at $100\mathrm{K}\ \text{tok/s}$.
* **Empirical Fleet Numbers:**
  - *Upstream Metrics Audit:* Confirmed `llama-server` on port 8080 exports raw token sums, but TTFT and TPOT histograms are completely absent.
  - *Ingress Telemetry Gateway (Port 9095):* Captured 4 concurrent streams in $1.11\ \mathrm{s}$ (Median TTFT $111.40\ \mathrm{ms}$, Mean TPOT $13.92\ \mathrm{ms/tok}$).
  - *Span Cardinality Fix:* Emitted 1 parent span with exactly 2 structured child spans (`prefill` and `decode`), capturing full phase granularity with zero collector overhead.
* **The SRE Takeaway:** An ingress telemetry gateway is non-negotiable for enterprise SLO tracking. Track the 5 Golden Signals of LLM serving: TTFT, TPOT/ITL, KV cache saturation, queue depth, and hardware utilization.

---

### Stage 08: Serving Engine Optimizations & Ablation Matrix
* **The Question:** What is the isolated, measurable impact of each individual serving optimization knob?
* **The Ground Truth:** Inference performance cannot be treated as a monolithic black box. Tuning runtime knobs yields dramatic architectural trade-offs:
* **Empirical Fleet Numbers:**
  - *FlashAttention:* Boosted M3 Ultra decode speed by **+31.9%** ($104.76 \rightarrow 138.20\ \text{tok/s}$); collapsed L4 prefill latency by $3.35\times$ ($108.41 \rightarrow 32.36\ \mathrm{ms}$) via on-chip SRAM tiling.
  - *KV Cache Quantization (`q8_0`):* Cuts memory footprint in half ($28.0\ \mathrm{GB} \rightarrow 14.0\ \mathrm{GB}$ on 32K context) with negligible decode speed variation ($134.8 \rightarrow 119.8\ \text{tok/s}$).
  - *Chunked Prefill (`-ub 64` vs `-ub 512`):* Slicing prompts into 64-token chunks added $+39.0\ \mathrm{ms}$ prefill overhead, but kept decode speed completely invariant at $137.3\ \text{tok/s}$, eliminating multi-user stutter.
  - *Prefix KV Caching:* Collapsed Turn 2 TTFT by **$2.22\times$** on M3 Ultra ($75.87 \rightarrow 34.11\ \mathrm{ms}$) and **$3.86\times$** on L4 ($46.56 \rightarrow 12.06\ \mathrm{ms}$).
* **The SRE Takeaway:** FlashAttention and prefix caching provide free performance wins. Chunked prefill is a deliberate engineering trade-off: sacrifice milliseconds of prompt latency to shield concurrent users from stutter.

---

### Stage 09: Kubernetes Orchestration & KEDA Scaling Lag
* **The Question:** Why does standard Kubernetes Horizontal Pod Autoscaling (HPA) fail for LLM inference workloads?
* **The Ground Truth:** NVIDIA GPU utilization measures compute kernel execution time, not memory or slot headroom. A pod serving 2 streams reads $90\%$ GPU utilization. More importantly, reactive container scheduling takes over 50 seconds to complete.
* **Empirical Fleet Numbers:**
  - *The 7-Step Scaling Lag Chain:* Prometheus scrape ($15\ \mathrm{s}$) + KEDA polling ($15\ \mathrm{s}$) + HPA sync ($15\ \mathrm{s}$) + scheduler binding ($1.5\ \mathrm{s}$) + containerd init ($2.0\ \mathrm{s}$) + model weights page-in ($1.23\ \mathrm{s}$) + readiness probe pass ($2.0\ \mathrm{s}$) $= \mathbf{51.73\ \mathrm{seconds}}$.
  - *Policy Surge Benchmark ($4\times$ burst):* During the 51-second lag, static and GPU HPA suffered $49.9\%$ SLO breaches. KEDA queue-depth triggers reduced violations to $43.2\%$ by firing on the very first queued request.
* **The SRE Takeaway:** Reactive autoscaling alone can never protect tight sub-second latency SLOs against sudden traffic spikes. Clusters must maintain standby headroom ($N+1$ buffer pods) or schedule predictive Cron HPA scale-ups ahead of known peak hours.

---

### Stage 10: Capacity Planning, Token Economics & Hardware Sizing
* **The Question:** How many concurrent users can a given silicon setup sustain within SLOs, and what does it cost per 1 million tokens?
* **The Ground Truth:** The Dual-Constraint Capacity Axiom dictates:
  $$\text{Production Concurrency Limit} = \min\left(N_{\mathrm{mem}}, N_{\mathrm{SLO}}\right)$$
  For models $\le 7\mathrm{B}$ parameters, memory is cheap. The latency knee binds long before VRAM runs out.
* **Empirical Fleet Numbers:**
  - *Dual-Constraint Matrix ($4\mathrm{K}$ context, $\mathrm{TTFT} \le 250\ \mathrm{ms}$):*
    - Node 3 (NVIDIA L4): $N_{\mathrm{mem}} = 414$ slots, but $N_{\mathrm{SLO}} = \mathbf{8\ \mathrm{slots}}$.
    - Node 2 (M3 Ultra): $N_{\mathrm{mem}} = 4,656$ slots, but $N_{\mathrm{SLO}} = \mathbf{32\ \mathrm{slots}}$.
    - Node 1 (GTX 1050 Ti): $N_{\mathrm{mem}} = 48$ slots, but $N_{\mathrm{SLO}} = \mathbf{2\ \mathrm{slots}}$.
  - *Unit Economics ($/1M Tokens at 50% Daytime Load):*
    - NVIDIA L4 Cloud Spot ($0.35/hr): **$1.01 per 1M tokens** ($2.5\times$ cheaper than closed APIs at $2.50+/1M$).
    - Mac Studio M3 Ultra ($0.288/hr amortized): **$0.67 per 1M tokens**.
    - GTX 1050 Ti Edge ($0.017/hr): **$0.22 per 1M tokens**.
  - *24-Hour Diurnal Autoscaling:* KEDA reduced daily spend by **33.3%** ($25.20 \rightarrow 16.80\ \text{USD/day}$) with only $0.86\ \text{minutes}$ of morning scaling lag.
* **The SRE Takeaway:** Sizing an inference cluster on VRAM alone overprovisions concurrency by up to $50\times$, destroying user TTFT. Size strictly to the latency knee, and use KEDA to capture $33\%$ cloud cost savings.

---

## Master Fleet Calibration Matrix

```
┌─────────────────────────────────────────┬──────────────────────────┬──────────────────────────┬──────────────────────────┐
│ Systems Engineering Dimension           │ Node 1: Edge GTX 1050 Ti │ Node 2: Mac Studio M3    │ Node 3: Cloud NVIDIA L4  │
│                                         │ Pascal GDDR5 (112 GB/s)  │ Apple UMA (800 GB/s)     │ Ada Lovelace (300 GB/s)  │
├─────────────────────────────────────────┼──────────────────────────┼──────────────────────────┼──────────────────────────┤
│ Single-Stream Decode TPOT               │ 19.6 ms/tok (51.0 tok/s) │ 7.63 ms/tok (131.0 tok/s)│ 5.54 ms/tok (183.5 tok/s)│
│ Peak In-Slot Aggregate Throughput       │ 51.0 tok/s (N = 1)       │ 262.01 tok/s (N = 8)     │ 457.58 tok/s (N = 8)     │
│ Within-Slot P50 TTFT                    │ 104.3 ms                 │ 76.67 ms                 │ 43.00 ms                 │
│ Over-Capacity P50 TTFT (N = 16)         │ Crashed / Unserviced     │ 729.65 ms (9.51x cliff)  │ 428.21 ms (10.0x cliff)  │
│ Over-Capacity SLO Goodput (<500ms)      │ N/A                      │ 6.2% (catastrophic drop) │ 96.6% (looser margin)    │
│ Cold-Start Mechanism                    │ PCI DMA Page Fault       │ Contiguous Malloc (-850ms│ PCIe DMA Page Fault      │
│ Self-Healing Hard Crash MTTR (P50)      │ N/A                      │ 2,661.99 ms (sigma 26.7ms│ 2,231.83 ms (sigma 29.7ms│
│ SLO Latency Knee (Effective Slots)      │ 2 slots                  │ 32 slots                 │ 8 slots                  │
│ Cost per 1M Tokens (50% Daytime Load)   │ $0.22 / 1M               │ $0.67 / 1M               │ $1.01 / 1M               │
└─────────────────────────────────────────┴──────────────────────────┴──────────────────────────┴──────────────────────────┘
```
