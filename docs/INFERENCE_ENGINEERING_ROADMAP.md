# Inference Engineering Master Roadmap & Systems Curriculum

**Target Audience:** DevOps / Platform / SRE Engineers transitioning into Senior AI Infrastructure & LLM Systems Engineering.  
**Hardware Fleet:**
* **Node 1 (Constrained Edge Linux CUDA):** NVIDIA GeForce GTX 1050 Ti (4 GB GDDR5, Pascal, CC 6.1, ~112 GB/s bandwidth) on PCIe 3.0 x16.
* **Node 2 (Enterprise Bare-Metal UMA Server):** Apple Mac Studio M3 Ultra (28 Cores, 256 GB Unified LPDDR5X Memory, ~800 GB/s bandwidth).
* **Node 3 (Production Edge Inference Workhorse):** Apple Mac mini M4 (10 Cores, 24 GB Unified Memory).

**Zero-Cloud Budget:** All learning, benchmarking, and profiling is executed on local silicon. vLLM, CUDA profiling, and multi-GPU distributed serving are learned conceptually, by reading source code, running toy schedulers, and doing the physical bandwidth math.

---

## 1. Hardware Strategy: The Dual-Node Advantage

Running experiments across both machines provides an immediate, profound lesson in systems engineering: **the bottleneck moves depending on the architecture.**

| Hardware Node | Good For (Learning Vector) | Cannot Teach (Boundary) |
| :--- | :--- | :--- |
| **GTX 1050 Ti (4 GB)** | Memory pressure within minutes (KV cache OOM, slot exhaustion), roofline math (decode ceiling = bandwidth / model bytes), real NVIDIA driver / NVML behavior, K8s GPU device plugins. Capacity-bound. | Tensor Cores, FP16 throughput, vLLM/SGLang (requires CC 7.0+), modern Nsight. Stay on 1B–3B Q4 models. |
| **Mac Studio M3 Ultra (256 GB)** | Large models (70B-class, MoE), deep context (32k–128k) KV cache growth, `llama.cpp` parallel slots + continuous batching, `mlx_lm.server`, prefill vs decode trade-offs. Bandwidth/prefill-bound. | CUDA tooling, NCCL, multi-GPU NVLink, PagedAttention CUDA kernels, NVIDIA Xid / ECC failure modes. |

---

## 2. What an Inference Engineer Actually Does

**The Job:** Turn a "pile of weights" into a production endpoint with **predictable latency, throughput, and cost**, and keep it online under overload and hardware failure.

### The 8 Domains of Inference Engineering
1. **Serving Engine & Scheduler:** Owns continuous batching, KV allocation, admission control, chunked prefill, and preemption. (Skip writing CUDA kernels).
2. **GPU & Performance Engineering:** Diagnoses compute-bound vs memory-bandwidth-bound regimes, roofline ceilings, and memory accounting.
3. **Request Path (Gateway & Streaming):** Owns reverse proxying, auth, rate limiting, SSE streaming, cancellation propagation, and backpressure.
4. **Capacity & Cost:** Sizes fleets from token length distributions and P99 latency targets, not simple average QPS.
5. **Reliability & Operations:** Manages overload behavior, OOM recovery, GPU driver stalls, graceful drain, and readiness probes.
6. **Observability:** Measures metrics, structured JSON logs, and OpenTelemetry distributed traces separating queue wait from compute from network.
7. **Fleet & Deployment:** Manages GPU node pools, driver compatibility, model weight caching, and deadlock-free rolling deployments.
8. **Distributed Inference:** Owns Tensor Parallelism (TP), Pipeline Parallelism (PP), and Prefill-Decode Disaggregation.

### Where Your DevOps/SRE Background is an Unfair Advantage
Most real inference outages are **not** GPU mysteries. They are **queueing, timeouts, proxy buffer bloat, failed rollouts, health check flap, and dependency saturation**. 

As a Platform/SRE engineer, you already master Domains 3, 5, 6, and 7. Your gap is bridging Domains 1, 2, 4, and 8. That is what this platform builds.

---

## 3. The Minimum ML You Need (The P0 Filter)

* **P0 (Required):**
  * Tensors, shapes, and dtypes (`FP16`, `BF16`, `INT8`, `INT4`, `FP8`). Memory = elements $\times$ bytes.
  * Transformer forward pass (Attention + MLP layers). Shape level only.
  * Self-Attention ($Q, K, V$): explains why $K$ and $V$ are cached and why prefill is compute-heavy.
  * **KV Cache Size Formula** (drives all capacity planning).
  * MHA vs GQA vs MQA (changes KV cache size by 4x–8x).
  * Tokenization & Chat Templates (tokenizer mismatches produce garbage output).
  * Autoregressive Decode Loop (one forward pass per generated token).
  * Quantization (GGUF `Q4_K_M`, AWQ, FP8) trading precision for memory bandwidth.
* **P1 (Useful):** RoPE, Mixture-of-Experts (MoE), Embeddings, LoRA adapter serving, Speculative Decoding, Prefix Caching.
* **IGNORE (Stop Here):**
  * Pretraining & Scaling laws (different job).
  * Fine-tuning loss & backprop (training-side only; inference does not compute gradients).
  * RLHF / DPO / Reward models (researcher territory).
  * Linear algebra beyond matrix multiplication shapes.

### The Two Mandatory Formulas to Memorize

#### Formula 1: KV Cache Memory per Token
$$\text{KV Cache Bytes per Token} = 2 \times \text{layers} \times \text{kv\_heads} \times \text{head\_dim} \times \text{bytes\_per\_element}$$

*Example (Llama 3.2 3B in FP16):*
$$2 \times 28 \text{ layers} \times 8 \text{ KV heads} \times 128 \text{ head dim} \times 2 \text{ bytes} = 114,688 \text{ bytes} \approx 112 \text{ KiB per token}$$
At an 8K context window, one sequence consumes **~0.9 GB of VRAM**. On a 4 GB GTX 1050 Ti with ~2 GB of model weights, there is physical room for only **two concurrent long sequences**.

#### Formula 2: Single-Stream Decode Ceiling
$$\text{Decode Ceiling (tok/s)} \approx \frac{\text{Memory Bandwidth (GB/s)}}{\text{Model Weight Bytes Read per Token (GB)}}$$

*Example (GTX 1050 Ti with ~2 GB Q4 Model):*
$$\frac{112 \text{ GB/s}}{2 \text{ GB}} \approx 56 \text{ tokens/second}$$
Batching helps because loading the model weights once across the memory bus serves multiple concurrent tokens.

---

## 4. The 10 Build Projects Mapped to Repository Stages

```
ai-infra-systems/stages/
├── 01-baseline-inference/       ──▶ Project 1: Single-GPU Inference Server & Roofline Prediction
├── 02-cold-start-behavior/      ──▶ Project 2: Cold vs Warm Breakdown (drop_caches & page cache)
├── 03-concurrency-and-queues/   ──▶ Project 3: Concurrency, Queueing & Finding the Knee
├── 04-memory-pressure-and-oom/  ──▶ Project 5 & 9: KV Cache Pressure & GPU OOM Triage
├── 05-streaming-and-edge-proxy/ ──▶ Project 6: Reverse Proxy Streaming & Buffer Bloat
├── 06-failure-injection/        ──▶ Project 8: Overload, Backpressure & 429/503 Shedding
├── 07-observability-telemetry/  ──▶ Project 7: End-to-End Metrics, Logs & Distributed Traces
├── 08-serving-optimizations/    ──▶ Project 4: Continuous Batching & Toy Scheduler Simulator
├── 09-production-deployment/    ──▶ Project 10: Multi-GPU Placement, K8s Pods & Topology
└── 10-capacity-planning/        ──▶ Project 10: Token Sizing, Fleet Economics & SLO Math
```

---

## 5. Performance Engineering: The Diagnostic Loop

### Step 0: Split TTFT from ITL
* **TTFT up, ITL normal:** Queue wait, long prompt prefill, prefix cache miss, or reverse proxy / gateway transit lag.
* **ITL up, TTFT normal:** Batch size too large, deep KV cache reading, interleaved prefills, thermal throttling, or CPU layer offloading.
* **Both up:** Hardware saturation, GPU throttling, or KV preemption/eviction.
* **Tokens arrive in bursts:** Reverse proxy buffer bloat (`proxy_buffering on;`), Nginx chunking, or gzip enabled on event streams.
* **Errors / Timeouts up:** Queue overflow, OOM restart, upstream timeout disconnects.

### The Scientific Troubleshooting Loop
$$\text{SYMPTOM} \longrightarrow \text{HYPOTHESIS} \longrightarrow \text{MEASUREMENT} \longrightarrow \text{EVIDENCE} \longrightarrow \text{ROOT CAUSE} \longrightarrow \text{FIX} \longrightarrow \text{VERIFICATION}$$

#### The 2:00 AM Production Scenario
*"Users are timing out. GPU utilization looks weird. TTFT is 8.2 seconds. Throughput appears normal."*
1. **Diagnosis:** Normal throughput means the GPU is actively executing matrix operations—it is not crashed. An 8.2s TTFT means requests are **waiting in a queue**.
2. **Investigation:**
   * Check queue depth and wait time histogram.
   * Inspect prompt token length distribution (did a client start sending 30k-token prompts, monopolizing prefill?).
   * Check KV utilization: a full KV pool keeps throughput high while blocking all incoming admissions.
   * Check prefix cache hit rates and client retry storm rates.
3. **Immediate Mitigation:** Enforce admission limits, shed excess traffic with HTTP 429 and `Retry-After`, cap `max_tokens`, and prioritize interactive traffic. Root-cause after bleeding stops.

---

## 6. Interview War Room: Model Answers & Traps

### Q1: Why did throughput increase while P99 latency became terrible?
* **Junior Answer:** *"More requests were being processed."*
* **Strong Answer:** *"Offered load passed the knee of the latency curve. The engine saturated its batch capacity, causing subsequent arrivals to queue. Aggregate tokens/second stayed high or increased because larger batches amortize the cost of reading weights across the memory bus, but individual requests experienced severe queue wait. Throughput measures engine productivity; P99 measures user experience."*
* **Expert Follow-Up:** *"How do you pick the operating point?"* $\rightarrow$ Plot goodput (% within SLO) versus load. Operate strictly to the left of the knee, cap queue depth, and scale based on queue wait time.

### Q2: Why can a GPU show 95% utilization but perform terribly?
* **Junior Answer:** *"The GPU is overloaded."*
* **Strong Answer:** *"GPU utilization in `nvidia-smi` only measures the percentage of sample time during which at least one kernel was executing. At small batch sizes, decode is memory-bandwidth bound and can register 95% utilization while executing a fraction of peak FLOPs with low SM occupancy. Alternatively, the GPU is thermally throttled or stalled waiting on CPU-offloaded layers over PCIe."*
* **Expert Follow-Up:** *"What metrics do you check instead?"* $\rightarrow$ Memory bandwidth utilization (DCGM), SM activity, clock throttling reasons, batch size, and tokens/sec versus the roofline model.

### Q3: Why can the KV cache cause an OOM even when model weights fit comfortably?
* **Junior Answer:** *"It's a memory leak."*
* **Strong Answer:** *"Model weights are static, but the KV cache grows dynamically with tokens in flight: $\text{KV} = (\text{prompt} + \text{generated tokens}) \times \text{concurrency}$. With deep context lengths and concurrent users, the KV cache can exceed the weight size. If the serving runtime does not enforce KV-aware admission limits, incoming requests allocate beyond free VRAM and trigger an allocation crash."*

### Q4: Why does disabling proxy buffering matter for LLM serving?
* **Junior Answer:** *"It makes streaming faster."*
* **Strong Answer:** *"By default, Nginx and reverse proxies buffer upstream response chunks until an internal buffer (e.g. 4KB–16KB) fills or the connection closes, optimizing network packet density. For Server-Sent Events, this destroys the typewriter effect, delivering tokens in jarring bursts. We configure `proxy_buffering off;` and `X-Accel-Buffering: no;` so every generated token flushes immediately to the client socket."*

### Q5: How do you determine whether a bottleneck is compute, bandwidth, queueing, or network?
* **Strong Answer:**
  1. **Network vs Engine:** Compare timestamps across hops: Client TTFT vs Gateway TTFT vs Engine TTFT. A constant delta indicates network transit.
  2. **Queueing vs Execution:** Check queue wait time versus engine step time. High queue time with normal step time indicates queueing saturation.
  3. **Compute vs Bandwidth:** Vary batch size or quantization. If tokens/sec scales with bytes per weight, the workload is bandwidth-bound. If it scales with clocks and FLOPs, it is compute-bound.

---

## 7. What to Explicitly SKIP (Protecting Your Time)

* **CUDA Kernel Authoring / PTX / Triton Programming:** Kernel engineering is a specialist skill. You cannot verify it on a 1050 Ti, and its operational value for platform engineers is near zero.
* **Writing FlashAttention:** Understand that it tiles attention into SRAM to cut HBM round trips. Do not read the CUDA code line-by-line.
* **Training Infrastructure:** Backpropagation, gradient accumulation, Megatron, ZeRO. You are serving models, not training them.
* **Application Frameworks (LangChain, LlamaIndex):** Application glue code belongs to product developers, not infrastructure engineers.
* **Generic MLOps (Kubeflow, MLflow):** Traditional tabular training pipelines do not teach Foundation Model inference systems.

**Depth in the 20% beats breadth in everything.**
