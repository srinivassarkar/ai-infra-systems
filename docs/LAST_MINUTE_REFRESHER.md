# 10-Stage Inference Engineering Revision Notes & Overview

> **Single Source of Truth:** High-signal last-minute revision guide covering all 10 stages of the LLM Inference Reliability Lab.  
> **Evaluated Fleet:** Apple Mac Studio M3 Ultra (256 GB UMA), Cloud NVIDIA L4 (24 GB GDDR6), Edge GTX 1050 Ti (4 GB GDDR5).

---

## The 10-Stage Master Arsenal

```text
========================================================================================
THE 10-STAGE INFERENCE ENGINEERING MASTER ARSENAL (100% COMPLETE)
========================================================================================
[x] STAGE 01: Baseline Inference Profiling & Silicon Physics
    - Prefill = Compute-bound (TFLOPS) | Decode = Memory-bandwidth bound (GB/s).
    - 1 token = streaming the entire model weights. Doubling size halves speed.
    - WAN latency dominates TTFT (95%); SSE streaming renders distance invisible.

[x] STAGE 02: Cold-Start Behavior, Memory Paging & UNIX Signals
    - mmap on GPUs is an illusion. CudaMemcpy and Metal force eager loading anyway.
    - Eager malloc (--no-mmap) eliminates the 500ms first-request page-fault storm.
    - exec /app/llama-server guarantees PID 1 signal capture (SIGTERM 143 vs SIGKILL 137).

[x] STAGE 03: Multi-Slot Concurrency, Queuing & Head-of-Line Blocking
    - Sizing Law: -c = parallel * per_slot_context.
    - Prefill-Decode Interference: large prompts freeze active decode streams.
    - Fix: Chunked Prefill (--ubatch / -ub) breaks prompts into micro-batches.

[x] STAGE 04: Memory Pressure, Dynamic PagedAttention & OOM Triaging
    - Host RAM OOM (Exit 137 / SIGKILL) vs GPU VRAM OOM (cudaErrorMemoryAllocation).
    - Static Pre-allocation (llama.cpp) cannot OOM mid-request.
    - Dynamic PagedAttention (vLLM) swaps KV-blocks to CPU RAM, triggering 137 if unbudgeted.

[x] STAGE 05: Edge Streaming, Nginx Buffering & Client Backpressure
    - proxy_buffering off; kills the 4KB buffer delay and typewriter stutter.
    - proxy_set_header X-Accel-Buffering no; informs upstream CDNs.
    - proxy_ignore_client_abort off; halts GPU generation when clients disconnect.

[x] STAGE 06: Chaos Engineering, Deterministic MTTR & The Gray Failure Paradox
    - Process MTTR is mathematically deterministic (<2.7s on L4/M3 Ultra, sigma < 30ms).
    - The Gray Failure Trap: deadlocked processes pass Layer 4 TCP checks (kernel space).
    - AI workloads must ALWAYS use Layer 7 HTTP /health checks.

[x] STAGE 07: Observability, The 5 Golden Signals & Tracing
    - Total HTTP duration is a lie; track TTFT (Histogram) and TPOT/ITL (Histogram).
    - Queue Depth (requests_deferred) is the primary leading saturation indicator.
    - Hierarchical Tracing: Avoid span cardinality explosion (never 1 span per token).

[x] STAGE 08: Serving Engine Optimizations & The Ablation Matrix
    - FlashAttention (-fa 1): Tiled SRAM online softmax collapses prefill by 3.35x.
    - KV Cache Quantization (-ctk q8_0): Halves KV VRAM, doubling concurrency capacity.
    - Prompt Prefix Caching (--cache-prompt): Slashes multi-turn TTFT by 70-90%.

[x] STAGE 09: Kubernetes Orchestration, KEDA Autoscaling & Scaling Lag
    - The GPU Saturation Illusion: Never autoscale on GPU utilization; use Queue Depth.
    - The 51.7-second Scaling Lag: Reactive autoscaling cannot protect TTFT SLOs alone.
    - preStop drain hooks (/health/drain + sleep 15) eliminate 502 Bad Gateway errors.

[x] STAGE 10: Capacity Planning, Sizing Bounds & Token Economics
    - The Universal Capacity Law: Concurrency Limit = min(N_mem, N_SLO).
    - Never size by VRAM alone; latency knee bounds real production throughput.
    - Self-hosted economics: $1.01 / 1M tokens on L4 Spot vs $2.50+ on closed APIs.
========================================================================================
```

---

## Stage-by-Stage Rapid Revision Guide

### Stage 01: Baseline Inference Profiling & Silicon Physics
* **Core Distinction:** Prefill is compute-bound (GEMM, TFLOPS, governs TTFT); Decode is memory-bandwidth bound (GEMV, GB/s, governs TPOT).
* **The Golden Law:** To generate 1 token, the GPU must read the entire model weights from VRAM into registers.
* **Equation:** $\text{Achieved Bandwidth (GB/s)} = \text{Model Size (GB)} \times \text{Decode Speed (tok/s)}$.
* **Fleet Insight:** Over 94% of remote API wait time is internet transport (140 ms WAN vs 7.7 ms compute), but SSE streaming renders distance invisible after token 1.

### Stage 02: Cold-Start Behavior & Virtual Memory Paging
* **The `mmap` Illusion:** Memory-mapping only helps CPU-only inference. On GPUs, `cudaMemcpy` (PCIe) and Metal committed buffers force eager page loading upfront anyway.
* **The Trap:** With `mmap=ON` on unified memory, the first user triggers hundreds of virtual memory page faults, blowing up TTFT to 520 ms.
* **The Production SRE Rule:** Always use eager allocation (`--no-mmap` or `--load-mode none`). It boots 850 ms faster overall and guarantees a ~35 ms first-token latency.
* **UNIX Signals:** Always wrap your container entrypoint with `exec /app/llama-server` so the server becomes PID 1, properly receiving orchestrator `SIGTERM` (143) instead of being killed by `SIGKILL` (137).

### Stage 03: Multi-Slot Concurrency, Queuing & Head-of-Line Blocking
* **Context Sizing Law:** `llama-server` divides `-c` evenly across parallel slots. You must size `-c = parallel * per_slot_context` (e.g. `--parallel 8 -c 16384` for 2k tokens per user).
* **Prefill-Decode Interference:** A new client submitting a large prompt (e.g. 1,500 tokens) hogs compute cores, causing a 15x ITL stutter spike on active background streaming users.
* **The SRE Fix:** Enable Chunked Prefill (`--ubatch 128` or `-ub 64`) to chop the incoming prompt into micro-batches, interleaving ongoing decode steps.

### Stage 04: Memory Pressure, KV Cache Sizing & OOM Prevention
* **Host RAM OOM (137) vs GPU VRAM OOM:** Exit code 137 is sent by the Linux kernel `oom-killer` when Host RAM or Docker `cgroup` limits are breached. GPUs throw in-process runtime exceptions (`cudaErrorMemoryAllocation`).
* **Static vs Dynamic Allocation:** `llama-server` pre-allocates KV cache at boot (cannot OOM mid-request). `vLLM` allocates dynamically via PagedAttention and swaps blocks to CPU RAM during saturation, triggering Exit Code 137 if host RAM is unbudgeted.
* **KV Cache Formula:**
  $$\text{KV Bytes per Token} = 2 \times L \times H_{\text{kv}} \times D_{\text{head}} \times B_{\text{element}}$$

### Stage 05: Edge Streaming Reverse Proxy & Backpressure
* **The Token Vomit Bug:** Default Nginx buffers responses (`proxy_buffering on;`) waiting for 4KB, destroying the smooth typewriter UX and vomiting 80 tokens at once.
* **The Production Directives:**
  - `proxy_buffering off;` (flush every token down the TCP wire immediately).
  - `proxy_set_header X-Accel-Buffering no;` (disable CDN edge buffering).
  - `proxy_read_timeout 300s;` (prevent drops during deep reasoning).
  - `proxy_ignore_client_abort off;` (Backpressure: drops backend GPU generation if client disconnects).

### Stage 06: Chaos Engineering & The Gray Failure Paradox
* **Deterministic MTTR:** Process recovery time under hard crash (`SIGKILL`) is deterministic (<2.7s on L4 and M3 Ultra, $\sigma < 30\text{ ms}$).
* **The Gray Failure Trap:** When a GPU kernel deadlocks or freezes, the **Linux OS Kernel still completes the TCP 3-way handshake in kernel space**. A Layer 4 TCP socket check reports the dead pod as healthy!
* **The SRE Rule:** Always configure **Layer 7 HTTP `/health` probes** with a 1-second timeout.

### Stage 07: Observability & The 5 Golden Signals
* **Why Total HTTP Duration Fails:** A 12-second HTTP duration cannot tell whether the user received 1,000 tokens smoothly or waited 11 seconds for 10 tokens.
* **The 5 Golden Signals:**
  1. Prefill Latency: `llm_ttft_seconds` (Histogram, P95 $\le$ 250 ms)
  2. Decode Latency: `llm_tpot_seconds` (Histogram, P99 $\le$ 40 ms)
  3. KV Cache Saturation: `kv_cache_usage_ratio` (Gauge, < 85%)
  4. Queue Depth: `requests_deferred` (Gauge, primary autoscaling signal)
  5. Hardware Saturation: `gpu_utilization_ratio` (Gauge)
* **Tracing Anti-Pattern:** Never emit 1 OpenTelemetry span per token (cardinality explosion). Emit 1 trace per request with child spans: `queue_wait`, `prefill`, and `decode`.

### Stage 08: Serving Engine Optimizations (Ablation Matrix)
* **FlashAttention (`-fa 1`):** Tiled SRAM online softmax avoids off-chip HBM writes; collapses prefill by 3.35x on L4 and boosts decode by +31.9% on M3 Ultra.
* **KV Cache Quantization (`-ctk q8_0`):** Compresses KV tensors to 8-bit, cutting KV memory by 50% with zero perceptible perplexity loss ($\Delta\text{PPL} < 0.05$), doubling user concurrency.
* **Prompt Prefix Caching (`--cache-prompt`):** Reuses existing KV activations for identical system prompt prefixes, cutting multi-turn TTFT by 70–90%.

### Stage 09: Kubernetes Orchestration & KEDA Autoscaling
* **GPU Saturation Illusion:** A GPU running a 2-user memory-bound decode loop reads 90% SM utilization. Never autoscale on GPU utilization; autoscale on **Queue Depth** (`requests_deferred`).
* **The 51.7-Second Scaling Lag:** Prometheus scrape (15s) + KEDA polling (15s) + HPA sync (15s) + Scheduler/Init (6.7s) = 51.7s total delay. Purely reactive autoscaling fails TTFT SLOs during sudden spikes without buffer headroom.
* **PreStop Drain Hook:**
  ```yaml
  lifecycle:
    preStop:
      exec:
        command: ["/bin/sh", "-c", "curl -s http://127.0.0.1:8000/health/drain && sleep 15"]
  ```
  Prevents 502 Bad Gateway errors by allowing active streaming responses 15 seconds to finish before `SIGTERM` terminates the pod.

### Stage 10: Capacity Planning & Token Economics
* **The Universal Capacity Law:**
  $$\text{Production Concurrency Limit} = \min\left(N_{\text{mem}}, N_{\text{SLO}}\right)$$
  For models $\le 7\mathrm{B}$, memory fits hundreds of slots ($N_{\text{mem}} = 414$ on L4), but the Latency Knee collapses at $N_{\text{SLO}} = 8$ slots. Never size by VRAM alone.
* **Token Unit Economics:** Self-hosted NVIDIA L4 Spot costs **$1.01 per 1M tokens** at 50% daytime load ($2.5\times$ cheaper than closed APIs at $2.50+/1M$). Owned Apple Silicon delivers tokens at **$0.67 per 1M tokens**.
