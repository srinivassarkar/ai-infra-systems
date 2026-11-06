# The 2-Page Inference Systems Engineering Refresher
## High-Signal Solitary Study Harness & Production Triage Notes

> **Target Role:** Platform / AI Infrastructure Engineer & High-Scale Systems SRE  
> **Source of Truth:** Verified across physical silicon (Apple Mac Studio M3 Ultra, NVIDIA L4, GTX 1050 Ti)

---

# PAGE 1: The 10-Stage Master Arsenal & Physical Formulas

### 1. The 10-Stage Inference Engineering Arsenal

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

### 2. The First-Principles Mathematical Formulary

1. **Achieved Decode Memory Bandwidth:**
   $$\text{Achieved Bandwidth (GB/s)} = \text{Model Size (GB)} \times \text{Decode Speed (tokens/sec)}$$
   *Rule:* If model weights double (7B $\rightarrow$ 14B), bytes streamed per token double, halving decode speed on identical memory bus.

2. **KV Cache Memory Footprint (MHA / GQA):**
   $$\text{KV Bytes per Token} = 2 \times L \times H_{\mathrm{kv}} \times D_{\mathrm{head}} \times B_{\mathrm{element}}$$
   $$\text{Total KV RAM (GiB)} = \frac{\text{KV Bytes per Token} \times C \times N}{1024^3}$$
   *Variables:* Factor of $2$ (separate $K$ and $V$ tensors); $L$ = Layers; $H_{\mathrm{kv}}$ = KV Heads; $D_{\mathrm{head}}$ = Dimension; $B_{\mathrm{element}}$ = Bytes per scalar (FP16 = 2, Q8_0 = 1, FP4 = 0.5); $C$ = Context Length; $N$ = Concurrency.

3. **Context Window Sizing Law (`llama-server`):**
   $$-c = \text{parallel} \times \text{per\_slot\_context}$$
   *Rule:* Launching `-c 2048 --parallel 8` silently allocates only 256 tokens per user!

4. **The Universal Capacity Law:**
   $$\text{Production Concurrency Limit} = \min\left(N_{\mathrm{mem}}, N_{\mathrm{SLO}}\right)$$
   *Rule:* For models $\le 7\mathrm{B}$, memory fits hundreds of slots ($N_{\mathrm{mem}} = 414$ on L4), but the Latency Knee collapses at $N_{\mathrm{SLO}} = 8$ slots. Always bind by $N_{\mathrm{SLO}}$.

5. **Financial Token Unit Economics:**
   $$\text{Cost per 1M Tokens} = \frac{\text{Hourly Hardware Cost}}{\text{Tokens Generated per Hour}} \times 1,000,000$$

---

# PAGE 2: Production Incident Triage & SRE Cheat Codes

### 3. POSIX Signal Arithmetic & The Exit Code Law

When any Linux process is terminated by an unhandled signal, the kernel sets:
$$\text{Exit Code} = 128 + \text{Signal Number}$$

* **Exit Code 137 ($128 + 9 = \text{SIGKILL}$):**
  - **Case 1 (Kernel OOM Killer):** Physical Host RAM or container `resources.limits.memory` was breached. Look for `oom-killer` in `dmesg -T`.
  - **Case 2 (Kubelet Eviction):** Container failed `livenessProbe` or was being drained and refused to terminate within `terminationGracePeriodSeconds` (default 30s) after `SIGTERM`.
  - *Golden Rule:* **Exit Code 137 is NEVER a GPU VRAM error.** GPUs return in-process runtime exceptions (`cudaErrorMemoryAllocation`).
* **Exit Code 143 ($128 + 15 = \text{SIGTERM}$):** Graceful pod termination signal sent by Kubernetes orchestrator during scale-down or deployment rollout.
* **Exit Code 130 ($128 + 2 = \text{SIGINT}$):** Interactive terminal interrupt (`Ctrl+C`).
* **The PID 1 Trap:** Always invoke server processes via `exec /app/llama-server`. Without `exec`, the parent bash wrapper retains PID 1, swallows `SIGTERM` (143), and forces a brutal `SIGKILL` (137) after timeout.

---

### 4. Rapid Host Diagnostics: The 4-Column `vmstat 1 5` Triage

During high load, ignore the other 12 columns. Inspect these four in the first 5 seconds:

```text
procs -----------memory---------- ---swap-- -----io---- -system-- -------cpu-------
  r  b   swpd   free   buff  cache   si   so    bi    bo   in   cs  us sy id wa st gu
  5  0      0 736916 305916 1057944   0    0     0     0 2278 1266 100  0  0  0  0  0
```

1. **`r` (Runnable Queue) vs `nproc`:** If $r > \text{nproc}$, CPU execution cores are saturated by queuing threads.
2. **`us` vs `sy` (User vs Kernel):** `us = 100%` means application-level compute loop (bad code or matrix math). `sy > 40%` indicates kernel thrashing, lock contention, or excessive context switching.
3. **`wa` (I/O Wait):** If `wa > 20%` and `id` is low, CPU is idling because threads are stuck in uninterruptible `D` state waiting on storage (disk/EBS/NFS).
4. **`si` / `so` (Swap In / Swap Out):** Non-zero values mean physical RAM is exhausted and the kernel is thrashing disk swap. Precursor to Exit Code 137.

---

### 5. Production Nginx Streaming Reverse Proxy Blueprint

To prevent token buffering, reading stutter, and abandoned compute waste:

```nginx
location /v1/chat/completions {
    proxy_pass http://inference_backend;

    # 1. Kill the 4KB buffer: Flush every SSE token chunk down TCP immediately
    proxy_buffering off;

    # 2. Prevent caching dynamic token streams on disk
    proxy_cache off;

    # 3. Prevent upstream CDNs (Cloudflare / Fastly) from buffering chunks
    proxy_set_header X-Accel-Buffering no;

    # 4. Prevent proxy timeout on deep reasoning / large context outputs
    proxy_read_timeout 300s;

    # 5. BACKPRESSURE LEVER: Abort GPU computation if user closes client connection
    proxy_ignore_client_abort off;
}
```

---

### 6. Kubernetes Autoscaling, Gray Failures & PreStop Drain

1. **The Gray Failure Paradox:**
   - A GPU compute thread deadlock or frozen CUDA kernel passes Layer 4 TCP socket checks (`tcpSocket: port: 8080`) because the **Linux OS Kernel answers the TCP 3-way handshake in kernel space**.
   - *Fix:* Always configure **Layer 7 HTTP `/health` probes** (`httpGet: path: /health, port: 8080`) with a 1-second timeout.
2. **Autoscaling Leading Metric:**
   - Never autoscale on NVIDIA `gpu_utilization` (a memory-bound decode loop with 2 users reads 95% utilization!).
   - Autoscale on **Queue Depth** via KEDA (`llamacpp:requests_deferred` or vLLM waiting requests).
3. **The 51.7-Second Scaling Lag:**
   - Prometheus scrape (15s) + KEDA polling (15s) + HPA sync (15s) + Scheduler (1.5s) + Containerd (2s) + Model load (1.2s) + Probe (2s) = **51.73s total delay**.
   - Reactive autoscaling alone fails TTFT SLOs during sudden spikes. Always maintain pre-warmed buffer headroom.
4. **Zero-Downtime Scale-Down Hook:**
   ```yaml
   lifecycle:
     preStop:
       exec:
         command: ["/bin/sh", "-c", "curl -s http://127.0.0.1:8000/health/drain && sleep 15"]
   ```
   Marks readiness probe as 503 (draining) so load balancers stop sending new requests, while allowing active in-flight streams 15 seconds to finish generation before `SIGTERM` terminates the pod.
