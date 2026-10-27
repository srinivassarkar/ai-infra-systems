# Stage 02: Cold-Start Behavior & Virtual Memory Paging (`mmap`)

## 1. Objective & Hypothesis

When an LLM inference service boots or scales out in response to incoming traffic, it incurs a **cold-start latency penalty**. This stage deconstructs the operating system, virtual memory, and hardware interconnect mechanisms that govern cold-start behavior.

### Architectural Hypothesis
1. **`--mmap` (Default Lazy Virtual Paging):** Calls `mmap(2)` to map the GGUF file descriptor directly into the process's Virtual Memory Area (VMA). Physical pages are **not** loaded eagerly into DRAM; they are faulted into memory on first touch.
   * *Prediction:* Server `Time-to-Ready` ($t_{\mathrm{ready}}$) will be near-instantaneous ($< 150\ \mathrm{ms}$), but the **first inference request will pay the page-fault penalty**, yielding inflated $\mathrm{TTFT}$.
2. **`--no-mmap` (Eager Physical DRAM Allocation):** Allocates heap memory via `malloc()` and reads weights sequentially from storage into physical RAM via `read(2)` syscalls prior to signaling health.
   * *Prediction:* Server `Time-to-Ready` will be substantially slower (proportional to disk read throughput), but the **first inference request will execute with zero page-fault latency**, matching warm steady-state execution.
3. **The Interconnect Divergence (PCIe vs Unified Memory):**
   * **Node 1 (Linux CUDA):** Weights follow a 4-hop path: $\text{Disk} \to \text{OS Page Cache} \to \text{Host DRAM} \to \text{PCIe 3.0 Bus} \to \text{GPU VRAM}$.
   * **Node 2 (Apple Metal UMA):** Weights follow a 2-hop zero-copy path: $\text{Disk} \to \text{OS Page Cache / Unified DRAM} \to \text{Metal GPU Buffer Pointer}$.

---

## 2. The 4-Tier Cold-Start Taxonomy

To avoid methodological contamination from the OS filesystem buffer cache, we evaluate four distinct operational states:

| State | Lifecycle Phase | OS Page Cache State | Physical Hardware Reality |
| :--- | :--- | :--- | :--- |
| **C0 (True Cold)** | Daemon not running | **Purged** (`drop_caches` / `purge`) | True cold I/O from storage media (NVMe/SSD). |
| **C1 (Warm Cache Cold)** | Daemon not running | **Populated in RAM** | Process boot without disk penalty; memory copy/map only. |
| **C2 (Unwarmed Engine)** | Daemon running, `/health` 200 | Populated in RAM | First request ever with `--no-warmup`; uninitialized CUDA/Metal kernels. |
| **C3 (Fully Warm)** | Daemon running, serving traffic | Populated in RAM | Steady-state inference with warm kernels and pre-faulted memory. |

---

## 3. Metrics Deconstruction

Cold start must not be conflated with simple $\mathrm{TTFT}$. We isolate two independent latency phases:

```
[ Process Launch ] ───────────────▶ [ /health Returns 200 ] ───────────────▶ [ First Output Token ]
        │                                    │                                      │
        └────────── t_ready ─────────────────┴────────── first_token_TTFT ──────────┘
        │                                                                           │
        └────────────────── t_first_useful_token (Total Cold Start) ────────────────┘
```

1. **Time-to-Ready ($t_{\mathrm{ready}}$):**
   Time elapsed from process spawn until the `/health` endpoint returns HTTP 200 (returns HTTP 503 while initializing).
2. **First-Request TTFT ($\mathrm{TTFT}_{\mathrm{first}}$):**
   Client-measured Time-To-First-Token on the virgin inference request immediately following a healthy probe. Must be executed with `"cache_prompt": false` to prevent prefix-cache masking.
3. **Total Cold-Start Latency ($t_{\mathrm{useful}}$):**
   $$t_{\mathrm{useful}} = t_{\mathrm{ready}} + \mathrm{TTFT}_{\mathrm{first}}$$

---

## 4. Cache Dropping Protocols (Enforcing C0)

Prior to executing a C0 true-cold trial, the operating system page cache must be explicitly purged:

### On Linux (Node 1 - GTX 1050 Ti):
```bash
# Sync dirty pages to disk and purge pagecache, dentries, and inodes
sync && echo 3 | sudo tee /proc/sys/vm/drop_caches
```

### On macOS (Node 2 - Mac Studio M3 Ultra):
```bash
# Force purge of inactive disk and OS memory cache
sudo purge
```

---

## 5. Automated Execution Harness

The experiment is orchestrated via `benchmark_cold_start.py`, which manages server lifecycles, monitors `/health` transitions, executes virgin requests without prompt caching, and records byte allocations from startup logs.

```bash
# Example execution on Node 1 (Linux):
python stages/02-cold-start-behavior/benchmark_cold_start.py \
  --server-bin ~/llama-bin/llama.cpp/build/bin/llama-server \
  --model ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  --ngl 99 \
  --ctx-size 2048 \
  --port 8080
```

