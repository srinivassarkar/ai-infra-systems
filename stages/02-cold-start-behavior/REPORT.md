# Stage 02: Cold-Start Behavior & Virtual Memory Paging (`mmap`) Report

## 1. System Context & Hardware Fleet

* **Model Identifier:** `Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf` (~0.98 GB on disk / ~1.1 GB in memory)
* **Serving Binary:** Standalone `llama-server` (b11901)
* **Target Fleet Nodes:**
  * **Node 1 (Linux CUDA):** Custom Linux Box, NVIDIA GeForce GTX 1050 Ti (4 GB GDDR5, PCIe 3.0 x16, Ubuntu 20.04)
  * **Node 2 (Apple Metal):** Mac Studio M3 Ultra (28 Cores, 256 GB Unified LPDDR5X, macOS Sequoia)

---

## 2. The 4-Tier Cold-Start Experiment Matrix

To eliminate OS page cache contamination, trials are categorized by hardware and memory lifecycle state:

1. **Condition A (C0 + mmap):** OS cache purged (`drop_caches` / `purge`), server started with default `--mmap`.
2. **Condition B (C0 + no-mmap):** OS cache purged (`drop_caches` / `purge`), server started with `--no-mmap` (eager heap allocation).
3. **Condition C (C1 + mmap):** OS cache warm (file resident in system pagecache), server started with `--mmap`.
4. **Condition D (C2 + no-warmup):** Server started with `--no-warmup`; first request triggers initial kernel/shader compilation.
5. **Condition E (C3 Steady State):** Server fully warm, active request serving.

---

## 3. Observed Telemetry Data

### Node 1: Custom Linux Box (NVIDIA GTX 1050 Ti - PCIe 3.0 x16)

| Condition | Description | Time-to-Ready ($t_{\mathrm{ready}}$) | First TTFT | Total Cold Start ($t_{\mathrm{useful}}$) | Steady Warm TTFT | Steady Warm TPOT |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A: C0 + mmap** | Cold disk + Lazy paging | - | - | - | - | - |
| **B: C0 + no-mmap**| Cold disk + Eager DRAM read | - | - | - | - | - |
| **C: C1 + mmap** | Warm pagecache + mmap | - | - | - | - | - |
| **D: C2 no-warmup**| Uninitialized CUDA kernels | - | - | - | - | - |
| **E: C3 Warm** | Steady-state baseline | N/A | N/A | N/A | - | - |

### Node 2: Apple Mac Studio M3 Ultra (256 GB Unified Memory)

| Condition | Description | Time-to-Ready ($t_{\mathrm{ready}}$) | First TTFT | Total Cold Start ($t_{\mathrm{useful}}$) | Steady Warm TTFT | Steady Warm TPOT |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A: C0 + mmap** | Cold disk + Lazy paging | - | - | - | - | - |
| **B: C0 + no-mmap**| Cold disk + Eager DRAM read | - | - | - | - | - |
| **C: C1 + mmap** | Warm pagecache + mmap | - | - | - | - | - |
| **D: C2 no-warmup**| Uninitialized Metal shaders | - | - | - | - | - |
| **E: C3 Warm** | Steady-state baseline | N/A | N/A | N/A | - | - |

---

## 4. Operating System & Virtual Memory Architectural Findings

### 1. Where the Latency is Paid: `--mmap` vs `--no-mmap`
* **Lazy Virtual Paging (`--mmap`):**
  * `mmap(2)` creates virtual memory address bindings without reading data into physical DRAM.
  * Time-to-Ready ($t_{\mathrm{ready}}$) is fast because the OS only creates kernel page table entries (`VMA`).
  * However, during the first forward pass, the GPU kernel touches weight tensors, triggering **major page faults (`majflt`)**. The execution pipeline halts while blocks are fetched from NVMe/SSD into physical memory.
* **Eager Allocation (`--no-mmap`):**
  * The server sequentially issues `read(2)` syscalls for all 1.1 GB of weights at boot time.
  * Time-to-Ready is delayed by physical I/O throughput.
  * Once the `/health` endpoint returns 200, **zero page faults occur during inference**, delivering immediate steady-state TTFT on request 1.

### 2. The Interconnect Memory Path: PCIe Copy vs Unified Zero-Copy
* **Linux CUDA (GTX 1050 Ti):**
  $$\text{Storage (SSD)} \xrightarrow{\text{I/O}} \text{OS Page Cache} \xrightarrow{\text{memcpy}} \text{Host Virtual Memory} \xrightarrow{\text{PCIe 3.0 x16}} \text{GPU VRAM (GDDR5)}$$
  * Weights must traverse the PCIe bus ($\sim 15.75\ \mathrm{GB/s}$ theoretical max), creating an inescapable physical serialization barrier before execution can begin.
* **Apple Silicon (M3 Ultra):**
  $$\text{Storage (SSD)} \xrightarrow{\text{I/O}} \text{Unified LPDDR5X DRAM} \xrightarrow{\text{zero-copy}} \text{Metal GPU Execution Context}$$
  * The Metal framework points directly to the mapped buffer in unified memory without an intermediate host-to-device PCIe DMA transfer.

### 3. Production Recommendation for Cloud Autoscaling (KEDA / Serverless)
* If scaling from 0 under latency-critical SLOs:
  * **Option A:** Use `--no-mmap` with a health-check grace period so requests are never routed to an instance that will suffer page-fault spikes.
  * **Option B:** If using `--mmap`, mandate a startup warmup pass (`--warmup`) so page tables are pre-faulted before the container marks itself `Ready` in Kubernetes.
