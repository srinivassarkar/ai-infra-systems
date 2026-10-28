# Stage 02: Cold-Start Behavior & Virtual Memory Paging Report

## 1. Executive Summary & The Core Architectural Law

We evaluated the cold-start lifecycle of large language model serving using `Qwen2.5-Coder-1.5B` across both discrete cloud GPU silicon (NVIDIA L4, 24 GB GDDR6) and CPU execution regimes.

### The Central Finding:
> **"On discrete PCIe GPUs with `-ngl 99`, `mmap` vs `no-mmap` is a false dichotomy for cold-start latency. The CUDA device memory copy forces eager population at process launch. The real `mmap` virtual memory trade-off exists exclusively for CPU inference or Apple Silicon Unified Memory."**

---

## 2. Empirical Telemetry Matrix (NVIDIA L4 - 24 GB GDDR6)

Here is the complete telemetry captured directly on the enterprise Ada Lovelace cloud node (Release b11963, Driver 580.126.20, CUDA 13.0, Ubuntu 24.04 LTS):

| Run | Silicon Target | Loading Levers | Time-to-Ready ($t_{\mathrm{ready}}$) | First-Request TTFT | Total Cold Start ($t_{\mathrm{useful}}$) | Steady Warm TTFT | Steady Warm TPOT |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1** | NVIDIA L4 (GPU) | `mmap=ON, warmup=ON` (`-ngl 99`) | **1096.08 ms** | **34.47 ms** | **1130.55 ms** | 16.32 ms | 5.54 ms |
| **2** | NVIDIA L4 (GPU) | `mmap=OFF, warmup=ON` (`--no-mmap -ngl 99`)| **1040.64 ms** | **34.61 ms** | **1075.25 ms** | 15.48 ms | 5.60 ms |
| **3** | NVIDIA L4 (GPU) | `mmap=ON, warmup=OFF` (`--no-warmup -ngl 99`)| **1072.54 ms** | **35.00 ms** | **1107.54 ms** | 17.12 ms | 5.68 ms |
| **4** | Host CPU | `mmap=ON, warmup=OFF` (`--no-warmup -ngl 0`) | **1322.32 ms** | **172.75 ms** | **1495.07 ms** | 627.16 ms | 17.48 ms |
| **5** | Host CPU | `mmap=OFF, warmup=OFF` (`--no-mmap -ngl 0`) | **1701.47 ms** | **136.49 ms** | **1837.95 ms** | 85.34 ms | 53.27 ms |

---

## 3. Two Critical Systems Discoveries

### Discovery 1: Why Discrete GPUs Break the Promise of `mmap`
In standard operating system theory, `mmap(2)` provides near-instantaneous process startup by mapping virtual memory pointers and lazily loading physical pages on demand when touched.

However, on a discrete GPU with `-ngl 99`:
1. The GPU executes code exclusively from its dedicated VRAM (GDDR6).
2. To offload the model, the CUDA runtime must execute `cudaMemcpy` to push all 1.1 GB of weights across the physical PCIe 4.0 bus into GPU memory.
3. **To transmit those bytes over PCIe, the host OS kernel must read every single byte immediately.**
4. Therefore, the PCIe transfer completely forces eager loading during startup, whether you request `mmap` or not. As evidenced in Runs 1, 2, and 3, **$t_{\mathrm{ready}}$ is identical (~1050 ms) across all GPU configurations**.

### Discovery 2: The Real Virtual Memory Trade-off on CPU (`-ngl 0`)
When layers are kept on the CPU (`-ngl 0`), the execution units access host virtual memory directly, unmasking the true OS paging trade-off:
* **Lazy Paging (`mmap=ON`):** $t_{\mathrm{ready}}$ was **1322.32 ms**. The server boots **379.15 ms faster** because it maps virtual address space (`VMA`) without waiting to pull all weights into heap memory upfront.
* **Eager Allocation (`mmap=OFF`):** $t_{\mathrm{ready}}$ ballooned to **1701.47 ms** (+28.7% boot delay). The process sequentially issues `read(2)` syscalls to fill physical RAM before signaling HTTP ready.
* **The First-Touch Penalty:** While lazy `mmap` booted 379 ms faster, its First-Request TTFT paid a page-fault penalty (**172.75 ms** vs **136.49 ms** on eager read).

### Discovery 3: The "First-Request Kernel Tax" ($2.1\times$ Latency Spike on GPU)
Across all discrete GPU runs, observe the gap between Request 1 and Request 2:
```text
First-Request TTFT:       34.47 ms
Steady-State Warm TTFT:   16.32 ms
Latency Spike:             2.11x
```

Even after the weights are resident in VRAM and the server returns HTTP 200, the first request experiences more than double the latency of subsequent requests. 

**Root Cause:** During the virgin forward pass, the CUDA driver must:
1. Initialize cuBLAS library execution context handles.
2. Allocate internal execution scratchpad buffers.
3. JIT-compile and validate dynamic CUDA kernel launch parameters.

**Production Takeaway for Autoscaling (KEDA / Serverless):**
If a Kubernetes horizontal autoscaler routes customer traffic to an LLM container the millisecond its HTTP health probe passes, the first customer will always receive a degraded experience unless an internal dummy warmup request is executed during the container's readiness probe grace period.

---

## 4. Architectural Interconnect Comparison: PCIe vs. Apple Unified Memory

| Architecture | Physical Interconnect | Memory Path | Does `mmap` allow lazy inference paging? |
| :--- | :--- | :--- | :--- |
| **Discrete GPU (NVIDIA L4 / 1050 Ti)** | PCIe Bus (3.0 / 4.0) | `Storage -> Page Cache -> Host DRAM -> PCIe Bus -> VRAM` | **No.** PCIe DMA copy forces eager page loading at boot. |
| **Host CPU** | System Memory Bus | `Storage -> Page Cache -> Virtual Memory -> CPU Registers` | **Yes.** Paging occurs on demand as CPU cores access pointers. |
| **Apple Silicon (Mac Studio M3 Ultra)** | Unified Memory Architecture (UMA)| `Storage -> Unified LPDDR5X DRAM -> Metal GPU Pointer` | **Yes.** Zero-copy architecture allows Metal to read directly from mapped unified DRAM. |

---

## 5. How to Replicate These Trials

To replicate these five permutations on any node, execute the commands detailed in the [Stage 02 README](README.md):

```bash
# Discrete GPU:
python3 benchmark_cold_start.py --server-bin <PATH_TO_SERVER> --model <PATH_TO_MODEL>

# CPU Paging Ablation:
python3 benchmark_cold_start.py --server-bin <PATH_TO_SERVER> --model <PATH_TO_MODEL> --no-warmup --ngl 0
python3 benchmark_cold_start.py --server-bin <PATH_TO_SERVER> --model <PATH_TO_MODEL> --no-mmap --no-warmup --ngl 0
```
