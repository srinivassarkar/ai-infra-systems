# Stage 02: Cold-Start Behavior & Virtual Memory Paging

## The Core Question
When an LLM inference service scales from zero under incoming user traffic, how long does it take before the first word is generated?
More importantly: **Does memory-mapping (`mmap`) actually eliminate cold-start latency, or is it an architectural illusion for GPU serving?**

---

## The Central Architectural Finding

> **"For GPU-offloaded inference (`-ngl 99`), `mmap=ON` vs `mmap=OFF` shows no cold-start benefit on either architecture. Discrete GPU (NVIDIA L4 PCIe) forces eager population via `cudaMemcpy` page-faulting the mmap region at startup. Unified GPU (Apple Metal) forces eager population via `MTLBuffer` allocation faulting the mmap region into physical DRAM. Measured Total Cold Start on Apple Silicon: `mmap=ON` 3,114.39 ms vs `mmap=OFF` 2,264.04 ms. Eager `malloc` is 850 ms faster because it eliminates first-request virtual page faulting. True lazy paging benefits exist exclusively for CPU-only inference (`-ngl 0`)."**

---

## The 3 Physical Hardware Realities Explained

```
Scenario 1: Discrete PCIe GPU (NVIDIA L4 - -ngl 99)
[ GGUF File on Disk ] ──▶ [ Host Virtual Memory ] ──▶ [ PCIe 4.0 Bus DMA Copy ] ──▶ [ GPU GDDR6 VRAM ]
* The GPU cannot execute code from CPU RAM or Linux virtual memory.
* To offload layers, the CUDA driver MUST issue cudaMemcpy to push all weights into VRAM at startup.
* This copy forces the OS to read all pages immediately, completely defeating lazy virtual memory paging!
* Result: t_ready is identical (~1050 ms) whether mmap is ON or OFF.

Scenario 2: Unified Memory GPU (Apple Mac Studio M3 Ultra - -ngl 99)
[ GGUF File on SSD ] ──▶ [ Unified LPDDR5X DRAM ] ──▶ [ MTLBuffer Allocation ] ──▶ [ Metal GPU Cores ]
* There is no PCIe bus, but Apple Metal requires a committed MTLBuffer pointer for GPU tensor math.
* When Metal creates the buffer, the kernel commits and faults the pages into physical DRAM on boot.
* Eager malloc + sequential read (2226 ms) is 368 ms FASTER than mmap (2594 ms) because contiguous
  allocation avoids piecemeal virtual memory fault overhead.

Scenario 3: CPU Virtual Paging (Host CPU - -ngl 0)
[ GGUF File on Disk ] ──▶ [ Virtual Memory Pointer (mmap) ] ──▶ [ CPU Execution Cores ]
* There is no GPU buffer and no PCIe bus. The CPU executes directly from virtual address space.
* With mmap ON: The server boots in 1322 ms without reading all weights upfront (fast scale-to-ready).
* With mmap OFF: The server boots in 1701 ms (+379 ms slower) because it sequentially reads all bytes into RAM.
* Here—and only here—does the classical OS lazy paging trade-off actually exist.
```

---

## Quickstart: Step-by-Step Reproduction Guide

### On macOS (Apple Mac Studio M3 Ultra):
```bash
cd ~/ai-infra-systems/stages/02-cold-start-behavior

# 1. Apple Silicon Lazy mmap
python3 benchmark_cold_start.py \
  --server-bin /opt/homebrew/bin/llama-server \
  --model /Users/blustudio/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  --port 8181

# 2. Apple Silicon Eager Heap Read (Contiguous Malloc)
python3 benchmark_cold_start.py \
  --server-bin /opt/homebrew/bin/llama-server \
  --model /Users/blustudio/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  --port 8181 \
  --no-mmap
```

### On Linux (NVIDIA L4 Cloud Pod):
```bash
# 1. Discrete GPU (Baseline mmap)
python3 benchmark_cold_start.py \
  --server-bin /root/llama-bin/llama-b11963/llama-server \
  --model /root/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf

# 2. Discrete GPU (Eager heap allocation)
python3 benchmark_cold_start.py \
  --server-bin /root/llama-bin/llama-b11963/llama-server \
  --model /root/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf \
  --no-mmap
```

---

## 5. Measured Physical Fleet Results

Here are the real measured numbers from the Apple Mac Studio M3 Ultra and NVIDIA L4:

| Metric | M3 Ultra (Lazy mmap) | M3 Ultra (Eager Malloc) | NVIDIA L4 (mmap) | NVIDIA L4 (Eager Malloc) |
| :--- | :---: | :---: | :---: | :---: |
| **Time-to-Ready ($t_{\mathrm{ready}}$)** | 2,594.54 ms | **2,226.30 ms** (-368 ms) | 1,096.08 ms | **1,040.64 ms** (-55 ms) |
| **First-Request TTFT** | 519.85 ms | **37.74 ms** (13.8x faster!) | 34.47 ms | 34.61 ms |
| **Total Cold Start ($t_{\mathrm{useful}}$)** | 3,114.39 ms | **2,264.04 ms** (-850 ms!) | 1,130.55 ms | **1,075.25 ms** |
| **Steady-State Warm TTFT** | 33.44 ms | 34.06 ms | 16.32 ms | 15.48 ms |
| **Decode Speed (tok/s)** | 124.2 tok/s | 123.5 tok/s | 180.5 tok/s | 178.6 tok/s |

### What These Numbers Mean in Simple Terms:
1. **The mmap Illusion on GPUs:** In textbooks, programmers are taught that `mmap` is faster because it delays reading files until needed. But on GPUs (both NVIDIA CUDA and Apple Metal), the GPU needs memory allocated and committed upfront.
2. **Why Eager Malloc Wins by 850 ms on Mac:** With `mmap=OFF`, the Mac reads the 1 GB model file sequentially in one clean burst from its 5,000 MB/s SSD. With `mmap=ON`, the first user request encounters hundreds of virtual memory page faults while computing, stalling the first word by **519.85 ms**!
3. **The SRE Production Rule:** Always configure GPU containers with eager loading (`--load-mode none`). It boots faster and eliminates first-request latency spikes for paying clients.

For the full architectural deep dive and CPU ablation results, read the [Stage 02 Full Report](REPORT.md).
