# Stage 02: Cold-Start Behavior & Virtual Memory Paging

## The Core Question
When an LLM inference service scales from zero under incoming user traffic, how long does it take before the first word is generated?
More importantly: **Does memory-mapping (`mmap`) actually eliminate cold-start latency, or is it an architectural illusion for GPU serving?**

---

## The Central Architectural Finding

> **"For GPU-offloaded inference (`-ngl 99`), `mmap=ON` vs `mmap=OFF` shows no cold-start benefit on either architecture. Discrete GPU (NVIDIA L4 PCIe) forces eager population via `cudaMemcpy` page-faulting the mmap region at startup. Unified GPU (Apple Metal) forces eager population via `MTLBuffer` allocation faulting the mmap region into physical DRAM. Measured Total Cold Start on Apple Silicon: `mmap=ON` 2407 ms vs `mmap=OFF` 2305 ms. Eager `malloc` is 102 ms faster due to contiguous allocation. True lazy paging benefits exist exclusively for CPU-only inference (`-ngl 0`)."**

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
* Eager malloc + sequential read (2240 ms) is actually 100 ms FASTER than mmap (2340 ms) because contiguous
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

### On Linux (NVIDIA L4 Cloud Pod):
```bash
cd ai-infra-systems/stages/02-cold-start-behavior

# 1. Discrete GPU (Baseline mmap)
python3 benchmark_cold_start.py \
  --server-bin /workspace/ai-infra/llama-b11963/llama-server \
  --model /workspace/ai-infra/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf

# 2. Discrete GPU (Eager heap allocation)
python3 benchmark_cold_start.py \
  --server-bin /workspace/ai-infra/llama-b11963/llama-server \
  --model /workspace/ai-infra/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf \
  --no-mmap

# 3. CPU Paging Ablation (Shows the real mmap trade-off)
python3 benchmark_cold_start.py \
  --server-bin /workspace/ai-infra/llama-b11963/llama-server \
  --model /workspace/ai-infra/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf \
  --no-warmup --ngl 0

python3 benchmark_cold_start.py \
  --server-bin /workspace/ai-infra/llama-b11963/llama-server \
  --model /workspace/ai-infra/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf \
  --no-mmap --no-warmup --ngl 0
```

### On macOS (Apple Mac Studio M3 Ultra):
Purge the macOS filesystem cache before each trial to enforce a true cold disk read:
```bash
cd ~/ai-infra-systems/stages/02-cold-start-behavior

# 1. Apple Silicon Lazy mmap (Zero-Copy UMA)
sudo purge; python3 benchmark_cold_start.py \
  --server-bin ~/llama-bin/llama-b11901/llama-server \
  --model ~/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf \
  --no-warmup

# 2. Apple Silicon Eager Heap Read (Contiguous Malloc)
sudo purge; python3 benchmark_cold_start.py \
  --server-bin ~/llama-bin/llama-b11901/llama-server \
  --model ~/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf \
  --no-mmap --no-warmup
```

---

## The Core Metrics Explained Simply

```text
1. Time-to-Ready (t_ready):
   How long the server takes from process start until the HTTP /health endpoint returns 200.
   Measures: Disk I/O + VRAM allocation + PCIe DMA transfer or MTLBuffer commit.

2. First-Request TTFT:
   Time-To-First-Token on the virgin inference request immediately following startup.
   Measures: Prompt prefill compute + CUDA/Metal dynamic kernel graph initialization.

3. Total Cold Start (Time-to-First-Useful-Token):
   Total latency experienced by an end-user during a scale-from-zero event.
   Formula: Total Cold Start = t_ready + First_Request_TTFT

4. Steady-State Warm TTFT:
   Time-To-First-Token on subsequent requests with warm, pre-compiled GPU kernels.
```
