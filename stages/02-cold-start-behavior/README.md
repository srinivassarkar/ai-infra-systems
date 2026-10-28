# Stage 02: Cold-Start Behavior & Virtual Memory Paging

## The Core Question
When an LLM inference service scales out from zero under incoming user traffic, how long does it take before the first word is generated?
More importantly: **Does memory-mapping (`mmap`) actually eliminate cold-start latency, or is it an architectural illusion on GPU hardware?**

---

## The Central Architectural Finding

> **"On discrete PCIe GPUs with `-ngl 99`, `mmap` vs `no-mmap` is a false dichotomy for cold-start latency. The CUDA device memory copy forces eager population at process launch. The real `mmap` virtual memory trade-off exists exclusively for CPU inference or Apple Silicon Unified Memory."**

---

## The Physical Mechanisms Explained

When an inference daemon loads a ~1 GB model (`Qwen2.5-Coder-1.5B`), it behaves fundamentally differently depending on whether it targets a discrete GPU or the CPU:

```
Scenario A: Discrete PCIe GPU (-ngl 99)
[ GGUF File on Disk ] ──▶ [ Host OS Page Cache ] ──▶ [ PCIe 4.0 Bus DMA Copy ] ──▶ [ GPU GDDR6 VRAM ]
* The GPU CANNOT execute from host CPU RAM.
* To offload layers, the CUDA driver MUST issue cudaMemcpy to push all weights into VRAM at startup.
* This copy forces the OS to read all pages immediately, completely defeating lazy virtual memory paging!
* Result: t_ready is identical (~1050 ms) regardless of whether mmap is ON or OFF.

Scenario B: CPU / Shared Memory (-ngl 0)
[ GGUF File on Disk ] ──▶ [ Virtual Memory Pointer (mmap) ] ──▶ [ CPU Execution Cores ]
* There is no PCIe bus transfer.
* With mmap ON: The server boots in 1322 ms without reading all bytes into heap memory upfront.
* With mmap OFF: The server boots in 1701 ms (+379 ms slower) because it sequentially reads all bytes into RAM.
```

---

## Quickstart: Step-by-Step Reproduction Guide

Follow these exact steps to reproduce the cold-start benchmarks:

### 1. Ensure the Model Weights and Binary Are Present
```bash
# Model path:
~/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf

# Standalone llama-server binary:
~/llama-bin/llama-b11963/llama-server
```

### 2. Run the Benchmark Across the 5 Test Permutations

```bash
cd ai-infra-systems/stages/02-cold-start-behavior

# Run 1: Discrete GPU with default lazy mmap & warmup (Baseline)
python3 benchmark_cold_start.py \
  --server-bin ~/llama-bin/llama-b11963/llama-server \
  --model ~/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf

# Run 2: Discrete GPU with eager heap allocation (--no-mmap)
python3 benchmark_cold_start.py \
  --server-bin ~/llama-bin/llama-b11963/llama-server \
  --model ~/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf \
  --no-mmap

# Run 3: Discrete GPU with uninitialized kernels (--no-warmup)
python3 benchmark_cold_start.py \
  --server-bin ~/llama-bin/llama-b11963/llama-server \
  --model ~/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf \
  --no-warmup

# Run 4: CPU Paging with lazy mmap (--ngl 0 --no-warmup)
python3 benchmark_cold_start.py \
  --server-bin ~/llama-bin/llama-b11963/llama-server \
  --model ~/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf \
  --no-warmup \
  --ngl 0

# Run 5: CPU Paging with eager DRAM allocation (--no-mmap --no-warmup --ngl 0)
python3 benchmark_cold_start.py \
  --server-bin ~/llama-bin/llama-b11963/llama-server \
  --model ~/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf \
  --no-mmap \
  --no-warmup \
  --ngl 0
```

---

## The Metrics Explained Simply

```text
1. Time-to-Ready (t_ready):
   Elapsed time from process spawn until the /health endpoint returns HTTP 200.
   Measures: Disk I/O + VRAM weight allocation + PCIe DMA transfer time.

2. First-Request TTFT:
   Time-To-First-Token on the virgin inference request immediately following startup.
   Measures: Prompt prefill compute + CUDA kernel compilation / memory graph initialization.

3. Time-to-First-Useful-Token (Total Cold Start):
   Total latency experienced by an end-user during a scale-from-zero event.
   Formula: Total Cold Start = t_ready + First_Request_TTFT

4. Steady-State Warm TTFT:
   Time-To-First-Token on subsequent requests with pre-compiled GPU kernels.
```

---

## Architectural Decision Record (ADR): Deprecation of `--no-mmap`

During empirical testing against `llama.cpp` build `b11963`, we identified that the legacy boolean flag `--no-mmap` has been deprecated and replaced with an explicit multi-mode loading parameter:

```bash
-lm, --load-mode MODE
  auto: mmap (default)
  none: eager heap allocation (replaces legacy --no-mmap)
  dio:  Direct I/O (bypasses OS filesystem page cache completely)
  mlock: locks pages in RAM to prevent OS swapping under memory pressure
```

Our profiler (`benchmark_cold_start.py`) detects and supports both modern `--load-mode` and legacy syntax automatically.
