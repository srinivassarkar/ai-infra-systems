# Stage 01: Baseline Inference Profiling

## The Core Question
Before adding complex optimizations, load balancers, or proxies, we must answer one fundamental question:
**How fast does a single large language model generate text on different physical hardware when only one user is requesting it?**

This is our baseline. Every future optimization (continuous batching, KV-cache quantization, speculative decoding, edge proxies) will be judged against these numbers.

---

## The 3 Hardware Archetypes Evaluated

Rather than testing only one machine, we profile the exact same model (`Qwen2.5-Coder-1.5B-Instruct-Q4_K_M`, ~986 MB) across three fundamentally different hardware and memory architectures:

1. **Node 1: Edge Consumer GPU (NVIDIA GTX 1050 Ti - 4 GB GDDR5)**
   * *Architecture:* Pascal (`sm_61`), 128-bit memory bus, 112 GB/s peak bandwidth, PCIe 3.0 x16.
   * *Purpose:* Represents budget, power-constrained edge hardware.
2. **Node 2: Unified Memory Workstation (Apple Mac Studio M3 Ultra - 256 GB LPDDR5X)**
   * *Architecture:* 80 GPU cores, Unified Memory Architecture (UMA), 800 GB/s peak bandwidth.
   * *Purpose:* Represents high-end unified silicon where CPU, GPU, and RAM share a single zero-copy memory pool.
3. **Node 3: Enterprise Cloud GPU (NVIDIA L4 - 24 GB GDDR6)**
   * *Architecture:* Ada Lovelace (`sm_89`), 192-bit bus, 300 GB/s peak bandwidth, 48 MB L2 cache, PCIe 4.0 x16.
   * *Purpose:* Represents modern cost-effective cloud datacenter serving (~₹40/hr on RACE).

---

## Quickstart: Step-by-Step Reproduction

Follow these exact steps to reproduce the baseline on any Linux/Mac environment:

### Step 1: Download Model Weights (Identical Across All Nodes)
```bash
mkdir -p ~/models && cd ~/models
curl -L -o Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf \
  "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
```

### Step 2: Get `llama-server` Binary

* **On macOS (Apple Silicon via Homebrew or Release Tarball):**
  ```bash
  # Option A: Install via Homebrew
  brew install llama.cpp
  # Binary will be at: /opt/homebrew/bin/llama-server

  # Option B: Download pre-built Metal release
  mkdir -p ~/llama-bin && cd ~/llama-bin
  curl -L -o llama-metal-macos.tar.gz \
    "https://github.com/ggml-org/llama.cpp/releases/download/b11901/llama-b11901-bin-macos-arm64.tar.gz"
  tar -xzf llama-metal-macos.tar.gz
  ```

* **On Linux with CUDA (e.g., Ubuntu 24.04 with NVIDIA L4):**
  ```bash
  mkdir -p ~/llama-bin && cd ~/llama-bin
  wget https://github.com/ggml-org/llama.cpp/releases/download/b11963/llama-b11963-bin-ubuntu-cuda-12.8-x64.tar.gz
  tar -xzf llama-b11963-bin-ubuntu-cuda-12.8-x64.tar.gz
  ```

### Step 3: Launch `llama-server` in the Background

Run the server with full GPU layer offloading (`-ngl 99`) and locked context window (`-c 2048`):

```bash
# On Mac Studio M3 Ultra:
/opt/homebrew/bin/llama-server \
  -m ~/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf \
  -ngl 99 \
  -c 2048 \
  --port 8080 \
  --host 0.0.0.0

# On Linux CUDA:
export LD_LIBRARY_PATH=~/llama-bin/llama-b11963:$LD_LIBRARY_PATH
~/llama-bin/llama-b11963/llama-server \
  -m ~/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf \
  -ngl 99 \
  -c 2048 \
  --port 8080 \
  --host 0.0.0.0
```

Verify that the server is ready:
```bash
curl http://127.0.0.1:8080/health
# Expected output: {"status":"ok"}
```

### Step 4: Run the Benchmark Profiler

Clone the repository and run the profiler:
```bash
cd ~/ai-infra-systems/stages/01-baseline-inference

# On Apple Mac Studio M3 Ultra (800 GB/s theoretical peak):
python3 benchmark_baseline.py \
  --endpoint http://127.0.0.1:8080/v1/chat/completions \
  --model Qwen2.5-Coder-1.5B \
  --samples 5 \
  --model-size-gb 0.986 \
  --peak-bandwidth-gbps 800.0

# On NVIDIA L4 (300 GB/s theoretical peak):
python3 benchmark_baseline.py \
  --endpoint http://127.0.0.1:8080/v1/chat/completions \
  --model Qwen2.5-Coder-1.5B \
  --samples 5 \
  --model-size-gb 0.986 \
  --peak-bandwidth-gbps 300.0
```

---

## 5. Measured Physical Fleet Results

Here are the actual measured results across our physical fleet for single-stream inference:

| Metric | Node 1: GTX 1050 Ti | Node 2: Mac Studio M3 Ultra | Node 3: NVIDIA L4 |
| :--- | :--- | :--- | :--- |
| **Silicon Architecture** | Pascal GDDR5 (112 GB/s) | Apple UMA LPDDR5X (800 GB/s) | Ada Lovelace GDDR6 (300 GB/s) |
| **Warm Time-to-First-Token (TTFT)** | 104.3 ms | **15.1 ms** (Cold: 861.7 ms) | **7.7 ms** (Local 0ms wire) |
| **Time-Per-Output-Token (TPOT)** | 19.6 ms/token | **7.6 ms/token** | **5.5 ms/token** |
| **Decode Generation Speed** | 51.0 tokens/sec | **131.0 tokens/sec** | **183.5 tokens/sec** |
| **Achieved Memory Bandwidth** | 53.7 GB/s | **127.54 GB/s** | **181.26 GB/s** |
| **Memory Bus Saturation (B = 1)** | 47.9% | **15.9%** (Scales to 32.3% at B=8) | **60.4%** |

### What These Numbers Mean in Simple Terms:
1. **Cold Start vs Warm Cache:** On the Mac Studio, the very first request took **861.7 ms** because the Metal GPU compiler had to compile shaders and set up the compute graph. Once warm, TTFT dropped immediately to **15.1 ms** (a $57\times$ speedup).
2. **Decode Generation Speed:** Once the model starts writing words, the M3 Ultra generates **131 tokens every second** (7.6 milliseconds per token). This is over $2.5\times$ faster than the GTX 1050 Ti ($51\ \mathrm{tok/s}$).
3. **The Bandwidth Saturation Question:** Why did the 800 GB/s Mac Studio show 15.9% bus saturation for a single user? Because a 1.5B model is narrow ($d_{\mathrm{model}} = 1536$). Generating one token at a time does not create enough parallel work to keep all 80 GPU cores busy. In Stage 03, we show that when 8 users query concurrently, achieved bandwidth jumps to **258.34 GB/s**.

For the complete deep-dive engineering analysis, read the [Stage 01 Full Report](REPORT.md).

---

## The Core Metrics Explained Simply

```text
1. Time-To-First-Token (TTFT):
   How long the user waits before the first word appears on screen.
   TTFT = Prefill Compute Time + Network Round-Trip Time + HTTP Framing

2. Time-Per-Output-Token (TPOT / Inter-Token Latency):
   How long it takes the GPU to generate each subsequent word.
   TPOT = (Total Generation Time - TTFT) / (Output Tokens - 1)

3. Decode Throughput (Speed in tok/s):
   How many tokens the model outputs per second while writing.
   Decode Speed = 1 / TPOT

4. Achieved Decode Memory Bandwidth (GB/s):
   For every token generated, the GPU must read the entire model from memory.
   Achieved Bandwidth = Model Size (GB) * Decode Speed (tok/s)

5. Memory Bus Saturation (%):
   How much of the GPU's physical memory pipe is being utilized.
   Saturation = (Achieved Bandwidth / Hardware Peak Bandwidth) * 100%
```

---

## Architectural Decision Record (ADR): Why Raw `llama-server`?

We deliberately test raw C++ `llama-server` rather than high-level frameworks like Ollama:
1. **Zero Runtime Noise:** Ollama wraps `llama.cpp` inside a Go daemon, which adds garbage collection pauses, HTTP-to-IPC serialization, and auto-unload timeouts (`keep_alive`). Raw `llama-server` reads directly from the model context in memory with zero intermediate proxy layers.
2. **Deterministic Hardware Allocation:** `llama-server` prints exact byte-level VRAM allocations (weight buffer, KV-cache, scratch space) directly on boot, which is required for mathematical Roofline validation.
3. **Explicit Kernel Levers:** Exposes direct CLI control over memory-mapping (`-lm`), context size (`-c`), and GPU offloading (`-ngl`) needed for our 10 progressive engineering stages.
