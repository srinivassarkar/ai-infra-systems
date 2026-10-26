# Stage 01: Baseline Inference Profiling

## Objective
Measure and isolate the fundamental performance characteristics of large language model serving before introducing concurrency, proxies, or optimization techniques.

Specifically:
1. **Deconstruct TTFT (Time-To-First-Token):** Measure prompt prefill latency as a function of prompt token length.
2. **Deconstruct TPOT (Time-Per-Output-Token):** Measure autoregressive decode token generation latency across single-token iterations.
3. **Establish Baseline Throughput:** Measure raw generation speed in tokens per second under single-tenant (concurrency = 1) conditions.
4. **Quantify Memory Bus Saturation:** Calculate achieved memory bandwidth versus theoretical hardware bus ceilings.

---

## Architecture

```
[ benchmark_baseline.py ]
         │
         │ HTTP/1.1 POST /v1/chat/completions (stream: true)
         ▼
[ llama-server (Port 8080 - Native C++ Daemon) ]
         │
         ▼
[ Physical Silicon (CUDA GDDR5 / Apple UMA LPDDR5X) ]
```

---

## Architectural Decision Record (ADR): Why Raw `llama-server` over High-Level Wrappers

For an empirical systems engineering reliability lab, we explicitly select raw **`llama-server` (`llama.cpp`)** over high-level runtime wrappers such as Ollama or vLLM based on three architectural distinctions:

### 1. Zero-Overhead Silicon Telemetry
High-level wrappers like Ollama wrap `llama.cpp` dynamic libraries within a Go daemon. This introduces Go runtime garbage collection pauses, HTTP-to-IPC serialization latency, and uncommanded background idle-unload routines (`keep_alive=5m`). 

Raw `llama-server` is written in pure C++ using `httplib`. The HTTP stream reads directly from the model execution context in memory with zero intermediate proxy layers, guaranteeing that recorded TTFT and TPOT latencies reflect physical silicon rather than runtime wrapper overhead.

### 2. Granular Kernel & Memory Levers
Ollama conceals low-level parameters behind heuristic defaults. In contrast, `llama-server` exposes explicit command-line controls required for our 10-stage investigations:
* `--mmap` vs `--no-mmap`: Explicitly toggles memory-mapped virtual memory paging versus eager DRAM allocation (vital for Stage 02 cold-start profiling).
* `-fa`: Toggles FlashAttention kernels to test memory bandwidth bound vs. compute bound regimes.
* `-ngl <layers>`: Explicitly partitions layers across GPU VRAM and CPU system RAM to measure PCIe bus saturation.
* `-c <tokens>`: Locks exact context window allocations to measure KV-cache memory pressure without magic fallback defaults (vital for Stage 04).

### 3. Deterministic Hardware Accounting on Boot
Upon startup, `llama-server` emits a precise byte-level breakdown of VRAM allocations to `stdout`:
* Model weight footprint in physical VRAM
* Dedicated KV-cache tensor size (`kv_self`)
* Compute scratch buffer allocations

This deterministic telemetry is essential for computing our Roofline arithmetic intensity and memory bus saturation percentages.

---

## Metrics Formulas

1. **TTFT (Time-To-First-Token):**
   $$\mathrm{TTFT} = t_{\mathrm{first}} - t_{\mathrm{start}}$$
2. **TPOT (Time-Per-Output-Token / Inter-Token Latency):**
   $$\mathrm{TPOT} = \frac{t_{\mathrm{end}} - t_{\mathrm{first}}}{N_{\mathrm{out}} - 1}$$
3. **System Output Throughput:**
   $$\mathrm{Throughput} = \frac{N_{\mathrm{out}}}{t_{\mathrm{total}}}$$
4. **Achieved Memory Bandwidth:**
   $$\mathrm{BW}_{\mathrm{achieved}} = \mathrm{ModelSize}_{\mathrm{VRAM}} \times \mathrm{Throughput}$$
5. **Memory Bus Saturation Percentage:**
   $$\mathrm{Saturation} = \frac{\mathrm{BW}_{\mathrm{achieved}}}{\mathrm{BW}_{\mathrm{peak}}} \times 100\%$$

---

## Execution Guide

### 1. Fetch & Build Standalone `llama-server`

#### On Node 1 (Ubuntu 20.04 LTS, NVIDIA GTX 1050 Ti - CUDA 12.8):
*Prebuilt GitHub release binaries fail on Ubuntu 20.04 due to glibc 2.38 / libssl3 dependencies. Build locally from source linking against local glibc 2.31:*

```bash
# 1. Ensure modern CMake (>= 3.22) and CUDA 12.8 nvcc are in PATH
pip3 install --upgrade cmake
export PATH=/usr/local/cuda-12.8/bin:$PATH

# 2. Clone and build specifically for Pascal (sm_61) without libcurl dependency
mkdir -p ~/llama-bin && cd ~/llama-bin
if [ ! -d "llama.cpp" ]; then
  git clone https://github.com/ggml-org/llama.cpp
fi
cd llama.cpp
rm -rf build

cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=61 \
  -DLLAMA_CURL=OFF \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --target llama-server -j $(nproc)
```

#### On Node 2 (Apple Mac Studio, M3 Ultra - Metal):
```bash
# Download official standalone b11901 Metal release binary
mkdir -p ~/llama-bin && cd ~/llama-bin
curl -L -o llama-metal-macos.tar.gz \
  "https://github.com/ggml-org/llama.cpp/releases/download/b11901/llama-b11901-bin-macos-arm64.tar.gz"
tar -xzf llama-metal-macos.tar.gz
chmod +x llama-b11901/llama-server
```

---

### 2. Download Baseline Model Weights (`Qwen2.5-Coder-1.5B-Q4_K_M`)

Download the unified ~986 MB GGUF weights from Hugging Face:
```bash
mkdir -p ~/models
curl -L -o ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
```

---

### 3. Launch the Native `llama-server` Daemon

Run the server with full layer offloading (`-ngl 99`) and explicit context window sizing (`-c 2048`):

```bash
# On NVIDIA Server (100.117.38.36 / GTX 1050 Ti - 4 GB GDDR5):
~/llama-bin/llama.cpp/build/bin/llama-server \
  -m ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  -ngl 99 \
  -c 2048 \
  --port 8080 \
  --host 0.0.0.0

# On Mac Studio M3 Ultra (100.102.40.99 / 256 GB UMA):
~/llama-bin/llama-b11901/llama-server \
  -m ~/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  -ngl 99 \
  -c 2048 \
  --port 8080 \
  --host 0.0.0.0
```

---

### 4. Run Baseline Profiling Harness

From your development environment, execute the Stage 01 profiling benchmark against the active server:

```bash
# Against NVIDIA Server (100.117.38.36 - GTX 1050 Ti, 112 GB/s peak):
python stages/01-baseline-inference/benchmark_baseline.py \
  --endpoint http://100.117.38.36:8080/v1/chat/completions \
  --model qwen2.5-coder:1.5b \
  --samples 5 \
  --model-size-gb 1.1 \
  --peak-bandwidth-gbps 112

# Against Mac Studio M3 Ultra (100.102.40.99 - M3 Ultra, 800 GB/s peak):
python stages/01-baseline-inference/benchmark_baseline.py \
  --endpoint http://100.102.40.99:8080/v1/chat/completions \
  --model qwen2.5-coder:1.5b \
  --samples 5 \
  --model-size-gb 1.1 \
  --peak-bandwidth-gbps 800
```
