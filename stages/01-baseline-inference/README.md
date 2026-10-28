# Stage 01: Baseline Inference Profiling

## Objective
Measure and isolate the fundamental performance characteristics of large language model serving before introducing concurrency, proxies, or optimization techniques.

Specifically:
1. **Deconstruct TTFT (Time-To-First-Token):** Measure prompt prefill latency as a function of prompt token length and isolate compute prefill from network transport.
2. **Deconstruct TPOT (Time-Per-Output-Token):** Measure autoregressive decode token generation latency across single-token iterations.
3. **Establish Baseline Throughput:** Measure raw generation speed in tokens per second under single-tenant (concurrency = 1) streaming conditions.
4. **Quantify Memory Bus Saturation:** Calculate achieved memory bandwidth versus theoretical hardware bus ceilings across three distinct silicon architectures.
5. **Network Topology Ablation:** Quantify the exact impact of network round-trip time (RTT) on client-perceived TTFT versus streaming token arrival.

---

## The 3-Tier Silicon Hardware Fleet

To avoid single-hardware bias, baseline inference is benchmarked across three fundamentally distinct hardware and memory paradigms:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             THE 3-TIER SILICON FLEET                             │
├─────────────────────────┬──────────────────────────┬─────────────────────────────┤
│ Node 1: Constrained     │ Node 2: Unified Memory   │ Node 3: Cloud Enterprise    │
│ Edge CUDA               │ Workstation (Apple UMA)  │ Datacenter (NVIDIA Ada)     │
├─────────────────────────┼──────────────────────────┼─────────────────────────────┤
│ NVIDIA GeForce          │ Apple Mac Studio         │ NVIDIA L4                   │
│ GTX 1050 Ti             │ M3 Ultra                 │ (Cloud Pod - Delhi)         │
│                         │                          │                             │
│ • VRAM: 4 GB GDDR5      │ • RAM: 256 GB LPDDR5X    │ • VRAM: 24 GB GDDR6 (ECC)   │
│ • Bus: 128-bit PCIe 3.0 │ • Bus: Unified Memory    │ • Bus: 192-bit PCIe 4.0     │
│ • Peak BW: 112 GB/s     │ • Peak BW: 800 GB/s      │ • Peak BW: 300 GB/s         │
│ • Arch: Pascal (sm_61)  │ • Arch: Apple Metal GPU  │ • Arch: Ada Lovelace (sm_89)│
│ • L2 Cache: 768 KB      │ • System Cache: 128 MB   │ • L2 Cache: 48 MB           │
└─────────────────────────┴──────────────────────────┴─────────────────────────────┘
```

---

## Architecture & Network Topologies

```
Topology A: Distributed Wire Benchmark (Nodes 1, 2, and 3-Remote)
[ Workstation Client (benchmark_baseline.py) ]
                    │
                    │ HTTP/1.1 POST /v1/chat/completions (stream: true)
                    │ Tailscale WireGuard Encrypted Mesh (WAN / LAN)
                    ▼
[ llama-server (Port 8080 - Native C++ Daemon) ]
                    │
                    ▼
[ Physical Silicon (GDDR5 / GDDR6 / Unified LPDDR5X) ]

----------------------------------------------------------------------------------

Topology B: Zero-Wire On-Chip Baseline (Node 3-Local)
[ On-Pod Client (benchmark_baseline.py) ] ──▶ loopback (127.0.0.1:8080) ──▶ [ llama-server ]
* Eliminates network transport delay to isolate pure on-chip prompt prefill compute.
```

---

## Architectural Decision Record (ADR): Why Raw `llama-server` over High-Level Wrappers

For an empirical systems engineering reliability lab, we explicitly select raw **`llama-server` (`llama.cpp`)** over high-level runtime wrappers such as Ollama or vLLM based on three architectural distinctions:

### 1. Zero-Overhead Silicon Telemetry
High-level wrappers like Ollama wrap `llama.cpp` dynamic libraries within a Go daemon. This introduces Go runtime garbage collection pauses, HTTP-to-IPC serialization latency, and uncommanded background idle-unload routines (`keep_alive=5m`). 

Raw `llama-server` is written in pure C++ using `httplib`. The HTTP stream reads directly from the model execution context in memory with zero intermediate proxy layers, guaranteeing that recorded TTFT and TPOT latencies reflect physical silicon rather than runtime wrapper overhead.

### 2. Granular Kernel & Memory Levers
Ollama conceals low-level parameters behind heuristic defaults. In contrast, `llama-server` exposes explicit command-line controls required for our 10-stage investigations:
* `-lm, --load-mode {mmap, none, dio, mlock}`: Explicitly toggles memory-mapped virtual memory paging, eager DRAM allocation, Direct I/O, or locked physical pages (vital for Stage 02 cold-start profiling).
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
   *Decomposition:* $\mathrm{TTFT} = \mathrm{TTFT}_{\mathrm{prefill\_compute}} + \mathrm{RTT}_{\mathrm{network}} + t_{\mathrm{framing}}$

2. **TPOT (Time-Per-Output-Token / Inter-Token Latency):**
   $$\mathrm{TPOT} = \frac{t_{\mathrm{end}} - t_{\mathrm{first}}}{N_{\mathrm{out}} - 1}$$

3. **Autoregressive Decode Throughput:**
   $$\mathrm{Throughput}_{\mathrm{decode}} = \frac{N_{\mathrm{out}} - 1}{t_{\mathrm{end}} - t_{\mathrm{first}}} = \frac{1}{\mathrm{TPOT}}$$

4. **End-to-End System Throughput:**
   $$\mathrm{Throughput}_{\mathrm{e2e}} = \frac{N_{\mathrm{out}}}{t_{\mathrm{total}}}$$

5. **Achieved Decode Memory Bandwidth:**
   $$\mathrm{BW}_{\mathrm{decode}} = \mathrm{ModelSize}_{\mathrm{GB}} \times \mathrm{Throughput}_{\mathrm{decode}} = \frac{\mathrm{ModelSize}_{\mathrm{GB}}}{\mathrm{TPOT}_{\mathrm{seconds}}}$$

6. **Memory Bus Saturation Percentage:**
   $$\mathrm{Saturation} = \frac{\mathrm{BW}_{\mathrm{decode}}}{\mathrm{BW}_{\mathrm{peak}}} \times 100\%$$

---

## Execution Guide: Running the Benchmarks Across All 3 Nodes

### 1. Download Model Weights (`Qwen2.5-Coder-1.5B-Q4_K_M`)
Download the identical ~986 MB GGUF weights across all nodes:
```bash
mkdir -p ~/models
curl -L -o ~/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf \
  "https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf"
```

---

### 2. Launch the `llama-server` Daemon

#### On Node 1 (GTX 1050 Ti - 4 GB GDDR5, Tailscale: 100.117.38.36):
```bash
~/llama-bin/llama.cpp/build/bin/llama-server \
  -m ~/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf \
  -ngl 99 \
  -c 2048 \
  --port 8080 \
  --host 0.0.0.0
```

#### On Node 2 (Mac Studio M3 Ultra - 256 GB UMA, Tailscale: 100.102.40.99):
```bash
~/llama-bin/llama-b11901/llama-server \
  -m ~/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf \
  -ngl 99 \
  -c 2048 \
  --port 8080 \
  --host 0.0.0.0
```

#### On Node 3 (NVIDIA L4 Pod - 24 GB GDDR6, Tailscale: 100.85.128.0):
```bash
LD_LIBRARY_PATH=/workspace/ai-infra/llama-b11963 nohup /workspace/ai-infra/llama-b11963/llama-server \
  -m /workspace/ai-infra/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf \
  -ngl 99 \
  -c 2048 \
  --port 8080 \
  --host 0.0.0.0 > /workspace/ai-infra/server.log 2>&1 &
```

---

### 3. Execute the Profiling Harness

#### Run 1: Node 1 (GTX 1050 Ti - 112 GB/s Peak) Over Tailscale:
```bash
python stages/01-baseline-inference/benchmark_baseline.py \
  --endpoint http://100.117.38.36:8080/v1/chat/completions \
  --model Qwen2.5-Coder-1.5B \
  --samples 5 \
  --model-size-gb 0.986 \
  --peak-bandwidth-gbps 112.0
```

#### Run 2: Node 2 (Mac Studio M3 Ultra - 800 GB/s Peak) Over Tailscale:
```bash
python stages/01-baseline-inference/benchmark_baseline.py \
  --endpoint http://100.102.40.99:8080/v1/chat/completions \
  --model Qwen2.5-Coder-1.5B \
  --samples 5 \
  --model-size-gb 0.986 \
  --peak-bandwidth-gbps 800.0
```

#### Run 3: Node 3 Local (NVIDIA L4 - 300 GB/s Peak) On-Pod Loopback (Zero-Wire):
Execute inside the L4 pod terminal:
```bash
python3 /workspace/ai-infra/repo/stages/01-baseline-inference/benchmark_baseline.py \
  --endpoint http://127.0.0.1:8080/v1/chat/completions \
  --model Qwen2.5-Coder-1.5B \
  --samples 5 \
  --model-size-gb 0.986 \
  --peak-bandwidth-gbps 300.0
```

#### Run 4: Node 3 Remote (NVIDIA L4 - 300 GB/s Peak) Over Tailscale WAN:
Execute from your workstation terminal:
```bash
python stages/01-baseline-inference/benchmark_baseline.py \
  --endpoint http://100.85.128.0:8080/v1/chat/completions \
  --model Qwen2.5-Coder-1.5B \
  --samples 5 \
  --model-size-gb 0.986 \
  --peak-bandwidth-gbps 300.0
```
