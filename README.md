# Production-Grade LLM Inference Platform

A progressive systems engineering platform demonstrating end-to-end reliability, latency profiling, memory economics, and edge delivery for large language models.

---

## 1. System Architecture

Rather than a collection of disconnected tutorials, this repository builds and benchmarks **one cohesive, production-grade inference system**:

```
[ Client Traffic / Load Generator (k6 / Locust) ]
                       │
                       ▼  (HTTP/1.1 & HTTP/2 Streaming SSE)
       [ Edge Reverse Proxy (Nginx / Envoy) ]
         - Buffer bloat mitigation (proxy_buffering off)
         - Connection keep-alive & backpressure
                       │
                       ▼  (Reverse Proxy Ingress)
       [ Platform Gateway / Admission Controller ]
         - Request queueing & rate limiting
         - Health checks & graceful degradation
                       │
                       ▼  (OpenAI Protocol / Unix Socket)
          [ LLM Serving Runtime Engine ]
         - Static vs Continuous Batching (vLLM / llama.cpp / mlx)
         - Iteration-level scheduling & PagedAttention
                       │
                       ▼  (Memory Bus: PCIe 3.0/4.0 vs Unified Memory)
         [ Physical Silicon & Hardware Layer ]
         - Node 1: NVIDIA CUDA (GTX 1050 Ti, 4GB GDDR5, PCIe 3.0 x16)
         - Node 2: Apple Silicon UMA (Mac Studio M3 Ultra, 256GB Unified RAM, ~800 GB/s)
         - Node 3: Apple Silicon Edge (Mac mini M4, 24GB Unified RAM)

[ Observability Plane (Parallel) ]
  ├── Prometheus (GPU utilization, queue depth, TTFT, TPOT)
  ├── Loki (Structured streaming runtime logs)
  └── OpenTelemetry (Distributed tracing from proxy to token generation)
```

---

## 2. The 10 Progressive Engineering Stages

Every stage is a self-contained, reproducible engineering experiment designed to answer a specific operational question:

| Stage | Focus Area | Core Operational Question | Key Deliverable |
| :--- | :--- | :--- | :--- |
| **01** | **Baseline Inference** | How do TTFT, TPOT, and throughput differ between prefill and decode? | Benchmark harness + mathematical timing models. |
| **02** | **Cold-Start Behavior** | What happens inside memory and disk during model load and warm-up? | Memory allocation timeline + preload scripts. |
| **03** | **Concurrency & Queues** | How does queue depth inflate P90/P99 latency under concurrent traffic? | Head-of-line blocking measurements + load test. |
| **04** | **Memory Pressure & OOM** | What causes KV cache explosion, and how do we prevent OOM crashes? | Exact VRAM budgeting formula + crash triage. |
| **05** | **Edge Streaming Proxy** | Why does reverse proxy buffering break Server-Sent Events (SSE)? | Tuned Nginx proxy configs + backpressure tests. |
| **06** | **Failure Injection** | How does the platform survive sudden daemon crashes and timeouts? | systemd/launchd auto-recovery + health probes. |
| **07** | **Observability & Tracing** | How do we observe token generation latency inside distributed traces? | Prometheus alerts + OpenTelemetry span propagation. |
| **08** | **Serving Optimization** | How do continuous batching and quantization affect throughput? | vLLM PagedAttention benchmarks + AWQ profiling. |
| **09** | **Production Deployment** | How do we deploy and supervise the stack with production guarantees? | systemd units, launchd plists, and container specs. |
| **10** | **Capacity Planning** | How many concurrent users can a given GPU/memory configuration support? | Sizing calculator + cost-per-token economics. |

---

## 3. The Scientific Experiment Standard

Every stage in this repository follows the strict empirical loop:

1. **Architecture:** Component diagram and request flow.
2. **Hypothesis:** Expected system behavior under specific conditions.
3. **Experiment:** Reproducible script injecting load or measuring hardware.
4. **Observed Output:** Raw metrics, telemetry tables, and latency percentiles.
5. **Root Cause Analysis:** Technical explanation of hardware, kernel, or software constraints.
6. **Remediation / Fix:** Production configuration, code change, or architecture fix.
7. **Verification:** Retest proving the issue is resolved.
8. **Engineering Write-up:** High-signal technical summary.

---

## 4. Hardware Fleet Specifications

The experiments in this repository are executed and verified across real, heterogeneous hardware:

* **Node 1 (Constrained Edge Linux CUDA):**
  * Model: Custom Linux Box
  * GPU: NVIDIA GeForce GTX 1050 Ti (4,096 MiB GDDR5, Driver 570.133.07, CUDA 12.8)
  * Interconnect: PCIe 3.0 x16
  * Host RAM: 16 GB DDR4
* **Node 2 (Enterprise Bare-Metal Inference Server):**
  * Model: Apple Mac Studio (M3 Ultra)
  * Cores: 28 Cores (System Firmware 13822.61.10)
  * Unified Memory: 256 GB LPDDR5X (~800 GB/s bandwidth)
  * Ingress: Tailscale Zero-Trust private mesh
* **Node 3 (Production Edge Inference Workhorse):**
  * Model: Apple Mac mini M4 (`Mac16,10`)
  * Cores: 10 Cores (4 Performance, 6 Efficiency)
  * Unified Memory: 24 GB
* **Node 4 (Client / Load Generator):**
  * Model: Apple Mac mini (Intel Core i5 6-Core, 16 GB RAM)
  * Role: Dedicated traffic injection (k6) without stealing target node compute.

---

## 5. Repository Structure

```
ai-infra-systems/
├── README.md
├── docs/
│   ├── ARCHITECTURE.md          # End-to-end platform architecture deep-dive
│   ├── GLOSSARY.md              # Foundational systems & inference mental models
│   └── EXPERIMENT_STANDARD.md   # Standard template for all stage reports
├── stages/
│   ├── 01-baseline-inference/
│   ├── 02-cold-start-behavior/
│   ├── 03-concurrency-and-queues/
│   ├── 04-memory-pressure-and-oom/
│   ├── 05-streaming-and-edge-proxy/
│   ├── 06-failure-injection/
│   ├── 07-observability-telemetry/
│   ├── 08-serving-optimizations/
│   ├── 09-production-deployment/
│   └── 10-capacity-planning/
└── platform/
    ├── ingress/                 # Nginx configurations & buffer tuning
    ├── gateway/                 # Queue, scheduler, and health check daemons
    ├── engine/                  # Serving runtimes (vLLM / llama.cpp / mlx)
    └── observability/           # Prometheus exporters, dashboards, and alerts
```

---

## 6. Operating Philosophy

* **Zero Bluffing:** We only document what has been measured on physical silicon.
* **No Endless Syllabi:** We do not study optimization techniques in isolation; we solve concrete bottlenecks in a running system.
* **Systems First:** We focus on memory layout, kernel syscalls, network sockets, and process resilience.
