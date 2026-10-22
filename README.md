# LLM Inference Reliability Lab

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

| Stage | Focus Area | Core Operational Question | Key Deliverable | Target Silicon |
| :--- | :--- | :--- | :--- | :--- |
| **01** | **Baseline Inference** | How do TTFT, TPOT, and throughput differ between prefill and decode? | Benchmark harness + mathematical timing models. | GTX 1050 Ti & M3 Ultra |
| **02** | **Cold-Start Behavior** | What happens inside memory and disk during model load and warm-up? | Memory allocation timeline + page cache warm-up scripts. | GTX 1050 Ti & M3 Ultra |
| **03** | **Concurrency & Queues** | How does queue depth inflate P90/P99 latency under concurrent traffic? | Head-of-line blocking measurements + saturation knee graph. | Mac mini M4 + k6 Client |
| **04** | **Memory Pressure & OOM** | What causes KV cache explosion, and how do we prevent OOM crashes? | Exact VRAM budgeting formula + memory admission controller. | GTX 1050 Ti (4 GB) |
| **05** | **Edge Streaming Proxy** | Why does reverse proxy buffering break Server-Sent Events (SSE)? | Tuned Nginx proxy configs (proxy_buffering off) + backpressure tests. | Linux Box / macOS |
| **06** | **Failure Injection** | How does the platform survive sudden daemon crashes and stalls? | Dual health probes (/health/live vs /health/ready) + auto-recovery. | Linux / macOS |
| **07** | **Observability & Tracing** | How do we observe token generation latency inside distributed traces? | Prometheus alerting rules + OpenTelemetry span propagation. | M3 Ultra + Prometheus |
| **08** | **Serving Optimization & Economics** | How do continuous batching and quantization impact throughput and cost? | Empirical vLLM (PagedAttention + AWQ) vs llama.cpp run; measured $/1M tokens. | Cloud NVIDIA L4 (24GB) |
| **09** | **Production K8s & KEDA** | How do we autoscale inference workloads without thrashing accelerators? | k3s manifests + KEDA event-driven autoscaling on Prometheus queue depth. | Local k3s Cluster |
| **10** | **Capacity Planning** | How many concurrent users can a given silicon setup sustain within SLOs? | Hardware sizing calculator + empirical unit economics ($/1M tokens). | Workstation Calculator |

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
* **Node 5 (Cloud Accelerator Target - Dedicated Benchmark Run):**
  * Model: Cloud Instance (NVIDIA L4 24GB GDDR6, Ada Lovelace, Compute Capability 8.9)
  * Role: Dedicated Stage 08 production benchmark (vLLM + PagedAttention + AWQ vs llama.cpp) to establish empirical $/1M tokens baseline.

---

## 5. Repository Structure

```
ai-infra-systems/
├── README.md
├── docs/
│   ├── 00-FOUNDATIONS-MODELS-ARCHITECTURES-INFERENCE.md  # Complete ML taxonomy & decision matrix
│   ├── INFERENCE_ENGINEERING_ROADMAP.md                  # Master 8-domain curriculum & war-room Q&A
│   ├── ARCHITECTURE.md                                  # End-to-end platform architecture deep-dive
│   ├── GLOSSARY.md                                      # Foundational systems & inference mental models
│   └── EXPERIMENT_STANDARD.md                           # Standard template for all stage reports
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
    ├── ingress/                         # Nginx configurations & buffer tuning
    ├── gateway/                         # Queue, scheduler, and health check daemons
    ├── engine/                          # Serving runtimes (vLLM / llama.cpp / mlx)
    └── observability/                   # Prometheus exporters, dashboards, and alerts
```

---

## 6. Operating Philosophy & Career Alignment

* **Zero Bluffing & Empirical Metrics:** We only document what has been measured on physical silicon. We report cost strictly as measured **$/1M tokens** derived from verified benchmark runs, never quoting hypothetical or unverified savings.
* **Event-Driven Orchestration:** We autoscale inference on **Prometheus queue depth** via KEDA, rejecting naive CPU-based autoscaling that leads to accelerator thrashing.
* **Systems First:** We focus on memory layout, kernel syscalls, network sockets, and process resilience.
* **Two-Tier Market Alignment:** This repository is engineered to outclass competitors across two distinct market tiers:
  1. *Primary Strike:* Platform / DevOps / SRE / MLOps roles at tech companies and scale-ups operating LLM workloads.
  2. *Secondary Strike:* Dedicated AI Platform / Inference Infrastructure roles at fast-growing AI startups.
