# System Architecture Specification

## 1. Executive Summary

This architecture defines a production-grade Large Language Model (LLM) inference platform engineered for high availability, low-latency streaming, and memory efficiency across heterogeneous compute environments.

Unlike standard web APIs, LLM inference introduces asymmetric resource bottlenecks:
* Compute-bound prefill vs memory-bandwidth-bound decode.
* Dynamic KV cache memory expansion.
* Long-lived HTTP streaming connections (Server-Sent Events).
* Hardware bus constraints (PCIe transfer overhead vs Unified Memory bandwidth).

---

## 2. Request Lifecycle & Data Flow

```
[ Client / SDK ]
       │
       │ 1. POST /v1/chat/completions (Stream: true)
       ▼
┌────────────────────────────────────────────────────────┐
│ Edge Ingress Layer (Nginx / Envoy)                     │
│  - SSL Termination & TLS Handshake                     │
│  - proxy_buffering off; (Flush chunks immediately)     │
│  - Connection keep-alive & HTTP/2 Multiplexing         │
│  - Client disconnect detection & backpressure          │
└────────────────────────────────────────────────────────┘
       │
       │ 2. Reverse Proxy Pass (Keep-alive Unix/TCP socket)
       ▼
┌────────────────────────────────────────────────────────┐
│ Platform Gateway & Admission Controller                │
│  - Health & Readiness Probing (/healthz, /ready)       │
│  - Request Concurrency Throttling (Semaphore Queue)    │
│  - Token Budget Estimation (Context Length Check)      │
│  - Token Cancellation Propagation                      │
└────────────────────────────────────────────────────────┘
       │
       │ 3. Forward to Serving Engine Worker
       ▼
┌────────────────────────────────────────────────────────┐
│ LLM Serving Runtime Engine (vLLM / llama.cpp / mlx)    │
│  - Request Scheduling (Continuous Batching)            │
│  - Memory Allocator (PagedAttention KV Cache)          │
│  - Model Execution Loop (Prefill GEMM -> Decode steps) │
└────────────────────────────────────────────────────────┘
       │
       │ 4. Hardware Memory Bus Transfer
       ▼
┌────────────────────────────────────────────────────────┐
│ Physical Silicon Execution                             │
│  - Discrete GPU: GDDR5/GDDR6 VRAM over PCIe bus        │
│  - Unified Memory: LPDDR5X shared RAM (Apple Silicon)  │
└────────────────────────────────────────────────────────┘
```

---

## 3. Plane Breakdown

### A. Data Plane (Token Generation & Delivery)
1. **Client Connection:** Long-lived HTTP/1.1 or HTTP/2 TCP connection holding open an SSE stream.
2. **Buffering Rule:** The edge proxy must disable all response buffering (`X-Accel-Buffering: no; proxy_buffering off;`). Any intermediate buffer turns real-time token delivery into jerky paragraph bursts.
3. **Cancellation Handling:** If the client closes the browser or aborts the TCP socket, the proxy and gateway must detect the broken pipe (`SIGPIPE` / `EPIPE`) and immediately abort the GPU decode loop to avoid burning compute cycles on orphaned requests.

### B. Control Plane (Scheduling, Admission & Health)
1. **Admission Control:** Rejects or queues incoming requests when KV cache utilization exceeds a safe threshold (e.g., 90%), preventing out-of-memory (OOM) crashes.
2. **Health Probes:**
   * **Liveness:** Is the daemon process alive and responding to pings?
   * **Readiness:** Is the model weights file loaded into silicon memory and warmed up?

### C. Telemetry Plane (Observability)
* **Metrics:** Scraped via Prometheus every 5s (GPU memory, GPU compute utilization, KV cache usage percentage, request queue depth).
* **Logs:** Structured JSON emitted to standard out, captured by Loki.
* **Tracing:** OpenTelemetry spans measuring:
  * Proxy latency
  * Queue wait time (Time in Queue)
  * TTFT (Time to First Token)
  * TPOT (Time Per Output Token / Inter-Token Latency)
