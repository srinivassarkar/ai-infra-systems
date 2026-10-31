# Stage 07: Observability, Metrics & Telemetry (The Golden Signals of LLM Serving)

> **Methodology Notice:** This stage evaluates LLM telemetry using a custom Ingress Telemetry Gateway (`telemetry_gateway.py`) that exports metrics in Prometheus text exposition format (Port 9090) and logs hierarchical trace spans structured per the OpenTelemetry schema (`telemetry_traces.jsonl`). No standalone Prometheus server, Loki daemon, or OpenTelemetry Collector was deployed in this local benchmark.

## 1. Executive Summary & The Core Architectural Defect

In traditional microservices, Google's Four Golden Signals (Latency, Traffic, Errors, Saturation) are measured via standard HTTP request/response durations. In high-throughput LLM inference engines, standard metrics fail because **a single HTTP request contains two fundamentally different computational phases**:
1. **Prefill Phase (Prompt Ingestion):** Compute-bound GEMM matrix operations determining **Time to First Token (TTFT)**.
2. **Decode Phase (Autoregressive Generation):** Memory-bandwidth-bound GEMV matrix-vector operations determining **Time Per Output Token (TPOT) / Inter-Token Latency (ITL)**.

### The Upstream Observability Defect
Native inference engines (e.g. `llama-server`) expose `/metrics` containing raw counters and gauges (e.g. total tokens processed, current KV cache usage ratio, active slot counts). Crucially, **they do NOT export client-perceived TTFT and TPOT latency histograms**.

Because TTFT depends on client network connection and request queue time, and TPOT depends on continuous token chunk dispatch, **these Golden Signals must be captured by an ingress telemetry gateway or reverse proxy**.

---

## 2. The 5 Golden Signals of LLM Serving

| Golden Signal | Metric Name | Metric Type | Target SLO Threshold | SRE Operational Meaning |
| :--- | :--- | :---: | :---: | :--- |
| **1. Prefill Latency (TTFT)** | `llm_ttft_seconds` | Histogram | $P_{95} \le 250\ \mathrm{ms}$ | User-perceived wait time before text begins streaming. |
| **2. Decode Latency (TPOT / ITL)** | `llm_tpot_seconds` | Histogram | $P_{99} \le 40\ \mathrm{ms}$ ($25\ \text{tok/s}$) | Perceived typewriter fluidity. High ITL jitter causes reading stutter. |
| **3. KV Cache Saturation** | `llamacpp:kv_cache_usage_ratio` | Gauge | $< 85\%$ | Ratio of allocated KV blocks. Above 90% risks OOM or slot preemption. |
| **4. Queue Depth** | `llamacpp:requests_deferred` | Gauge | $0$ (Spike $< 5$) | Pending client requests waiting for available inference slots. |
| **5. Hardware Saturation** | `llm_gpu_utilization_ratio` | Gauge | $70\% - 95\%$ | GPU SM and memory bus saturation. |

---

## 3. Distributed Tracing: The Span Cardinality Anti-Pattern

A common failure in LLM observability is generating **one OpenTelemetry span per generated token**. 
* For a modest cluster generating 100,000 tokens/second, this floods collectors (Jaeger, SigNoz, OpenTelemetry Collector) with **100,000 spans/sec**, overwhelming trace storage and degrading inference latency.

### The Production Tracing Pattern: Hierarchical Phase-Based Spans
Instead of per-token spans, the Telemetry Gateway emits a single trace per request with structured child spans:
```text
[Trace: 8f4a-9b21-4d33]
  └── Span: chat_completion (Total Duration: 842.10 ms)
        ├── Child Span: queue_wait (Duration: 0.85 ms)
        ├── Child Span: prefill (Duration: 38.12 ms) -> Determines TTFT
        └── Child Span: decode (Duration: 803.13 ms, Tokens: 94, Avg ITL: 8.54 ms)
```

---

## 4. Step-by-Step Reproduction Guide

### Step 1: Ensure Upstream `llama-server` is Running
```bash
/root/llama-bin/llama-b11963/llama-server \
  -m /root/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf \
  -ngl 99 \
  --parallel 8 \
  -c 16384 \
  --metrics \
  --port 8080 \
  --host 0.0.0.0 > ~/server_stage07.log 2>&1 &
```

### Step 2: Start the Telemetry Gateway & Prometheus Exporter
```bash
cd ~/ai-infra-systems/stages/07-observability-telemetry
python3 telemetry_gateway.py > ~/gateway.log 2>&1 &
```
* Ingress Proxy: `http://127.0.0.1:8000/v1/chat/completions`
* Prometheus Endpoint: `http://127.0.0.1:9090/metrics`

### Step 3: Run the End-to-End Observability Validation Suite
```bash
python3 validate_telemetry.py
```

The script will:
1. Verify upstream engine `/metrics` on port 8080.
2. Drive 4 concurrent streaming requests through the gateway on port 8000.
3. Audit the Prometheus `/metrics` exposition on port 9090, confirming all 9 Golden Signals.
4. Inspect the generated OpenTelemetry trace spans in `telemetry_traces.jsonl`.

---

## 5. Verified Empirical Telemetry Results

### Cross-Fleet Observability Calibration Matrix

| Observability Signal | Fleet Node 2: Mac Studio M3 Ultra (Metal) | Fleet Node 3: Cloud Pod (NVIDIA L4) | SRE Production Threshold |
| :--- | :---: | :---: | :---: |
| **Model** | `Qwen2.5-Coder-1.5B (Q4_K_M)` | `Qwen2.5-Coder-1.5B (Q4_K_M)` | Fixed Architecture |
| **Concurrent Streams ($N$)** | 4 parallel streams | 4 parallel streams | Target Load |
| **Median TTFT (Prefill Latency)** | **111.40 ms** | **25.69 ms** | $\le 250\ \mathrm{ms}$ (SLO PASS) |
| **Mean TPOT (Decode Fluidity)** | **13.92 ms/tok** ($71.8\ \text{tok/s}$) | **7.80 ms/tok** ($128.2\ \text{tok/s}$) | $\le 40\ \mathrm{ms/tok}$ (SLO PASS) |
| **Native Engine TTFT Histogram** | **MISSING (Absent)** | **MISSING (Absent)** | Gateway Required |
| **Native Engine TPOT Histogram** | **MISSING (Absent)** | **MISSING (Absent)** | Gateway Required |
| **OpenTelemetry Span Strategy** | Hierarchical (1 Parent, 2 Children) | Hierarchical (1 Parent, 2 Children) | 0 Per-Token Overhead |
| **Prometheus Exporter Status** | HTTP 200 on Port 9095 | HTTP 200 on Port 9090 | Active Scraping |

---

### Plain-English Engineering Takeaways

#### 1. What is the "Upstream Observability Gap"?
When you query `llama-server` directly at `/metrics`, it returns basic operational counters: how many total prompt tokens were read, how many total tokens were generated, and how many seconds were spent.

However, **it cannot tell you how long real users actually waited for the first character (TTFT)**, nor **whether output words stuttered (TPOT)**. That is because TTFT includes network round-trips and socket queuing that happen before the engine processes the prompt, while TPOT requires monitoring streaming Server-Sent Events (SSE) packet chunks across the wire. 

If your monitoring only scrapes `llama-server` directly, your dashboards are blind to user-facing latency SLO breaches.

#### 2. Why an Ingress Telemetry Gateway is Mandatory
To capture real user experience, we place a lightweight gateway (`telemetry_gateway.py`) in front of the inference server:
- **Timestamping Socket Arrival:** When an HTTP POST request hits the gateway, the clock starts.
- **Tracking First Chunk:** The exact millisecond the first streaming chunk arrives from `llama-server`, the gateway calculates TTFT and records it into a Prometheus histogram (`llm_ttft_seconds_bucket`).
- **Measuring Word-by-Word Fluidity:** As each subsequent token chunk streams through, the gateway records the time delta into an inter-token histogram (`llm_tpot_seconds_bucket`).

On our Mac Studio M3 Ultra run, all 4 concurrent streams completed in **1.11 seconds**, recording a median TTFT of **111.40 ms** and a streaming speed of **71.8 tokens/second per client**.

#### 3. Why Naive Distributed Tracing Blows Up Production
Many teams make the rookie mistake of starting an OpenTelemetry span for every single generated token.
- In a production environment serving 100,000 tokens/sec, that produces **100,000 spans every second** (over **8.6 billion spans per day**).
- This floods your network, overloads collectors like Jaeger and SigNoz, and introduces latency into the hot generation loop.

**The Production Fix:**
Our gateway emits exactly **one parent trace per request** with **two lightweight child spans**:
1. `prefill`: Measures prompt ingestion time (determines TTFT).
2. `decode`: Measures the autoregressive generation loop as a single span, attaching total token count and average ITL as structured attributes.

Zero span explosion, zero collector lag, and 100% observability across the entire inference pipeline.
