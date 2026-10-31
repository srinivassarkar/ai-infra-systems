# Stage 07: Observability, Metrics & Telemetry (The Golden Signals of LLM Serving)

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
