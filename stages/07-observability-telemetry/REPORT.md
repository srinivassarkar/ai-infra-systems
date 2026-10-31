# Stage 07: Observability, Metrics & Telemetry Report

## 1. Executive Summary & The Upstream Observability Gap

High-throughput LLM serving platforms deviate radically from traditional stateless microservices. While traditional REST APIs evaluate latency as a monolithic duration from socket ingress to socket egress, LLM inference comprises two asynchronous, hardware-differentiated phases:
1. **Prefill Phase (Prompt Ingestion):** Compute-bound matrix-matrix multiplication ($\mathcal{O}(N)$ GEMM) determining **Time to First Token (TTFT)**.
2. **Decode Phase (Autoregressive Generation):** Memory-bandwidth-bound matrix-vector multiplication ($\mathcal{O}(1)$ GEMV) determining **Time Per Output Token (TPOT) / Inter-Token Latency (ITL)**.

### The Upstream Observability Gap Verified Empirically
Our audit of `llama-server`'s native `/metrics` endpoint (`http://127.0.0.1:8080/metrics`) proved that while the engine exposes internal counters (`llamacpp:prompt_tokens_total`, `llamacpp:tokens_predicted_total`, `llamacpp:n_decode_total`), **it natively lacks TTFT and TPOT latency histograms**.

Because TTFT depends on client socket connection, network transit, and request queue depth, and TPOT requires measuring continuous token chunk delivery intervals over HTTP chunked transfer / SSE, **these Golden Signals cannot be derived from internal engine counters alone**. They must be captured by an ingress telemetry gateway or reverse proxy layer.

---

## 2. Empirical Validation Telemetry: The 5 Golden Signals

*Hardware: NVIDIA L4 (24 GB GDDR6, 300 GB/s bandwidth) on RACE Cloud Pod.*  
*Model: `Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf`.*  
*Ingress Layer: `telemetry_gateway.py` (Port 8002) $\rightarrow$ Upstream `llama-server` (Port 8080).*  
*Metrics Exporter: Prometheus text exposition on Port 9090.*

### Ingress Streaming Latency Profile ($N=4$ Concurrent Clients)

| Client | Max Tokens | Output Tokens | Measured TTFT | Decode Duration | E2E Request Duration | Effective TPOT / ITL |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Client 1** | 64 | 48 tokens | **27.22 ms** | 321.94 ms | 349.16 ms | **6.83 ms/tok** (146.4 tok/s) |
| **Client 2** | 64 | 66 tokens | **25.84 ms** | 522.80 ms | 548.64 ms | **8.06 ms/tok** (124.0 tok/s) |
| **Client 3** | 64 | 66 tokens | **25.54 ms** | 522.75 ms | 548.29 ms | **8.06 ms/tok** (124.0 tok/s) |
| **Client 4** | 64 | 61 tokens | **25.46 ms** | 493.80 ms | 519.26 ms | **8.26 ms/tok** (121.0 tok/s) |

### Prometheus Unified Exposition Snapshot (Port 9090)
```text
# HELP llm_gateway_active_streams Current active concurrent streaming requests
# TYPE llm_gateway_active_streams gauge
llm_gateway_active_streams 0

# HELP llm_gateway_requests_total Total number of inference requests handled
# TYPE llm_gateway_requests_total counter
llm_gateway_requests_total 4

# HELP llm_gateway_tokens_total Total tokens processed
# TYPE llm_gateway_tokens_total counter
llm_gateway_tokens_total{type="prompt"} 252
llm_gateway_tokens_total{type="completion"} 235

# HELP llm_ttft_seconds Time To First Token (Prefill Latency Histogram)
# TYPE llm_ttft_seconds histogram
llm_ttft_seconds_bucket{le="0.010"} 0
llm_ttft_seconds_bucket{le="0.025"} 0
llm_ttft_seconds_bucket{le="0.050"} 4
llm_ttft_seconds_bucket{le="+Inf"} 4

# HELP llm_tpot_seconds Time Per Output Token (Inter-Token Latency Histogram)
# TYPE llm_tpot_seconds histogram
llm_tpot_seconds_bucket{le="0.005"} 102
llm_tpot_seconds_bucket{le="0.010"} 231
llm_tpot_seconds_bucket{le="+Inf"} 231

# HELP llm_gpu_utilization_ratio NVIDIA GPU Compute Utilization (0.0 - 1.0)
# TYPE llm_gpu_utilization_ratio gauge
llm_gpu_utilization_ratio 0.80

# HELP llm_gpu_memory_used_bytes NVIDIA VRAM Used in Bytes
# TYPE llm_gpu_memory_used_bytes gauge
llm_gpu_memory_used_bytes 1763704832.0  (1.64 GiB)

# HELP llm_gpu_power_watts NVIDIA GPU Power Draw in Watts
# TYPE llm_gpu_power_watts gauge
llm_gpu_power_watts 40.95
```

---

## 3. Distributed Tracing: Eliminating the Span Cardinality Anti-Pattern

### The Cardinality Failure Mode
A pervasive anti-pattern in naive LLM observability is creating a distinct OpenTelemetry span for every individual generated token. In an enterprise inference deployment generating 100,000 tokens/sec, this creates:
$$\text{Span Ingestion Rate} = 100{,}000\ \text{spans/sec} \implies 8.64\ \text{billion spans/day}$$
This swamps collector pipelines (OpenTelemetry Collector, SigNoz, Jaeger), exhausts network I/O, and injects serialization latency into the hot token generation loop.

### The Production Solution: Hierarchical Phase-Based Spans
Our gateway emits a single trace per request with structured child spans representing the fundamental hardware states:

```json
{
  "trace_id": "fbce0d7b-9258-4218-baa0-496af45805e3",
  "span_id": "e6a9c81f-6a4f-43",
  "name": "chat_completion",
  "model": "qwen2.5-coder-1.5b",
  "total_duration_ms": 547.04,
  "ttft_prefill_ms": 39.02,
  "decode_phase_ms": 508.02,
  "tpot_itl_ms": 8.06,
  "tokens_emitted": 64,
  "child_spans": [
    {
      "name": "prefill",
      "duration_ms": 39.02
    },
    {
      "name": "decode",
      "duration_ms": 508.02,
      "tokens": 64
    }
  ]
}
```

This captures complete phase latency granularity without expanding trace cardinality.

---

## 4. Production PromQL Alert Rules & SLO Definitions

Implemented in `prometheus_rules.yml`:

```yaml
groups:
  - name: llm_inference_slos
    rules:
      # Prefill Phase Latency (TTFT) SLO: 95% of requests stream within 250ms
      - alert: LLMHighTimeToFirstTokenP95
        expr: histogram_quantile(0.95, sum(rate(llm_ttft_seconds_bucket[2m])) by (le)) > 0.250
        for: 1m
        labels:
          severity: critical
          tier: inference-gateway

      # Decode Phase Stutter (TPOT): P99 inter-token delay > 40ms (<25 tok/s floor)
      - alert: LLMTokenGenerationStutterP99
        expr: histogram_quantile(0.99, sum(rate(llm_tpot_seconds_bucket[2m])) by (le)) > 0.040
        for: 1m
        labels:
          severity: warning
          tier: inference-worker

      # Gray Failure Alert: Active connections exist, but 0 tokens emitted
      - alert: LLMGrayFailureDetected
        expr: llm_gateway_active_streams > 0 and rate(llm_gateway_tokens_total{type="completion"}[30s]) == 0
        for: 15s
        labels:
          severity: page
          tier: fault-injection
```
