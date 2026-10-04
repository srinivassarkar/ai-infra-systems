# Stage 07: Observability, Metrics & Distributed Tracing

## Objective
Implement end-to-end observability across the inference platform using Prometheus metrics, Grafana dashboards, and OpenTelemetry distributed tracing.

## Key Questions
1. What are the key Golden Signals for LLM serving (Queue Depth, Active Streams, TTFT, TPOT, VRAM Saturation)?
2. How do we extract GPU hardware telemetry (`nvidia-smi` exporter or macOS Metal counters) into Prometheus?
3. How do we propagate OpenTelemetry trace contexts through the reverse proxy to measure the exact latency breakdown of each token span?
