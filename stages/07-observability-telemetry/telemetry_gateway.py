#!/usr/bin/env python3
"""
telemetry_gateway.py - Stage 07 LLM Ingress Telemetry Gateway & Prometheus Exporter
Author: Srinivas Sarkar (ai-infra-systems)

Solves the Core LLM Observability Defect:
- llama-server natively provides raw counters/gauges (/metrics) but lacks TTFT & TPOT latency histograms.
- This gateway proxies incoming client traffic to upstream llama-server (127.0.0.1:8080),
  measures exact Time to First Token (TTFT), Inter-Token Latency (ITL / TPOT), and active streams,
  and combines them with upstream engine metrics and NVIDIA GPU hardware telemetry into a
  unified Prometheus exposition endpoint on port 9090.
- Emits hierarchical OpenTelemetry trace spans (Request -> Prefill -> Decode) without wasteful
  per-token span cardinality explosion.
"""

import http.server
import socketserver
import urllib.request
import urllib.error
import time
import json
import threading
import sys
import os
import subprocess
import uuid
import argparse
from typing import Dict, List, Any

UPSTREAM_URL = "http://127.0.0.1:8080"
DEFAULT_GATEWAY_PORT = 8002  # Using 8002 to avoid collision with Stage 05 Nginx (8000/8001)
DEFAULT_METRICS_PORT = 9090

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRACE_LOG_FILE = os.path.join(SCRIPT_DIR, "telemetry_traces.jsonl")

# Histogram Buckets (in seconds)
TTFT_BUCKETS = [0.010, 0.025, 0.050, 0.075, 0.100, 0.250, 0.500, 1.000, 2.500, 5.000]
TPOT_BUCKETS = [0.005, 0.008, 0.010, 0.015, 0.020, 0.030, 0.050, 0.100, 0.200]

class MetricsStore:
    def __init__(self):
        self.lock = threading.Lock()
        self.active_streams = 0
        self.total_requests = 0
        self.total_tokens_prompt = 0
        self.total_tokens_completion = 0
        self.ttft_sum = 0.0
        self.ttft_count = 0
        self.ttft_buckets = {b: 0 for b in TTFT_BUCKETS}
        self.tpot_sum = 0.0
        self.tpot_count = 0
        self.tpot_buckets = {b: 0 for b in TPOT_BUCKETS}

    def record_request_start(self):
        with self.lock:
            self.active_streams += 1
            self.total_requests += 1

    def record_request_end(self):
        with self.lock:
            if self.active_streams > 0:
                self.active_streams -= 1

    def record_ttft(self, ttft_seconds: float):
        with self.lock:
            self.ttft_sum += ttft_seconds
            self.ttft_count += 1
            for b in TTFT_BUCKETS:
                if ttft_seconds <= b:
                    self.ttft_buckets[b] += 1

    def record_tpot(self, tpot_seconds: float):
        with self.lock:
            self.tpot_sum += tpot_seconds
            self.tpot_count += 1
            for b in TPOT_BUCKETS:
                if tpot_seconds <= b:
                    self.tpot_buckets[b] += 1

    def record_tokens(self, prompt: int, completion: int):
        with self.lock:
            self.total_tokens_prompt += prompt
            self.total_tokens_completion += completion

METRICS = MetricsStore()


def get_gpu_telemetry() -> Dict[str, float]:
    """Extracts NVIDIA GPU utilization, VRAM usage, and power via nvidia-smi."""
    gpu_data = {
        "utilization_gpu_ratio": 0.0,
        "memory_used_bytes": 0.0,
        "memory_total_bytes": 0.0,
        "power_draw_watts": 0.0
    }
    if sys.platform != "linux":
        return gpu_data
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
            "--format=csv,noheader,nounits"
        ]
        out = subprocess.check_output(cmd, text=True, timeout=1.0).strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 4:
            gpu_data["utilization_gpu_ratio"] = float(parts[0]) / 100.0
            gpu_data["memory_used_bytes"] = float(parts[1]) * 1024 * 1024
            gpu_data["memory_total_bytes"] = float(parts[2]) * 1024 * 1024
            gpu_data["power_draw_watts"] = float(parts[3])
    except Exception:
        pass
    return gpu_data


def fetch_upstream_metrics() -> str:
    """Fetches native engine counters from llama-server /metrics."""
    try:
        req = urllib.request.Request(f"{UPSTREAM_URL}/metrics", headers={"User-Agent": "TelemetryGateway/1.0"})
        with urllib.request.urlopen(req, timeout=1.0) as resp:
            if resp.status == 200:
                return resp.read().decode("utf-8")
    except Exception:
        pass
    return ""


class GatewayRequestHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress default noisy console logs

    def do_GET(self):
        if self.path == "/health":
            self.handle_health()
        else:
            self.proxy_generic("GET")

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self.handle_chat_completions()
        else:
            self.proxy_generic("POST")

    def handle_health(self):
        try:
            req = urllib.request.Request(f"{UPSTREAM_URL}/health")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                content = resp.read()
                self.send_response(resp.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
        except Exception as e:
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "unavailable", "error": str(e)}).encode())

    def proxy_generic(self, method: str):
        content_len = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_len) if content_len > 0 else None
        target = f"{UPSTREAM_URL}{self.path}"
        try:
            req = urllib.request.Request(target, data=body, method=method)
            for k, v in self.headers.items():
                if k.lower() not in ["host", "content-length"]:
                    req.add_header(k, v)
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                out = resp.read()
                self.send_response(resp.status)
                for hk, hv in resp.getheaders():
                    self.send_header(hk, hv)
                self.end_headers()
                self.wfile.write(out)
        except urllib.error.HTTPError as he:
            self.send_response(he.code)
            self.end_headers()
            self.wfile.write(he.read())
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            self.wfile.write(str(e).encode())

    def handle_chat_completions(self):
        content_len = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_len)
        try:
            payload = json.loads(raw_body.decode())
        except Exception:
            payload = {}

        is_streaming = payload.get("stream", False)
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())[:16]

        METRICS.record_request_start()
        t_req_start = time.perf_counter()
        t_first_token = None
        token_timestamps = []
        tokens_emitted = 0

        target = f"{UPSTREAM_URL}/v1/chat/completions"
        try:
            req = urllib.request.Request(target, data=raw_body, method="POST")
            req.add_header("Content-Type", "application/json")
            if is_streaming:
                req.add_header("Accept", "text/event-stream")

            with urllib.request.urlopen(req, timeout=300.0) as upstream_resp:
                self.send_response(200)
                self.send_header("Content-Type", upstream_resp.headers.get("Content-Type", "application/json"))
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("X-Trace-ID", trace_id)
                self.end_headers()

                if is_streaming:
                    # Stream chunks incrementally while measuring latency deltas
                    buffer = b""
                    while True:
                        chunk = upstream_resp.read(512)
                        if not chunk:
                            break
                        now = time.perf_counter()
                        buffer += chunk

                        # Look for SSE token boundaries
                        while b"\n\n" in buffer:
                            event_data, buffer = buffer.split(b"\n\n", 1)
                            for line in event_data.split(b"\n"):
                                if line.startswith(b"data: "):
                                    payload_str = line[6:].strip()
                                    if payload_str == b"[DONE]":
                                        continue
                                    try:
                                        data_obj = json.loads(payload_str.decode())
                                        content_delta = data_obj.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                        if content_delta:
                                            tokens_emitted += 1
                                            if t_first_token is None:
                                                t_first_token = now
                                                ttft = t_first_token - t_req_start
                                                METRICS.record_ttft(ttft)
                                            token_timestamps.append(now)
                                    except Exception:
                                        pass

                        # Write HTTP 1.1 chunk
                        chunk_size_hex = f"{len(chunk):X}\r\n".encode()
                        self.wfile.write(chunk_size_hex + chunk + b"\r\n")
                        self.wfile.flush()

                    # Terminate HTTP chunked transfer
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
                else:
                    full_body = upstream_resp.read()
                    self.wfile.write(f"{len(full_body):X}\r\n".encode() + full_body + b"\r\n0\r\n\r\n")
                    self.wfile.flush()

            t_req_end = time.perf_counter()
            total_duration = t_req_end - t_req_start

            # Calculate and record TPOT / ITL
            if len(token_timestamps) > 1:
                itls = [token_timestamps[i] - token_timestamps[i-1] for i in range(1, len(token_timestamps))]
                for itl in itls:
                    METRICS.record_tpot(itl)

            METRICS.record_tokens(prompt=len(payload.get("messages", [{}])[0].get("content", "")), completion=tokens_emitted)

            # Emit Hierarchical OpenTelemetry Span JSON (Request -> Prefill -> Decode)
            ttft_ms = ((t_first_token - t_req_start) * 1000.0) if t_first_token else 0.0
            decode_ms = ((t_req_end - t_first_token) * 1000.0) if t_first_token else 0.0
            tpot_ms = (decode_ms / (tokens_emitted - 1)) if tokens_emitted > 1 else 0.0

            trace_record = {
                "trace_id": trace_id,
                "span_id": span_id,
                "name": "chat_completion",
                "model": payload.get("model", "unknown"),
                "total_duration_ms": round(total_duration * 1000.0, 2),
                "ttft_prefill_ms": round(ttft_ms, 2),
                "decode_phase_ms": round(decode_ms, 2),
                "tpot_itl_ms": round(tpot_ms, 2),
                "tokens_emitted": tokens_emitted,
                "child_spans": [
                    {"name": "prefill", "duration_ms": round(ttft_ms, 2)},
                    {"name": "decode", "duration_ms": round(decode_ms, 2), "tokens": tokens_emitted}
                ]
            }
            with open(TRACE_LOG_FILE, "a") as f:
                f.write(json.dumps(trace_record) + "\n")

        except Exception:
            pass
        finally:
            METRICS.record_request_end()


class PrometheusMetricsHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path != "/metrics":
            self.send_response(404)
            self.end_headers()
            return

        upstream_text = fetch_upstream_metrics()
        gpu = get_gpu_telemetry()

        # Build Prometheus Metrics Exposition
        lines = []
        lines.append("# HELP llm_gateway_active_streams Current active concurrent streaming requests")
        lines.append("# TYPE llm_gateway_active_streams gauge")
        lines.append(f"llm_gateway_active_streams {METRICS.active_streams}")

        lines.append("# HELP llm_gateway_requests_total Total number of inference requests handled")
        lines.append("# TYPE llm_gateway_requests_total counter")
        lines.append(f"llm_gateway_requests_total {METRICS.total_requests}")

        lines.append("# HELP llm_gateway_tokens_total Total tokens processed")
        lines.append("# TYPE llm_gateway_tokens_total counter")
        lines.append(f'llm_gateway_tokens_total{{type="prompt"}} {METRICS.total_tokens_prompt}')
        lines.append(f'llm_gateway_tokens_total{{type="completion"}} {METRICS.total_tokens_completion}')

        # TTFT Histogram
        lines.append("# HELP llm_ttft_seconds Time To First Token (Prefill Latency Histogram)")
        lines.append("# TYPE llm_ttft_seconds histogram")
        cumulative_ttft = 0
        for b in TTFT_BUCKETS:
            cumulative_ttft += METRICS.ttft_buckets[b]
            lines.append(f'llm_ttft_seconds_bucket{{le="{b:.3f}"}} {cumulative_ttft}')
        lines.append(f'llm_ttft_seconds_bucket{{le="+Inf"}} {METRICS.ttft_count}')
        lines.append(f"llm_ttft_seconds_sum {METRICS.ttft_sum:.6f}")
        lines.append(f"llm_ttft_seconds_count {METRICS.ttft_count}")

        # TPOT Histogram
        lines.append("# HELP llm_tpot_seconds Time Per Output Token (Inter-Token Latency Histogram)")
        lines.append("# TYPE llm_tpot_seconds histogram")
        cumulative_tpot = 0
        for b in TPOT_BUCKETS:
            cumulative_tpot += METRICS.tpot_buckets[b]
            lines.append(f'llm_tpot_seconds_bucket{{le="{b:.3f}"}} {cumulative_tpot}')
        lines.append(f'llm_tpot_seconds_bucket{{le="+Inf"}} {METRICS.tpot_count}')
        lines.append(f"llm_tpot_seconds_sum {METRICS.tpot_sum:.6f}")
        lines.append(f"llm_tpot_seconds_count {METRICS.tpot_count}")

        # GPU Hardware Telemetry
        lines.append("# HELP llm_gpu_utilization_ratio NVIDIA GPU Compute Utilization (0.0 - 1.0)")
        lines.append("# TYPE llm_gpu_utilization_ratio gauge")
        lines.append(f"llm_gpu_utilization_ratio {gpu['utilization_gpu_ratio']}")

        lines.append("# HELP llm_gpu_memory_used_bytes NVIDIA VRAM Used in Bytes")
        lines.append("# TYPE llm_gpu_memory_used_bytes gauge")
        lines.append(f"llm_gpu_memory_used_bytes {gpu['memory_used_bytes']}")

        lines.append("# HELP llm_gpu_memory_total_bytes NVIDIA VRAM Total in Bytes")
        lines.append("# TYPE llm_gpu_memory_total_bytes gauge")
        lines.append(f"llm_gpu_memory_total_bytes {gpu['memory_total_bytes']}")

        lines.append("# HELP llm_gpu_power_watts NVIDIA GPU Power Draw in Watts")
        lines.append("# TYPE llm_gpu_power_watts gauge")
        lines.append(f"llm_gpu_power_watts {gpu['power_draw_watts']}")

        # Append upstream engine metrics
        output = "\n".join(lines) + "\n\n"
        if upstream_text:
            output += "# --- UPSTREAM LLAMA.CPP ENGINE METRICS ---\n"
            output += upstream_text

        data = output.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class ThreadedHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def start_gateway_server(port: int):
    server = ThreadedHTTPServer(("0.0.0.0", port), GatewayRequestHandler)
    server.serve_forever()


def start_metrics_server(port: int):
    server = ThreadedHTTPServer(("0.0.0.0", port), PrometheusMetricsHandler)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description="Stage 07 LLM Ingress Telemetry Gateway")
    parser.add_argument("--gateway-port", type=int, default=DEFAULT_GATEWAY_PORT, help="Port for OpenAI proxy")
    parser.add_argument("--metrics-port", type=int, default=DEFAULT_METRICS_PORT, help="Port for Prometheus exporter")
    args = parser.parse_args()

    print("==========================================================================")
    print(" AI INFRA SYSTEMS - STAGE 07: LLM TELEMETRY & OBSERVABILITY GATEWAY")
    print("==========================================================================")
    print(f" Ingress Proxy Port:     http://0.0.0.0:{args.gateway_port} -> {UPSTREAM_URL}")
    print(f" Prometheus Metrics:     http://0.0.0.0:{args.metrics_port}/metrics")
    print(f" Tracing Span Log:       {TRACE_LOG_FILE}")
    print("==========================================================================")

    t_metrics = threading.Thread(target=start_metrics_server, args=(args.metrics_port,), daemon=True)
    t_metrics.start()

    try:
        start_gateway_server(args.gateway_port)
    except KeyboardInterrupt:
        print("\n[!] Shutting down telemetry gateway.")


if __name__ == "__main__":
    main()
