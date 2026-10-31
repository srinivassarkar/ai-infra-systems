#!/usr/bin/env python3
"""
calibrated_inference_server.py - Stage 09 Calibrated Kubernetes Inference Pod
Author: Srinivas Sarkar (ai-infra-systems)

Calibrated directly against NVIDIA L4 Stage 01-08 empirical benchmarks:
- Parallel Slots: 8 concurrent slots (llama-server --parallel 8)
- Prefill Latency: 32.36 ms (FlashAttention on)
- Decode Speed: 191.62 aggregate tok/s (~8.06 ms per token)
- Max Concurrency: 8 active requests before queueing in `requests_deferred`
- KV Cache: Preallocated 8 slots x 2048 ctx (0.125 usage ratio per slot)
- Graceful Drain: Supports preStop drain to eliminate dropped streams during KEDA scale-down
"""

import http.server
import socketserver
import time
import json
import threading
import sys
import os
import signal
from typing import Dict, List

PORT = 8000
MAX_SLOTS = 8
PREFILL_LATENCY_S = 0.032  # 32 ms prefill
DECODE_ITL_S = 0.008       # 8 ms per token decode

class InferenceEngineState:
    def __init__(self):
        self.lock = threading.Lock()
        self.active_slots = 0
        self.deferred_requests = 0
        self.draining = False
        self.total_tokens_predicted = 0
        self.total_prompt_tokens = 0

    def acquire_slot(self) -> bool:
        with self.lock:
            if self.draining:
                return False
            if self.active_slots < MAX_SLOTS:
                self.active_slots += 1
                return True
            else:
                self.deferred_requests += 1
                return False

    def release_slot(self):
        with self.lock:
            if self.active_slots > 0:
                self.active_slots -= 1

    def release_deferred(self):
        with self.lock:
            if self.deferred_requests > 0:
                self.deferred_requests -= 1

    @property
    def kv_cache_usage(self) -> float:
        with self.lock:
            return round(self.active_slots / float(MAX_SLOTS), 3)

STATE = InferenceEngineState()


class CalibratedHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        if self.path == "/health/live":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"OK")
        elif self.path == "/health/ready":
            if STATE.draining:
                self.send_response(503)
                self.end_headers()
                self.wfile.write(b"DRAINING")
            else:
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"READY")
        elif self.path == "/health/drain":
            # preStop hook endpoint
            with STATE.lock:
                STATE.draining = True
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"DRAIN_INITIATED")
        elif self.path == "/metrics":
            self.handle_metrics()
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            self.handle_inference()
        else:
            self.send_response(404)
            self.end_headers()

    def handle_metrics(self):
        with STATE.lock:
            active = STATE.active_slots
            deferred = STATE.deferred_requests
            kv_ratio = STATE.kv_cache_usage
            tokens = STATE.total_tokens_predicted

        metrics_text = (
            f"# HELP llamacpp:requests_processing Active inference slots\n"
            f"# TYPE llamacpp:requests_processing gauge\n"
            f"llamacpp:requests_processing {active}\n"
            f"# HELP llamacpp:requests_deferred Requests queued in waiting line\n"
            f"# TYPE llamacpp:requests_deferred gauge\n"
            f"llamacpp:requests_deferred {deferred}\n"
            f"# HELP llamacpp:kv_cache_usage_ratio KV cache pool saturation (0.0 - 1.0)\n"
            f"# TYPE llamacpp:kv_cache_usage_ratio gauge\n"
            f"llamacpp:kv_cache_usage_ratio {kv_ratio}\n"
            f"# HELP llamacpp:tokens_predicted_total Total generated tokens\n"
            f"# TYPE llamacpp:tokens_predicted_total counter\n"
            f"llamacpp:tokens_predicted_total {tokens}\n"
        )
        data = metrics_text.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def handle_inference(self):
        content_len = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_len) if content_len > 0 else b"{}"
        try:
            payload = json.loads(raw_body.decode())
        except Exception:
            payload = {}

        max_tokens = payload.get("max_tokens", 32)
        stream = payload.get("stream", True)

        # Attempt to acquire an active execution slot
        slot_acquired = STATE.acquire_slot()
        if not slot_acquired:
            # Model queue wait loop
            t_wait_start = time.perf_counter()
            while not STATE.acquire_slot():
                time.sleep(0.01)
                if time.perf_counter() - t_wait_start > 10.0:
                    STATE.release_deferred()
                    self.send_response(503)
                    self.end_headers()
                    self.wfile.write(b"Queue timeout")
                    return
            STATE.release_deferred()

        try:
            # Simulate Prefill Phase (Compute bound)
            time.sleep(PREFILL_LATENCY_S)

            self.send_response(200)
            if stream:
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                # Simulate Autoregressive Decode Phase
                for i in range(max_tokens):
                    time.sleep(DECODE_ITL_S)
                    chunk_payload = json.dumps({
                        "choices": [{"delta": {"content": f" tok_{i}"}}]
                    })
                    chunk_body = f"data: {chunk_payload}\n\n".encode("utf-8")
                    self.wfile.write(f"{len(chunk_body):X}\r\n".encode() + chunk_body + b"\r\n")
                    self.wfile.flush()
                    with STATE.lock:
                        STATE.total_tokens_predicted += 1

                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            else:
                total_decode_time = max_tokens * DECODE_ITL_S
                time.sleep(total_decode_time)
                resp_body = json.dumps({"choices": [{"message": {"content": "completed"}}]}).encode("utf-8")
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(resp_body)))
                self.end_headers()
                self.wfile.write(resp_body)
                with STATE.lock:
                    STATE.total_tokens_predicted += max_tokens
        finally:
            STATE.release_slot()


class ThreadedServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    print("==========================================================================")
    print(" AI INFRA SYSTEMS - STAGE 09: CALIBRATED INFERENCE POD")
    print("==========================================================================")
    print(f" Port:             {PORT}")
    print(f" Calibrated Fleet: NVIDIA L4 (Max Slots={MAX_SLOTS}, TTFT=32ms, TPOT=8ms)")
    print(" Prometheus:       http://0.0.0.0:8000/metrics")
    print(" Health Ready:     http://0.0.0.0:8000/health/ready")
    print(" Drain Hook:       http://0.0.0.0:8000/health/drain")
    print("==========================================================================")
    server = ThreadedServer(("0.0.0.0", PORT), CalibratedHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
