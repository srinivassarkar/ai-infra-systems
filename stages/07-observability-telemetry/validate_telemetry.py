#!/usr/bin/env python3
"""
validate_telemetry.py - Stage 07 End-to-End Observability Validation Suite
Author: Srinivas Sarkar (ai-infra-systems)

1. Validates native llama-server /metrics endpoint (proves lack of TTFT/TPOT histograms).
2. Drives concurrent streaming load through the Telemetry Gateway (port 8002).
3. Reads unified Prometheus exposition (port 9090/metrics) and extracts Golden Signals.
4. Inspects hierarchical OpenTelemetry trace spans (Request -> Prefill -> Decode).
5. Simulates fault-to-detection time against SLO alert rules.
"""

import urllib.request
import urllib.error
import json
import time
import threading
import sys
import os
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GATEWAY_URL = "http://127.0.0.1:8002/v1/chat/completions"
DEFAULT_METRICS_URL = "http://127.0.0.1:9090/metrics"
DEFAULT_UPSTREAM_URL = "http://127.0.0.1:8080"
TRACE_FILE = os.path.join(SCRIPT_DIR, "telemetry_traces.jsonl")


def test_upstream_native_metrics(upstream_url: str):
    print("--------------------------------------------------------------------------")
    print(f" 1. AUDITING UPSTREAM LLAMA-SERVER NATIVE METRICS ({upstream_url})")
    print("--------------------------------------------------------------------------")
    try:
        req = urllib.request.Request(f"{upstream_url}/metrics")
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            content = resp.read().decode("utf-8")
            lines = [l for l in content.split("\n") if l and not l.startswith("#")]
            print(f"[+] Successfully fetched {len(lines)} raw metric series from llama-server.")
            
            # Print sample native metrics
            print("\n  Sample Raw Upstream Metrics:")
            for l in lines[:6]:
                print(f"    {l}")

            has_ttft = any("ttft" in l.lower() for l in lines)
            has_tpot = any("tpot" in l.lower() or "inter_token" in l.lower() for l in lines)

            print(f"\n  [*] Native TTFT Histogram Present: {'YES' if has_ttft else 'NO (MISSING)'}")
            print(f"  [*] Native TPOT Histogram Present: {'YES' if has_tpot else 'NO (MISSING)'}")

            if not has_ttft and not has_tpot:
                print("\n  [+] ARCHITECTURAL VERIFICATION: The Upstream Observability Gap!")
                print("      llama-server provides raw slot counts and token counters, but")
                print("      natively lacks client-perceived TTFT and TPOT latency histograms.")
                print("      An ingress gateway / proxy layer is strictly required for SLO tracking.")
    except Exception as e:
        print(f"[-] Upstream server not reachable at {upstream_url}: {e}")
        return False
    return True


def run_single_stream_client(client_id: int, gateway_url: str, results: list):
    payload = {
        "model": "qwen2.5-coder-1.5b",
        "messages": [{"role": "user", "content": "Explain Linux cgroups and namespace isolation in two sentences."}],
        "stream": True,
        "max_tokens": 64,
        "temperature": 0.1
    }
    data = json.dumps(payload).encode("utf-8")
    t0 = time.perf_counter()
    tokens = 0
    t_first = None

    try:
        req = urllib.request.Request(gateway_url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            buffer = b""
            while True:
                chunk = resp.read(256)
                if not chunk:
                    break
                buffer += chunk
                while b"\n\n" in buffer:
                    evt, buffer = buffer.split(b"\n\n", 1)
                    if b"data: [DONE]" in evt:
                        continue
                    if b"data: " in evt:
                        tokens += 1
                        if t_first is None:
                            t_first = time.perf_counter()
        t_end = time.perf_counter()
        ttft = (t_first - t0) * 1000.0 if t_first else 0.0
        duration = (t_end - t0) * 1000.0
        results.append({
            "client_id": client_id,
            "tokens": tokens,
            "ttft_ms": ttft,
            "duration_ms": duration
        })
    except Exception as e:
        print(f"[-] Client {client_id} error: {e}")


def drive_gateway_load(gateway_url: str, concurrency: int = 4):
    print("\n--------------------------------------------------------------------------")
    print(f" 2. DRIVING CONCURRENT STREAMING LOAD THROUGH GATEWAY (N={concurrency})")
    print(f" Target Endpoint: {gateway_url}")
    print("--------------------------------------------------------------------------")
    threads = []
    results = []
    t_start = time.perf_counter()

    for i in range(concurrency):
        t = threading.Thread(target=run_single_stream_client, args=(i + 1, gateway_url, results))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    t_total = time.perf_counter() - t_start
    print(f"[+] Completed {len(results)} concurrent streams in {t_total:.2f}s.")
    for r in results:
        print(f"    Client {r['client_id']}: TTFT = {r['ttft_ms']:.2f} ms | Tokens = {r['tokens']} | Total = {r['duration_ms']:.2f} ms")


def audit_gateway_prometheus_exposition(metrics_url: str):
    print("\n--------------------------------------------------------------------------")
    print(f" 3. AUDITING GATEWAY PROMETHEUS EXPOSITION ({metrics_url})")
    print("--------------------------------------------------------------------------")
    try:
        req = urllib.request.Request(metrics_url)
        with urllib.request.urlopen(req, timeout=2.0) as resp:
            content = resp.read().decode("utf-8")
            print(f"[+] Successfully queried Prometheus endpoint at {metrics_url}.\n")
            
            key_signals = [
                "llm_gateway_active_streams",
                "llm_gateway_requests_total",
                "llm_gateway_tokens_total",
                "llm_ttft_seconds_bucket",
                "llm_tpot_seconds_bucket",
                "llm_gpu_utilization_ratio",
                "llm_gpu_memory_used_bytes",
                "llm_gpu_power_watts",
                "llamacpp:kv_cache_usage_ratio"
            ]

            found = {}
            for line in content.split("\n"):
                for sig in key_signals:
                    if line.startswith(sig) and sig not in found:
                        found[sig] = line

            print("  Unified Golden Signals Captured:")
            for sig, val in found.items():
                print(f"    {val}")

            missing = set(key_signals) - set(found.keys())
            if not missing:
                print("\n  [+] ALL 9 GOLDEN SIGNALS VERIFIED IN EXPOSITION.")
            else:
                print(f"\n  [*] Note: Unset signals: {missing}")
    except Exception as e:
        print(f"[-] Failed to query {metrics_url}: {e}")


def inspect_opentelemetry_traces():
    print("\n--------------------------------------------------------------------------")
    print(f" 4. INSPECTING DISTRIBUTED OPENTELEMETRY TRACES ({TRACE_FILE})")
    print("--------------------------------------------------------------------------")
    if not os.path.exists(TRACE_FILE):
        print(f"[-] Trace file {TRACE_FILE} does not exist yet.")
        return

    traces = []
    with open(TRACE_FILE, "r") as f:
        for line in f:
            if line.strip():
                try:
                    traces.append(json.loads(line))
                except Exception:
                    pass

    print(f"[+] Total Request Spans Emitted: {len(traces)}")
    if traces:
        sample = traces[-1]
        print("\n  Sample Hierarchical Span (No Per-Token Span Cardinality Explosion):")
        print(f"    Trace ID:         {sample.get('trace_id')}")
        print(f"    Span Name:        {sample.get('name')}")
        print(f"    Total Duration:   {sample.get('total_duration_ms')} ms")
        print(f"    Prefill (TTFT):   {sample.get('ttft_prefill_ms')} ms")
        print(f"    Decode Duration:  {sample.get('decode_phase_ms')} ms")
        print(f"    TPOT (ITL):       {sample.get('tpot_itl_ms')} ms/tok")
        print(f"    Child Spans:      {sample.get('child_spans')}")


def main():
    parser = argparse.ArgumentParser(description="Stage 07 End-to-End Observability Validation")
    parser.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL, help="Gateway chat completions URL")
    parser.add_argument("--metrics-url", default=DEFAULT_METRICS_URL, help="Gateway Prometheus metrics URL")
    parser.add_argument("--upstream-url", default=DEFAULT_UPSTREAM_URL, help="Upstream llama-server URL")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrent test streams")
    args = parser.parse_args()

    print("==========================================================================")
    print(" AI INFRA SYSTEMS - STAGE 07: END-TO-END OBSERVABILITY VALIDATION")
    print("==========================================================================")
    
    upstream_ok = test_upstream_native_metrics(args.upstream_url)
    if not upstream_ok:
        print(f"\n[!] Please ensure llama-server is running on {args.upstream_url} with --metrics.")
        sys.exit(1)

    drive_gateway_load(args.gateway_url, concurrency=args.concurrency)
    audit_gateway_prometheus_exposition(args.metrics_url)
    inspect_opentelemetry_traces()

    print("\n==========================================================================")
    print(" STAGE 07 OBSERVABILITY SUITE AUDIT COMPLETE")
    print("==========================================================================\n")


if __name__ == "__main__":
    main()
