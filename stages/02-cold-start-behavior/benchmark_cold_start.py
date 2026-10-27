#!/usr/bin/env python3
"""
benchmark_cold_start.py - Stage 02 Cold-Start & Virtual Memory Paging Profiler
Author: Srinivas Sarkar (ai-infra-systems)

Measures Time-to-Ready (t_ready), First-Request TTFT, and Total Cold-Start Latency
across --mmap vs --no-mmap and --warmup vs --no-warmup conditions.
Executes locally against a dedicated llama-server instance to eliminate network noise.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple


def poll_health_endpoint(url: str, timeout_s: float = 60.0) -> Tuple[bool, float]:
    """Polls http://host:port/health until HTTP 200 is returned. Returns (success, duration_s)."""
    start = time.perf_counter()
    while time.perf_counter() - start < timeout_s:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "cold-start-bench/1.0"})
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    return True, time.perf_counter() - start
        except (urllib.error.HTTPError, urllib.error.URLError, ConnectionRefusedError, OSError):
            pass
        time.sleep(0.01)  # 10ms polling interval
    return False, time.perf_counter() - start


def send_inference_request(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int = 64,
    cache_prompt: bool = False,
) -> Dict:
    """Sends a streaming chat completion request and measures high-precision TTFT and TPOT."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "cache_prompt": cache_prompt,
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=req_data,
        headers={"Content-Type": "application/json", "User-Agent": "cold-start-bench/1.0"},
    )

    t_start = time.perf_counter()
    t_first_token = None
    token_timestamps: List[float] = []
    received_tokens = 0

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            for line in resp:
                now = time.perf_counter()
                line = line.decode("utf-8").strip()

                if not line or not line.startswith("data:"):
                    continue

                data_str = line[len("data:") :].strip()
                if data_str == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                    choices = chunk.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if t_first_token is None:
                                t_first_token = now
                            token_timestamps.append(now)
                            received_tokens += 1
                except json.JSONDecodeError:
                    continue

    except urllib.error.URLError as e:
        return {"error": str(e)}

    t_end = time.perf_counter()

    if t_first_token is None or received_tokens == 0:
        return {"error": "No tokens received from stream."}

    ttft_ms = (t_first_token - t_start) * 1000.0
    decode_duration_s = t_end - t_first_token
    tpot_ms = (decode_duration_s / (received_tokens - 1) * 1000.0) if received_tokens > 1 else 0.0
    decode_speed = ((received_tokens - 1) / decode_duration_s) if decode_duration_s > 0 and received_tokens > 1 else 0.0

    return {
        "tokens": received_tokens,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "decode_speed_tok_s": decode_speed,
        "total_duration_s": t_end - t_start,
    }


def run_single_trial(
    server_bin: str,
    model_path: str,
    ngl: int = 99,
    ctx_size: int = 2048,
    port: int = 8080,
    host: str = "127.0.0.1",
    mmap: bool = True,
    warmup: bool = True,
    prompt: str = "Explain the difference between virtual memory paging and eager memory allocation in 50 words.",
) -> Dict:
    """Spawns llama-server, times health transition, executes virgin request, and terminates."""
    cmd = [
        server_bin,
        "-m", model_path,
        "-ngl", str(ngl),
        "-c", str(ctx_size),
        "--port", str(port),
        "--host", host,
    ]

    if not mmap:
        cmd.append("--no-mmap")
    if not warmup:
        cmd.append("--no-warmup")

    print(f"[*] Spawning server: {' '.join(cmd)}")
    t_process_start = time.perf_counter()

    # Launch daemon
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    health_url = f"http://{host}:{port}/health"
    completions_url = f"http://{host}:{port}/v1/chat/completions"

    healthy, ready_duration_s = poll_health_endpoint(health_url, timeout_s=90.0)
    if not healthy:
        proc.kill()
        return {"error": f"Server failed to become healthy within 90s. Status: {proc.poll()}"}

    print(f"[*] Server healthy in {ready_duration_s * 1000.0:.1f}ms. Sending first-touch inference request...")

    # First request (Virgin - touches mapped pages/initializes kernels)
    req1 = send_inference_request(completions_url, "test-model", prompt, max_tokens=48, cache_prompt=False)
    if "error" in req1:
        proc.kill()
        return {"error": f"First inference request failed: {req1['error']}"}

    # Second request (Warm - steady-state comparison)
    # Use a varied prompt to ensure prompt caching doesn't mask warm decode
    prompt2 = "Summarize the primary benefits of zero-copy file mapping for database storage engines in 50 words."
    req2 = send_inference_request(completions_url, "test-model", prompt2, max_tokens=48, cache_prompt=False)

    # Terminate daemon gracefully
    proc.terminate()
    try:
        proc.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        proc.kill()

    t_first_useful_token_ms = (ready_duration_s * 1000.0) + req1["ttft_ms"]

    return {
        "mmap": mmap,
        "warmup": warmup,
        "t_ready_ms": ready_duration_s * 1000.0,
        "first_ttft_ms": req1["ttft_ms"],
        "first_tpot_ms": req1["tpot_ms"],
        "first_useful_token_ms": t_first_useful_token_ms,
        "warm_ttft_ms": req2.get("ttft_ms", 0.0),
        "warm_tpot_ms": req2.get("tpot_ms", 0.0),
    }


def main():
    parser = argparse.ArgumentParser(description="Profile Stage 02 Cold-Start and Virtual Memory Paging.")
    parser.add_argument("--server-bin", required=True, help="Absolute path to llama-server executable")
    parser.add_argument("--model", required=True, help="Absolute path to GGUF model file")
    parser.add_argument("--ngl", type=int, default=99, help="Number of GPU layers to offload")
    parser.add_argument("--ctx-size", type=int, default=2048, help="Context size")
    parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host IP to bind")
    parser.add_argument("--no-mmap", action="store_true", help="Disable mmap (force eager heap allocation)")
    parser.add_argument("--no-warmup", action="store_true", help="Disable server startup warmup pass")
    args = parser.parse_args()

    mmap_enabled = not args.no_mmap
    warmup_enabled = not args.no_warmup

    print("==========================================================================")
    print(" AI INFRA SYSTEMS - STAGE 02: COLD-START & VIRTUAL MEMORY PROFILER")
    print("==========================================================================")
    print(f" Server Bin:    {args.server_bin}")
    print(f" Model:         {args.model}")
    print(f" Mode:          mmap={'ON' if mmap_enabled else 'OFF'} | warmup={'ON' if warmup_enabled else 'OFF'}")
    print("--------------------------------------------------------------------------\n")

    res = run_single_trial(
        server_bin=args.server_bin,
        model_path=args.model,
        ngl=args.ngl,
        ctx_size=args.ctx_size,
        port=args.port,
        host=args.host,
        mmap=mmap_enabled,
        warmup=warmup_enabled,
    )

    if "error" in res:
        print(f"[!] Benchmark Failed: {res['error']}")
        sys.exit(1)

    print("\n==========================================================================")
    print(" COLD-START TRIAL SUMMARY")
    print("==========================================================================")
    print(f" Condition:                 mmap={'ON' if res['mmap'] else 'OFF'}, warmup={'ON' if res['warmup'] else 'OFF'}")
    print(f" Time-to-Ready (t_ready):    {res['t_ready_ms']:8.2f} ms")
    print(f" First-Request TTFT:        {res['first_ttft_ms']:8.2f} ms")
    print(f" Time-to-First-Useful-Token:{res['first_useful_token_ms']:8.2f} ms (Total Cold Start)")
    print(f" Steady-State Warm TTFT:    {res['warm_ttft_ms']:8.2f} ms")
    print(f" Warm TPOT (ms/token):      {res['warm_tpot_ms']:8.2f} ms")
    print("==========================================================================")


if __name__ == "__main__":
    main()
