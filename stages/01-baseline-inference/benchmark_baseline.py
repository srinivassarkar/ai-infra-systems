#!/usr/bin/env python3
"""
benchmark_baseline.py - Stage 01 Baseline Inference Profiler
Author: Srinivas Sarkar (ai-infra-systems)

Measures and isolates TTFT, TPOT, and throughput across streaming OpenAI-compatible
chat completion endpoints (/v1/chat/completions). Zero heavy dependencies (uses standard library).
"""

import argparse
import json
import statistics
import time
import urllib.error
import urllib.request
from typing import Dict, List, Tuple


def profile_single_stream(
    endpoint: str, model: str, prompt: str, max_tokens: int = 128
) -> Dict:
    """Sends a single streaming chat completion request and records high-precision timestamps."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.2,
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=req_data,
        headers={"Content-Type": "application/json", "User-Agent": "ai-infra-bench/1.0"},
    )

    t_start = time.perf_counter()
    t_first_token = None
    token_timestamps: List[float] = []
    received_tokens = 0
    full_text = []

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
                            full_text.append(content)
                except json.JSONDecodeError:
                    continue

    except urllib.error.URLError as e:
        return {"error": str(e)}

    t_end = time.perf_counter()

    if t_first_token is None or received_tokens == 0:
        return {"error": "No tokens received from stream."}

    ttft_ms = (t_first_token - t_start) * 1000.0
    total_duration_s = t_end - t_start
    decode_duration_s = t_end - t_first_token

    # Inter-token latencies
    itls_ms: List[float] = []
    for i in range(1, len(token_timestamps)):
        itls_ms.append((token_timestamps[i] - token_timestamps[i - 1]) * 1000.0)

    tpot_ms = (decode_duration_s / (received_tokens - 1) * 1000.0) if received_tokens > 1 else 0.0
    throughput = received_tokens / total_duration_s if total_duration_s > 0 else 0.0

    return {
        "tokens": received_tokens,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "itls_ms": itls_ms,
        "throughput_tok_per_sec": throughput,
        "total_duration_s": total_duration_s,
    }


def main():
    parser = argparse.ArgumentParser(description="Profile baseline inference performance.")
    parser.add_argument(
        "--endpoint",
        default="http://localhost:11434/v1/chat/completions",
        help="Target OpenAI-compatible endpoint URL",
    )
    parser.add_argument("--model", default="qwen2.5-coder:1.5b", help="Model name identifier")
    parser.add_argument("--samples", type=int, default=5, help="Number of benchmark iterations")
    parser.add_argument(
        "--prompt",
        default="Explain the difference between prefill and decode phases in LLM inference in 80 words.",
        help="Prompt text to evaluate",
    )
    args = parser.parse_args()

    print(f"============================================================")
    print(f" AI INFRA SYSTEMS - STAGE 01: BASELINE INFERENCE BENCHMARK")
    print(f"============================================================")
    print(f" Endpoint: {args.endpoint}")
    print(f" Model:    {args.model}")
    print(f" Samples:  {args.samples}")
    print(f" Prompt:   \"{args.prompt}\"")
    print(f"------------------------------------------------------------\n")

    results = []

    for run_idx in range(1, args.samples + 1):
        print(f"[*] Executing iteration {run_idx}/{args.samples}...", end="", flush=True)
        res = profile_single_stream(args.endpoint, args.model, args.prompt)
        if "error" in res:
            print(f" FAILED: {res['error']}")
            continue

        results.append(res)
        print(
            f" Done! Tokens: {res['tokens']} | TTFT: {res['ttft_ms']:.1f}ms | TPOT: {res['tpot_ms']:.1f}ms | Throughput: {res['throughput_tok_per_sec']:.1f} tok/s"
        )
        time.sleep(0.5)

    if not results:
        print("\n[!] Error: No benchmark iterations succeeded. Ensure the model server is running.")
        return

    ttfts = [r["ttft_ms"] for r in results]
    tpots = [r["tpot_ms"] for r in results]
    throughputs = [r["throughput_tok_per_sec"] for r in results]

    print(f"\n============================================================")
    print(f" BENCHMARK SUMMARY (N = {len(results)})")
    print(f"============================================================")
    print(f" Metric                  |   Min   |   Avg   |   Max   |   P90   ")
    print(f"-------------------------+---------+---------+---------+---------")
    print(
        f" TTFT (ms)               | {min(ttfts):7.1f} | {statistics.mean(ttfts):7.1f} | {max(ttfts):7.1f} | {sorted(ttfts)[int(0.9 * len(ttfts))]:7.1f} "
    )
    print(
        f" TPOT (ms/token)         | {min(tpots):7.1f} | {statistics.mean(tpots):7.1f} | {max(tpots):7.1f} | {sorted(tpots)[int(0.9 * len(tpots))]:7.1f} "
    )
    print(
        f" Throughput (tokens/sec) | {min(throughputs):7.1f} | {statistics.mean(throughputs):7.1f} | {max(throughputs):7.1f} | {sorted(throughputs)[int(0.9 * len(throughputs))]:7.1f} "
    )
    print(f"============================================================")


if __name__ == "__main__":
    main()
