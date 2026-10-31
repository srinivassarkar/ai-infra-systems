#!/usr/bin/env python3
"""
benchmark_streaming.py - Stage 05 Edge Streaming & Buffer Bloat Profiler
Author: Srinivas Sarkar (ai-infra-systems)

Measures the impact of reverse proxy buffering on Server-Sent Events (SSE) streaming:
- Time-to-First-Token (TTFT)
- Inter-Token Latency (ITL) distribution and jitter
- Tokens per second and chunk burstiness
"""

import asyncio
import time
import json
import statistics
import argparse
import sys
from typing import List, Dict, Optional

try:
    import httpx
except ImportError:
    print("[!] httpx is required: pip install httpx")
    sys.exit(1)

DEFAULT_URL = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_MODEL = "Qwen2.5-Coder-1.5B"
PROMPT = "Explain the difference between TCP TIME_WAIT and CLOSE_WAIT in three crisp bullet points with deep technical detail."


async def send_stream(client: httpx.AsyncClient, user_id: int, url: str, model: str, max_tokens: int) -> Optional[Dict]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.1
    }
    
    start_time = time.perf_counter()
    first_token_time = None
    token_timestamps: List[float] = []
    chunk_sizes: List[int] = []
    total_tokens = 0

    try:
        async with client.stream("POST", url, json=payload, timeout=60.0) as resp:
            if resp.status_code != 200:
                print(f"[User {user_id}] HTTP {resp.status_code}")
                return None
                
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                data_str = line[6:].strip()
                if data_str == "[DONE]":
                    break
                
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        now = time.perf_counter()
                        if first_token_time is None:
                            first_token_time = now
                        token_timestamps.append(now)
                        chunk_sizes.append(len(content))
                        total_tokens += 1
                except Exception:
                    continue

        end_time = time.perf_counter()
        if first_token_time is None or total_tokens < 2:
            return None

        ttft_ms = (first_token_time - start_time) * 1000.0
        gen_duration = end_time - first_token_time
        tps = total_tokens / gen_duration if gen_duration > 0 else 0.0
        
        itls = [
            (token_timestamps[i] - token_timestamps[i-1]) * 1000.0 
            for i in range(1, len(token_timestamps))
        ]
        avg_itl = statistics.mean(itls) if itls else 0.0
        p99_itl = sorted(itls)[int(len(itls) * 0.99)] if itls else 0.0
        max_itl = max(itls) if itls else 0.0

        return {
            "user": user_id,
            "tokens": total_tokens,
            "ttft_ms": ttft_ms,
            "tps": tps,
            "avg_itl_ms": avg_itl,
            "p99_itl_ms": p99_itl,
            "max_itl_ms": max_itl,
            "total_s": end_time - start_time
        }
    except Exception as e:
        print(f"[User {user_id}] Error: {e}")
        return None


async def run_benchmark(concurrency: int, url: str, model: str, max_tokens: int):
    print(f"============================================================")
    print(f"STAGE 05 STREAMING BENCHMARK: Concurrency = {concurrency}")
    print(f"Target Endpoint: {url}")
    print(f"============================================================")
    
    limits = httpx.Limits(max_keepalive_connections=concurrency, max_connections=concurrency * 2)
    async with httpx.AsyncClient(limits=limits) as client:
        tasks = [send_stream(client, i, url=url, model=model, max_tokens=max_tokens) for i in range(concurrency)]
        results = await asyncio.gather(*tasks)

    valid = [r for r in results if r is not None]
    if not valid:
        print(f"[ERROR] All requests failed. Is your Nginx proxy running at {url}?")
        return

    ttfts = sorted([r["ttft_ms"] for r in valid])
    tpss = [r["tps"] for r in valid]
    itls = [r["avg_itl_ms"] for r in valid]
    p99_itls = [r["p99_itl_ms"] for r in valid]
    max_itls = [r["max_itl_ms"] for r in valid]

    print(f"Success Rate:            {len(valid)}/{concurrency} ({(len(valid)/concurrency)*100:.1f}%)")
    print(f"Average Tokens Emitted:  {statistics.mean([r['tokens'] for r in valid]):.1f}")
    print(f"Median TTFT:             {statistics.median(ttfts):.2f} ms")
    print(f"P99 TTFT:                {ttfts[int(len(ttfts)*0.99)]:.2f} ms")
    print(f"Median ITL:              {statistics.median(itls):.2f} ms")
    print(f"P99 ITL (Tail Jitter):   {statistics.median(p99_itls):.2f} ms")
    print(f"Max ITL (Burst Stutter): {max(max_itls):.2f} ms")
    print(f"Aggregate Throughput:    {sum(tpss):.2f} tokens/sec")
    print(f"============================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Stage 05 Streaming Benchmark.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Target URL (e.g. http://127.0.0.1:8000/v1/chat/completions)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model name")
    parser.add_argument("--concurrency", type=int, default=4, help="Number of concurrent streams")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate")
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.concurrency, args.url, args.model, args.max_tokens))


if __name__ == "__main__":
    main()
