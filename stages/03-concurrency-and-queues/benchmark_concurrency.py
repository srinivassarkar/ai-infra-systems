#!/usr/bin/env python3
"""
benchmark_concurrency.py - Stage 03 Concurrency, Queueing & Head-of-Line Profiler
Author: Srinivas Sarkar (ai-infra-systems)

Supports:
1. Closed-Loop Concurrency (N = 1, 2, 4, 8, 16 concurrent worker loops)
2. Open-Loop Poisson Arrivals (independent arrival rate lambda req/s)
3. Mixed-Workload Intruder Injection (short requests + 1,500 token intruder)
4. Full Inter-Token Latency (ITL) and Tail Percentile (P50, P90, P99) Tracking
5. Little's Law Validation (Concurrency = Throughput * Latency)
6. Raw Per-Request CSV Export

Requires: httpx (pip install httpx)
"""

import argparse
import asyncio
import csv
import json
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

try:
    import httpx
except ImportError:
    print("[!] httpx is required for async load generation: pip install httpx")
    sys.exit(1)


def percentile(data: List[float], p: float) -> float:
    """Computes the p-th percentile (0 <= p <= 100) using linear interpolation."""
    if not data:
        return 0.0
    k = (len(data) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    return data[int(f)] * (c - k) + data[int(c)] * (k - f)


class RequestRecord:
    def __init__(self, req_id: int, prompt_type: str = "standard"):
        self.req_id = req_id
        self.prompt_type = prompt_type
        self.t_dispatched: float = 0.0
        self.t_first_token: Optional[float] = None
        self.t_completed: Optional[float] = None
        self.token_timestamps: List[float] = []
        self.output_tokens: int = 0
        self.prompt_tokens: int = 0
        self.error: Optional[str] = None

    @property
    def ttft_ms(self) -> float:
        if self.t_first_token and self.t_dispatched:
            return (self.t_first_token - self.t_dispatched) * 1000.0
        return 0.0

    @property
    def total_latency_ms(self) -> float:
        if self.t_completed and self.t_dispatched:
            return (self.t_completed - self.t_dispatched) * 1000.0
        return 0.0

    @property
    def tpot_ms(self) -> float:
        if self.t_first_token and self.t_completed and self.output_tokens > 1:
            return ((self.t_completed - self.t_first_token) / (self.output_tokens - 1)) * 1000.0
        return 0.0

    @property
    def decode_speed_tok_s(self) -> float:
        if self.tpot_ms > 0:
            return 1000.0 / self.tpot_ms
        return 0.0

    @property
    def itl_list_ms(self) -> List[float]:
        """Calculates inter-token intervals in milliseconds."""
        if len(self.token_timestamps) < 2:
            return []
        return [
            (self.token_timestamps[i] - self.token_timestamps[i - 1]) * 1000.0
            for i in range(1, len(self.token_timestamps))
        ]

    @property
    def itl_p99_ms(self) -> float:
        itls = sorted(self.itl_list_ms)
        return percentile(itls, 99.0) if itls else 0.0


async def send_streaming_request(
    client: httpx.AsyncClient,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    req_id: int,
    prompt_type: str = "standard",
) -> RequestRecord:
    record = RequestRecord(req_id=req_id, prompt_type=prompt_type)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "cache_prompt": False,  # Prevent prefix caching from flattening load
    }

    record.t_dispatched = time.perf_counter()

    try:
        async with client.stream(
            "POST",
            endpoint,
            json=payload,
            timeout=180.0,
            headers={"Content-Type": "application/json", "User-Agent": "ai-infra-loadgen/1.0"},
        ) as response:
            if response.status_code != 200:
                record.error = f"HTTP {response.status_code}"
                record.t_completed = time.perf_counter()
                return record

            async for line in response.aiter_lines():
                now = time.perf_counter()
                line = line.strip()
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
                        if delta.get("content"):
                            if record.t_first_token is None:
                                record.t_first_token = now
                            record.token_timestamps.append(now)
                            record.output_tokens += 1
                except json.JSONDecodeError:
                    pass

        record.t_completed = time.perf_counter()
    except Exception as e:
        record.error = str(e)
        record.t_completed = time.perf_counter()

    return record


# -------------------------------------------------------------------------
# Load Generators: Closed-Loop vs. Open-Loop Poisson
# -------------------------------------------------------------------------

async def run_closed_loop(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
    duration_s: float,
) -> List[RequestRecord]:
    """Generates closed-loop load with N concurrent workers."""
    records: List[RequestRecord] = []
    stop_event = asyncio.Event()
    req_counter = 0

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrency * 2)) as client:
        async def worker(worker_id: int):
            nonlocal req_counter
            while not stop_event.is_set():
                req_counter += 1
                curr_id = req_counter
                rec = await send_streaming_request(
                    client, endpoint, model, prompt, max_tokens, curr_id, "standard"
                )
                records.append(rec)

        workers = [asyncio.create_task(worker(i)) for i in range(concurrency)]

        await asyncio.sleep(duration_s)
        stop_event.set()
        await asyncio.gather(*workers, return_exceptions=True)

    return records


async def run_open_loop(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    arrival_rate: float,
    duration_s: float,
) -> List[RequestRecord]:
    """Generates open-loop load with Poisson arrivals (mean arrival rate lambda req/s)."""
    records: List[RequestRecord] = []
    active_tasks = []
    t_start = time.perf_counter()
    req_counter = 0

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=500)) as client:
        while time.perf_counter() - t_start < duration_s:
            req_counter += 1
            curr_id = req_counter

            # Dispatch task without waiting
            task = asyncio.create_task(
                send_streaming_request(
                    client, endpoint, model, prompt, max_tokens, curr_id, "standard"
                )
            )
            active_tasks.append(task)

            # Sample inter-arrival interval from exponential distribution (Poisson process)
            inter_arrival = random.expovariate(arrival_rate)
            await asyncio.sleep(inter_arrival)

        # Await completion of in-flight requests (drain)
        print(f"[*] Draining {len(active_tasks)} open-loop requests...")
        done_records = await asyncio.gather(*active_tasks, return_exceptions=False)
        records.extend(done_records)

    return records


async def run_mixed_workload(
    endpoint: str,
    model: str,
    short_prompt: str,
    long_prompt: str,
    background_concurrency: int,
    duration_s: float,
) -> List[RequestRecord]:
    """Runs background short streams and injects a 1,500-token intruder prompt mid-run."""
    records: List[RequestRecord] = []
    stop_event = asyncio.Event()
    req_counter = 0

    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=50)) as client:
        async def background_worker(worker_id: int):
            nonlocal req_counter
            while not stop_event.is_set():
                req_counter += 1
                curr_id = req_counter
                rec = await send_streaming_request(
                    client, endpoint, model, short_prompt, 128, curr_id, "background_short"
                )
                records.append(rec)

        # Start background workers
        workers = [asyncio.create_task(background_worker(i)) for i in range(background_concurrency)]

        # Let background run for 10 seconds to establish baseline
        print("[*] Running background traffic (10s baseline)...")
        await asyncio.sleep(10.0)

        # Inject Intruder
        print("[!] >>> INJECTING 1,500-TOKEN INTRUDER PROMPT <<<")
        req_counter += 1
        intruder_id = req_counter
        intruder_task = asyncio.create_task(
            send_streaming_request(
                client, endpoint, model, long_prompt, 64, intruder_id, "intruder_long"
            )
        )

        await intruder_task
        print("[*] Intruder completed. Continuing background traffic...")
        records.append(intruder_task.result())

        # Let background run for remainder of duration
        remaining = max(5.0, duration_s - 15.0)
        await asyncio.sleep(remaining)
        stop_event.set()
        await asyncio.gather(*workers, return_exceptions=True)

    return records


# -------------------------------------------------------------------------
# Telemetry Analytics & Reporting
# -------------------------------------------------------------------------

def analyze_and_print_results(
    records: List[RequestRecord],
    mode_name: str,
    target_metric_val: float,
    duration_s: float,
    csv_filename: str = "stage03_telemetry.csv",
):
    valid_records = [r for r in records if r.error is None and r.output_tokens > 1]
    failed_count = len(records) - len(valid_records)

    if not valid_records:
        print(f"[!] All {len(records)} requests failed!")
        return

    # Extract latency lists
    ttfts = sorted([r.ttft_ms for r in valid_records])
    tpots = sorted([r.tpot_ms for r in valid_records])
    latencies_s = [(r.t_completed - r.t_dispatched) for r in valid_records]
    all_itls = []
    for r in valid_records:
        all_itls.extend(r.itl_list_ms)
    all_itls.sort()

    total_tokens_generated = sum(r.output_tokens for r in valid_records)
    actual_test_duration = max(r.t_completed for r in valid_records) - min(r.t_dispatched for r in valid_records)
    aggregate_tok_s = total_tokens_generated / actual_test_duration if actual_test_duration > 0 else 0.0
    request_throughput_req_s = len(valid_records) / actual_test_duration if actual_test_duration > 0 else 0.0
    avg_latency_s = sum(latencies_s) / len(latencies_s)

    # Little's Law check: Concurrency L = Throughput * Latency
    littles_law_concurrency = request_throughput_req_s * avg_latency_s

    # SLO Goodput: % of requests meeting TTFT < 500ms AND TPOT < 50ms
    slo_met = sum(1 for r in valid_records if r.ttft_ms < 500.0 and r.tpot_ms < 50.0)
    slo_goodput_pct = (slo_met / len(valid_records)) * 100.0

    print("\n" + "=" * 76)
    print(f" STAGE 03 CONCURRENCY BENCHMARK: {mode_name.upper()} = {target_metric_val}")
    print("=" * 76)
    print(f" Completed Requests:  {len(valid_records)} (Errors: {failed_count})")
    print(f" Total Output Tokens: {total_tokens_generated} tokens")
    print(f" Aggregate Throughput:{aggregate_tok_s:8.2f} tokens/sec")
    print(f" Request Throughput:  {request_throughput_req_s:8.2f} requests/sec")
    print(f" SLO Goodput:         {slo_goodput_pct:8.1f}% (TTFT < 500ms, TPOT < 50ms)")
    print(f" Little's Law (L):    {littles_law_concurrency:8.2f} (Target Concurrency: {target_metric_val})")
    print("-" * 76)
    print(f" METRIC                |    P50    |    P90    |    P99    |    Max    ")
    print("-" * 76)
    print(f" TTFT (ms)             | {percentile(ttfts, 50):9.2f} | {percentile(ttfts, 90):9.2f} | {percentile(ttfts, 99):9.2f} | {max(ttfts):9.2f} ")
    print(f" TPOT (ms/token)       | {percentile(tpots, 50):9.2f} | {percentile(tpots, 90):9.2f} | {percentile(tpots, 99):9.2f} | {max(tpots):9.2f} ")
    if all_itls:
        print(f" Inter-Token Lat (ITL) | {percentile(all_itls, 50):9.2f} | {percentile(all_itls, 90):9.2f} | {percentile(all_itls, 99):9.2f} | {max(all_itls):9.2f} ")
    print("=" * 76)

    # Export raw data to CSV
    file_exists = os.path.isfile(csv_filename)
    with open(csv_filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "req_id", "prompt_type", "t_dispatched", "output_tokens",
                "ttft_ms", "tpot_ms", "total_latency_ms", "itl_p99_ms", "mode", "target_val"
            ])
        for r in valid_records:
            writer.writerow([
                r.req_id, r.prompt_type, f"{r.t_dispatched:.4f}", r.output_tokens,
                f"{r.ttft_ms:.2f}", f"{r.tpot_ms:.2f}", f"{r.total_latency_ms:.2f}",
                f"{r.itl_p99_ms:.2f}", mode_name, target_metric_val
            ])
    print(f"[*] Raw per-request telemetry exported to: {csv_filename}\n")


# -------------------------------------------------------------------------
# Main CLI Entry Point
# -------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 03 Concurrency & Queueing Benchmark.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/chat/completions", help="Endpoint URL")
    parser.add_argument("--model", default="Qwen2.5-Coder-1.5B", help="Model identifier")
    parser.add_argument("--mode", choices=["closed", "open", "intruder"], default="closed", help="Test mode")
    parser.add_argument("--concurrency", type=int, default=4, help="Concurrency for closed-loop mode")
    parser.add_argument("--rate", type=float, default=2.0, help="Arrival rate lambda (req/s) for open-loop mode")
    parser.add_argument("--duration", type=float, default=30.0, help="Test duration in seconds")
    parser.add_argument("--max-tokens", type=int, default=64, help="Tokens to generate per request")
    parser.add_argument("--csv", default="stage03_telemetry.csv", help="CSV export destination")
    args = parser.parse_args()

    short_prompt = "Explain the difference between prefill and decode phases in LLM serving in 50 words."
    long_intruder_prompt = (
        "Analyze the following architectural trace across distributed systems and compare all memory access "
        "patterns, cache line invalidations, NUMA node balance, and PCIe bandwidth saturation in depth: "
        + ("word " * 1400)
    )

    print("==========================================================================")
    print(" AI INFRA SYSTEMS - STAGE 03: CONCURRENCY & QUEUEING PROFILER")
    print("==========================================================================")
    print(f" Endpoint:     {args.endpoint}")
    print(f" Mode:         {args.mode.upper()}")
    print(f" Duration:     {args.duration}s")
    if args.mode == "closed":
        print(f" Concurrency:  {args.concurrency} concurrent workers")
    elif args.mode == "open":
        print(f" Arrival Rate: {args.rate} req/s (Poisson process)")
    elif args.mode == "intruder":
        print(f" Background:   {args.concurrency} workers | Intruder: ~1500 tokens")
    print("--------------------------------------------------------------------------")

    if args.mode == "closed":
        records = asyncio.run(
            run_closed_loop(
                args.endpoint, args.model, short_prompt, args.max_tokens, args.concurrency, args.duration
            )
        )
        analyze_and_print_results(records, "concurrency", args.concurrency, args.duration, args.csv)
    elif args.mode == "open":
        records = asyncio.run(
            run_open_loop(
                args.endpoint, args.model, short_prompt, args.max_tokens, args.rate, args.duration
            )
        )
        analyze_and_print_results(records, "rate_req_s", args.rate, args.duration, args.csv)
    elif args.mode == "intruder":
        records = asyncio.run(
            run_mixed_workload(
                args.endpoint, args.model, short_prompt, long_intruder_prompt, args.concurrency, args.duration
            )
        )
        analyze_and_print_results(records, "intruder_test", args.concurrency, args.duration, args.csv)


if __name__ == "__main__":
    main()
