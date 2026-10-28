#!/usr/bin/env python3
"""
batching_simulator.py - Discrete-Event Simulator: Static vs. Continuous Batching
Author: Srinivas Sarkar (ai-infra-systems)

Compares Queue Head-of-Line (HoL) blocking and slot utilization between:
1. Static Batching (Batch runs until the longest sequence completes)
2. Continuous Batching (Iteration-level scheduling: slots backfill instantly)

Given an identical arrival trace of requests with variable output token lengths.
"""

import math
import random
from typing import List, Tuple


class Request:
    def __init__(self, req_id: int, arrival_step: int, prompt_tokens: int, output_tokens: int):
        self.req_id = req_id
        self.arrival_step = arrival_step
        self.prompt_tokens = prompt_tokens
        self.output_tokens = output_tokens
        self.start_step: int = -1
        self.finish_step: int = -1

    @property
    def queue_wait_steps(self) -> int:
        return self.start_step - self.arrival_step


def generate_trace(num_requests: int = 100, arrival_rate: float = 0.5) -> List[Request]:
    """Generates a synthetic Poisson arrival trace with variable output lengths."""
    trace = []
    curr_step = 0
    for i in range(num_requests):
        inter_arrival = math.ceil(random.expovariate(arrival_rate))
        curr_step += inter_arrival
        # Prompt ~50 tokens, Output between 16 and 128 tokens
        out_tokens = random.randint(16, 128)
        trace.append(Request(req_id=i, arrival_step=curr_step, prompt_tokens=50, output_tokens=out_tokens))
    return trace


def simulate_static_batching(requests: List[Request], num_slots: int = 4) -> Tuple[List[Request], float]:
    """Simulates Static Batching: All slots lock until the slowest sequence finishes."""
    queue = [Request(r.req_id, r.arrival_step, r.prompt_tokens, r.output_tokens) for r in requests]
    completed: List[Request] = []
    current_step = 0
    total_slot_steps_used = 0

    while queue or any(r.finish_step > current_step for r in completed):
        # Find requests that have arrived by current_step
        available = [r for r in queue if r.arrival_step <= current_step]
        if not available:
            # Advance time to next arrival
            current_step = queue[0].arrival_step
            continue

        # Form a static batch up to num_slots
        batch = available[:num_slots]
        for r in batch:
            queue.remove(r)
            r.start_step = current_step
            # Batch runtime is determined by the LONGEST sequence in this batch
            r.finish_step = current_step + max(item.output_tokens for item in batch)
            total_slot_steps_used += r.output_tokens
            completed.append(r)

        # Static batch locks all slots until the longest request completes
        batch_duration = max(item.output_tokens for item in batch)
        current_step += batch_duration

    total_possible_slot_steps = current_step * num_slots
    utilization = (total_slot_steps_used / total_possible_slot_steps) * 100.0 if total_possible_slot_steps > 0 else 0.0
    return completed, utilization


def simulate_continuous_batching(requests: List[Request], num_slots: int = 4) -> Tuple[List[Request], float]:
    """Simulates Continuous Batching: Slots backfill at every single iteration step."""
    queue = [Request(r.req_id, r.arrival_step, r.prompt_tokens, r.output_tokens) for r in requests]
    active_slots: List[Tuple[Request, int]] = []  # (request, remaining_tokens)
    completed: List[Request] = []
    current_step = 0
    total_slot_steps_used = 0

    while queue or active_slots:
        current_step += 1

        # 1. Backfill any free slots from arrived requests in queue
        while len(active_slots) < num_slots:
            available = [r for r in queue if r.arrival_step <= current_step]
            if not available:
                break
            next_req = available[0]
            queue.remove(next_req)
            next_req.start_step = current_step
            active_slots.append((next_req, next_req.output_tokens))

        # 2. Advance 1 decode iteration for all active slots
        next_active = []
        for req, remaining in active_slots:
            total_slot_steps_used += 1
            if remaining - 1 == 0:
                req.finish_step = current_step
                completed.append(req)
            else:
                next_active.append((req, remaining - 1))
        active_slots = next_active

    total_possible_slot_steps = current_step * num_slots
    utilization = (total_slot_steps_used / total_possible_slot_steps) * 100.0 if total_possible_slot_steps > 0 else 0.0
    return completed, utilization


def main():
    random.seed(42)
    num_requests = 100
    num_slots = 4
    trace = generate_trace(num_requests, arrival_rate=0.08)

    static_res, static_util = simulate_static_batching(trace, num_slots=num_slots)
    cont_res, cont_util = simulate_continuous_batching(trace, num_slots=num_slots)

    def stats(reqs: List[Request]):
        waits = sorted([r.queue_wait_steps for r in reqs])
        avg = sum(waits) / len(waits)
        p50 = waits[int(len(waits) * 0.50)]
        p90 = waits[int(len(waits) * 0.90)]
        p99 = waits[int(len(waits) * 0.99)]
        return avg, p50, p90, p99

    s_avg, s_p50, s_p90, s_p99 = stats(static_res)
    c_avg, c_p50, c_p90, c_p99 = stats(cont_res)

    print("==========================================================================")
    print(" BATCHING SIMULATOR: STATIC VS. CONTINUOUS ITERATION SCHEDULING")
    print(f" Requests: {num_requests} | Slots: {num_slots} | Identical Arrival Trace")
    print("==========================================================================")
    print(f" METRIC                      | STATIC BATCHING    | CONTINUOUS BATCHING")
    print("-----------------------------+--------------------+--------------------")
    print(f" Average Queue Wait (steps)  | {s_avg:14.2f}     | {c_avg:14.2f}")
    print(f" P50 Queue Wait (steps)      | {s_p50:14.2f}     | {c_p50:14.2f}")
    print(f" P90 Queue Wait (steps)      | {s_p90:14.2f}     | {c_p90:14.2f}")
    print(f" P99 Queue Wait (steps)      | {s_p99:14.2f}     | {c_p99:14.2f}")
    print(f" Hardware Slot Utilization   | {static_util:13.1f}%     | {cont_util:13.1f}%")
    print(f" Speedup / Wait Reduction    |       BASELINE     | {s_avg / c_avg:13.2f}x faster")
    print("==========================================================================")


if __name__ == "__main__":
    main()
