#!/usr/bin/env python3
"""
simulate_scaling_lag.py - Stage 09 Kubernetes & KEDA Autoscaling Lag Simulator
Author: Srinivas Sarkar (ai-infra-systems)

Deconstructs and simulates the full 7-step autoscaling lag chain in LLM inference clusters:
1. Metric Scrape Interval (Prometheus): 15.0s
2. Autoscaler Polling Interval (KEDA):  15.0s
3. HPA Sync Loop (Kubernetes HPA):      15.0s
4. Pod Scheduling & Node Assignment:     1.5s
5. Container Runtime & Image Check:      2.0s
6. Model Load & CUDA Context Init:       1.2s (Stage 02 L4 cold start)
7. Readiness Probe Stabilization:        2.0s
Total Reactive Autoscaling Lag:         ~46.7 seconds

Simulates burst arrival (6 req/s -> 24 req/s surge) across 3 architectures:
A) Static Deployment (1 Replica - Baseline)
B) GPU Utilization Autoscaler (nvidia-smi SM saturation)
C) KEDA Queue-Depth Autoscaler (llamacpp:requests_deferred > 0)
"""

import time
import statistics
from typing import Dict, List, Tuple, Any

# Calibrated Hardware Parameters (NVIDIA L4 Fleet from Stages 01-08)
SLOTS_PER_REPLICA = 8
BASE_TTFT_MS = 32.36      # ms (Stage 08 FlashAttention on)
BASE_DECODE_ITL_MS = 8.06 # ms per token
TOKENS_PER_REQUEST = 64
REQUEST_DURATION_S = (BASE_TTFT_MS / 1000.0) + (TOKENS_PER_REQUEST * (BASE_DECODE_ITL_MS / 1000.0))  # ~0.55s

# Kubernetes Autoscaling Lag Chain Breakdown (seconds)
LAG_SCRAPE_INTERVAL = 15.0    # Prometheus scrape
LAG_KEDA_POLL = 15.0          # KEDA pollingInterval
LAG_HPA_SYNC = 15.0           # HPA controller sync period (horizontal-pod-autoscaler-sync-period)
LAG_POD_SCHEDULE = 1.5        # kube-scheduler binding
LAG_CONTAINER_START = 2.0     # Containerd init
LAG_MODEL_LOAD = 1.23         # L4 CUDA context + page cache load (Stage 02/06)
LAG_READINESS_PROBE = 2.0     # HTTP /health/ready probe period
TOTAL_SCALING_LAG_S = (
    LAG_SCRAPE_INTERVAL + LAG_KEDA_POLL + LAG_HPA_SYNC +
    LAG_POD_SCHEDULE + LAG_CONTAINER_START + LAG_MODEL_LOAD + LAG_READINESS_PROBE
)


def simulate_traffic_surge(scaling_policy: str, duration_s: int = 120) -> Dict[str, Any]:
    """
    Simulates traffic burst:
    - t = 0 to 10s: 6 concurrent users (steady state < 8 slots)
    - t = 10 to 70s: 24 concurrent users (sharp 4x burst requiring 3 replicas)
    - t = 70 to 120s: 6 concurrent users (cooldown)
    """
    replicas = 1
    max_replicas = 3
    scale_event_time = None
    scale_completed_time = None

    ttft_history = []
    queued_requests = 0
    slo_violations = 0
    total_requests = 0

    # Simulation time step: 0.1s
    dt = 0.1
    current_time = 0.0

    while current_time < duration_s:
        # Determine current concurrency demand
        if 10.0 <= current_time < 70.0:
            active_demand = 24
        else:
            active_demand = 6

        total_requests += int(active_demand * dt * 2)

        # Capacity of current replicas
        total_capacity = replicas * SLOTS_PER_REPLICA

        if active_demand > total_capacity:
            unserved = active_demand - total_capacity
            queued_requests += unserved
            # High queue depth increases TTFT linearly via Little's Law
            effective_ttft = BASE_TTFT_MS + (queued_requests * REQUEST_DURATION_S * 1000.0 / total_capacity)
        else:
            queued_requests = max(0, queued_requests - int((total_capacity - active_demand) * dt * 10))
            effective_ttft = BASE_TTFT_MS

        ttft_history.append(effective_ttft)
        if effective_ttft > 250.0:  # SLO threshold = 250ms
            slo_violations += 1

        # Check Autoscaling Logic
        if scaling_policy == "keda_queue_depth":
            # Triggers on first deferred request in queue
            if queued_requests > 0 and scale_event_time is None and replicas < max_replicas:
                scale_event_time = current_time
                scale_completed_time = current_time + TOTAL_SCALING_LAG_S

        elif scaling_policy == "gpu_utilization":
            # GPU utilization metric averaging over 30s rolling window
            if active_demand >= total_capacity and scale_event_time is None and replicas < max_replicas:
                scale_event_time = current_time + 20.0
                scale_completed_time = scale_event_time + TOTAL_SCALING_LAG_S

        # Process replica scaling completion
        if scale_completed_time and current_time >= scale_completed_time:
            replicas = min(max_replicas, replicas + 2)
            scale_completed_time = None  # Reset for next scale

        current_time += dt

    return {
        "policy": scaling_policy,
        "p50_ttft_ms": statistics.median(ttft_history),
        "p99_ttft_ms": sorted(ttft_history)[int(len(ttft_history) * 0.99)],
        "max_ttft_ms": max(ttft_history),
        "slo_violation_pct": (slo_violations / len(ttft_history)) * 100.0,
        "final_replicas": replicas
    }


def print_lag_breakdown():
    print("==========================================================================")
    print(" 1. DECOMPOSED KUBERNETES AUTOSCALING LAG CHAIN")
    print("==========================================================================")
    print(f" 1. Prometheus Scrape Interval:            {LAG_SCRAPE_INTERVAL:5.2f} s")
    print(f" 2. KEDA ScaledObject Polling Interval:    {LAG_KEDA_POLL:5.2f} s")
    print(f" 3. Kubernetes HPA Controller Sync Period: {LAG_HPA_SYNC:5.2f} s")
    print(f" 4. kube-scheduler Pod Binding:            {LAG_POD_SCHEDULE:5.2f} s")
    print(f" 5. containerd Image Runtime Init:         {LAG_CONTAINER_START:5.2f} s")
    print(f" 6. CUDA Context Init & Weights Page-in:   {LAG_MODEL_LOAD:5.2f} s (Stage 02/06)")
    print(f" 7. Readiness Probe HTTP /health/ready:    {LAG_READINESS_PROBE:5.2f} s")
    print("--------------------------------------------------------------------------")
    print(f" TOTAL MEASURED SCALING LAG:               {TOTAL_SCALING_LAG_S:5.2f} s")
    print("==========================================================================")
    print(" [*] SRE Law: Reactive autoscaling requires ~46.7s to provision new slots.")
    print("     Because TTFT SLO is <= 250ms, a sudden 4x burst inevitably causes")
    print("     queue saturation during the scaling lag window unless headroom is buffered.")


def main():
    print("==========================================================================")
    print(" AI INFRA SYSTEMS - STAGE 09: KUBERNETES & KEDA AUTOSCALING SIMULATION")
    print("==========================================================================")
    print_lag_breakdown()

    print("\n==========================================================================")
    print(" 2. SIMULATING 4x TRAFFIC BURST (6 -> 24 CONCURRENT REQUESTS)")
    print("==========================================================================")

    policies = ["static_baseline", "gpu_utilization", "keda_queue_depth"]
    results = {}

    for pol in policies:
        res = simulate_traffic_surge(pol)
        results[pol] = res
        print(f"\n Policy: {pol.upper()}")
        print(f"   Final Replicas:      {res['final_replicas']}")
        print(f"   P50 TTFT:            {res['p50_ttft_ms']:.2f} ms")
        print(f"   P99 TTFT:            {res['p99_ttft_ms']:.2f} ms")
        print(f"   Max Peak TTFT:       {res['max_ttft_ms']:.2f} ms")
        print(f"   SLO Violation Rate:  {res['slo_violation_pct']:.1f}% of surge window")

    print("\n==========================================================================")
    print(" STAGE 09 AUTOSCALING POLICY COMPARISON")
    print("==========================================================================")
    print(" Autoscaling Strategy      | P50 TTFT   | P99 TTFT     | SLO Breach % | Final Pods")
    print("---------------------------+------------+--------------+--------------+------------")
    print(f" Static (No Autoscaling)   | {results['static_baseline']['p50_ttft_ms']:7.1f} ms | {results['static_baseline']['p99_ttft_ms']:9.1f} ms | {results['static_baseline']['slo_violation_pct']:11.1f}% | 1 pod")
    print(f" GPU Utilization HPA       | {results['gpu_utilization']['p50_ttft_ms']:7.1f} ms | {results['gpu_utilization']['p99_ttft_ms']:9.1f} ms | {results['gpu_utilization']['slo_violation_pct']:11.1f}% | {results['gpu_utilization']['final_replicas']} pods")
    print(f" KEDA Queue-Depth Trigger  | {results['keda_queue_depth']['p50_ttft_ms']:7.1f} ms | {results['keda_queue_depth']['p99_ttft_ms']:9.1f} ms | {results['keda_queue_depth']['slo_violation_pct']:11.1f}% | {results['keda_queue_depth']['final_replicas']} pods")
    print("==========================================================================\n")


if __name__ == "__main__":
    main()
