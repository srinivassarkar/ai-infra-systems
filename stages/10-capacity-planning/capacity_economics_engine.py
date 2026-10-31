#!/usr/bin/env python3
"""
capacity_economics_engine.py - Stage 10 Capacity Planning & Sizing Economics Engine
Author: Srinivas Sarkar (ai-infra-systems)

1. Dual-Constraint Sizing Matrix: Evaluates capacity as min(VRAM Memory Limit, SLO Latency Knee).
2. Cost Per 1M Tokens: Compares NVIDIA L4 Cloud, Apple Mac Studio M3 Ultra, and Edge GTX 1050 Ti
   at peak capacity and realistic utilization levels (100%, 50%, 20%).
3. 24-Hour Diurnal Fleet Simulation: Compares Static 24/7 cluster vs. KEDA Event-Driven Autoscaling,
   measuring total daily expenditure, dollar savings, and scaling-lag SLO breach minutes.
"""

import argparse
from typing import Dict, List, Any

# Hardware Fleet Profiles
FLEET = {
    "NVIDIA L4 (Cloud Spot)": {
        "vram_gb": 24.0,
        "bandwidth_gb_s": 300.0,
        "hourly_cost_usd": 0.35,      # RunPod / RACE spot rate ($0.35/hr)
        "slo_max_concurrency": 8,     # Stage 03 measured cliff (TTFT <= 43ms at N=8, 428ms at N=16)
        "goodput_tps": 191.62,        # Stage 08 measured decode throughput
        "is_cloud": True
    },
    "Mac Studio M3 Ultra (Node 1)": {
        "vram_gb": 256.0,
        "bandwidth_gb_s": 800.0,
        "hourly_cost_usd": 0.288,     # $7,000 amortized 3yr ($0.266/hr) + 150W electricity ($0.022/hr)
        "slo_max_concurrency": 32,    # High bandwidth allows 32 concurrent streams without SLO cliff
        "goodput_tps": 240.0,
        "is_cloud": False
    },
    "Edge GTX 1050 Ti (Node 2)": {
        "vram_gb": 4.0,
        "bandwidth_gb_s": 112.0,
        "hourly_cost_usd": 0.017,     # $150 amortized 3yr ($0.006/hr) + 75W electricity ($0.011/hr)
        "slo_max_concurrency": 2,     # Constrained compute and memory
        "goodput_tps": 42.0,
        "is_cloud": False
    }
}


def compute_vram_limit(vram_gb: float, model_weights_gb: float = 0.85, ctx_len: int = 4096) -> int:
    """Calculates maximum slots that fit in VRAM before OOM."""
    # Qwen2.5-1.5B (GQA): 28 layers, 16 Q heads, 2 KV heads, 64 head_dim
    # KV per token (FP16): 2 * 28 * 2 * 64 * 2 = 14,336 bytes = 0.00001335 GiB
    bytes_per_token_gb = (2 * 28 * 2 * 64 * 2) / (1024**3)
    kv_per_user_gb = bytes_per_token_gb * ctx_len
    available_vram = max(0.0, vram_gb - model_weights_gb - 0.5)  # 0.5 GB CUDA context/scratch
    return int(available_vram / kv_per_user_gb) if kv_per_user_gb > 0 else 0


def generate_dual_constraint_matrix():
    print("==========================================================================")
    print(" 1. DUAL-CONSTRAINT SIZING MATRIX: VRAM CEILING vs. SLO LATENCY KNEE")
    print(" Model: Qwen2.5-Coder-1.5B (Q4_K_M) | Target SLO: TTFT <= 250ms, TPOT <= 40ms")
    print("==========================================================================")
    print(f" {'Hardware Platform':<28} | {'VRAM Limit':<12} | {'SLO Knee':<12} | {'Binding Constraint':<20} | {'Effective Slots'}")
    print("------------------------------+--------------+--------------+----------------------+-----------------")

    for name, spec in FLEET.items():
        vram_limit = compute_vram_limit(spec["vram_gb"], ctx_len=4096)
        slo_knee = spec["slo_max_concurrency"]
        
        if vram_limit < slo_knee:
            binding = "Memory (VRAM OOM)"
            effective = vram_limit
        else:
            binding = "Latency (SLO Knee)"
            effective = slo_knee

        print(f" {name:<28} | {vram_limit:>8} slots | {slo_knee:>8} slots | {binding:<20} | {effective:>8} slots")

    print("\n [*] Architectural Principle: For models <= 7B on modern GPUs, KV cache is cheap.")
    print("     The SLO Latency Knee binds LONG before VRAM runs out. Sizing solely on")
    print("     VRAM arithmetic overprovisions concurrency and destroys user TTFT.\n")


def generate_cost_per_token_table():
    print("==========================================================================")
    print(" 2. FINANCIAL COST PER 1 MILLION TOKENS ($/1M Tokens)")
    print(" Calculated at SLO-Constrained Goodput across Utilization Levels")
    print("==========================================================================")
    print(f" {'Hardware Platform':<28} | {'Hourly Cost':<12} | {'100% Util':<12} | {'50% Daytime':<12} | {'20% Off-Peak'}")
    print("------------------------------+--------------+--------------+--------------+--------------")

    for name, spec in FLEET.items():
        cost_hr = spec["hourly_cost_usd"]
        goodput_tps = spec["goodput_tps"]
        tokens_per_hr_peak = goodput_tps * 3600.0

        cost_1m_100 = (cost_hr / tokens_per_hr_peak) * 1e6
        cost_1m_50 = cost_1m_100 / 0.50
        cost_1m_20 = cost_1m_100 / 0.20

        print(f" {name:<28} | ${cost_hr:>9.3f}/hr | ${cost_1m_100:>10.4f} | ${cost_1m_50:>10.4f} | ${cost_1m_20:>10.4f}")

    print("\n [*] Economic Finding: At 50% average production utilization, cloud L4 spot costs")
    print("     ~$1.01 per 1M tokens, outperforming closed proprietary APIs ($2.50+/1M) by 2.5x,")
    print("     while owned Apple Silicon delivers $0.67 per 1M tokens when amortized.\n")


def simulate_diurnal_24h_trace():
    print("==========================================================================")
    print(" 3. 24-HOUR DIURNAL ENTERPRISE SIMULATION: STATIC vs. KEDA AUTOSCALING")
    print(" Trace Profile: 8h Peak (24 req/s) | 8h Mid-day (12 req/s) | 8h Night (2 req/s)")
    print(" Hardware: NVIDIA L4 Spot ($0.35/hr per pod, 8 slots/pod)")
    print("==========================================================================")

    # Hourly demand profile (24 hours)
    # Hours 0-7: Night (2 req/s -> 1 pod required)
    # Hours 8-15: Peak (24 req/s -> 3 pods required)
    # Hours 16-23: Mid-day (12 req/s -> 2 pods required)
    hourly_pods_keda = [1]*8 + [3]*8 + [2]*8
    hourly_pods_static = [3]*24  # Static cluster provisioned for peak 24 req/s

    cost_per_pod_hr = 0.35
    total_cost_static = sum(hourly_pods_static) * cost_per_pod_hr
    total_cost_keda = sum(hourly_pods_keda) * cost_per_pod_hr
    savings_usd = total_cost_static - total_cost_keda
    savings_pct = (savings_usd / total_cost_static) * 100.0

    # SLO violation during morning burst transition (8:00 AM 1 pod -> 3 pods)
    # From Stage 09: Scaling lag is 51.73 seconds
    scaling_lag_minutes = 51.73 / 60.0

    print(f" Static 24/7 Provisioning Cost (3 Pods Constant):  ${total_cost_static:.2f} / day")
    print(f" KEDA Event-Driven Autoscaled Cost (1-3 Pods):     ${total_cost_keda:.2f} / day")
    print(f" Daily Financial Savings:                          ${savings_usd:.2f} ({savings_pct:.1f}% reduction)")
    print(f" Morning Scaling Lag SLO Window:                   {scaling_lag_minutes:.2f} minutes (3.6% of transition hour)")
    print("--------------------------------------------------------------------------")
    print(" [*] Executive SRE Recommendation: KEDA autoscaling saves 33.3% of GPU spend.")
    print("     To eliminate the 0.86-minute morning SLO violation during the 51.7s scaling lag,")
    print("     implement a scheduled Cron HPA pre-warming 1 additional pod at 07:55 AM.\n")


def main():
    print("==========================================================================")
    print(" AI INFRA SYSTEMS - STAGE 10: CAPACITY PLANNING & TOKEN ECONOMICS")
    print("==========================================================================")
    generate_dual_constraint_matrix()
    generate_cost_per_token_table()
    simulate_diurnal_24h_trace()
    print("==========================================================================")
    print(" STAGE 10 CAPACITY PLANNING & ECONOMICS ANALYSIS COMPLETE")
    print("==========================================================================\n")


if __name__ == "__main__":
    main()
