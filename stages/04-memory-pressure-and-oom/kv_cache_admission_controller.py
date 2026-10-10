#!/usr/bin/env python3
"""
Stage 04: KV Cache Capacity Planner & Admission Controller
===========================================================
First-principles memory accounting and admission control budgeting for LLM inference.
Calculates exact KV cache footprints across attention architectures (MHA, GQA, MQA, MLA)
and simulates GPU admission control to eliminate CUDA OOM panics.
"""

import sys
import argparse
from typing import Dict, Any, Optional

# Standard Model Architectures for Capacity Modeling
MODEL_CATALOG: Dict[str, Dict[str, Any]] = {
    "qwen2.5-coder-1.5b": {
        "description": "Qwen 2.5 Coder 1.5B (GQA)",
        "layers": 28,
        "kv_heads": 2,
        "head_dim": 64,
        "params_b": 1.54,
        "arch": "GQA",
        "default_context": 32768,
    },
    "llama-3.2-3b": {
        "description": "Llama 3.2 3B (GQA)",
        "layers": 28,
        "kv_heads": 8,
        "head_dim": 128,
        "params_b": 3.21,
        "arch": "GQA",
        "default_context": 131072,
    },
    "qwen2.5-7b": {
        "description": "Qwen 2.5 7B (GQA)",
        "layers": 28,
        "kv_heads": 4,
        "head_dim": 128,
        "params_b": 7.61,
        "arch": "GQA",
        "default_context": 32768,
    },
    "llama-3.1-8b": {
        "description": "Llama 3.1 8B (GQA)",
        "layers": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "params_b": 8.03,
        "arch": "GQA",
        "default_context": 131072,
    },
    "llama-3.1-70b": {
        "description": "Llama 3.1 70B (GQA)",
        "layers": 80,
        "kv_heads": 8,
        "head_dim": 128,
        "params_b": 70.6,
        "arch": "GQA",
        "default_context": 131072,
    },
    "deepseek-v3-mla": {
        "description": "DeepSeek-V3 671B (MLA - Multi-Head Latent Attention)",
        "layers": 61,
        "kv_heads": 1,  # Compressed latent dimension
        "head_dim": 576,  # d_c (512) + d_r (64) = 576 compressed latent vector
        "params_b": 671.0,
        "arch": "MLA",
        "default_context": 131072,
    },
    "gpt-3-175b-mha": {
        "description": "GPT-3 175B (Legacy MHA - Multi-Head Attention)",
        "layers": 96,
        "kv_heads": 96,
        "head_dim": 128,
        "params_b": 175.0,
        "arch": "MHA",
        "default_context": 4096,
    }
}

# Physical Hardware Profiles
HARDWARE_PROFILES: Dict[str, Dict[str, Any]] = {
    "gtx-1050-ti": {
        "name": "NVIDIA GeForce GTX 1050 Ti (Edge CUDA)",
        "total_memory_gb": 4.0,
        "memory_bandwidth_gbps": 112.0,
        "safe_headroom_gb": 0.4,
    },
    "mac-mini-m4": {
        "name": "Apple Mac mini M4 (Edge UMA)",
        "total_memory_gb": 24.0,
        "memory_bandwidth_gbps": 120.0,
        "safe_headroom_gb": 3.0,
    },
    "mac-studio-m3-ultra": {
        "name": "Apple Mac Studio M3 Ultra (Server UMA)",
        "total_memory_gb": 256.0,
        "memory_bandwidth_gbps": 800.0,
        "safe_headroom_gb": 16.0,
    },
    "nvidia-a10g": {
        "name": "AWS EC2 g5.xlarge (NVIDIA A10G)",
        "total_memory_gb": 24.0,
        "memory_bandwidth_gbps": 600.0,
        "safe_headroom_gb": 2.0,
    },
    "nvidia-h100-80gb": {
        "name": "NVIDIA H100 SXM5 (80GB HBM3)",
        "total_memory_gb": 80.0,
        "memory_bandwidth_gbps": 3350.0,
        "safe_headroom_gb": 4.0,
    }
}


def calculate_kv_cache_bytes_per_token(
    layers: int,
    kv_heads: int,
    head_dim: int,
    bytes_per_elem: int = 2,
    arch: str = "GQA"
) -> int:
    """
    Calculates the memory required to store KV cache for a single token.
    Standard Formula: 2 * layers * kv_heads * head_dim * bytes_per_elem
    Factor of 2 accounts for both Key (K) and Value (V) tensors.
    """
    if arch == "MLA":
        # Multi-Head Latent Attention compresses K and V into a shared latent vector
        # (d_c + d_r) = head_dim
        return layers * head_dim * bytes_per_elem
    return 2 * layers * kv_heads * head_dim * bytes_per_elem


def calculate_total_kv_cache_gb(
    layers: int,
    kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int = 1,
    bytes_per_elem: int = 2,
    arch: str = "GQA"
) -> float:
    """Calculates total KV cache memory footprint in Gigabytes (GiB)."""
    bytes_per_token = calculate_kv_cache_bytes_per_token(layers, kv_heads, head_dim, bytes_per_elem, arch)
    total_bytes = bytes_per_token * seq_len * batch_size
    return total_bytes / (1024 ** 3)


def estimate_weight_memory_gb(params_b: float, quant_format: str = "fp16") -> float:
    """
    Estimates model weight memory footprint in Gigabytes.
    FP16 / BF16: 2.0 bytes/param
    Q8: 1.0 byte/param
    Q4 (GGUF Q4_K_M / AWQ): ~0.55 bytes/param (including scale overhead)
    """
    quant_multipliers = {
        "fp16": 2.0,
        "bf16": 2.0,
        "int8": 1.0,
        "q8": 1.0,
        "q4": 0.55,
        "awq-4bit": 0.55,
    }
    multiplier = quant_multipliers.get(quant_format.lower(), 2.0)
    return (params_b * 1e9 * multiplier) / (1024 ** 3)


class AdmissionController:
    """
    Simulates production LLM serving admission control.
    Prevents CUDA Out-Of-Memory (OOM) by tracking available KV cache slots.
    """
    def __init__(
        self,
        hardware_key: str,
        model_key: str,
        quant_format: str = "q4",
        runtime_overhead_gb: Optional[float] = None
    ):
        self.hw = HARDWARE_PROFILES[hardware_key]
        self.model = MODEL_CATALOG[model_key]
        self.quant_format = quant_format

        # Compute static memory allocations
        self.weight_memory_gb = estimate_weight_memory_gb(self.model["params_b"], quant_format)
        self.headroom_gb = runtime_overhead_gb if runtime_overhead_gb is not None else self.hw["safe_headroom_gb"]

        self.available_kv_memory_gb = max(
            0.0,
            self.hw["total_memory_gb"] - self.weight_memory_gb - self.headroom_gb
        )
        self.bytes_per_token = calculate_kv_cache_bytes_per_token(
            self.model["layers"],
            self.model["kv_heads"],
            self.model["head_dim"],
            bytes_per_elem=2,
            arch=self.model["arch"]
        )

        self.max_allocatable_tokens = int((self.available_kv_memory_gb * (1024 ** 3)) / self.bytes_per_token)
        self.current_allocated_tokens = 0

    def evaluate_request(self, prompt_tokens: int, max_gen_tokens: int) -> Dict[str, Any]:
        """
        Evaluates whether an incoming request can be admitted without risking OOM.
        Uses worst-case reservation: prompt_tokens + max_gen_tokens.
        """
        required_tokens = prompt_tokens + max_gen_tokens
        required_gb = (required_tokens * self.bytes_per_token) / (1024 ** 3)

        free_tokens = self.max_allocatable_tokens - self.current_allocated_tokens
        can_admit = free_tokens >= required_tokens

        return {
            "admitted": can_admit,
            "required_tokens": required_tokens,
            "required_gb": required_gb,
            "free_tokens_remaining": free_tokens,
            "free_kv_memory_gb": (free_tokens * self.bytes_per_token) / (1024 ** 3),
            "utilization_pct": (self.current_allocated_tokens / self.max_allocatable_tokens * 100) if self.max_allocatable_tokens > 0 else 100.0,
            "reason": "OK" if can_admit else "CAPACITY_EXHAUSTED_REJECT_429_OR_QUEUE"
        }

    def allocate(self, tokens: int) -> bool:
        """Reserve token slots in the KV cache."""
        if self.current_allocated_tokens + tokens <= self.max_allocatable_tokens:
            self.current_allocated_tokens += tokens
            return True
        return False

    def release(self, tokens: int) -> None:
        """Release token slots upon request completion."""
        self.current_allocated_tokens = max(0, self.current_allocated_tokens - tokens)


def print_fleet_capacity_matrix():
    """Generates the enterprise capacity matrix across hardware and model combinations."""
    print("\n" + "=" * 90)
    print("ENTERPRISE INFERENCE FLEET: KV CACHE & CONCURRENCY MATRIX")
    print("=" * 90)

    test_combinations = [
        ("gtx-1050-ti", "qwen2.5-coder-1.5b", "q4"),
        ("gtx-1050-ti", "llama-3.2-3b", "q4"),
        ("mac-mini-m4", "llama-3.1-8b", "q4"),
        ("mac-studio-m3-ultra", "llama-3.1-70b", "q4"),
        ("mac-studio-m3-ultra", "deepseek-v3-mla", "int8"),
        ("nvidia-h100-80gb", "llama-3.1-70b", "fp16"),
    ]

    context_targets = [2048, 8192, 32768, 131072]

    for hw_key, model_key, quant in test_combinations:
        ac = AdmissionController(hw_key, model_key, quant)
        hw = HARDWARE_PROFILES[hw_key]
        model = MODEL_CATALOG[model_key]

        print(f"\nHardware: {hw['name']} ({hw['total_memory_gb']} GB)")
        print(f"Model:    {model['description']} [{quant.upper()}] | Weights: {ac.weight_memory_gb:.1f} GB | Headroom: {ac.headroom_gb:.1f} GB")
        print(f"KV Pool:  {ac.available_kv_memory_gb:.2f} GB available for KV cache ({ac.bytes_per_token:,} bytes/token)")
        print("-" * 90)
        print(f"{'Context Window':<18} {'KV Cache / User':<20} {'Max Concurrency (Users)':<26} {'Status':<15}")
        print("-" * 90)

        for ctx in context_targets:
            kv_per_user_mb = (ac.bytes_per_token * ctx) / (1024 ** 2)
            kv_per_user_gb = kv_per_user_mb / 1024
            max_concurrency = int(ac.available_kv_memory_gb / kv_per_user_gb) if kv_per_user_gb > 0 else 0

            status = "HEALTHY" if max_concurrency >= 10 else ("CONSTRAINED" if max_concurrency >= 1 else "OOM_IMPOSSIBLE")
            print(f"{ctx:<18,d} {kv_per_user_mb:>10.1f} MB ({kv_per_user_gb:.2f} GB)   {max_concurrency:>12d} slots           {status:<15}")


def main():
    parser = argparse.ArgumentParser(description="KV Cache Capacity Planner & Admission Controller")
    parser.add_argument("--matrix", action="store_true", help="Print full fleet capacity matrix")
    parser.add_argument("--hw", type=str, default="mac-studio-m3-ultra", choices=list(HARDWARE_PROFILES.keys()))
    parser.add_argument("--model", type=str, default="llama-3.1-70b", choices=list(MODEL_CATALOG.keys()))
    parser.add_argument("--quant", type=str, default="q4", choices=["fp16", "bf16", "int8", "q8", "q4"])
    parser.add_argument("--prompt", type=int, default=4096, help="Incoming prompt token length")
    parser.add_argument("--max-gen", type=int, default=1024, help="Max generation token length")
    args = parser.parse_args()

    if args.matrix or len(sys.argv) == 1:
        print_fleet_capacity_matrix()
        return

    ac = AdmissionController(args.hw, args.model, args.quant)
    result = ac.evaluate_request(args.prompt, args.max_gen)

    print("\n" + "=" * 60)
    print("ADMISSION CONTROL DECISION REPORT")
    print("=" * 60)
    print(f"Hardware:          {ac.hw['name']}")
    print(f"Model:             {ac.model['description']} [{args.quant.upper()}]")
    print(f"KV Memory Pool:    {ac.available_kv_memory_gb:.2f} GB ({ac.max_allocatable_tokens:,} tokens total)")
    print(f"Incoming Request:  {args.prompt:,} prompt + {args.max_gen:,} gen = {result['required_tokens']:,} tokens")
    print(f"Required KV RAM:   {result['required_gb'] * 1024:.1f} MB ({result['required_gb']:.3f} GB)")
    print(f"Decision:          {'ADMITTED (200 OK)' if result['admitted'] else 'REJECTED (429 / 503 OVERLOAD)'}")
    print(f"Reason:            {result['reason']}")
    print(f"Tokens Remaining:  {result['free_tokens_remaining']:,} ({result['free_kv_memory_gb']:.2f} GB)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
