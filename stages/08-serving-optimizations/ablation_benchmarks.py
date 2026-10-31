#!/usr/bin/env python3
"""
ablation_benchmarks.py - Stage 08 Serving Optimization Ablation Suite
Author: Srinivas Sarkar (ai-infra-systems)

Empirically benchmarks the core serving optimization levers on LLM inference engines:
1. FlashAttention (-fa on vs -fa off): Prefill speedup and VRAM efficiency.
2. KV Cache Quantization (-ctk q8_0 -ctv q8_0 vs FP16): 50% VRAM reduction.
3. Chunked Prefill (-ub 64 vs -ub 512): Prompt ingestion micro-batching.
4. Prompt Prefix Caching: TTFT collapse on shared context reuse.
"""

import argparse
import subprocess
import time
import urllib.request
import urllib.error
import json
import statistics
import os
import signal
import sys
from typing import Dict, Any, List, Tuple

DEFAULT_BIN = "/root/llama-bin/llama-b11963/llama-server"
DEFAULT_MODEL = "/root/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf"
PORT = 8080
BASE_URL = f"http://127.0.0.1:{PORT}"
LOG_FILE_PATH = "/tmp/llama_ablation.log"


def get_vram_mb() -> float:
    """Returns currently allocated VRAM in MiB via nvidia-smi."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True
        ).strip()
        return float(out.split("\n")[0])
    except Exception:
        return 0.0


def wait_for_health(timeout: float = 20.0) -> bool:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        try:
            req = urllib.request.Request(f"{BASE_URL}/health")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


def kill_existing_server():
    try:
        subprocess.run(["pkill", "-9", "-x", "llama-server"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass
    time.sleep(1.2)


def run_completion(prompt: str, max_tokens: int = 64, stream: bool = True) -> Dict[str, Any]:
    url = f"{BASE_URL}/v1/chat/completions"
    payload = {
        "model": "qwen2.5-coder-1.5b",
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "max_tokens": max_tokens,
        "temperature": 0.0
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    
    t0 = time.perf_counter()
    t_first = None
    tokens = 0
    token_times = []

    try:
        with urllib.request.urlopen(req, timeout=30.0) as resp:
            buffer = b""
            while True:
                chunk = resp.read(256)
                if not chunk:
                    break
                now = time.perf_counter()
                buffer += chunk
                while b"\n\n" in buffer:
                    evt, buffer = buffer.split(b"\n\n", 1)
                    if b"data: [DONE]" in evt:
                        continue
                    if b"data: " in evt:
                        tokens += 1
                        if t_first is None:
                            t_first = now
                        token_times.append(now)
        t_end = time.perf_counter()
        ttft_ms = (t_first - t0) * 1000.0 if t_first else 0.0
        total_s = t_end - t0
        decode_s = t_end - t_first if t_first else 0.0
        tps = (tokens - 1) / decode_s if decode_s > 0 and tokens > 1 else 0.0

        itls = [(token_times[i] - token_times[i-1]) * 1000.0 for i in range(1, len(token_times))]
        median_itl = statistics.median(itls) if itls else 0.0
        max_itl = max(itls) if itls else 0.0

        return {
            "tokens": tokens,
            "ttft_ms": ttft_ms,
            "tps": tps,
            "median_itl_ms": median_itl,
            "max_itl_ms": max_itl,
            "total_s": total_s
        }
    except Exception as e:
        return {"error": str(e)}


def test_flash_attention_ablation(binary: str, model: str) -> Dict[str, Any]:
    print("\n--------------------------------------------------------------------------")
    print(" 1. ABLATION 1: FLASHATTENTION (-fa on vs -fa off)")
    print("--------------------------------------------------------------------------")
    results = {}
    long_prompt = "Write a comprehensive analysis of distributed systems concensus protocols. " * 30  # ~450 tokens

    configs = [
        ("FlashAttention=OFF", ["-fa", "off"]),
        ("FlashAttention=ON", ["-fa", "on"])
    ]

    for label, extra_flags in configs:
        kill_existing_server()
        cmd = [
            binary, "-m", model, "-ngl", "99", "-c", "4096",
            "--port", str(PORT), "--host", "127.0.0.1"
        ] + extra_flags

        with open(LOG_FILE_PATH, "w") as log_f:
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
            healthy = wait_for_health(timeout=15.0)

        if not healthy:
            print(f"  [-] {label} failed to boot healthy within 15s.")
            if os.path.exists(LOG_FILE_PATH):
                with open(LOG_FILE_PATH, "r") as f:
                    err_lines = f.readlines()
                print("      Engine Error Output:")
                for l in err_lines[-10:]:
                    print(f"        {l.strip()}")
            results[label] = {"error": "Server boot failed"}
            continue

        time.sleep(0.5)
        vram = get_vram_mb()
        run_completion("Hello", max_tokens=8)
        res = run_completion(long_prompt, max_tokens=64)
        
        results[label] = {
            "vram_mib": vram,
            "ttft_ms": res.get("ttft_ms", 0.0),
            "decode_tps": res.get("tps", 0.0)
        }
        print(f"  {label}:")
        print(f"    VRAM Usage:   {vram:.1f} MiB")
        print(f"    Prefill TTFT:  {res.get('ttft_ms', 0.0):.2f} ms")
        print(f"    Decode Speed:  {res.get('tps', 0.0):.2f} tok/s")

        proc.kill()
        proc.wait()

    return results


def test_kv_cache_quantization_ablation(binary: str, model: str) -> Dict[str, Any]:
    print("\n--------------------------------------------------------------------------")
    print(" 2. ABLATION 2: KV CACHE QUANTIZATION (FP16 vs Q8_0)")
    print(" Context Window: 32,768 tokens (High-Context Scale)")
    print("--------------------------------------------------------------------------")
    results = {}

    configs = [
        ("FP16 (Standard)", []),
        ("Q8_0 (Quantized)", ["-ctk", "q8_0", "-ctv", "q8_0"])
    ]

    for label, extra_flags in configs:
        kill_existing_server()
        cmd = [
            binary, "-m", model, "-ngl", "99",
            "-c", "32768",
            "--port", str(PORT), "--host", "127.0.0.1"
        ] + extra_flags

        with open(LOG_FILE_PATH, "w") as log_f:
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
            healthy = wait_for_health(timeout=25.0)

        if not healthy:
            print(f"  [-] {label} failed to boot healthy.")
            results[label] = {"error": "Boot failed"}
            continue

        time.sleep(1.0)
        vram = get_vram_mb()
        res = run_completion("Explain Kubernetes pod lifecycles.", max_tokens=32)
        results[label] = {
            "vram_mib": vram,
            "ttft_ms": res.get("ttft_ms", 0.0),
            "decode_tps": res.get("tps", 0.0)
        }
        print(f"  Configuration: {label}")
        print(f"    VRAM Footprint: {vram:.1f} MiB")
        print(f"    Prefill TTFT:   {res.get('ttft_ms', 0.0):.2f} ms")
        print(f"    Decode Speed:   {res.get('tps', 0.0):.2f} tok/s")

        proc.kill()
        proc.wait()

    return results


def test_chunked_prefill_ablation(binary: str, model: str) -> Dict[str, Any]:
    print("\n--------------------------------------------------------------------------")
    print(" 3. ABLATION 3: CHUNKED PREFILL (-ub 64 vs -ub 512)")
    print(" Measures Prompt Micro-batch Chunking Impact on TTFT & Throughput")
    print("--------------------------------------------------------------------------")
    results = {}
    long_prompt = "Explain distributed consensus in raft algorithms. " * 30  # ~450 tokens

    configs = [
        ("-ub 512 (Standard Large Batch)", ["-ub", "512"]),
        ("-ub 64 (Fine-Grained Chunking)", ["-ub", "64"])
    ]

    for label, extra_flags in configs:
        kill_existing_server()
        cmd = [
            binary, "-m", model, "-ngl", "99",
            "-c", "4096",
            "--port", str(PORT), "--host", "127.0.0.1"
        ] + extra_flags

        with open(LOG_FILE_PATH, "w") as log_f:
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
            healthy = wait_for_health(timeout=15.0)

        if not healthy:
            print(f"  [-] {label} failed to boot healthy.")
            continue

        time.sleep(0.5)
        res = run_completion(long_prompt, max_tokens=64)
        results[label] = {
            "ttft_ms": res.get("ttft_ms", 0.0),
            "decode_tps": res.get("tps", 0.0)
        }
        print(f"  Configuration: {label}")
        print(f"    Prefill TTFT:  {res.get('ttft_ms', 0.0):.2f} ms")
        print(f"    Decode Speed:  {res.get('tps', 0.0):.2f} tok/s")

        proc.kill()
        proc.wait()

    return results


def test_prompt_caching_ablation(binary: str, model: str) -> Dict[str, Any]:
    print("\n--------------------------------------------------------------------------")
    print(" 4. ABLATION 4: PROMPT / PREFIX KV CACHE REUSE")
    print("--------------------------------------------------------------------------")
    kill_existing_server()
    cmd = [
        binary, "-m", model, "-ngl", "99",
        "-c", "4096",
        "--port", str(PORT), "--host", "127.0.0.1"
    ]
    with open(LOG_FILE_PATH, "w") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        healthy = wait_for_health(timeout=15.0)

    if not healthy:
        print("  [-] Prompt caching test failed to boot server.")
        return {}

    time.sleep(0.5)
    system_prefix = "You are a Linux Kernel Systems Architect. Here is the operational specification: " + ("sys_call_entry_point_audit " * 40)
    
    # First turn: Cold prompt ingestion (full prefill required)
    prompt_turn1 = system_prefix + "\nQuestion 1: What is the purpose of eBPF?"
    res_turn1 = run_completion(prompt_turn1, max_tokens=32)

    # Second turn: Warm prompt reuse (cached KV prefix hit)
    prompt_turn2 = system_prefix + "\nQuestion 2: What is the purpose of cgroups v2?"
    res_turn2 = run_completion(prompt_turn2, max_tokens=32)

    print(f"  Turn 1 (Cold Prompt Prefill): TTFT = {res_turn1.get('ttft_ms', 0.0):.2f} ms")
    print(f"  Turn 2 (Prefix KV Cache Hit): TTFT = {res_turn2.get('ttft_ms', 0.0):.2f} ms")
    speedup = (res_turn1.get('ttft_ms', 1.0) / res_turn2.get('ttft_ms', 1.0)) if res_turn2.get('ttft_ms', 0) > 0 else 0
    print(f"  [+] TTFT Speedup Factor:      {speedup:.2f}x reduction via KV Cache Reuse!")

    proc.kill()
    proc.wait()

    return {
        "turn1_cold_ttft_ms": res_turn1.get("ttft_ms", 0.0),
        "turn2_cached_ttft_ms": res_turn2.get("ttft_ms", 0.0),
        "speedup_factor": speedup
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 08 Serving Optimization Ablation Suite")
    parser.add_argument("--bin", default=DEFAULT_BIN, help="Path to llama-server")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to GGUF model")
    args = parser.parse_args()

    print("==========================================================================")
    print(" AI INFRA SYSTEMS - STAGE 08: SERVING ENGINE OPTIMIZATIONS ABLATION")
    print("==========================================================================")
    print(f" Binary: {args.bin}")
    print(f" Model:  {args.model}")

    try:
        fa_res = test_flash_attention_ablation(args.bin, args.model)
        kv_res = test_kv_cache_quantization_ablation(args.bin, args.model)
        chunk_res = test_chunked_prefill_ablation(args.bin, args.model)
        cache_res = test_prompt_caching_ablation(args.bin, args.model)

        print("\n==========================================================================")
        print(" STAGE 08 ABLATION MATRIX SUMMARY")
        print("==========================================================================")
        print(" Optimization Lever            | Metric Tested     | Measured Impact")
        print("-------------------------------+-------------------+---------------------------------")
        if "FlashAttention=OFF" in fa_res and "FlashAttention=ON" in fa_res and "error" not in fa_res["FlashAttention=ON"]:
            fa_diff = fa_res["FlashAttention=OFF"]["ttft_ms"] - fa_res["FlashAttention=ON"]["ttft_ms"]
            print(f" FlashAttention (-fa on)       | Prefill TTFT      | {fa_res['FlashAttention=ON']['ttft_ms']:.1f} ms (vs {fa_res['FlashAttention=OFF']['ttft_ms']:.1f} ms without FA)")
        elif "error" in fa_res.get("FlashAttention=ON", {}):
            print(" FlashAttention (-fa)          | Compiler/Backend  | Unsupported / Not Compiled in Build")
        
        if "FP16 (Standard)" in kv_res and "Q8_0 (Quantized)" in kv_res:
            vram_saved = kv_res["FP16 (Standard)"]["vram_mib"] - kv_res["Q8_0 (Quantized)"]["vram_mib"]
            pct_saved = (vram_saved / kv_res["FP16 (Standard)"]["vram_mib"]) * 100.0 if kv_res["FP16 (Standard)"]["vram_mib"] > 0 else 0
            print(f" KV Quantization (q8_0)        | VRAM Footprint    | -{vram_saved:.1f} MiB ({pct_saved:.1f}% total VRAM reduction)")
        
        if "-ub 512 (Standard Large Batch)" in chunk_res and "-ub 64 (Fine-Grained Chunking)" in chunk_res:
            print(f" Chunked Prefill (-ub 64)      | Micro-batch TTFT  | {chunk_res['-ub 64 (Fine-Grained Chunking)']['ttft_ms']:.1f} ms (vs {chunk_res['-ub 512 (Standard Large Batch)']['ttft_ms']:.1f} ms at ub=512)")

        if "speedup_factor" in cache_res:
            print(f" Prefix Caching                | Shared Prompt TTFT| {cache_res['speedup_factor']:.1f}x speedup ({cache_res['turn2_cached_ttft_ms']:.1f} ms vs {cache_res['turn1_cold_ttft_ms']:.1f} ms)")
        print("==========================================================================\n")

    finally:
        kill_existing_server()


if __name__ == "__main__":
    main()
