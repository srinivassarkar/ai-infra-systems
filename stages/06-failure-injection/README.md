# Stage 06: Failure Injection, Daemon Crashes & Recovery

## 1. Objective
Simulate real-world production failure modes (sudden process termination via `SIGKILL`, GPU driver stalls, and port conflicts) and verify automated daemon recovery and readiness probe protection.

---

## 2. Key Failure Modes Evaluated

1. **Hard Process Failure (`SIGKILL` / `kill -9`):**
   * Simulates kernel OOM kills (Exit Code 137), hardware segfaults, or host panics.
   * Measures **Mean Time to Recovery (MTTR)**: from kill execution to supervisor respawn and HTTP `/health` probe recovery.
2. **Graceful Degradation (`SIGTERM` / `kill -15`):**
   * Verifies whether the process cleanly releases POSIX locks, GPU context memory, and socket bindings before shutdown.
3. **Readiness Probe Traffic Eviction:**
   * Why load balancers (K8s ingress, Nginx) must probe `/health` to immediately pull a crashing container from active upstream pools, preventing cascading `502 Bad Gateway` errors.

---

## 3. Step-by-Step Reproduction Guide

### Run the Automated Chaos Recovery Harness:

```bash
cd ~/ai-infra-systems/stages/06-failure-injection

python3 chaos_recovery_harness.py \
  --bin /root/llama-bin/llama-b11963/llama-server \
  --model /root/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf
```

The harness will:
1. Boot the server and measure initial cold startup time.
2. Fire a hard `SIGKILL` directly into the process group.
3. Supervise and respawn the daemon, recording total outage duration (MTTR).
4. Issue a clean `SIGTERM` to measure graceful shutdown duration.
