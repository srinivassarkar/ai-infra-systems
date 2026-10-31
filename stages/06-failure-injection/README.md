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
3. **The Gray Failure Paradox (`SIGSTOP`):**
   * Simulates a frozen GPU kernel or deadlocked compute thread.
   * Proves why load balancers must NEVER rely on Layer 4 TCP port checks alone.

---

## 3. Step-by-Step Reproduction Guide

### Run the Automated Chaos Recovery Harness:

```bash
cd ~/ai-infra-systems/stages/06-failure-injection

# On Apple Mac Studio M3 Ultra:
python3 chaos_recovery_harness.py \
  --bin /opt/homebrew/bin/llama-server \
  --model /Users/blustudio/models/qwen2.5-coder-1.5b-instruct-q4_k_m.gguf \
  --trials 3

# On Linux (NVIDIA L4 Cloud Pod):
python3 chaos_recovery_harness.py \
  --bin /root/llama-bin/llama-b11963/llama-server \
  --model /root/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf \
  --trials 3
```

---

## 4. Verified Physical Fleet Results: Gray Failure & MTTR

Here are the real measured numbers from our Apple Mac Studio M3 Ultra and NVIDIA L4:

### A. Experiment 1: The Gray Failure Paradox
When `SIGSTOP` was delivered to the running inference daemon:
- **Layer 4 TCP Port Probe (SYN-ACK):** **OPEN (PASS)**
- **Layer 7 HTTP Health Probe (/health):** **UNRESPONSIVE (FAIL)**

**What This Means in Plain English:**
The Linux and macOS operating system kernels handle TCP network handshakes in kernel space. Even when the AI server is completely frozen or locked up, the kernel continues saying "Yes, the port is open!" If an SRE configures an AWS Network Load Balancer or Kubernetes probe to check only the TCP port, traffic continues flooding into the dead server. **You must always configure Layer 7 HTTP `/health` probes with a 1-second timeout.**

---

### B. Experiment 2: Hard Crash MTTR Distribution (`kill -9`)

| Hardware Node | Trials | Mean MTTR | P50 MTTR | Std Dev ($\sigma$) | Target SLA (<3.0s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Node 2: Mac Studio M3 Ultra** | 3 | **2,674.33 ms** | 2,661.99 ms | **26.70 ms** | **PASS** (325 ms margin) |
| **Node 3: NVIDIA L4 Cloud Pod** | 3 | **2,224.86 ms** | 2,231.83 ms | **29.74 ms** | **PASS** (775 ms margin) |

### What These Numbers Mean in Plain English:
1. **Predictable Auto-Healing:** Across both Apple Silicon and enterprise NVIDIA GPUs, the standard deviation is under **$30\ \mathrm{ms}$**. Once killed, the server respawns with near-deterministic timing.
2. **Breakdown of the 2.6-Second MTTR:**
   - **1.0 second:** Intentional supervisor backoff delay (preventing rapid crash loops).
   - **1.6 seconds:** Disk reload and GPU kernel/shader initialization.
3. **SLA Compliance:** Both platforms recover cleanly and begin passing health probes in under **2.7 seconds**, satisfying enterprise 99.9% availability SLAs.

For the production systemd supervisor unit and Kubernetes eviction manifests, read the [Stage 06 Full Report](REPORT.md).
