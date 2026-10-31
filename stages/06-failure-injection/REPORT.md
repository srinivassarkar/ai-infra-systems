# Stage 06: Failure Injection, Daemon Crashes & Self-Healing MTTR Report

## 1. Executive Summary & SRE Reliability Axioms

Production LLM inference infrastructure operates at the volatile intersection of user-space web daemons, kernel virtual memory, and proprietary hardware accelerators (NVIDIA CUDA drivers, Apple Metal). Unhandled driver panics, out-of-memory kernel reaps, and frozen compute threads inevitably strike. 

Stage 06 empirically validates the failure modes, automated recovery mechanics, and health-probing architectures required to maintain an enterprise 99.9% uptime SLA.

### Key Empirical Findings:
1. **Deterministic Auto-Healing Across Silicon Architectures:** Across multiple failure trials under a hard uncatchable `SIGKILL` (simulating Linux kernel OOM-killer Exit 137 or hardware panic):
   - **Node 2 (Apple Mac Studio M3 Ultra):** Mean MTTR averaged **2,674.33 ms** ($\sigma = 26.70\ \mathrm{ms}$).
   - **Node 3 (Enterprise NVIDIA L4):** Mean MTTR averaged **2,224.86 ms** ($\sigma = 29.74\ \mathrm{ms}$).
   Both platforms consistently auto-heal within 2.2 to 2.7 seconds (inclusive of a 1.0-second supervisor backoff), easily beating our strict 3.0-second production SLA.
2. **The Gray Failure Paradox Empirically Verified:** Injecting `SIGSTOP` (simulating a frozen GPU driver, thread deadlock, or GIL lockup) proved that **Layer 4 TCP checks remain OPEN (PASS)** while **Layer 7 HTTP health checks FAIL (UNRESPONSIVE)**. Relying on Layer 4 network probes in inference clusters causes catastrophic traffic black-holing.

---

## 2. Empirical Telemetry: Cross-Fleet Recovery Distribution

*Model: `Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf`.*

### A. Node 2: Apple Mac Studio M3 Ultra (256 GB Unified LPDDR5X)
*Executed on bare-metal macOS Apple Silicon:*

| Trial Number | Injected Fault | Supervisor Backoff | Respawn & Metal Graph Init | Total Measured MTTR | Health Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Trial 1** | `SIGKILL` (kill -9) | 1,000.00 ms | 1,661.99 ms | **2,661.99 ms** | Restored (PID 7194) |
| **Trial 2** | `SIGKILL` (kill -9) | 1,000.00 ms | 1,704.97 ms | **2,704.97 ms** | Restored (PID 7198) |
| **Trial 3** | `SIGKILL` (kill -9) | 1,000.00 ms | 1,656.03 ms | **2,656.03 ms** | Restored (PID 7217) |

* **Mean MTTR:** $2,674.33\ \mathrm{ms}$ ($2.67\ \mathrm{s}$)
* **P50 MTTR:** $2,661.99\ \mathrm{ms}$
* **Std Dev ($\sigma$):** $26.70\ \mathrm{ms}$ (Tight, predictable variance)
* **Production SLA:** **PASS** (Comfortably beneath 3-second SLA window)

---

### B. Node 3: Cloud Enterprise NVIDIA L4 (24 GB GDDR6)
*Executed on Ubuntu 24.04 LTS (Driver 580.126.20, CUDA 13.0):*

| Trial Number | Injected Fault | Supervisor Backoff | Respawn & CUDA Context Init | Total Measured MTTR | Health Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Trial 1** | `SIGKILL` (kill -9) | 1,000.00 ms | 1,231.83 ms | **2,231.83 ms** | Restored (PID 4209) |
| **Trial 2** | `SIGKILL` (kill -9) | 1,000.00 ms | 1,250.49 ms | **2,250.49 ms** | Restored (PID 4255) |
| **Trial 3** | `SIGKILL` (kill -9) | 1,000.00 ms | 1,192.25 ms | **2,192.25 ms** | Restored (PID 4307) |

* **Mean MTTR:** $2,224.86\ \mathrm{ms}$ ($2.22\ \mathrm{s}$)
* **Std Dev ($\sigma$):** $29.74\ \mathrm{ms}$
* **Production SLA:** **PASS** (775 ms margin beneath 3-second SLA)

---

## 3. The Gray Failure Paradox: Deep Systems Breakdown

### The Empirical Anomaly
During Experiment 1, a `SIGSTOP` signal was delivered to the active `llama-server` process, immediately pausing execution threads:
```text
[*] Layer 4 TCP Port Probe (SYN-ACK):    OPEN (PASS)
[*] Layer 7 HTTP Health Probe (/health): UNRESPONSIVE (FAIL)
```

### The Kernel Mechanics
Why does a completely frozen process continue passing Layer 4 TCP health checks?
1. **Kernel-Space TCP Handshake:** The host kernel TCP/IP stack implements the TCP three-way handshake (`SYN` $\rightarrow$ `SYN-ACK` $\rightarrow$ `ACK`) entirely inside kernel space.
2. **The Socket Listen Backlog:** When a listening socket is created (`bind()` + `listen(backlog=512)`), the kernel manages incoming connection handshakes in the `TCP_SYN_RECV` and `TCP_ESTABLISHED` backlog queues without requiring user-space intervention.
3. **The Deadlock:** If the user-space application thread hangs (e.g., waiting on a stalled CUDA/Metal kernel or locked mutex), it cannot call `accept()` or parse incoming HTTP requests. However, the kernel continues responding to TCP port checks with instant `SYN-ACK` flags!
4. **The Production Consequence:** If an SRE configures an AWS Network Load Balancer (NLB) with a TCP ping or a Kubernetes probe with `tcpSocket: {port: 8080}`, the load balancer marks the stalled container as **HEALTHY**. User inference requests continue flooding the dead pod, timing out and returning 504 errors across the cluster.
5. **Architectural Rule:** Ingress routers and orchestrators must exclusively probe **Layer 7 HTTP `/health` endpoints** with stringent client timeouts ($T_{\mathrm{timeout}} \le 1.0\ \mathrm{s}$).

---

## 4. Production Daemon Supervision Architectures

To ensure automated self-healing matching our empirical MTTR, production systems must utilize declarative process managers.

### 1. Hardened systemd Daemon Unit (`inference.service`)
```ini
[Unit]
Description=High-Throughput LLM Inference Server (llama.cpp)
After=network.target nvidia-persistenced.service
Wants=nvidia-persistenced.service

[Service]
Type=simple
User=inference
Group=inference
WorkingDirectory=/opt/models
ExecStart=/usr/local/bin/llama-server \
  -m /opt/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf \
  -ngl 99 \
  --parallel 8 \
  -c 16384 \
  --port 8080 \
  --host 0.0.0.0

# SELF-HEALING SUPERVISORY RULES:
Restart=always
RestartSec=1s
KillMode=mixed
TimeoutStopSec=15s
OOMScoreAdjust=-500

# RESOURCE CONSTRAINTS:
LimitNOFILE=65536
LimitMEMLOCK=infinity

[Install]
WantedBy=multi-user.target
```

### 2. Kubernetes Probing & Auto-Eviction Spec
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-inference-worker
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: inference-engine
        image: ghcr.io/srinivassarkar/llama-server:latest
        ports:
        - containerPort: 8080
        resources:
          limits:
            nvidia.com/gpu: 1
        # CRITICAL: L7 PROBE EVACUATION PREVENTS GRAY FAILURES
        readinessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 3
          periodSeconds: 2
          timeoutSeconds: 1
          failureThreshold: 2
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 2
          failureThreshold: 3
```
