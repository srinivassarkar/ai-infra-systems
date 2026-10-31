# Stage 06: Failure Injection, Daemon Crashes & Self-Healing MTTR Report

## 1. Executive Summary & SRE Reliability Axioms

Production LLM inference infrastructure operates at the volatile intersection of user-space web daemons, kernel virtual memory, and proprietary hardware accelerators (NVIDIA CUDA drivers). Unhandled driver panics, out-of-memory kernel reaps, and frozen compute threads inevitably strike. 

Stage 06 empirically validates the failure modes, automated recovery mechanics, and health-probing architectures required to maintain an enterprise 99.9% uptime SLA.

### Key Empirical Findings:
1. **Empirical MTTR on Enterprise Accelerator:** Under a hard uncatchable `SIGKILL` (simulating Linux kernel OOM-killer Exit 137 or hardware panic), total Mean Time to Recovery (MTTR) averaged **2,224.86 ms** (P50: **2,231.83 ms**, $\sigma = 29.74\ \mathrm{ms}$) on the NVIDIA L4 fleet.
2. **Deterministic Auto-Healing:** Across multiple failure trials, recovery was exceptionally stable with standard deviation under 30 ms, proving that model reload from cached page cache + CUDA context initialization operates within a tight 1.2-second window (after a 1.0-second supervisor backoff).
3. **The Gray Failure Paradox Empirically Verified:** Injecting `SIGSTOP` (simulating a frozen GPU driver, thread deadlock, or GIL lockup) proved that **Layer 4 TCP checks remain OPEN (PASS)** while **Layer 7 HTTP health checks FAIL (UNRESPONSIVE)**. Relying on Layer 4 network probes in inference clusters causes catastrophic traffic black-holing.

---

## 2. Empirical Telemetry: Hardware & Recovery Distribution

*Hardware: NVIDIA L4 (24 GB GDDR6, PCIe Gen4) on RACE Cloud Pod.*  
*Model: `Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf` (1.5B parameters, GQA).*  
*Runtime: `llama-server` (b11963, CUDA 13.0, Driver 580.126.20).*

### Hard SIGKILL MTTR Telemetry

| Trial Number | Injected Fault | Supervisor Backoff | Cold Respawn & CUDA Init | Total Measured MTTR | Health Status |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Trial 1** | `SIGKILL` (kill -9) | 1,000.00 ms | 1,231.83 ms | **2,231.83 ms** | Restored (PID 4209) |
| **Trial 2** | `SIGKILL` (kill -9) | 1,000.00 ms | 1,250.49 ms | **2,250.49 ms** | Restored (PID 4255) |
| **Trial 3** | `SIGKILL` (kill -9) | 1,000.00 ms | 1,192.25 ms | **2,192.25 ms** | Restored (PID 4307) |

### Statistical Distribution Summary
* **Trials Executed:** 3
* **Mean MTTR:** $2,224.86\ \mathrm{ms}$ ($2.22\ \mathrm{s}$)
* **P50 MTTR:** $2,231.83\ \mathrm{ms}$ ($2.23\ \mathrm{s}$)
* **Std Dev ($\sigma$):** $29.74\ \mathrm{ms}$
* **Target SLA Threshold:** $< 3,000.00\ \mathrm{ms}$
* **SLA Verification:** **PASS** (775 ms margin beneath strict 3-second SLA)

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
1. **Kernel-Space TCP Handshake:** The Linux kernel TCP/IP stack implements the TCP three-way handshake (`SYN` $\rightarrow$ `SYN-ACK` $\rightarrow$ `ACK`) entirely inside kernel space.
2. **The Socket Listen Backlog:** When a listening socket is created (`bind()` + `listen(backlog=512)`), the kernel manages incoming connection handshakes in the `TCP_SYN_RECV` and `TCP_ESTABLISHED` backlog queues without requiring user-space intervention.
3. **The Deadlock:** If the user-space application thread hangs (e.g., waiting on a stalled CUDA kernel or locked mutex), it cannot call `accept()` or parse incoming HTTP requests. However, the kernel continues responding to TCP port checks with instant `SYN-ACK` flags!
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
