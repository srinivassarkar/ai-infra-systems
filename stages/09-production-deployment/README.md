# Stage 09: Kubernetes Orchestration, KEDA Autoscaling & Scaling Lag Analysis

## 1. Executive Summary & The Autoscaling Lag Dilemma

Scaling stateless microservices on Kubernetes is typically driven by CPU/Memory utilization via the standard Horizontal Pod Autoscaler (HPA). In LLM inference clusters, **CPU/GPU utilization metrics fail as autoscaling indicators**:
1. **The GPU Saturation Illusion:** NVIDIA `gpu_utilization` measures the percentage of time GPU kernels are running on the SMs. A memory-bound decode loop serving only 2 concurrent streams reads $80\% - 95\%$ GPU utilization, offering zero visibility into true slot headroom.
2. **The Decomposed Scaling Lag:** Reactive autoscaling takes **$\sim 46.7\ \mathrm{seconds}$** from the moment traffic bursts until a newly scheduled replica begins serving traffic.
3. **The TTFT SLO Breach:** Because user TTFT SLOs are bounded at $\le 250\ \mathrm{ms}$, a reactive autoscaler cannot prevent latency degradation during sudden sharp bursts unless provisioned with leading queue-depth signals and ingress buffer headroom.

---

## 2. The 7-Step Kubernetes Scaling Lag Decomposition

```text
Traffic Burst Event (t = 0.0s)
  │
  ├── 1. Prometheus Scrape Interval (15.0s)
  │     └─ Engine exposes llamacpp:requests_deferred; scraped on cadence
  ├── 2. KEDA Polling Interval (15.0s)
  │     └─ KEDA queries Prometheus for metric threshold evaluation
  ├── 3. HPA Controller Sync Period (15.0s)
  │     └─ horizontal-pod-autoscaler-sync-period updates Deployment replicas
  ├── 4. kube-scheduler Node Binding (1.5s)
  │     └─ Evaluates GPU resources and binds pod to worker node
  ├── 5. containerd Image Runtime Init (2.0s)
  │     └─ Container creation, volume mounts, network namespace setup
  ├── 6. CUDA Context Init & Model Weights Page-in (1.2s)
  │     └─ Stage 02 L4 measured cold-start duration
  └── 7. Readiness Probe Stabilization (2.0s)
        └─ HTTP /health/ready passes twice, adding pod to Service endpoints
  │
New Replica Serving Traffic (Total Elapsed Time: ~46.7s)
```

---

## 3. Scale-Down & The In-Flight Stream Termination Trap

When traffic recedes, Kubernetes initiates pod termination. In naive deployments:
* Kubelet sends `SIGTERM` directly to the container.
* In-flight autoregressive streaming responses (e.g., token 32 of 128) are abruptly terminated.
* Paying clients receive truncated responses and `HTTP 502 Bad Gateway` errors.

### The Production SRE Solution: Graceful Drain PreStop Hook
```yaml
spec:
  terminationGracePeriodSeconds: 60
  containers:
    - name: inference-engine
      lifecycle:
        preStop:
          exec:
            command: ["/bin/sh", "-c", "curl -s http://127.0.0.1:8000/health/drain && sleep 15"]
```
1. Kubelet invokes the `preStop` hook before issuing `SIGTERM`.
2. `/health/drain` marks readiness probe as `503 DRAINING`, immediately removing the pod from active Service endpoints.
3. Existing in-flight requests finish generating tokens within the 15-second grace window before the container is terminated.

---

## 4. Step-by-Step Reproduction Guide

### Run the Autoscaling Lag Simulation:
```bash
cd ~/ai-infra-systems/stages/09-production-deployment

python3 simulate_scaling_lag.py
```

The simulator models a 4x burst surge ($6 \rightarrow 24$ concurrent streams) against our calibrated NVIDIA L4 telemetry, comparing **Static Baseline**, **GPU Utilization HPA**, and **KEDA Queue-Depth Triggering**.
