# Stage 09: Kubernetes Orchestration, KEDA Autoscaling & Scaling Lag Analysis

> **Methodology Notice:** This stage evaluates the 7-step Kubernetes scaling lag chain via a calibrated discrete-event simulation (`simulate_scaling_lag.py`) to model controller sync loops, image runtime init, and container startup times before physical cluster deployment. The included Kubernetes manifests (`01-namespace.yaml` through `04-service-ingress.yaml`) are verified declarative blueprints designed for production clusters.

## 1. Executive Summary & The Autoscaling Lag Dilemma

Scaling stateless microservices on Kubernetes is typically driven by CPU/Memory utilization via the standard Horizontal Pod Autoscaler (HPA). In LLM inference clusters, **CPU/GPU utilization metrics fail as autoscaling indicators**:
1. **The GPU Saturation Illusion:** NVIDIA `gpu_utilization` measures the percentage of time GPU kernels are running on the compute cores. A memory-bound decode loop serving only 2 concurrent streams reads $80\% - 95\%$ GPU utilization, offering zero visibility into true slot headroom.
2. **The Decomposed Scaling Lag:** Reactive autoscaling takes **$51.73\ \mathrm{seconds}$** from the moment traffic bursts until a newly scheduled replica begins serving traffic.
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
  ├── 6. CUDA Context Init & Model Weights Page-in (1.23s)
  │     └─ Measured cold-start duration (Stage 02)
  └── 7. Readiness Probe Stabilization (2.0s)
        └─ HTTP /health/ready passes twice, adding pod to Service endpoints
  │
New Replica Serving Traffic (Total Elapsed Time: 51.73s)
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

---

## 5. Measured Simulation Results: The 51.73s Lag Chain

Here is the exact simulation output modeling a $4\times$ burst surge ($6 \rightarrow 24$ concurrent streams) against our calibrated telemetry:

```text
==========================================================================
 AI INFRA SYSTEMS - STAGE 09: KUBERNETES & KEDA AUTOSCALING SIMULATION
==========================================================================
 1. Prometheus Scrape Interval:            15.00 s
 2. KEDA ScaledObject Polling Interval:    15.00 s
 3. Kubernetes HPA Controller Sync Period: 15.00 s
 4. kube-scheduler Pod Binding:             1.50 s
 5. containerd Image Runtime Init:          2.00 s
 6. CUDA Context Init & Weights Page-in:    1.23 s
 7. Readiness Probe HTTP /health/ready:     2.00 s
--------------------------------------------------------------------------
 TOTAL MEASURED SCALING LAG:               51.73 s
==========================================================================
```

### Policy Comparison Under a Sudden 4x Traffic Surge:

| Autoscaling Strategy | P50 TTFT | P99 TTFT | SLO Breach % | Final Pods |
| :--- | :---: | :---: | :---: | :---: |
| **Static Baseline (1 Pod)** | 32.4 ms | 643,619 ms | **49.9%** | 1 pod |
| **GPU Utilization HPA** | 32.4 ms | 643,619 ms | **49.9%** | 3 pods |
| **KEDA Queue-Depth Trigger** | 32.4 ms | 555,907 ms | **43.2%** | 3 pods |

### What These Numbers Mean in Simple Terms:
1. **Why GPU Utilization Fails as an Autoscaler:** When traffic surged from 6 to 24 streams, GPU utilization was already at 90% during normal load. Standard HPA took over 50 seconds to scale up, during which half of all incoming requests breached the latency SLO.
2. **The KEDA Advantage:** Using queue depth (`requests_deferred > 0`) signals overload the exact moment requests start queuing, scaling pods faster and cutting SLO violation time.
3. **The SRE Golden Rule:** Reactive autoscaling alone can never protect tight latency SLOs ($\le 250\ \mathrm{ms}$) against sudden spikes. Clusters must maintain pre-warmed headroom (e.g. 1 buffer pod or scheduled Cron HPA ahead of morning peak hours).

For the complete manifest definitions and deployment guide, read the [Stage 09 Full Report](REPORT.md).
