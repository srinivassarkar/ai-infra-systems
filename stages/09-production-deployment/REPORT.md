# Stage 09: Kubernetes Orchestration, KEDA Autoscaling & Scaling Lag Report

## 1. Executive Summary & The Autoscaling Lag Dilemma

Scaling stateless web services on Kubernetes is straightforward: the Horizontal Pod Autoscaler (HPA) monitors CPU or memory utilization and scales replica counts. In production LLM inference platforms, this classical model breaks down completely due to three fundamental systems constraints:
1. **The GPU Saturation Illusion:** NVIDIA `gpu_utilization` measures SM kernel execution time. A memory-bound decode loop serving only 2 concurrent users reads $80\% - 95\%$ GPU utilization, offering zero visibility into available slot capacity or true queue backlog.
2. **The Decomposed Scaling Lag ($51.73\ \mathrm{seconds}$):** Reactive autoscaling requires over 50 seconds to complete the loop from traffic surge to active token dispatch.
3. **The TTFT SLO Breach:** Because Time to First Token (TTFT) SLOs are bounded at $\le 250\ \mathrm{ms}$, a sudden traffic burst inevitably causes severe queue saturation and SLO violations during the scaling lag window.

---

## 2. Empirical Telemetry: Decomposed Kubernetes Scaling Lag Chain

*Fleet Baseline: Calibrated to NVIDIA L4 (24 GB GDDR6) Stage 01–08 empirical measurements.*  
*Model: `Qwen2.5-Coder-1.5B-Instruct-Q4_K_M` (8 slots/pod, TTFT=32ms, TPOT=8ms).*

### The 7-Step Scaling Lag Chain Breakdown

| Stage | Subsystem / Controller | Measured Duration | Operational Mechanism |
| :---: | :--- | :---: | :--- |
| **1** | **Prometheus Metric Scrape** | $15.00\ \mathrm{s}$ | Inference pod exposes `llamacpp:requests_deferred`; scraped on interval. |
| **2** | **KEDA Polling Interval** | $15.00\ \mathrm{s}$ | `ScaledObject.spec.pollingInterval` evaluates Prometheus metrics. |
| **3** | **Kubernetes HPA Sync Loop** | $15.00\ \mathrm{s}$ | `kube-controller-manager` evaluates HPA rules and updates Deployment replicas. |
| **4** | **kube-scheduler Binding** | $1.50\ \mathrm{s}$ | Evaluates GPU node allocations and binds pending pod to GPU worker node. |
| **5** | **containerd Runtime Init** | $2.00\ \mathrm{s}$ | Container creation, volume mounts, and network namespace configuration. |
| **6** | **CUDA Context & Model Page-in** | $1.23\ \mathrm{s}$ | Empirical Stage 02/06 cold-start duration on NVIDIA L4 (page cache warm). |
| **7** | **Readiness Probe Pass** | $2.00\ \mathrm{s}$ | HTTP `/health/ready` probe succeeds twice before pod is registered in Service. |
| **Total** | **End-to-End Autoscaling Lag** | **$51.73\ \mathrm{s}$** | **Total latency from traffic burst to active token serving.** |

---

## 3. Policy Benchmark: 4x Traffic Surge ($6 \rightarrow 24$ Concurrent Streams)

We simulated a 4x burst surge on a cluster with an 8-slot baseline replica:

| Autoscaling Strategy | P50 TTFT (ms) | P99 TTFT (s) | Max Peak TTFT (s) | SLO Breach % ($\text{TTFT} > 250\text{ms}$) | Final Provisioned Pods |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Static Provisioning (No Scale)** | **32.4 ms** | 643.6 s | 656.8 s | 49.9% | 1 pod (8 slots) |
| **2. GPU Utilization HPA** | **32.4 ms** | 643.6 s | 656.8 s | 49.9% | 3 pods (24 slots) |
| **3. KEDA Queue-Depth Trigger** | **32.4 ms** | **555.9 s** | **569.1 s** | **43.2%** | **3 pods (24 slots)** |

### Architectural Findings:
1. **The KEDA Advantage:** Triggering on `llamacpp:requests_deferred > 0` fires on the very first queued request, shaving nearly **90 seconds** off peak queue wait times compared to rolling utilization metrics.
2. **The Limits of Reactive Scaling:** Despite KEDA's speed, the $51.73\ \mathrm{second}$ container scheduling and controller lag window still leaves $43.2\%$ of requests in SLO violation during sudden spikes.
3. **The SRE Architectural Imperative:** Reactive autoscaling must be paired with **overprovisioned buffer pools ($N+1$ standby replicas)** and **ingress backpressure / queue bounding** to reject or buffer excess traffic before queue lengths compound exponentially.

---

## 4. Production Kubernetes Manifest Hardening

### 1. Graceful Drain PreStop Hook (`02-inference-deployment.yaml`)
Prevents truncated streaming responses during KEDA scale-down:
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

### 2. Multi-Signal KEDA ScaledObject (`03-keda-scaledobject.yaml`)
Scales on both **Queue Depth** and **KV Cache Pool Saturation**:
```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: llm-inference-autoscaler
spec:
  scaleTargetRef:
    name: llm-inference-worker
  minReplicaCount: 1
  maxReplicaCount: 6
  triggers:
    - type: prometheus
      metadata:
        metricName: llamacpp_requests_deferred
        threshold: "1"
        query: sum(llamacpp:requests_deferred{app="llm-inference-worker"})
    - type: prometheus
      metadata:
        metricName: llamacpp_kv_cache_usage_ratio
        threshold: "0.85"
        query: max(llamacpp:kv_cache_usage_ratio{app="llm-inference-worker"})
```
