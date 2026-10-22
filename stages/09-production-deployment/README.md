# Stage 09: Production Kubernetes Deployment, Simulated GPU Scheduling & KEDA Autoscaling

## 1. Architectural Objective

Demonstrate senior-grade Kubernetes control-plane orchestration for accelerated AI workloads by:
1. Decoupling Kubernetes control-plane scheduling mechanics from expensive physical cloud compute.
2. Simulating multi-GPU resource advertising and scheduler saturation using the Kubernetes Device Plugin API.
3. Implementing event-driven horizontal pod autoscaling via KEDA based on Prometheus queue depth (`http_requests_in_flight`), proving why traditional CPU/GPU-utilization autoscaling fails for LLM serving.
4. Stress-testing scheduler capacity limits, verifying pod `Pending` transitions when GPU capacity is exhausted, and validating node taints, tolerations, and affinity.

---

## 2. The Core Problem: The Single GPU Dilemma

A single physical GPU (such as Node 2's NVIDIA GTX 1050 Ti) can prove that a single containerized pod can bind CUDA drivers, allocate VRAM, and execute tensor kernels. 

However, a single physical GPU **cannot demonstrate horizontal scaling mechanics**:
- It cannot demonstrate multiple GPU pods scheduling across discrete allocatable resource slots.
- It cannot demonstrate scheduler queue saturation when all GPU slots are occupied.
- It cannot demonstrate horizontal pod autoscaling from 1 to $N$ replicas without thrashing physical memory.

Renting multiple multi-GPU cloud instances on AWS or GCP simply to watch the Kubernetes scheduler manage resource integers costs hundreds of dollars in idle compute waste. 

**The Solution:** Use the Kubernetes Device Plugin API locally on `kind` or `k3s` to advertise simulated GPU extended resources (`nvidia.com/gpu: 4`). This allows full validation of scheduling, taints, tolerations, and KEDA queue-depth autoscaling at zero cost, reserving real silicon for single-pod CUDA execution and spot benchmarking.

---

## 3. Architecture & Control-Plane Mechanics

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Local Kubernetes (kind)                         │
│                                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │                    Control Plane (kube-scheduler)                │  │
│  └──────────────────────────────────┬───────────────────────────────┘  │
│                                     │                                  │
│                 Schedules Pods with │ requests:                        │
│                                     │   limits: nvidia.com/gpu: 1      │
│                                     ▼                                  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Worker Node (Capacity: nvidia.com/gpu = 4)                       │  │
│  │                                                                  │  │
│  │  [Fake GPU Device Plugin DaemonSet]                              │  │
│  │    Registers via /var/lib/kubelet/device-plugins/kubelet.sock    │  │
│  │                                                                  │  │
│  │  ┌───────────────┐ ┌───────────────┐ ┌─────────────────────────┐ │  │
│  │  │ Pod Replica 1 │ │ Pod Replica 2 │ │ Pod Replica 3           │ │  │
│  │  │ (GPU Slot 0)  │ │ (GPU Slot 1)  │ │ (GPU Slot 2)            │ │  │
│  │  └───────────────┘ └───────────────┘ └─────────────────────────┘ │  │
│  │  ┌───────────────┐                                               │  │
│  │  │ Pod Replica 4 │ ◄── Node GPU Capacity Saturated (4/4)         │  │
│  │  │ (GPU Slot 3)  │                                               │  │
│  │  └───────────────┘                                               │  │
│  └──────────────────────────────────┬───────────────────────────────┘  │
│                                     │                                  │
│                                     ▼                                  │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │ Pod Replica 5: State = Pending                                   │  │
│  │ Event: 0/1 nodes available: Insufficient nvidia.com/gpu.         │  │
│  │ (In cloud EKS, this event triggers Karpenter/Cluster Autoscaler) │  │
│  └──────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
                                      ▲
                                      │ Scales Replicas (1 -> 5)
                                      │ based on queue depth
┌─────────────────────────────────────┴──────────────────────────────────┐
│ KEDA Controller (ScaledObject)                                         │
│ Trigger: Prometheus metric 'http_requests_in_flight > 2'               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Why LLM Inference Cannot Autoscale on CPU / GPU Percentage

In traditional web applications, Horizontal Pod Autoscaler (HPA) scales pods on CPU utilization (e.g. `targetCPUUtilizationPercentage: 70%`). For LLM inference servers, this fails catastrophically:

1. **The Prefill vs. Decode Asymmetry:**
   - **Prefill (Prompt Processing):** Compute-bound (high matrix GEMM utilization, GPU utilization spikes to 95–100%).
   - **Decode (Token Generation):** Memory-bandwidth-bound (sequential memory reads, GPU compute utilization often drops to 30–50% despite full memory bus saturation).
2. **False Scaling Signals:**
   - A server generating tokens for 10 concurrent streams may show only 40% GPU compute utilization, causing traditional HPA to refuse to scale or scale down.
   - However, the server's KV cache is completely full. New incoming requests are queued, inflating Time-To-First-Token (TTFT) from 50 ms to 4,000 ms (head-of-line blocking).
3. **The Solution: Queue-Depth Driven Autoscaling:**
   - KEDA monitors Prometheus metric: `http_requests_in_flight` or `llm_pending_requests`.
   - When requests exceed concurrent processing capacity (e.g. queue depth > 2 per replica), KEDA triggers pod scaling immediately, before latency degrades.

---

## 5. The 5-Step Experimental Procedure

### Step 1: Provision Multi-Node Cluster
Deploy a local `kind` cluster with dedicated worker labels:
```bash
kind create cluster --config manifests/00-cluster-kind.yaml
```

### Step 2: Deploy the Mock GPU Device Plugin
Deploy the lightweight mock device plugin DaemonSet that binds `/var/lib/kubelet/device-plugins/` and advertises 4 virtual GPU slots:
```bash
kubectl apply -f manifests/01-fake-gpu-plugin.yaml
kubectl describe node kind-worker | grep "nvidia.com/gpu"
```
*Expected Output:* `Allocatable: nvidia.com/gpu: 4`

### Step 3: Deploy the Inference Server with Extended Resource Limits
Deploy the inference workload with dual semantic probes (`/health/live` vs `/health/ready` checking model warm-up):
```bash
kubectl apply -f manifests/02-inference-deployment.yaml
```

### Step 4: Configure KEDA Event-Driven Autoscaler
Deploy KEDA and configure the `ScaledObject` pointing to Prometheus queue depth:
```bash
kubectl apply -f manifests/03-keda-scaledobject.yaml
```

### Step 5: Execute Saturation Load Test & Observe Controller Triage
From Node 3, launch concurrent traffic using `k6`:
```bash
k6 run manifests/04-load-test-k6.js
```
*Observation Checklist:*
- Traffic spike increases Prometheus `http_requests_in_flight`.
- KEDA detects threshold breach and scales Deployment replicas: $1 \to 2 \to 3 \to 4$.
- Replicas 1 through 4 successfully bind to `kind-worker` (slots 0 to 3).
- Replica 5 triggers and enters `Pending` state:
  ```bash
  kubectl get events --sort-by='.metadata.creationTimestamp'
  ```
  `Warning FailedScheduling pod/inference-worker-xyz 0/2 nodes available: Insufficient nvidia.com/gpu.`
- Verify that when traffic subsides, KEDA cleanly scales replicas back to the baseline.

---

## 6. Real Silicon Execution vs. Control Plane Simulation

To ensure complete credibility, Stage 09 documents the strict separation between control-plane simulation and hardware execution:

| Concern | Simulated Lab (`kind`) | Physical Verification (Node 2 & Spot GPU) |
| :--- | :--- | :--- |
| **Scheduler Limits** | Proves integer device allocation and `Pending` behavior. | Confirms single-pod NVIDIA Container Toolkit / CDI hooks. |
| **Autoscaling Policy** | Validates KEDA reactive loop against Prometheus metrics. | Validates cold-start model weight loading latency into VRAM. |
| **Tolerations & Taints** | Tests `nvidia.com/gpu=present:NoSchedule` rules. | Tests real CUDA driver stability and PCIe bus access. |
| **Production EKS Bridge**| Identical control plane YAML used in production. | In AWS EKS, Karpenter listens to the `Insufficient nvidia.com/gpu` event to launch real `g5.xlarge` instances. |

---

## 7. Deliverables & Artifacts

- [ ] `manifests/00-cluster-kind.yaml`: Multi-node cluster definition with worker taints.
- [ ] `manifests/01-fake-gpu-plugin.yaml`: Mock GPU Device Plugin DaemonSet advertising 4 devices.
- [ ] `manifests/02-inference-deployment.yaml`: Inference pod spec with `nvidia.com/gpu: 1` limits and dual health probes.
- [ ] `manifests/03-keda-scaledobject.yaml`: KEDA ScaledObject configured on Prometheus queue depth.
- [ ] `manifests/04-load-test-k6.js`: k6 concurrency test triggering autoscaling saturation.
- [ ] `docs/STAGE_09_BENCHMARK_REPORT.md`: Documented event logs, Grafana replica scaling graphs, and architectural post-mortem.
