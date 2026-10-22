# Stage 09: Production Kubernetes Deployment & Event-Driven KEDA Autoscaling

## Objective
Package, supervise, and deploy the entire inference stack for long-term production reliability using Linux `systemd`, container specs, and Kubernetes manifests with event-driven queue-depth autoscaling via KEDA.

## Target Silicon & Topology
* **Local k3s Cluster:** Demonstrating multi-replica horizontal autoscaling without hardware thrashing.
* **Worker Replicas:** Lightweight CPU/accelerator worker pods scaling dynamically based on Prometheus metrics.

## Key Questions
1. How do we configure Kubernetes Pod resources, tolerations, and dual semantic health probes (`/health/live` vs `/health/ready`) for inference containers?
2. Why does scaling on CPU/GPU percentage fail for LLM inference, and how do we configure KEDA `ScaledObject` to scale worker replicas based on Prometheus queue depth (`http_requests_in_flight`)?
3. How do we configure scale-to-zero during idle windows and verify cold-start buffering behavior under sudden traffic spikes?
4. How do we supervise daemon execution on bare-metal nodes using Linux `systemd` units (`LimitNOFILE=65536`, `After=nvidia-persistenced.service`)?
