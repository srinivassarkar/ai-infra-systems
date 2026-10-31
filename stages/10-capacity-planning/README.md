# Stage 10: Capacity Planning, Token Economics & Hardware Sizing

## 1. Executive Summary & Objective

The ultimate mandate of a Platform / AI Infrastructure Engineer is translating hardware physics into predictable capacity boundaries and defensible cost economics.

Stage 10 establishes the **Enterprise Capacity Planning & Token Economics Framework**:
1. **The Dual-Constraint Sizing Matrix:** Proves that inference capacity is bounded by $\min(N_{\mathrm{VRAM}}, N_{\mathrm{SLO}})$, demonstrating why memory-only VRAM calculators fatally overprovision concurrency.
2. **SLO-Constrained Goodput Economics:** Measures the financial cost per 1M tokens across the heterogeneous fleet (NVIDIA L4 Cloud Spot, Apple Mac Studio M3 Ultra, Edge GTX 1050 Ti) across realistic utilization levels ($100\%$, $50\%$, $20\%$).
3. **24-Hour Diurnal Trace Simulation:** Replays enterprise traffic patterns to compare 24/7 static provisioning against KEDA event-driven autoscaling, measuring both dollar savings and scaling-lag SLO breach minutes.

---

## 2. Theoretical Framework: Sizing Bounds

### The Memory Capacity Ceiling ($N_{\mathrm{mem}}$)
$$N_{\mathrm{mem}} = \left\lfloor \frac{V_{\mathrm{total}} - V_{\mathrm{weights}} - V_{\mathrm{overhead}}}{S \times L \times 2 \times H_{\mathrm{kv}} \times D_{\mathrm{head}} \times B_{\mathrm{element}}} \right\rfloor$$

### The Latency Knee Bound ($N_{\mathrm{SLO}}$)
From Stage 03 Little's Law and slot saturation telemetry, when active requests exceed the physical parallel slots ($N > S_{\mathrm{engine}}$), prompt ingestion queues in FIFO backlog, triggering exponential TTFT degradation:
$$\mathrm{TTFT}_{\mathrm{effective}} = \mathrm{TTFT}_{\mathrm{prefill}} + \frac{Q \times T_{\mathrm{request}}}{S_{\mathrm{engine}}}$$

### The Universal Capacity Law
$$\text{Production Concurrency Limit} = \min\left(N_{\mathrm{mem}}, N_{\mathrm{SLO}}\right)$$

---

## 3. Step-by-Step Reproduction Guide

### Run the Capacity Planning & Economics Engine:
```bash
cd ~/ai-infra-systems/stages/10-capacity-planning

python3 capacity_economics_engine.py
```

The script will output:
1. The Dual-Constraint Sizing Matrix across the 3 fleet platforms.
2. The Financial Cost per 1M Tokens table at $100\%$, $50\%$, and $20\%$ utilization.
3. The 24-Hour Diurnal Fleet Simulation comparing static cluster spend vs KEDA event-driven autoscaling.
