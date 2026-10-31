# Stage 10: Capacity Planning, Token Economics & Hardware Sizing

> **Methodology Notice:** This stage evaluates hardware concurrency bounds and token unit economics using a mathematical capacity planning engine (`capacity_economics_engine.py`) and a 24-hour diurnal traffic trace simulation, parameterized with empirical latency measurements from Stages 01–08.

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
From Stage 03 Little's Law and slot saturation telemetry, when active requests exceed physical parallel slots ($N > S_{\mathrm{engine}}$), prompt ingestion queues in FIFO backlog, triggering exponential TTFT degradation:
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

---

## 4. Verified Sizing Matrix & Fleet Economics Output

Here is the exact output produced by `capacity_economics_engine.py`:

### A. Dual-Constraint Sizing Matrix
Model: `Qwen2.5-Coder-1.5B (Q4_K_M)` | Target SLO: $\mathrm{TTFT} \le 250\ \mathrm{ms}$, $\mathrm{TPOT} \le 40\ \mathrm{ms}$:

| Hardware Platform | Memory Limit ($N_{\mathrm{mem}}$) | SLO Latency Knee ($N_{\mathrm{SLO}}$) | Binding Constraint | Effective Production Slots |
| :--- | :---: | :---: | :--- | :---: |
| **Node 3: NVIDIA L4 (Cloud Spot)** | 414 slots | **8 slots** | Latency (SLO Knee) | **8 slots** |
| **Node 2: Mac Studio M3 Ultra** | 4,656 slots | **32 slots** | Latency (SLO Knee) | **32 slots** |
| **Node 1: Edge GTX 1050 Ti** | 48 slots | **2 slots** | Latency (SLO Knee) | **2 slots** |

**The Plain Takeaway:** For models under 7B parameters, memory is cheap. The latency cliff hits **long before VRAM runs out**. If you size your cluster based only on memory math, you will allocate hundreds of slots to a single GPU and completely destroy user latency.

---

### B. Financial Cost Per 1 Million Tokens ($/1M Tokens)
Calculated across realistic enterprise utilization levels:

| Hardware Platform | Hourly Cost | 100% Utilized | 50% Daytime Load | 20% Off-Peak Load |
| :--- | :---: | :---: | :---: | :---: |
| **Node 3: NVIDIA L4 Spot** | $0.350 / hr | $0.51 / 1M | **$1.01 / 1M** | $2.54 / 1M |
| **Node 2: Mac Studio M3 Ultra** | $0.288 / hr | $0.33 / 1M | **$0.67 / 1M** | $1.67 / 1M |
| **Node 1: Edge GTX 1050 Ti** | $0.017 / hr | $0.11 / 1M | **$0.22 / 1M** | $0.56 / 1M |

**The Economic Takeaway:** At typical 50% daytime utilization, running self-hosted models on NVIDIA L4 cloud spot costs **$1.01 per 1 million tokens**, which is over **$2.5\times$ cheaper than commercial closed APIs** ($2.50+/1M). Owned Apple Silicon brings this down to **$0.67 per 1M tokens**.

---

### C. 24-Hour Diurnal Fleet Simulation: Static vs. KEDA Autoscaling
Simulating a typical enterprise traffic day (8h peak at 24 req/s, 8h mid-day at 12 req/s, 8h night at 2 req/s):

```text
Static 24/7 Provisioning Cost (3 Pods Constant):  $25.20 / day
KEDA Event-Driven Autoscaled Cost (1-3 Pods):     $16.80 / day
--------------------------------------------------------------------------
Daily Financial Savings:                          $8.40 (33.3% reduction)
Morning Scaling Lag SLO Window:                   0.86 minutes
```

**The SRE Takeaway:** Event-driven autoscaling with KEDA reduces daily cloud GPU costs by **33.3%**. To prevent the 0.86-minute latency violation during the morning traffic jump, schedule a pre-warming rule (Cron HPA) 5 minutes before peak hours.

For the full unit economics mathematical derivations, read the [Stage 10 Full Report](REPORT.md).
