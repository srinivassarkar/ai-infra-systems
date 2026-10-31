# Stage 10: Capacity Planning, Token Economics & Hardware Sizing Report

## 1. Executive Summary & The Dual-Constraint Capacity Axiom

Capacity planning in enterprise LLM serving is frequently miscalculated by dividing available GPU VRAM by token KV cache sizes. This naive memory calculation assumes that if tokens physically fit into memory, the hardware can serve them within SLA boundaries.

Stage 10 empirically repudiates this approach, proving the **Dual-Constraint Capacity Axiom**:
$$\text{Effective Concurrency Capacity} = \min\left(N_{\mathrm{VRAM\ Ceiling}},\ N_{\mathrm{SLO\ Latency\ Knee}}\right)$$

For models $\le 7\mathrm{B}$ parameters operating on modern accelerators (such as the NVIDIA L4 with 24 GB GDDR6), Grouped-Query Attention (GQA) renders KV cache memory remarkably cheap. Consequently, **the latency knee binds at $N=8$ slots long before VRAM runs out** ($414$ slots). Sizing an inference cluster on VRAM arithmetic alone overprovisions concurrency by $50\times$, destroying user TTFT.

---

## 2. Empirical Sizing Matrix: Memory Ceiling vs. Latency Knee

*Model: `Qwen2.5-Coder-1.5B-Instruct-Q4_K_M` (0.85 GB weights, GQA 28 layers, 2 KV heads, 64 head_dim).*  
*Context Length: $4{,}096\ \mathrm{tokens}$ per user.*  
*Target SLO: $\mathrm{TTFT} \le 250\ \mathrm{ms}$, $\mathrm{TPOT} \le 40\ \mathrm{ms}$.*

| Hardware Platform | Total Memory (GB) | Memory Bandwidth (GB/s) | Theoretical VRAM Limit ($N_{\mathrm{mem}}$) | Measured SLO Knee ($N_{\mathrm{SLO}}$) | Binding Constraint | Effective Production Capacity |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NVIDIA L4 (Cloud Spot)** | 24.0 GB | 300 GB/s | 414 slots | **8 slots** | **Latency (SLO Knee)** | **8 slots** |
| **Mac Studio M3 Ultra (Node 1)** | 256.0 GB | 800 GB/s | 4,656 slots | **32 slots** | **Latency (SLO Knee)** | **32 slots** |
| **Edge GTX 1050 Ti (Node 2)** | 4.0 GB | 112.0 GB/s | 48 slots | **2 slots** | **Latency (SLO Knee)** | **2 slots** |

### Key Systems Analysis:
* On the **NVIDIA L4**, $24\ \mathrm{GB}$ of VRAM can hold $414$ concurrent user KV caches. However, our Stage 03 Little's Law benchmarks proved that beyond $N=8$, TTFT exploded from $43\ \mathrm{ms}$ to **$428.21\ \mathrm{ms}$**, breaching the $250\ \mathrm{ms}$ SLO. Thus, the effective capacity is strictly **8 slots**.
* On the **Mac Studio M3 Ultra**, $800\ \mathrm{GB/s}$ of unified memory bandwidth sustains up to **32 concurrent streams** within SLO limits.
* On the **GTX 1050 Ti**, despite holding 48 slots in VRAM, compute saturation and memory bandwidth limit SLO compliance to **2 concurrent slots**.

---

## 3. Financial Cost Per 1 Million Tokens ($/1M Tokens)

Because inference clusters rarely operate at 100% constant saturation without queue explosion, cost per token must be modeled across realistic production utilization levels:
* **100% Utilization:** Theoretical lower bound (continuous peak saturation).
* **50% Daytime Utilization:** Typical business day traffic with headroom buffers.
* **20% Off-Peak Utilization:** Nighttime / weekend background traffic.

| Hardware Platform | All-Inclusive Hourly Cost | 100% Peak Utilization | 50% Daytime Production | 20% Off-Peak Night |
| :--- | :---: | :---: | :---: | :---: |
| **NVIDIA L4 (Cloud Spot)** | $\$0.350\ /\ \mathrm{hr}$ | **$\$0.5074$** | **$\$1.0147$** | **$\$2.5368$** |
| **Mac Studio M3 Ultra (Node 1)** | $\$0.288\ /\ \mathrm{hr}$ | **$\$0.3333$** | **$\$0.6667$** | **$\$1.6667$** |
| **Edge GTX 1050 Ti (Node 2)** | $\$0.017\ /\ \mathrm{hr}$ | **$\$0.1124$** | **$\$0.2249$** | **$\$0.5622$** |

### Economic Takeaways:
1. **Self-Hosted Cloud vs Closed APIs:** At 50% daytime production utilization, self-hosting on NVIDIA L4 cloud spot costs **$\$1.01$ per 1M tokens**. This represents a **$60\% - 75\%$ cost savings** over closed proprietary API endpoints (which typically bill $\$2.50 - \$5.00$ per 1M output tokens).
2. **Owned Silicon Economics:** Owned hardware (Mac Studio M3 Ultra) amortized over a 3-year lifecycle delivers tokens at **$\$0.67$ per 1M tokens** at 50% utilization, making it an exceptional testbed for internal enterprise R&D.

---

## 4. 24-Hour Diurnal Fleet Simulation: Static vs. KEDA Autoscaling

We simulated a 24-hour enterprise business cycle:
* **08:00 - 16:00 (Peak Business Hours):** 24 concurrent requests $\implies 3$ pods required.
* **16:00 - 24:00 (Evening / Mid-Day):** 12 concurrent requests $\implies 2$ pods required.
* **00:00 - 08:00 (Nighttime Off-Peak):** 2 concurrent requests $\implies 1$ pod required.

### Financial & Operational Comparison (NVIDIA L4 Fleet @ $0.35/hr/pod)

| Metric | Static 24/7 Provisioning (3 Pods Constant) | KEDA Event-Driven Autoscaling (1 - 3 Pods) | Delta / Economic Impact |
| :--- | :---: | :---: | :---: |
| **Total Daily Expenditure** | **$\$25.20\ /\ \text{day}$** | **$\$16.80\ /\ \text{day}$** | **-$8.40 / day (33.3% Cost Reduction)** |
| **Monthly Projected Spend** | $\$756.00\ /\ \text{mo}$ | $\$504.00\ /\ \text{mo}$ | **-$252.00 / month saved per model** |
| **Scaling-Lag SLO Window** | $0.00\ \text{min}$ (Pre-warmed) | **$0.86\ \text{min}$ ($51.73\ \text{s}$)** | Morning burst queue wait during pod spin-up |

### Production SRE Synthesis & Recommendations:
1. **KEDA Delivers $33.3\%$ Direct Cost Reduction:** Scaling from 1 pod at night to 3 pods during peak hours saves $\$8.40$ per day per model instance.
2. **Mitigating the 51.7s Morning Scaling Lag:** To prevent the 0.86-minute SLO violation window during the 08:00 AM burst transition, configure a Kubernetes **Scheduled Cron HPA** to pre-warm 1 additional pod at **07:55 AM**, absorbing the burst seamlessly without paying for 24/7 static overprovisioning.
