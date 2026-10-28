# Stage 03: Concurrency, Queuing & Head-of-Line Blocking

## The Core Question
In single-tenant inference (Stage 01), we measured raw silicon speed under one user. But production inference servers handle concurrent streams.
**What happens when multiple users hit the server simultaneously? Where is the saturation knee point where queuing wait time overtakes actual generation time?**

---

## The Three Core Architectural Upgrades

### 1. Closed-Loop vs. Open-Loop Poisson Load Generation
* **Closed-Loop Concurrency ($N$ workers):** Each worker sends a request, waits for it to complete, and immediately sends the next. 
  * *Purpose:* Measures aggregate throughput scaling and validates **Little's Law** ($\text{Concurrency } N = \text{Throughput} \times \text{Latency}$).
  * *The Trap:* When the server slows down, clients naturally wait longer, throttling the request rate. This **coordinated omission** hides queuing delays.
* **Open-Loop Poisson Arrivals ($\lambda$ req/s):** Dispatches requests at a Poisson rate $\lambda$ independent of whether previous requests have completed.
  * *Purpose:* Exposes true queuing theory behavior ($W \propto \frac{1}{1 - \rho}$), revealing the non-linear tail latency explosion at the saturation knee.

### 2. The Context Window Sizing Law
`llama-server` divides `-c` evenly across all parallel slots. 
* *The Bug:* Launching with `-c 2048` and `--parallel 8` silently allocates only **256 tokens per slot**, causing long prompts to truncate.
* *The Rule:* You must set:
  ```text
  -c = parallel * per_slot_context
  ```
  *(e.g., `--parallel 8 -c 16384` ensures every slot has a full 2,048 token window).*

### 3. The Two Forms of Head-of-Line (HoL) Blocking
* **Queue-Level HoL (Request-Level):** When offered load exceeds available slots (`clients > parallel`), requests wait in a FIFO queue. One long request at the front of the line stalls all subsequent requests.
* **Prefill-Decode Interference (Iteration-Level):** In continuous batching, all active slots share one forward pass per step. When a user with a 1,500-token prompt arrives, its prompt prefill consumes compute cores, delaying the decode steps of other active streams and causing visible **inter-token latency (ITL) stutter**.
* **The Mitigation Lever (`-ub` Chunked Prefill):** Capping the physical micro-batch size (`-ub 128` vs `-ub 512` vs `-ub 2048`) chunks large prefills across multiple steps, protecting running decodes from stutter.

---

## The 3 Definitions of Saturation

1. **The Throughput Knee:** The concurrency level or arrival rate $\lambda$ where aggregate output tokens/sec stops growing (hardware compute/memory pipeline is fully saturated).
2. **The Queue-Dominated Point:** The inflection point where client-side queuing delay exceeds the actual time spent generating tokens:
   $$\text{Queue Wait} > \text{Generation Service Time}$$
3. **The SLO Goodput Limit:** The highest arrival rate where $\ge 95\%$ of requests satisfy production Service Level Objectives ($\mathrm{TTFT} < 500\ \mathrm{ms}$ and $\mathrm{TPOT} < 50\ \mathrm{ms}$).

---

## Quickstart: Step-by-Step Reproduction Guide

### Prerequisites
Install `httpx` for non-blocking asynchronous load generation:
```bash
pip install httpx
```

### Step 1: Launch `llama-server` with Multi-Slot Allocation & Metrics
Run the server with 8 slots, context scaled to $8 \times 2048 = 16384$, and `--metrics` enabled:

```bash
# On Mac Studio M3 Ultra or Linux:
~/llama-bin/llama-server \
  -m ~/models/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf \
  -ngl 99 \
  --parallel 8 \
  -c 16384 \
  --metrics \
  --port 8080 \
  --host 0.0.0.0
```

Verify the server is healthy:
```bash
curl http://127.0.0.1:8080/health
# Check active slots:
curl -s http://127.0.0.1:8080/slots | jq .
```

---

### Step 2: Run the Closed-Loop Concurrency Sweep (Experiment A)
Measure throughput scaling and validate Little's Law across $N \in \{1, 2, 4, 8, 16\}$:

```bash
cd ai-infra-systems/stages/03-concurrency-and-queues

# Run N = 1, 2, 4, 8, 16 (30 seconds per level):
for n in 1 2 4 8 16; do
  python3 benchmark_concurrency.py \
    --endpoint http://127.0.0.1:8080/v1/chat/completions \
    --mode closed \
    --concurrency $n \
    --duration 30 \
    --csv closed_loop_telemetry.csv
done
```

---

### Step 3: Run the Open-Loop Poisson Arrival Sweep (Experiment B)
Sweep arrival rates ($\lambda$) past the hardware capacity to witness the non-linear tail latency explosion:

```bash
# Sweep lambda from 1 req/s up to 16 req/s:
for rate in 1 2 4 8 12 16; do
  python3 benchmark_concurrency.py \
    --endpoint http://127.0.0.1:8080/v1/chat/completions \
    --mode open \
    --rate $rate \
    --duration 30 \
    --csv open_loop_telemetry.csv
done
```

---

### Step 4: Run the Mixed-Workload Intruder Test (Experiment C)
Inject a 1,500-token prompt mid-flight into background short streams to measure prefill-decode interference and ITL stutter:

```bash
python3 benchmark_concurrency.py \
  --endpoint http://127.0.0.1:8080/v1/chat/completions \
  --mode intruder \
  --concurrency 4 \
  --duration 30 \
  --csv intruder_telemetry.csv
```

---

### Step 5: Run the Discrete-Event Batching Simulator
Compare static batching versus continuous iteration-level scheduling on an identical synthetic arrival trace:

```bash
python3 batching_simulator.py
```
