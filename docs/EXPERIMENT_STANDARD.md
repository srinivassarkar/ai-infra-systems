# Scientific Experiment Standard & Report Template

Every stage in this repository follows this standardized engineering template to ensure all findings are reproducible, verified, and free of conjecture.

---

## Experiment Report Template

```markdown
# Stage [XX]: [Experiment Title]

## 1. System Context & Hardware
* Target Node: [e.g., Node 1 (NVIDIA GTX 1050 Ti) / Node 2 (Apple Mac Studio M3 Ultra)]
* Runtime Engine: [e.g., vLLM / llama.cpp / mlx_lm / Ollama]
* Model Identifier: [e.g., Qwen2.5-Coder-1.5B-Instruct-Q4_K_M]
* Ingress / Networking: [e.g., Direct Socket / Nginx Reverse Proxy / Tailscale]

---

## 2. Hypothesis
* Explicit statement of expected system behavior under test.
* Example: "Under concurrency > 4, default static batching will cause P90 TTFT to degrade by >500% due to Head-of-Line blocking in the prefill queue."

---

## 3. Test Methodology & Execution
* Command or script executed:
```bash
python benchmark_concurrency.py --concurrency 8 --prompts 32
```
* Telemetry collection commands:
  * GPU memory / utilization: `nvidia-smi` / `asitop` / Prometheus exporter
  * System diagnostics: `vmstat 1`, `pidstat -r 1`, `ss -s`

---

## 4. Observed Telemetry & Data
* Summary Table:
| Concurrency Level | TTFT P50 (ms) | TTFT P90 (ms) | TPOT Avg (ms) | Throughput (tok/s) | GPU VRAM Peak (MiB) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 User | ... | ... | ... | ... | ... |
| 4 Users | ... | ... | ... | ... | ... |
| 8 Users | ... | ... | ... | ... | ... |

---

## 5. Root Cause Analysis (RCA)
* Deep dive into the underlying hardware, kernel, or software mechanism causing the observed behavior.
* What physical bus or queue bottleneck was saturated?
* Why did the software stack behave this way?

---

## 6. Engineering Remediation & Configuration
* The production configuration, kernel parameter tuning, or architectural fix applied to resolve or mitigate the issue.
* Specific configuration files or code changes committed.

---

## 7. Verification & Post-Fix Benchmarking
* Re-run of the test proving the fix successfully resolved the bottleneck or stabilized the system.

---

## 8. SRE & Platform Interview Takeaways
* How to articulate this specific experiment during a systems or platform engineering interview.
```
