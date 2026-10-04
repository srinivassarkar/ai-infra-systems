# Stage 03: Concurrency, Queueing & Head-of-Line Blocking

## Objective
Profile how latency percentiles (P50, P90, P99) and token throughput behave as concurrent user connections scale from 1 to 16.

## Key Questions
1. Why does P90 TTFT explode non-linearly under concurrency?
2. What is Head-of-Line (HoL) prefill blocking, and how does static batching exacerbate it?
3. What is the saturation point of the target hardware before queue wait time exceeds generation time?
