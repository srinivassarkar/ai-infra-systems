# Stage 02: Cold-Start Behavior & Model Lifecycle

## Objective
Measure what happens to system memory, disk I/O, and token latency when a model is loaded into silicon from disk versus when it is preloaded in warm memory.

## Key Questions
1. How does cold TTFT (initial load + weight compilation) compare to warm TTFT?
2. What are the memory allocation steps (disk read -> host RAM -> GPU VRAM / UMA allocation)?
3. How do daemon preload strategies (`keep_alive`, background warmup requests) eliminate latency spikes?
