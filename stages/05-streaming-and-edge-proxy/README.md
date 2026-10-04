# Stage 05: Edge Streaming Reverse Proxy & Backpressure

## Objective
Configure, test, and tune a production reverse proxy (Nginx) to stream Server-Sent Events (SSE) from the inference engine to clients without buffer bloat or connection stalls.

## Key Questions
1. Why does standard reverse proxy buffering (`proxy_buffering on;`) break the typewriter streaming effect?
2. How do `X-Accel-Buffering: no;`, `proxy_read_timeout 300s;`, and TCP keep-alives stabilize long-lived generation sessions?
3. What happens when a slow or disconnected client drops the connection, and how does backpressure propagate to the inference engine?
