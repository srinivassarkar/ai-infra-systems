# Stage 05: Edge Streaming Reverse Proxy & Backpressure

## 1. Objective
Configure, test, and tune a production reverse proxy (Nginx) to stream Server-Sent Events (SSE) from the inference engine to clients without buffer bloat, packet stalls, or connection dropouts.

---

## 2. The Core Problem: Why Default Reverse Proxies Break LLM Streaming

In traditional web applications, Nginx acts as an asynchronous shock absorber:
* **`proxy_buffering on;` (Default):** Nginx reads responses from the backend as fast as possible into local memory buffers (`4k` or `8k` chunks) before transmitting them across the slow WAN client connection. This frees the backend worker quickly.
* **The LLM Streaming Disaster:** An LLM generates tokens autoregressively one by one (each token is 4 to 20 bytes). With `proxy_buffering on`, Nginx **holds back the tokens in memory** waiting for 4,096 bytes to accumulate!
* **The User Experience:** The user sees a frozen screen for several seconds, followed by an unnatural, sudden burst of 50 tokens at once. The smooth typewriter effect is destroyed.

---

## 3. The Production Nginx Configuration

To achieve zero-jitter streaming:
1. `proxy_buffering off;`: Flushes every SSE chunk immediately down the TCP socket to the client.
2. `proxy_cache off;`: Prevents storing streaming responses in disk caches.
3. `proxy_set_header X-Accel-Buffering no;`: Signals upstream proxies to bypass buffering.
4. `proxy_read_timeout 300s;`: Prevents Nginx from dropping connections during long-running reasoning generations (e.g. thinking models generating 2,000+ tokens).
5. `keepalive 64;`: Maintains persistent Layer 4 TCP connection pools to the inference backend, avoiding TCP 3-way handshake overhead per request.

---

## 4. Step-by-Step Reproduction Guide

### Prerequisites
Ensure `nginx` and `llama-server` are running on your Linux host.

### Step 1: Launch Nginx with Dual Configurations
Our `nginx.conf` sets up two parallel virtual servers:
* **Port 8000:** Tuned Unbuffered Streaming (`proxy_buffering off;`)
* **Port 8001:** Anti-Pattern Buffered Streaming (`proxy_buffering on;`)

```bash
# Start Nginx using the stage config:
nginx -c /root/ai-infra-systems/stages/05-streaming-and-edge-proxy/nginx.conf

# Verify both ports are listening:
curl -I http://127.0.0.1:8000/
curl -I http://127.0.0.1:8001/
```

### Step 2: Benchmark Unbuffered Streaming (Port 8000)
```bash
python3 benchmark_streaming.py \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --concurrency 4 \
  --max-tokens 128
```

### Step 3: Benchmark Buffered Streaming (Port 8001)
```bash
python3 benchmark_streaming.py \
  --url http://127.0.0.1:8001/v1/chat/completions \
  --concurrency 4 \
  --max-tokens 128
```

Observe the difference in **TTFT**, **P99 ITL (Tail Jitter)**, and **Max ITL (Burst Stutter)** between the two endpoints.
