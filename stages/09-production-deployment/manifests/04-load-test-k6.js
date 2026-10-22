// k6 Queue-Depth Saturation Load Test
// Generates sudden concurrent traffic bursts to trip KEDA threshold past 4 replicas

import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '15s', target: 2 },   // Baseline warm-up
    { duration: '30s', target: 20 },  // High concurrency burst (trips queue threshold)
    { duration: '30s', target: 35 },  // Saturation burst (pushes past 4 replicas to Pending state)
    { duration: '20s', target: 0 },   // Cool-down
  ],
  thresholds: {
    http_req_failed: ['rate<0.05'],
  },
};

export default function () {
  const url = __ENV.TARGET_URL || 'http://localhost:8000/generate';
  const payload = JSON.stringify({
    prompt: 'Explain the difference between compute-bound prefill and memory-bandwidth bound decode.',
    max_tokens: 128,
  });

  const params = {
    headers: {
      'Content-Type': 'application/json',
    },
    timeout: '10s',
  };

  const res = http.post(url, payload, params);
  check(res, {
    'status is 200': (r) => r.status === 200,
  });

  sleep(0.1);
}
