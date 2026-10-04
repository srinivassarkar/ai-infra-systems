# Stage 06: Failure Injection, Daemon Crashes & Recovery

## Objective
Simulate real-world production failure modes (sudden process termination, unhandled exceptions, GPU driver stalls, and port conflicts) and verify automated recovery mechanisms.

## Key Questions
1. How fast can a `systemd` daemon unit (with `Restart=always`, `RestartSec=3s`) restore service availability after `kill -9`?
2. How do health and readiness probes (`/healthz` vs `/ready`) prevent traffic from hitting an un-initialized model during restart?
3. How do we implement graceful shutdown (`SIGTERM` handling) so in-flight token streams are completed before the process terminates?
