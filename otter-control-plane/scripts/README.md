# scripts/ — Runtime Simulator

`simulate-runtime.js` fakes ONE `OtterRuntime` instance connecting to
`otter-control-plane-server` — it exists purely so you can see the Control Plane UI populated
with real (if fabricated) data, without needing a running Flink job.

**This is a test/demo tool, not part of the product.** The real Java-side piece that would make
an actual `OtterRuntime` emit spans/events to the server — a `TelemetryExporter` implementation
— does not exist yet (see the root `README.md`'s Control Plane section). This script simulates
what that exporter would eventually send, using the exact same wire format
(`otter-telemetry-model/schema`), so the server/UI code paths get real exercise even though the
data itself is fake.

## What it does

- Registers as runtime instance `taskmanager-1`, job `fraud-detection-job`, serving model
  `fraud-detector` v3.2
- Emits a realistic transaction every 150ms: `kafka-source → feature-lookup →
  inference:fraud-detector → rule-engine → sink`, with proper trace/span parent-child chaining,
  randomized realistic latencies, a ~2% simulated error rate, and randomized confidence scores
- Emits `VALIDATING → WARMING → ACTIVATED` lifecycle events on startup, then simulates a full
  hot-swap to a new version every 45 seconds — enough to watch the model timeline move
- Listens for real commands from the UI (rollback, promote canary, etc.) and acknowledges them,
  so the model action buttons in the UI have something to actually respond to

## Run it

```bash
npm install
node simulate-runtime.js
# or against a non-default server / with auth:
SERVER_URL=http://localhost:4200 RUNTIME_TOKEN=changeme node simulate-runtime.js
# or run several concurrent fake instances (e.g. approximating multi-job ingestion load):
RUNTIME_INSTANCE_ID=taskmanager-2 JOB_ID=fraud-detection-job node simulate-runtime.js
```

See the root `otter-control-plane/README.md` for the full three-process test-run sequence
(server, simulator, UI).

## Verified in this repo's own development

This exact sequence — build the server, boot it, run this simulator, hit the REST API — was
actually run (not just written) while building this: `/api/v1/topology` returned 5 correctly
-categorized nodes with real health/latency/throughput numbers, `/api/v1/models/fraud-detector/timeline`
returned the real `VALIDATING`/`WARMING`/`ACTIVATED` sequence, and `/api/v1/traces` returned
real trace ids from the hot tier. The UI itself was not part of that specific verification run
(no browser available) — see `otter-control-plane-ui/README.md` for what was and wasn't checked
there.
