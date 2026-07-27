# Otter Control Plane

Live topology graph, distributed tracing, and a rule dashboard for `OtterRuntime`. See
`ARCHITECTURE.md` in this directory for the full design.

## Is this "the full thing"?

No — read this before you spend time on it. What exists:

| Piece | Status |
|---|---|
| `otter-control-plane-server` (NestJS) | Implemented. Builds, boots, and responds correctly — actually verified in this repo's own development, see its README. |
| `otter-control-plane-ui` (Vite/TS) | Implemented. Builds and serves correctly — actually verified. Never rendered in a real browser. |
| `scripts/simulate-runtime.js` | A test/demo tool faking a Runtime connection, so you can see the above two actually working together. |
| **The real Java-side `TelemetryExporter`** | **Does not exist.** This is the piece that would let an actual `OtterRuntime` inside a real Flink job send data here. Without it, connecting a real Runtime isn't possible yet — you can only test-run with the simulator. |
| ClickHouse cold tier | Wired, degrades gracefully without it, never tested against a real cluster |
| Auth (`RUNTIME_AUTH_TOKEN` / `API_AUTH_TOKEN`) | Implemented, open-by-default for local dev |

If what you need is the `TelemetryExporter` so a real Runtime can connect, say so — that's a
different, smaller, well-scoped piece of work than "the whole Control Plane," and worth its own
pass rather than being an afterthought at the bottom of this list.

## Test-run it yourself (verified sequence — this exact process was actually run)

You need three terminals.

**Terminal 1 — the server:**
```bash
cd otter-control-plane-server
npm install
npm run build      # generates types from ../otter-telemetry-model, then compiles
npm run start:prod # or: node dist/main.js
```
Wait for `Otter Control Plane listening on port 4200`.

**Terminal 2 — the simulator** (fakes a Runtime so there's something to look at):
```bash
cd scripts
npm install
node simulate-runtime.js
```
Wait for `Registered as instance 'taskmanager-1' serving model 'fraud-detector'`.

**Terminal 3 — the UI:**
```bash
cd otter-control-plane-ui
npm install
npm run dev
```
Open the printed URL (default `http://localhost:5173`). You should see:
- **Topology** (`#topology`): five nodes — `kafka-source → feature-lookup →
  inference:fraud-detector → rule-engine → sink` — with live throughput/latency, animated
  edges, and real icons (Kafka, ONNX)
- **Traces** (`#traces`): a growing list of trace ids; click one for the span waterfall
- **Models** (`#models`): `fraud-detector`, active version `3.2`, a lifecycle timeline, and
  working action buttons (the simulator acknowledges commands sent from here)
- **Rule Dashboard** (`#rules`): empty, by design — nothing in this test setup pushes rule
  metrics; see `otter-stream-rules/README.md`'s "Metrics" section for how an application would

If you don't see data after ~10 seconds, check Terminal 1's logs for `Runtime connected` and
`Registered runtime instance` lines — if those aren't there, the simulator isn't reaching the
server (check `SERVER_URL`, check nothing else is bound to port 4200).

## Directory map

```
otter-control-plane/
├── ARCHITECTURE.md              # Full design doc
├── otter-telemetry-model/       # Shared JSON Schema — Java + TypeScript codegen source
├── otter-control-plane-server/  # NestJS backend
├── otter-control-plane-ui/      # Vite/TS frontend
└── scripts/                     # Runtime simulator (test/demo tool, not product code)
```
