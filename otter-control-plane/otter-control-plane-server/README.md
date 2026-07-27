# otter-control-plane-server

NestJS implementation of the Otter Control Plane's ingestion service, topology builder, trace
store, and REST/WebSocket API — see `../ARCHITECTURE.md` for the full design this implements.

**Status:** implemented and verified to build and boot in this environment — `npm install`,
type generation, `nest build`, and starting the compiled server were all actually run (not just
written) as part of building this. Not yet run against a live `OtterRuntime` instance or a real
ClickHouse cluster (neither is available in the environment this was built in) — see the
"What's verified vs. not" section below before treating this as production-ready.

## Run locally

```bash
npm install
npm run build      # runs the type-generation prebuild step against ../otter-telemetry-model
npm run start:dev  # or: npm run build && npm run start:prod
```

Server listens on `PORT` (default `4200`). Swagger docs at `/api/v1/docs`.

## Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `4200` | HTTP/WS listen port |
| `RUNTIME_AUTH_TOKEN` | *(unset)* | If set, required for a Runtime's WebSocket handshake (§13). Unset = open, dev only. |
| `API_AUTH_TOKEN` | *(unset)* | If set, required (`Authorization: Bearer <token>`) for mutating REST endpoints. Unset = open, dev only. |
| `CLICKHOUSE_URL` | *(unset)* | Cold-tier trace storage. Unset = hot tier only (safe degraded mode, not an error). |
| `CLICKHOUSE_USERNAME` / `CLICKHOUSE_PASSWORD` / `CLICKHOUSE_DATABASE` | `default` / `` / `otter` | ClickHouse connection details |

**Always set `RUNTIME_AUTH_TOKEN` and `API_AUTH_TOKEN` outside local development** — both
default to open access specifically so `npm run start:dev` works with zero config, not because
that's a safe production default.

## Docker

```bash
# from the repository root (build context matters — see the Dockerfile's header comment)
docker build -f otter-control-plane/otter-control-plane-server/Dockerfile -t otter-control-plane-server:latest .
docker run -p 4200:4200 -e RUNTIME_AUTH_TOKEN=changeme -e API_AUTH_TOKEN=changeme otter-control-plane-server:latest
```

Not verified in this environment — no Docker daemon available here. The Dockerfile itself was
reviewed carefully (multi-stage, non-root user, healthcheck) but treat "not yet run" the same
way as this project's other documented verification gaps: a solid starting point, not a
guarantee.

## What's verified vs. not

**Actually run in this environment, not just written:**
- `npm install` — succeeds, all dependencies resolve
- Type generation from `../otter-telemetry-model/schema` — succeeds, produces real `.ts` files
- `nest build` — succeeds with zero TypeScript errors
- Booting the compiled server — succeeds; `/health`, `/api/v1/topology`, `/api/v1/traces` all
  respond correctly with real (empty, since nothing is connected) data

**Not run in this environment (no infrastructure available here):**
- Against a real `OtterRuntime` instance sending real spans/lifecycle events over the WebSocket
- Against a real ClickHouse cluster (the code path exists and degrades gracefully without one,
  but the actual INSERT/query behavior against a live cluster is unverified)
- The Docker build (no Docker daemon in this environment)
- Any load/concurrency testing

## Module map

| Module | Responsibility |
|---|---|
| `ingestion/` | Runtime ⟷ Control Plane WebSocket (telemetry in, commands out) — ARCHITECTURE.md §6.1/§6.5/§6.6 |
| `topology/` | Sliding-window aggregation of spans into live nodes/edges, pushed to UI clients on a tick — §6.2/§7 |
| `traces/` | Hot-tier in-memory trace store + ClickHouse cold tier — §6.3 |
| `models/` | Per-model lifecycle timeline + REST endpoints that issue commands (deploy/rollback/canary/shadow) — §9 |
| `commands/` | Fans a command out to every instance serving a model, correlates acks, reports partial failures — §6.5 |
| `rules/` | Rule Dashboard backend — receives pushed `RuleMetricsSnapshot` + rule definitions from `otter-stream-rules`-based applications |
| `common/` | Health check, bearer-token auth guard |

Every consumer of ingested telemetry (`topology`, `traces`, `models`) subscribes via
`@nestjs/event-emitter`, not a direct import of `IngestionModule` — this avoids a circular
module dependency (Ingestion → Models → Commands → Ingestion) that a direct-import design would
hit; see `IngestionGateway`'s class doc comment for the full reasoning.
