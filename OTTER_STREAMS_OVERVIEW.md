# Otter Streams — Technical Overview

## What is Otter Streams?

Otter Streams is distributed as a set of Java libraries — Maven JARs you add as
`<dependency>` entries, no installer, no daemon required to get started. But calling it "a
library" undersells what it actually does once you use it, and it's worth being precise about
that distinction rather than glossing over it.

A pure library is passive: you call a function, it returns, nothing happens in the background.
`OtterRuntime` is not that. Once you build one, it owns and manages, on its own background
threads, independent of whether you're actively calling into it:

- A thread pool running shadow inference (fire-and-forget comparison calls against a shadow
  model version)
- A scheduler polling the model registry for new versions and deploying them automatically
- A scheduler watching GPU utilization and deciding when to scale an idle model back to CPU
- A drain loop that runs on every hot swap, waiting for in-flight requests to finish before
  closing the old model version

That's runtime behavior — self-managed lifecycle and concurrency, not a stateless utility you
invoke and forget. It's the same distinction as, say, Netty or an embedded actor system:
technically a Maven dependency, but once instantiated it's a runtime living inside your process.

So the accurate framing is two-layered: **Otter Streams ships as Java libraries that together
form an embedded runtime inside your Flink job** — this is the part covered in this document as
"the Otter Runtime" — **plus a separate, optional Control Plane service** that isn't part of the
library at all: it's an independent deployable (its own process, its own network port) that
your Flink job's Runtime talks to over a WebSocket, covered later in this document as "the
Otter Control Plane."

With that distinction in mind: Otter Streams brings production-grade AI/ML model inference to
Apache Flink streaming applications. It lets you embed a model call — ONNX, TensorFlow,
PyTorch, XGBoost, PMML, or a remote endpoint like SageMaker/Vertex AI/a custom REST service —
directly into a Flink `DataStream` pipeline or a Flink SQL query, without hand-rolling model
loading, caching, versioning, or lifecycle management yourself.

Positioned against the rest of the Flink ecosystem, the relationship is:

| Layer | Role |
|---|---|
| **Apache Flink** | Stream processing runtime |
| **Otter Streams** | Production AI runtime for that stream processing |

Flink tells you *when* and *in what order* events happen. Otter Streams tells you what a model
thinks about each one — reliably, with the model lifecycle (loading, versioning, hot-swapping,
rollback) handled for you rather than left as an exercise for every team that wants inference in
their pipeline.

---

## The problem this solves

Without Otter Streams, putting a model inside a Flink job usually means each team independently
solving the same handful of hard problems: how do you load a model once per TaskManager instead
of per record; how do you swap in a new model version without restarting the job; how do you
avoid one slow model call blocking the whole operator thread; how do you know which model
version actually processed a given transaction after the fact. Otter Streams solves these once,
centrally, so individual Flink jobs consume a stable API instead of re-solving them.

---

## Architecture: two parts, deliberately separated

Otter Streams is split into two halves that scale independently and fail independently:

```
┌─────────────────────────────┐        ┌──────────────────────────────┐
│        Otter Runtime         │        │      Otter Control Plane      │
│                               │        │                               │
│  Embedded in Flink            │  spans │  Independent service(s)       │
│  TaskManagers. Executes       │  and   │  Ingests telemetry, builds    │
│  inference, manages model     │─events→│  the live topology, stores    │
│  lifecycle, emits telemetry.  │        │  traces, serves REST/WS API.  │
└─────────────────────────────┘        └──────────────────────────────┘
```

**Why split them:** inference latency must never depend on whether an observability dashboard
is up or overloaded. If the Control Plane is down, Flink jobs keep running exactly as before —
telemetry is dropped, nothing else changes. This mirrors how Flink itself separates the
JobManager/TaskManager runtime from the Flink Web UI.

---

## The Otter Runtime

This is the part that actually runs inside your Flink job, and the part that's implemented
today (not a design document — real, working Java code across several Maven modules).

### Inference engines
A common `InferenceEngine` interface with concrete implementations per model format:
ONNX Runtime, TensorFlow SavedModel, PyTorch TorchScript (via DJL), XGBoost, PMML, and remote
engines (HTTP, gRPC, SageMaker, Vertex AI). Each lives in its own Maven module so a project only
pulls in the framework it actually uses.

### `OtterRuntime` — the central facade
Ties together everything below via a builder:

```java
OtterRuntime runtime = OtterRuntime.builder()
    .metrics(new MetricsCollector())
    .build(); // auto-discovers provider modules on the classpath
```

- **Provider SPI** — engine implementations are discovered via `ServiceLoader`, not hardcoded.
- **Model Registry SPI** — resolves a model reference (id + optional version) to a loadable
  config; pluggable for MLflow/S3/Nexus-backed registries.
- **Lifecycle Manager** — every deployment goes through validate → warm → atomic swap → drain →
  retire. A hot swap under load never severs an in-flight request: the old engine is only closed
  once its in-flight count hits zero (or a timeout).
- **Dynamic loading** — `runtime.watch(modelId, interval)` polls the registry and redeploys new
  versions automatically.
- **Rollback** — `runtime.rollback(modelId)` redeploys the last known-good version.
- **Canary deployments** — a candidate version takes a configurable percentage of traffic
  alongside the current primary; promote or discard once you're confident.
- **Shadow deployments** — a candidate version silently receives a sampled copy of traffic for
  comparison, with zero effect on what's actually served.

### Feature stores
Pluggable `FeatureProvider` implementations — Redis, JDBC (any driver), and Feast (HTTP feature
server) — for pulling additional features into a model call.

### Rule Engine
Turns an `InferenceResult` into a decision (`FRAUD`, `APPROVE`, `REVIEW`, or any flag you
define), using rules you configure — YAML by default, `.properties` or a hand-written class as
equally-supported alternatives. Conditions compile once at load time, not per call. Supports
single-match, multi-match, and batch evaluation. Can delegate to an external enterprise decision
engine (KIE Server, Camunda DMN, IBM ODM, or embedded Drools) instead of evaluating in-process.

### Publishing results
`otter-stream-kafka` publishes inference results and rule decisions to Kafka as JSON, built on
Flink's own `KafkaSink`. A generic `StreamResultSink` SPI covers any other target system.

### Hardware acceleration awareness
`ExecutionTargetManager` monitors GPU-capable engines and automatically scales an idle one back
to CPU, freeing GPU resources when traffic drops — with automatic scale-*up* left as an explicit
trigger rather than a guessed heuristic, since that decision needs a forward-looking signal the
runtime doesn't fabricate.

---

## The Otter Control Plane

This is the operational visibility layer — a live topology graph and distributed tracing for
everything the Runtime is doing, in the spirit of Flink's own execution graph or a
Jaeger/Zipkin-style trace viewer, but purpose-built around AI inference (model version,
confidence, provider, GPU) rather than generic RPC semantics.

**Status: implemented and verified to build/boot/serve — not yet run end-to-end against a live
Runtime, and not yet visually verified in a real browser.** Both halves exist as real code, not
just design: the NestJS server (`otter-control-plane-server`) actually installs, builds, boots,
and responds correctly on `/health`, `/api/v1/topology`, and `/api/v1/traces`; the UI
(`otter-control-plane-ui`) actually installs, type-checks, builds, and serves its JS bundle,
icons, and logo with real HTTP 200s. What's genuinely not done: ClickHouse has never been tested
against a live cluster (it degrades gracefully without one, by design), the UI has never
connected to a live `OtterRuntime` or rendered in an actual browser, and the Docker images for
both are unverified (no Docker daemon in the environment they were built in). Each module's own
README states exactly what was and wasn't run — treat those as the source of truth over any
summary here. What it shows:

- A **live topology graph** of your inference pipeline, colored by latency, confidence, or model
  version — not a static diagram, an animated one with traffic actually flowing across it.
- **Per-transaction tracing** — click one transaction, see every stage it passed through and how
  long each took, down to which specific model version and execution provider (CPU/GPU) handled
  it.
- **Shadow and canary visualization** — watch two model versions process the same traffic
  side by side, or a canary's traffic slider move in real time.
- **A rule dashboard** — per-rule and per-flag hit counts, rendered with each rule's configured
  color, backed by the Rule Engine's existing metrics.

---

## Key benefits

- **One API across five+ model formats.** Swap ONNX for TensorFlow without rewriting pipeline
  code — the `InferenceEngine` contract stays the same.
- **Hot swaps that don't drop requests.** Model updates go live without a job restart and
  without severing in-flight calls.
- **Safe rollout by default.** Canary and shadow deployments are first-class, not something you
  build yourself on top of raw model loading.
- **Decisions, not just scores.** The Rule Engine turns a raw model output into an actionable,
  auditable flag — with a path to your existing enterprise decision engine if you already have
  one.
- **Pay only for what you use.** Every integration (each model format, each feature store,
  Drools, Kafka) is its own Maven module with its own scoped dependencies — including only ONNX
  and the Rule Engine pulls in exactly `ml-inference-core` + ONNX Runtime + SnakeYAML, nothing
  else.
- **Built for the hot path.** Conditions and configs are compiled once, not re-parsed per
  request; blocking inference calls run on a dedicated executor, never the shared JVM-wide
  thread pool other libraries also depend on.
- **Operational visibility as a first-class feature**, not an afterthought bolted on later — the
  Control Plane's design already maps directly onto events the Runtime produces today.

---

## Where things stand today

| Component | Status |
|---|---|
| Inference engines (ONNX/TF/PyTorch/XGBoost/PMML/remote) | Implemented |
| `OtterRuntime` (Provider SPI, Lifecycle Manager, dynamic loading, rollback, canary, shadow) | Implemented |
| Feature store providers (Redis, JDBC, Feast) | Implemented |
| Rule Engine (YAML/properties/programmatic, REST/Drools connectors) | Implemented |
| Kafka result publishing | Implemented |
| GPU auto-scale-down | Implemented (scale-up is an explicit trigger, by design) |
| Otter Control Plane — server (ingestion, topology, traces, models, commands, rules) | Implemented and verified to build/boot (see `otter-control-plane-server/README.md` for exactly what's verified vs. not) |
| Otter Control Plane — UI (topology graph, tracing, rule dashboard, model controls) | Implemented and verified to build/serve (see `otter-control-plane-ui/README.md`); never rendered in a real browser or connected to a live Runtime |
| Otter Control Plane — ClickHouse cold tier, production hardening, RBAC | Not done — hot tier + degraded-mode ClickHouse wiring exist, full production validation doesn't |

See `README.md` for install instructions per module, `PERFORMANCE.md` for the concurrency/latency
review, and `otter-control-plane/ARCHITECTURE.md` for the full Control Plane design.
