# otter-benchmarks

JMH microbenchmarks for Otter Streams' own overhead, plus guidance below for the two kinds of
benchmarking JMH *can't* do: real end-to-end pipeline latency, and HTTP/WebSocket load on the
Control Plane.

**Not run in this environment.** Like every other Java module in this project, this was written
without Maven Central access to actually compile/run it (see `PERFORMANCE.md` for the same
caveat applied elsewhere). Reviewed carefully by hand against the actual current APIs
(`ManagedModel`, `DefaultRuleEngine`, `ReplicaPool`, etc.), but treat "not verified" as literal
here, more than usual — benchmark code that silently measures the wrong thing is worse than no
benchmark, so read each class before trusting its numbers.

## Build and run

```bash
mvn -pl otter-benchmarks -am clean package
java -jar otter-benchmarks/target/benchmarks.jar
```

Run one class, or filter by name:
```bash
java -jar otter-benchmarks/target/benchmarks.jar RuleEngineBenchmark
java -jar otter-benchmarks/target/benchmarks.jar ExpressionEvaluatorBenchmark.evaluatePrecompiled
```

Useful flags: `-f 3` (more forks, less noise), `-prof gc` (see allocation pressure), `-rff results.json` (machine-readable output for tracking over time).

## What's here, and what each one actually measures

| Class | Measures | Does NOT measure |
|---|---|---|
| `ManagedModelRoutingBenchmark` | `OtterRuntime`'s own per-call overhead (canary check, in-flight tracking) using a no-op engine | Any real model's inference time |
| `RuleEngineBenchmark` | `DefaultRuleEngine.evaluate()` against a realistic 4-rule set, SINGLE and MULTIPLE modes | Rule evaluation cost for *your* rule set (rule count and condition complexity change this) |
| `ExpressionEvaluatorBenchmark` | The exact cost of compile-once-vs-reparse-every-time for conditions | Anything beyond that one design decision |
| `ReplicaPoolBenchmark` | Load-balancing selection overhead, round-robin vs. least-connections | Whether replica pooling helps *your* model — that depends on whether your model is actually CPU-bound in a way one instance can't saturate |

**The single most important thing to understand about all four:** they measure Otter's own
code, not your model. A JMH result of "180ns per call" for `ManagedModelRoutingBenchmark` tells
you the routing layer adds essentially nothing to your latency budget — it says nothing about
whether your actual ONNX/XGBoost/PMML model call fits under 5ms, because that number isn't
Otter's to report. Nothing in this repository can benchmark your specific model for you.

## How to actually validate the sub-5ms target end to end

JMH benchmarks in-process Java method calls. Your real latency budget is: Kafka consume → any
feature lookups → the actual model call → rule evaluation → sink. To measure *that*:

1. **Instrument the real path.** Wrap your `AsyncModelInferenceFunction` (or wherever you call
   `runtime.infer(...)`) with `System.nanoTime()` before/after, emit it as a Flink metric
   (`getRuntimeContext().getMetricGroup().gauge(...)`), and watch it in the Flink Web UI or
   whatever metrics backend you already have (Prometheus/Grafana, etc.) — this is the real
   number, not a JMH number.
2. **Load-generate realistic traffic.** A Kafka producer script pushing your actual message
   shape at your actual expected QPS, into the actual topic your job consumes from — synthetic
   traffic that doesn't match your real payload sizes/distributions will give you a
   confidently wrong answer.
3. **Once the Control Plane's tracing is connected to a real Runtime** (see the root README —
   this isn't built yet, the `TelemetryExporter` that would make this automatic doesn't exist),
   the per-span breakdown in the trace waterfall view becomes the natural place to see exactly
   which stage of the pipeline is eating your latency budget. Until then, the Flink-metrics
   approach above is the concrete path.

## Load-testing the Control Plane's REST/WebSocket API

Different question from the above — this is "can the Control Plane server itself handle load,"
not "is my model fast." A minimal [k6](https://k6.io) script for the REST side:

```javascript
// control-plane-load-test.js — run with: k6 run --vus 50 --duration 30s control-plane-load-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';

const BASE_URL = __ENV.BASE_URL || 'http://localhost:4200/api/v1';

export default function () {
  const res = http.get(`${BASE_URL}/topology`);
  check(res, { 'status is 200': (r) => r.status === 200 });
  sleep(0.1);
}
```

For the ingestion side (spans arriving over the `/runtime` WebSocket namespace), adapt
`otter-control-plane/scripts/simulate-runtime.js` — it already emits realistic spans; running
several instances of it concurrently (different `RUNTIME_INSTANCE_ID` env values) is a
reasonable first approximation of multi-job ingestion load, though it wasn't built with load
-testing as its primary purpose (it's a demo tool — see its own README) so treat results from it
as directional, not authoritative.
