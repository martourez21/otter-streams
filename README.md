# Otter Streams

> The production AI runtime for Apache Flink — inference, context assembly, and decisioning, unified.

[![Otter Streams CI Pipeline](https://github.com/martourez21/otter-streams/actions/workflows/ci.yaml/badge.svg)](https://github.com/martourez21/otter-streams/actions/workflows/ci.yaml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-online-blue)](https://martourez21.github.io/otter-streams/docs/otter-docs/index.html)
[![GitHub stars](https://img.shields.io/github/stars/martourez21/otter-streams?style=social)](https://github.com/martourez21/otter-streams/stargazers)

<div align="center">
  <img src="docs/assets/otter-mark.png" alt="Otter Streams Logo" width="140"/>
</div>

---

## What is Otter Streams?

Otter Streams ships as a set of Java libraries — but once you build an `OtterRuntime`, it's
running an embedded runtime inside your Flink job, not a stateless utility: a shadow-inference
pool, a scheduler polling for new model versions, a GPU auto-scale-down scheduler, and a drain
loop on every hot swap all run on their own background threads for as long as the runtime lives.

It started as a way to embed ML inference — ONNX, TensorFlow, PyTorch, XGBoost, PMML, or a
remote endpoint — directly into Flink DataStream and SQL pipelines behind one consistent API,
with model lifecycle (hot swap, canary, shadow, rollback) handled for you. It has since grown
into four connected pieces:

| | |
|---|---|
| **Runtime** | Inference engines, model lifecycle, hardware auto-scaling, replica pooling |
| **Context Engine** | Parallel context assembly for RAG and memory-heavy AI — feature lookups, vector search, conversation memory |
| **Decisioning** | A YAML-configured Rule Engine and A/B testing on top of the runtime's canary mechanism |
| **Control Plane** | An optional, separate service for live topology, tracing, and a rule dashboard |

Every piece above the core Runtime is **opt-in** — a project that only needs ONNX inference
pulls in exactly `ml-inference-core` + `otter-stream-onnx`, nothing else.

---

## Quick Start

**1. Add the core dependency and an engine module:**

```xml
<dependency>
    <groupId>com.codedstreams</groupId>
    <artifactId>ml-inference-core</artifactId>
    <version>0.0.4</version>
</dependency>
<dependency>
    <groupId>com.codedstreams</groupId>
    <artifactId>otter-stream-onnx</artifactId>
    <version>0.0.4</version>
</dependency>
```

**2. DataStream API:**

```java
AsyncModelInferenceFunction<Transaction, ScoredTransaction> fn = new AsyncModelInferenceFunction<>(
    InferenceConfig.builder()
        .modelConfig(ModelConfig.builder()
            .modelId("fraud-detector")
            .modelPath("s3a://ml-models/fraud-detector/v1/fraud.onnx")
            .format(ModelFormat.ONNX)
            .modelVersion("v1")
            .build())
        .timeout(Duration.ofSeconds(5))
        .build(),
    cfg -> new OnnxInferenceEngine(),
    tx  -> Map.of("amount", (float) tx.getAmount(), "hour_of_day", (float) tx.getHourOfDay()),
    result -> new ScoredTransaction(tx, ((float[]) result.getOutputs().get("output"))[0]));

DataStream<ScoredTransaction> scored =
        AsyncDataStream.unorderedWait(transactionStream, fn, 5_000L, TimeUnit.MILLISECONDS, 100);
```

**3. Or Flink SQL — no Java operator needed at query time:**

```sql
ADD JAR '/var/www/udf-jars/otter-stream-sql-0.0.4-flink-udf.jar';
CREATE TEMPORARY FUNCTION ml_score AS 'com.codedstreams.otterstreams.sql.udf.MLInferenceFunction';

SELECT transaction_id,
       ml_score(MAP['amount', CAST(amount AS STRING)], 'fraud-detector') AS fraud_score
FROM transactions;
```

Full walkthroughs (MinIO model loading, SQL connector DDL, Studio registration): see
[Documentation](https://martourez21.github.io/otter-streams/docs/otter-docs/index.html).

---

## Capabilities

| Capability | Module(s) | What it does |
|---|---|---|
| **Inference engines** | `otter-stream-{onnx,tensorflow,pytorch,xgboost,pmml,remote}` | One `InferenceEngine` API across 5 model formats + remote endpoints (HTTP/gRPC/SageMaker/Vertex AI) |
| **Runtime & lifecycle** | `ml-inference-core` (`OtterRuntime`) | Provider SPI, hot swap with graceful draining, canary, shadow, rollback, dynamic loading |
| **Context Engine** | `otter-stream-context` | Parallel context assembly for RAG/memory — see [below](#context-engine) |
| **Connectors** | *(various, see [below](#connectors))* | Feature stores, caches, vector DBs, decision engines, streaming sinks |
| **Rule Engine** | `otter-stream-rules`, `-rules-drools` | YAML-configured decisions; delegate to KIE Server/Camunda/Drools |
| **A/B Testing** | `otter-stream-experiments` | Named experiments on top of the runtime's canary mechanism, with statistical comparison |
| **Distributed serving** | `ml-inference-core` (`runtime.serving`) | In-process replica pooling, load balancing, bidirectional auto-scaling |
| **Hardware acceleration** | `ml-inference-core` (`runtime.hardware`) | Automatic GPU→CPU scale-down when idle |
| **Control Plane** | `otter-control-plane/` | Optional, separate service: live topology graph, tracing, rule dashboard |
| **Benchmarking** | `otter-benchmarks` | JMH benchmarks for Otter's own overhead |

Full detail on any of these lives in that module's own README, linked throughout this document
and in [Modules](https://martourez21.github.io/otter-streams/docs/otter-docs/modules.html).

---

## Context Engine

For RAG and other context/memory-heavy AI applications: `otter-stream-context` assembles
context — feature lookups, vector search, conversation memory, cached state — from every
configured provider **in parallel**, on a dedicated thread pool, with a per-provider timeout so
one slow source degrades gracefully instead of blocking the whole request.

```java
ContextEngine engine = ContextEngine.builder()
        .provider(new FeatureProviderContextAdapter(redisFeatureProvider)) // reuse existing feature providers
        .provider(new ConversationMemoryProvider(20, 100_000, Duration.ofHours(2)))
        .provider(new PineconeContextProvider("docs", pineconeIndexHost, apiKey, "default"))
        .cache(new ContextCache(100_000, 30))
        .parallelism(64)
        .perProviderTimeout(Duration.ofMillis(300))
        .build();

Context context = engine.assemble(userId, Map.of("embedding", queryEmbedding, "topK", 5));

InferenceResult result = runtime.infer("rag-model", context.flatten());
```

This is a separate object you call before `runtime.infer(...)` — `OtterRuntime` itself has no
`.context(...)` builder method and isn't gaining one; keeping context assembly and inference
composition in your own pipeline code, rather than baked into the Runtime, is a deliberate
choice. See [`otter-stream-context/README.md`](otter-stream-context/README.md) for the full
provider list (including a Flink-native streaming-state provider) and its concurrency design.

---

## Connectors

Every external system Otter Streams talks to is a small, focused connector — pull in only the
ones you need:

| Kind | Connector | Backing system |
|---|---|---|
| Feature store | `RedisFeatureProvider`, `JdbcFeatureProvider`, `FeastFeatureProvider` | Redis, any JDBC source, Feast |
| Cache | `RedisFeatureProvider` (adapted), `MemcachedContextProvider` | Redis, Memcached |
| Vector search | `PineconeContextProvider`, or implement `VectorSearchProvider` | Pinecone; Milvus/OpenSearch via the same interface |
| Decision engine | `RestDecisionEngineConnector`, `DroolsDecisionEngineConnector` | Any REST decision service (KIE Server, Camunda DMN, IBM ODM); embedded Drools |
| Model registry | `ModelRegistry` SPI, `DefaultModelRegistry` | Pluggable — MLflow/S3/Nexus by implementing the interface |
| Streaming sink | `otter-stream-kafka` | Kafka, via Flink's own `KafkaSink`; `StreamResultSink` SPI for anything else |

```java
// Example: one REST connector, works against any vendor exposing a decision endpoint
DecisionEngineConnector kie = new RestDecisionEngineConnector(
        "kie-server",
        URI.create("https://decisions.mycompany.com/kie-server/decision"),
        System.getenv("KIE_SERVER_TOKEN"));

Decision decision = kie.evaluate(inferenceResult, Map.of());
```

Each connector follows the same shape: a small interface (`FeatureProvider`, `ContextProvider`,
`DecisionEngineConnector`, `VectorSearchProvider`) with one or two concrete implementations
shipped, and "implement the interface yourself" as the documented path for anything not
covered — no bespoke SDK per vendor.

---

## Architecture

Both the DataStream and SQL paths converge on `OtterRuntime`, which owns model lifecycle and
dispatches to whichever engine module your model format needs:

```
Data Sources (Kafka, S3, CDC)
      │
      ├── DataStream API path ──────┐    ┌── Flink SQL / Table API path
      │   AsyncModelInferenceFunction    │    ml_score() UDF / ml-inference connector
      │                             │    │
      └─────────────┬───────────────┴────┘
                     ▼
              OtterRuntime
        (Provider SPI · Lifecycle Manager · Model Registry)
                     │
       ┌─────────────┼──────────────┬─────────────┐
   OnnxEngine  TensorFlowEngine  XGBoostEngine  PmmlEngine ...
```

Flink is always `provided` scope — never bundled in the shaded JAR. The same artifact runs
across Flink 1.15 through 2.0 without recompilation.

---

## Project Structure

```
otter-streams/
├── ml-inference-core/           # InferenceEngine, OtterRuntime, lifecycle, serving, hardware
├── otter-stream-{onnx,tensorflow,pytorch,xgboost,pmml,remote}/  # Engine modules
├── otter-stream-feature-{redis,jdbc,feast}/  # Feature store providers
├── otter-stream-context/        # Context Engine (RAG/memory) — see above
├── otter-stream-rules/, -rules-drools/       # Rule Engine
├── otter-stream-experiments/    # A/B testing
├── otter-stream-kafka/          # Kafka publishing
├── otter-benchmarks/            # JMH benchmarks
├── otter-stream-sql/            # Flink SQL UDF, connector, lookup function
├── otter-stream-examples/       # Runnable example jobs
├── otter-control-plane/         # Optional Control Plane (server + UI + shared schema)
├── PERFORMANCE.md
└── docs/                        # HTML documentation site
```

---

## Compatibility

| Otter Streams | Flink | Java | ONNX Runtime | TensorFlow Java |
|---|---|---|---|---|
| **0.0.4** | 1.15 – 2.0 | 17 | 1.23.2 | 0.5.0 |
| 1.0.17 | 1.17 – 1.20 | 11, 17 | 1.23.2 | 0.5.0 |
| 1.0.0 | 1.16 | 11 | 1.14.0 | 0.4.2 |

Full matrix and upgrade notes: [Releases](https://martourez21.github.io/otter-streams/docs/otter-docs/releases.html).

---

## Documentation

| Page | Description |
|---|---|
| [Documentation](https://martourez21.github.io/otter-streams/docs/otter-docs/index.html) | Introduction, Quick Start, Architecture, troubleshooting |
| [Rule Engine](https://martourez21.github.io/otter-streams/docs/otter-docs/rules.html) | YAML rules, evaluation modes, external connectors, dashboard |
| [Modules](https://martourez21.github.io/otter-streams/docs/otter-docs/modules.html) | Per-module reference, dependency graph, Runtime Layer detail |
| [DataStream Guide](https://martourez21.github.io/otter-streams/docs/otter-docs/datastream.html) | Complete code for every engine module |
| [API Reference](https://martourez21.github.io/otter-streams/docs/otter-docs/api.html) | `InferenceEngine`, `InferenceResult`, connector DDL options |
| [Examples](https://martourez21.github.io/otter-streams/docs/otter-docs/examples.html) | Fraud detection, IoT anomaly detection, MinIO pipeline |
| [Javadoc](https://martourez21.github.io/otter-streams/docs/javadoc/0.0.4/) | Full generated API documentation |

Module-level READMEs (`otter-stream-context/README.md`, `otter-stream-rules/README.md`,
`otter-control-plane/README.md`, etc.) have deeper detail — including the specifics on what's
been verified to actually build/run vs. reviewed by hand — than this file attempts to hold.

---

## Dependency Management

Every optional integration lives in its own Maven module with its own scoped dependencies —
pulling in `otter-stream-onnx` never drags in TensorFlow, PyTorch, Drools, or anything else you
didn't ask for. Third-party versions are pinned once, centrally, in the root `pom.xml`'s
`<dependencyManagement>`. `flink-*` dependencies are always `provided` scope. Minimal example:
YAML rules + ONNX inference is exactly `ml-inference-core` + `otter-stream-onnx` +
`otter-stream-rules` — nothing else.

---

## Contributing

```bash
git clone https://github.com/your-username/otter-streams.git
cd otter-streams
mvn clean install -DskipTests
mvn test
```

Good first issues are labelled [`good first issue`](https://github.com/martourez21/otter-streams/issues?q=label%3A%22good+first+issue%22).
See [`CONTRIBUTION.md`](CONTRIBUTION.md) for the full guide.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

Built with passion by **Nestor Martourez Abiangang A.**

- GitHub: [@martourez21](https://github.com/martourez21)
- LinkedIn: [linkedin.com/in/nestor-abiangang](https://www.linkedin.com/in/nestor-abiangang/)
- Email: [nestorabiawuh@gmail.com](mailto:nestorabiawuh@gmail.com)

Special thanks to the Apache Flink community and all open-source ML framework maintainers.

<div align="center">

[Documentation](https://martourez21.github.io/otter-streams/docs/otter-docs/index.html) &nbsp;·&nbsp;
[Discussions](https://github.com/martourez21/otter-streams/discussions) &nbsp;·&nbsp;
[Star the project](https://github.com/martourez21/otter-streams/stargazers)

</div>
