# Otter Streams

> Production-grade machine learning inference for Apache Flink

[![Otter Streams CI Pipeline](https://github.com/martourez21/otter-streams/actions/workflows/ci.yaml/badge.svg)](https://github.com/martourez21/otter-streams/actions/workflows/ci.yaml)
[![Maven Central](https://img.shields.io/maven-central/v/com.codedstreams/otter-streams)](https://search.maven.org/artifact/com.codedstreams/otter-streams)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub stars](https://img.shields.io/github/stars/martourez21/otter-streams?style=social)](https://github.com/martourez21/otter-streams/stargazers)
[![Discussions](https://img.shields.io/badge/Discussions-Enabled-blue)](https://github.com/martourez21/otter-streams/discussions)

<div align="center">
  <img src="docs/assets/otter-mark.png" alt="Otter Streams Logo" width="140"/>
</div>

---

## What is Otter Streams?

Otter Streams is distributed as a set of Java libraries (Maven JARs, no installer or daemon
required) that together bring production-grade AI/ML inference to Apache Flink streaming
applications. Calling it "a library" is accurate for how you get it onto your classpath, but
undersells what it does once you use it — `OtterRuntime` isn't a stateless call-and-return
utility. Once built, it owns and manages its own background threads: a shadow-inference pool,
a scheduler polling for new model versions, a scheduler watching GPU utilization for
auto-scale-down, and a drain loop on every hot swap. That's runtime behavior, not passive
library behavior — the same distinction as an embedded Netty server or actor system.

So the accurate framing is two layers: **a set of libraries that together form an embedded
runtime inside your Flink job** (the "Otter Runtime" — everything in this README), **plus a
separate, optional Control Plane service** (`otter-control-plane/`) that isn't a library at all:
an independent NestJS process your Flink job's Runtime talks to over a WebSocket, for live
topology visualization, distributed tracing, and a rule dashboard.

With that distinction clear: Otter Streams was built first as a **DataStream API** library -
letting you embed ML inference directly into Flink operators via `AsyncDataStream` - and has
since grown a full **Flink SQL / Table API** layer, a **Rule Engine** for turning model output
into decisions, pluggable **feature store** providers, **Kafka** result publishing, and
**hot-swap/canary/shadow** model deployment — all covered below.

Models are stored externally (MinIO, AWS S3, or any S3-compatible store, or via the pluggable
Model Registry SPI) and loaded at engine initialisation time into a Caffeine LRU `ModelCache`
shared across both the DataStream and SQL paths.

### Why Otter Streams?

- **Real-time ML at scale** - millisecond-latency inference on unbounded Flink streams
- **Framework agnostic** - ONNX, TensorFlow SavedModel, PyTorch TorchScript, XGBoost, PMML, and remote endpoints (HTTP/gRPC, SageMaker, Vertex AI)
- **Two APIs, one cache** - DataStream operators and Flink SQL queries share the same `ModelCache` singleton; models are downloaded from MinIO once per session
- **SQL-native deployment** - register `ml_score()` as a Flink UDF and call it inline in any SQL query; no Java operator required at query time
- **Production ready** - Caffeine LRU caching, Micrometer metrics (Prometheus / InfluxDB / DataDog), async retry, configurable timeouts, dead-letter routing
- **MinIO / S3 model store** - the `ml-inference` connector downloads SavedModel directories and single-file formats (`onnx`, `xgboost-json`, `pmml`) from any S3-compatible endpoint at table-creation time

---

## Quick Start

### 1 - Add the core dependency

```xml
<!-- Core framework - required by all engine modules -->
<dependency>
    <groupId>com.codedstreams</groupId>
    <artifactId>ml-inference-core</artifactId>
    <version>0.0.4</version>
</dependency>

<!-- Pick the engine module(s) you need -->
<dependency>
    <groupId>com.codedstreams</groupId>
    <artifactId>otter-stream-onnx</artifactId>
    <version>0.0.4</version>
</dependency>

<!-- Flink SQL / Table API integration (optional) -->
<dependency>
    <groupId>com.codedstreams</groupId>
    <artifactId>otter-stream-sql</artifactId>
    <version>0.0.4</version>
</dependency>
```

### 2 - DataStream API

```java
// Configure the model
ModelConfig modelConfig = ModelConfig.builder()
    .modelId("fraud-detector")
    .modelPath("s3a://ml-models/fraud-detector/v1/fraud.onnx")
    .format(ModelFormat.ONNX)
    .modelVersion("v1")
    .build();

InferenceConfig inferenceConfig = InferenceConfig.builder()
    .modelConfig(modelConfig)
    .batchSize(32)
    .timeout(Duration.ofSeconds(5))
    .enableCaching(true)
    .enableMetrics(true)
    .build();

// Wrap in an async Flink function
AsyncModelInferenceFunction<Transaction, ScoredTransaction> fn =
    new AsyncModelInferenceFunction<>(
        inferenceConfig,
        cfg -> new OnnxInferenceEngine(),
        tx  -> Map.of(
            "amount",      (float) tx.getAmount(),
            "hour_of_day", (float) tx.getHourOfDay()
        ),
        result -> new ScoredTransaction(
            tx, ((float[]) result.getOutputs().get("output"))[0]
        )
    );

// Apply to the stream
DataStream<ScoredTransaction> scored = AsyncDataStream.unorderedWait(
    transactionStream, fn, 5_000L, TimeUnit.MILLISECONDS, 100
);
```

### 3 - Flink SQL / Table API

```sql
-- Load the shaded JAR and register the scalar UDF
ADD JAR '/var/www/udf-jars/otter-stream-sql-0.0.4-flink-udf.jar';

CREATE TEMPORARY FUNCTION IF NOT EXISTS ml_score
AS 'com.codedstreams.otterstreams.sql.udf.MLInferenceFunction'
LANGUAGE JAVA;

-- Preload the model from MinIO (populates ModelCache)
CREATE TEMPORARY TABLE fraud_model_source (score DOUBLE)
WITH (
    'connector'           = 'ml-inference',
    'model.name'          = 'fraud-detector',
    'model.path'          = 's3a://ml-models/fraud-detector/v1/',
    'model.format'        = 'tensorflow-savedmodel',
    'model.s3.endpoint'   = 'http://minio:9000',
    'model.s3.access-key' = 'minioadmin',
    'model.s3.secret-key' = 'minioadmin',
    'model.s3.path-style' = 'true',
    'cache.enabled'       = 'true'
);

-- Call ml_score() inline in a streaming query
INSERT INTO fraud_alerts
SELECT
    transaction_id,
    fraud_score,
    CASE
        WHEN fraud_score >= 0.85 THEN 'CRITICAL'
        WHEN fraud_score >= 0.65 THEN 'HIGH'
        ELSE                          'MEDIUM'
    END AS risk_tier
FROM transactions
CROSS JOIN LATERAL (
    SELECT COALESCE(
        ml_score(
            MAP['amount',      CAST(amount      AS STRING),
                'hour_of_day', CAST(hour_of_day AS STRING)],
            'fraud-detector'
        ), 0.0
    ) AS fraud_score
) scores
WHERE fraud_score >= 0.40;
```

---

## Documentation

| Page | Description |
|------|-------------|
| [Documentation](https://martourez21.github.io/otter-streams/docs/otter-docs/index.html) | Introduction, Quick Start, Architecture, UDF Reference, Shaded JAR, Troubleshooting |
| [DataStream API Guide](https://martourez21.github.io/otter-streams/docs/otter-docs/datastream.html) | Complete Java code for every engine module - ONNX, TensorFlow, PyTorch, XGBoost, PMML, Remote |
| [SQL Examples](https://martourez21.github.io/otter-streams/docs/otter-docs/examples.html) | MinIO pipeline demo (6 SQL sections), fraud detection, IoT anomaly detection |
| [Studio Demo](https://martourez21.github.io/otter-streams/docs/otter-docs/studio-demo.html) | Str:::Lab Studio Feature Engineering Manager + Inference Manager walkthrough |
| [API Reference](https://martourez21.github.io/otter-streams/docs/otter-docs/api.html) | InferenceEngine, InferenceResult, ModelCache, MLInferenceFunction, connector DDL options |
| [Modules](https://martourez21.github.io/otter-streams/docs/otter-docs/modules.html) | Dependency graph + per-module docs for all 7 modules |
| [Releases](https://martourez21.github.io/otter-streams/docs/otter-docs/releases.html) | Changelog, compatibility matrix, upgrade notes |
| [Javadoc](https://martourez21.github.io/otter-streams/docs/javadoc/0.0.4/) | Full generated API documentation |

---

## Architecture

Otter Streams has two parallel integration paths - both converge on the same engine and cache layer.

```
Data Sources  (Kafka, S3, CDC)
      │
      ├──────────────────────────────────────────────────────────┐
      │                                                          │
      ▼   DataStream API path                                    ▼   Flink SQL / Table API path
  SourceFunction                                          SQL Gateway Session
      │                                                          │
  AsyncDataStream.unorderedWait()                    ADD JAR  +  CREATE FUNCTION ml_score
      │                                                          │
  AsyncModelInferenceFunction                        MLInferenceDynamicTableFactory
      │                                               (SPI: 'connector' = 'ml-inference')
      │                                                          │
      │                                               MinioModelLoader  ──▶  MinIO / S3
      │                                                          │
      └────────────────────┬─────────────────────────────────────┘
                           │
                    ModelCache  (Caffeine LRU singleton)
                           │
                    InferenceEngine<C>  (interface)
                           │
          ┌────────────────┼──────────────────┬────────────────┐
          │                │                  │                │
    OnnxEngine     TensorFlowEngine    XGBoostEngine    PmmlEngine  ...
          │
         Sink  (Kafka, JDBC, S3, …)
```

Flink is always `provided` scope - never bundled in the shaded JAR. The same artifact runs across Flink 1.15 through 2.0 without recompilation.

---

## Supported Frameworks

| Module | Framework | Formats | Notes |
|--------|-----------|---------|-------|
| `otter-stream-onnx` | ONNX Runtime 1.23.2 | `.onnx` | CPU, CUDA, TensorRT execution providers |
| `otter-stream-tensorflow` | TensorFlow Java 0.5.0 | SavedModel directory | Automatic signature discovery; GPU optional |
| `otter-stream-pytorch` | DJL 0.25.0 | TorchScript `.pt` | Auto GPU via CUDA; NDManager memory scopes |
| `otter-streams-xgboost` | XGBoost4J 3.1.1 | `.bin`, `.json`, `.ubj` | Thread-safe; native NaN handling |
| `otter-stream-pmml` | JPMML 1.5.16 | PMML 4.x `.pmml` | Built-in transforms applied automatically |
| `otter-stream-remote` | OkHttp / gRPC / AWS SDK | REST, gRPC, SageMaker, Vertex AI | Retry, circuit-breaker, custom auth headers |
| `otter-stream-feature-redis` | Jedis | Redis hash | Per-entity feature lookups |
| `otter-stream-feature-jdbc` | `java.sql` (bring your own driver) | Any JDBC source | Raw URL or pooled `DataSource` |
| `otter-stream-feature-feast` | OkHttp | Feast HTTP feature server | `POST /get-online-features` |
| `otter-stream-rules` | SnakeYAML | YAML / properties / code | Post-inference decision engine — see [Rule Engine](#rule-engine) |
| `otter-stream-rules-drools` | Drools/KIE (isolated) | `.drl` | Embedded Drools connector; optional, separate module |
| `otter-stream-kafka` | flink-connector-kafka | JSON | Publishes `InferenceResult`/`Decision` to Kafka |
| `otter-stream-sql` | Flink Table API | DDL `WITH` options | ScalarFunction + connector + lookup function |

---

## Project Structure

```
otter-streams/
├── ml-inference-core/          # InferenceEngine, InferenceResult, ModelCache, AsyncFunction,
│                                #   OtterRuntime (Provider/Registry/Lifecycle SPIs, dynamic
│                                #   loading, shadow/canary, hardware auto-scaling)
├── otter-stream-onnx/          # ONNX Runtime engine
├── otter-stream-tensorflow/    # TensorFlow SavedModel engine
├── otter-stream-pytorch/       # PyTorch TorchScript engine (DJL)
├── otter-streams-xgboost/      # XGBoost engine
├── otter-stream-pmml/          # PMML engine (JPMML)
├── otter-stream-remote/        # HTTP / gRPC / SageMaker / Vertex AI clients
├── otter-stream-feature-redis/ # Redis FeatureProvider
├── otter-stream-feature-jdbc/  # JDBC FeatureProvider
├── otter-stream-feature-feast/ # Feast FeatureProvider
├── otter-stream-rules/         # Rule/decision engine (YAML default) — see Rule Engine below
├── otter-stream-rules-drools/  # Optional embedded-Drools connector (isolated dependency tree)
├── otter-stream-kafka/         # Publishes results/decisions to Kafka
├── otter-stream-sql/           # Flink SQL UDF, connector, lookup function
├── otter-stream-examples/      # Runnable example jobs
├── otter-control-plane/        # Topology/tracing/observability platform — design phase,
│                                #   see otter-control-plane/ARCHITECTURE.md
├── PERFORMANCE.md               # Concurrency/latency review notes
└── docs/                       # HTML documentation site
```

---

## Use Cases

**Fraud detection** - score payment transactions against an ONNX or TensorFlow model inline in a Kafka-to-Kafka Flink SQL pipeline, with no Java operator code at query time.

**Anomaly detection** - apply an XGBoost model loaded from MinIO to sensor readings in a HOP window aggregation pipeline; route anomalies to a Kafka alert topic.

**Sentiment analysis** - run a PyTorch TorchScript NLP model on a streaming review feed using the DataStream async function with DJL GPU acceleration.

**Credit scoring** - evaluate PMML logistic regression models via JPMML in a Flink DataStream job; PMML's built-in preprocessing transforms are applied automatically.

**Remote model serving** - route inference to an AWS SageMaker endpoint or a custom REST API from a DataStream operator, with configurable retry and circuit-breaker policies.

---

## Flink SQL - Shaded JAR Note

When deploying to the Flink SQL Gateway, you must use the **shaded classifier JAR** produced by the `maven-shade-plugin`:

```bash
cd otter-stream-sql
mvn clean package -DskipTests

# Use this file - note the -flink-udf classifier:
ls target/otter-stream-sql-0.0.4-flink-udf.jar
```

The shaded JAR bundles `ml-inference-core` and all other runtime dependencies except Flink and SLF4J. Without it, the Flink SQL Gateway raises `NoClassDefFoundError: InferenceException` when it reflects over the `eval()` method signature during UDF registration.

---

## Compatibility

| Otter Streams | Flink | Java | ONNX Runtime | TensorFlow Java |
|---------------|-------|------|--------------|-----------------|
| **0.0.4** | 1.15 – 2.0 | 17 | 1.23.2 | 0.5.0 |
| 1.0.17 | 1.17 – 1.20 | 11, 17 | 1.23.2 | 0.5.0 |
| 1.0.15 | 1.17 – 1.18 | 11, 17 | 1.23.2 | 0.5.0 |
| 1.0.0 | 1.16 | 11 | 1.14.0 | 0.4.2 |

**Note:** Version 0.0.4 requires Java 17 and is compatible with Flink 2.0+ due to the removal of Scala-specific artifacts and the introduction of the table planner loader.

---

## Rule Engine

Turn an `InferenceResult` into a decision — a flag like `FRAUD`, `APPROVE`, or `REVIEW`, plus
which rule(s) fired — using rules you configure. **YAML is the standard, default format**;
`.properties` and fully-programmatic (a hand-written `RuleSetSource`) are equally supported,
first-class alternatives, not fallbacks.

```yaml
name: fraud-detection-rules
version: "3"
evaluationMode: SINGLE
rules:
  - id: high-risk-score
    priority: 100
    condition: "output.risk_score > 0.85"
    flag: FRAUD
    color: "#c0392b"
  - id: default-approve
    priority: 0
    condition: "true"
    flag: APPROVE
    color: "#00875a"
```

```java
RuleEngine engine = new DefaultRuleEngine(YamlRuleSetSource.fromClasspath("rules.yaml"));
Decision decision = engine.evaluate(runtime.infer("fraud-detector", inputs), Map.of());
```

- **Single, multiple, or batch flagging** — `RuleEvaluationMode.SINGLE`/`MULTIPLE` on the rule
  set, plus `evaluateBatch(...)` (auto-parallelizing above 64 items).
- **Per-rule hex colors** for dashboard rendering, validated at load time.
- **External enterprise decision engines** — `RestDecisionEngineConnector` talks to any
  REST-exposed engine (KIE Server / Red Hat Decision Manager, Camunda DMN, IBM ODM, in-house
  services) via configuration, not a bespoke SDK per vendor; `otter-stream-rules-drools` (a
  separate, optional module) embeds Drools directly for teams that need that instead of REST.
- **Zero heavy dependencies by default** — `otter-stream-rules` pulls in only SnakeYAML; Drools'
  dependency tree is fully isolated to the opt-in `otter-stream-rules-drools` module.

Full documentation: [`otter-stream-rules/README.md`](otter-stream-rules/README.md) ·
[`otter-stream-rules-drools/README.md`](otter-stream-rules-drools/README.md) · rules dashboard
design: `otter-control-plane/ARCHITECTURE.md`.

---

## Publishing Results to Kafka (or any stream)

`otter-stream-kafka` publishes `InferenceResult`s and rule-engine `Decision`s to Kafka as JSON,
built on Flink's own `KafkaSink` (not a reimplementation of it):

```java
KafkaSink<InferenceResult> sink = OtterKafkaSinks.inferenceResultSink(
        "broker1:9092,broker2:9092", "fraud-inference-results");
resultStream.sinkTo(sink);
```

For any other target system (Kinesis, Pulsar, an internal event bus, a webhook), implement the
generic `StreamResultSink<T>` SPI in `ml-inference-core` — Kafka is the concretely-shipped
integration; the SPI is the extension point for everything else.

---

## Hardware Acceleration (GPU) Auto-Scaling

`ExecutionTargetManager` (in `ml-inference-core`) watches GPU-capable engines and automatically
switches an idle one back to CPU, freeing GPU memory/compute when traffic drops:

```java
ExecutionTargetManager manager = new ExecutionTargetManager(0.05, Duration.ofMinutes(10));
manager.register("fraud-detector", onnxEngine); // engine implements ExecutionTargetAware
manager.start(Duration.ofSeconds(30));
```

Read the class Javadoc before relying on this: automatic **scale-down** (GPU → CPU when idle) is
implemented; automatic **scale-up** is deliberately an explicit `requestScaleUp(...)` call rather
than a guessed heuristic — wire it to whatever real load signal your deployment has. This is
documented as a known scope boundary, not silently glossed over.

---

## Dependency Management

Every optional integration (each ONNX/TensorFlow/PyTorch/XGBoost/PMML engine, each feature
store, the Rule Engine, Drools, Kafka) lives in its **own Maven module** with its own scoped
dependencies — pulling in `otter-stream-onnx` never drags in TensorFlow, PyTorch, Drools, or
anything else you didn't ask for. All third-party versions are pinned once, centrally, in the
root `pom.xml`'s `<dependencyManagement>`, so two Otter modules never disagree on (say) which
Jackson version to bring in. `flink-*` dependencies are always `provided` scope — your own Flink
distribution supplies them, Otter never bundles a conflicting copy. If you only need YAML-based
rules and ONNX inference, your dependency tree is exactly `ml-inference-core` +
`otter-stream-onnx` + `otter-stream-rules` (+ SnakeYAML) — nothing else.

---

Contributions of all kinds are welcome - bug reports, documentation improvements, new engine adapters, or test coverage.

```bash
# Fork, then clone your fork
git clone https://github.com/your-username/otter-streams.git
cd otter-streams

# Build
mvn clean install -DskipTests

# Run tests
mvn test
```

Good first issues are labelled [`good first issue`](https://github.com/martourez21/otter-streams/issues?q=label%3A%22good+first+issue%22) on the tracker.

---

## License

Apache License 2.0 - see [LICENSE](LICENSE) for details.

---

## Author

Built with passion by **Nestor Martourez Abiangang A.**

- GitHub: [@martourez21](https://github.com/martourez21)
- LinkedIn: [linkedin.com/in/nestor-abiangang](https://www.linkedin.com/in/nestor-abiangang/)
- Email: [nestorabiawuh@gmail.com](mailto:nestorabiawuh@gmail.com)

Special thanks to the Apache Flink community and all open-source ML framework maintainers.

---

<div align="center">

[Documentation](https://martourez21.github.io/otter-streams/docs/otter-docs/index.html) &nbsp;·&nbsp;
[DataStream Guide](https://martourez21.github.io/otter-streams/docs/otter-docs/datastream.html) &nbsp;·&nbsp;
[Studio Demo](https://martourez21.github.io/otter-streams/docs/otter-docs/studio-demo.html) &nbsp;·&nbsp;
[Discussions](https://github.com/martourez21/otter-streams/discussions) &nbsp;·&nbsp;
[Star the project](https://github.com/martourez21/otter-streams/stargazers)

</div>