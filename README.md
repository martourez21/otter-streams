# Otter Streams

> Production-grade machine learning inference for Apache Flink

[![Otter Streams CI Pipeline](https://github.com/martourez21/otter-streams/actions/workflows/ci.yaml/badge.svg)](https://github.com/martourez21/otter-streams/actions/workflows/ci.yaml)
[![Maven Central](https://img.shields.io/maven-central/v/com.codedstreams/otter-streams)](https://search.maven.org/artifact/com.codedstreams/otter-streams)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![GitHub stars](https://img.shields.io/github/stars/martourez21/otter-streams?style=social)](https://github.com/martourez21/otter-streams/stargazers)
[![Discussions](https://img.shields.io/badge/Discussions-Enabled-blue)](https://github.com/martourez21/otter-streams/discussions)

<div align="center">
  <img src="docs/assets/otterstream-sdk-icon.ico" alt="Otter Streams Logo" width="120"/>
</div>

---

## What is Otter Streams?

Otter Streams is an open-source library that brings production-grade machine learning inference to Apache Flink streaming applications. It was built first as a **DataStream API** library — letting you embed ML inference directly into Flink operators via `AsyncDataStream` — and extended in v1.0.17 with a full **Flink SQL / Table API** layer, including a scalar UDF, a lookup table function, and the `ml-inference` dynamic table connector.

Models are stored externally (MinIO, AWS S3, or any S3-compatible store) and loaded at engine initialisation time into a Caffeine LRU `ModelCache` that is shared across both the DataStream and SQL paths.

### Why Otter Streams?

- **Real-time ML at scale** — millisecond-latency inference on unbounded Flink streams
- **Framework agnostic** — ONNX, TensorFlow SavedModel, PyTorch TorchScript, XGBoost, PMML, and remote endpoints (HTTP/gRPC, SageMaker, Vertex AI)
- **Two APIs, one cache** — DataStream operators and Flink SQL queries share the same `ModelCache` singleton; models are downloaded from MinIO once per session
- **SQL-native deployment** — register `ml_score()` as a Flink UDF and call it inline in any SQL query; no Java operator required at query time
- **Production ready** — Caffeine LRU caching, Micrometer metrics (Prometheus / InfluxDB / DataDog), async retry, configurable timeouts, dead-letter routing
- **MinIO / S3 model store** — the `ml-inference` connector downloads SavedModel directories and single-file formats (`onnx`, `xgboost-json`, `pmml`) from any S3-compatible endpoint at table-creation time

---

## Quick Start

### 1 — Add the core dependency

```xml
<!-- Core framework — required by all engine modules -->
<dependency>
    <groupId>com.codedstreams</groupId>
    <artifactId>ml-inference-core</artifactId>
    <version>1.0.17</version>
</dependency>

<!-- Pick the engine module(s) you need -->
<dependency>
    <groupId>com.codedstreams</groupId>
    <artifactId>otter-stream-onnx</artifactId>
    <version>1.0.17</version>
</dependency>

<!-- Flink SQL / Table API integration (optional) -->
<dependency>
    <groupId>com.codedstreams</groupId>
    <artifactId>otter-stream-sql</artifactId>
    <version>1.0.17</version>
</dependency>
```

### 2 — DataStream API

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

### 3 — Flink SQL / Table API

```sql
-- Load the shaded JAR and register the scalar UDF
ADD JAR '/var/www/udf-jars/otter-stream-sql-1.0.17-flink-udf.jar';

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
| [Documentation](https://martourez21.github.io/otter-streams/docs/index.html) | Introduction, Quick Start, Architecture, UDF Reference, Shaded JAR, Troubleshooting |
| [DataStream API Guide](https://martourez21.github.io/otter-streams/docs/datastream.html) | Complete Java code for every engine module — ONNX, TensorFlow, PyTorch, XGBoost, PMML, Remote |
| [SQL Examples](https://martourez21.github.io/otter-streams/docs/examples.html) | MinIO pipeline demo (6 SQL sections), fraud detection, IoT anomaly detection |
| [Studio Demo](https://martourez21.github.io/otter-streams/docs/studio-demo.html) | Str:::Lab Studio Feature Engineering Manager + Inference Manager walkthrough |
| [API Reference](https://martourez21.github.io/otter-streams/docs/api.html) | InferenceEngine, InferenceResult, ModelCache, MLInferenceFunction, connector DDL options |
| [Modules](https://martourez21.github.io/otter-streams/docs/modules.html) | Dependency graph + per-module docs for all 7 modules |
| [Releases](https://martourez21.github.io/otter-streams/docs/releases.html) | Changelog, compatibility matrix, upgrade notes |
| [Javadoc](https://martourez21.github.io/otter-streams/javadoc/) | Full generated API documentation |

---

## Architecture

Otter Streams has two parallel integration paths — both converge on the same engine and cache layer.

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

Flink is always `provided` scope — never bundled in the shaded JAR. The same artifact runs across Flink 1.15 through 1.20 without recompilation.

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
| `otter-stream-sql` | Flink Table API | DDL `WITH` options | ScalarFunction + connector + lookup function |

---

## Project Structure

```
otter-streams/
├── ml-inference-core/          # InferenceEngine, InferenceResult, ModelCache, AsyncFunction
├── otter-stream-onnx/          # ONNX Runtime engine
├── otter-stream-tensorflow/    # TensorFlow SavedModel engine
├── otter-stream-pytorch/       # PyTorch TorchScript engine (DJL)
├── otter-streams-xgboost/      # XGBoost engine
├── otter-stream-pmml/          # PMML engine (JPMML)
├── otter-stream-remote/        # HTTP / gRPC / SageMaker / Vertex AI clients
├── otter-stream-sql/           # Flink SQL UDF, connector, lookup function
├── otter-stream-examples/      # Runnable example jobs
└── docs/                       # HTML documentation site
```

---

## Use Cases

**Fraud detection** — score payment transactions against an ONNX or TensorFlow model inline in a Kafka-to-Kafka Flink SQL pipeline, with no Java operator code at query time.

**Anomaly detection** — apply an XGBoost model loaded from MinIO to sensor readings in a HOP window aggregation pipeline; route anomalies to a Kafka alert topic.

**Sentiment analysis** — run a PyTorch TorchScript NLP model on a streaming review feed using the DataStream async function with DJL GPU acceleration.

**Credit scoring** — evaluate PMML logistic regression models via JPMML in a Flink DataStream job; PMML's built-in preprocessing transforms are applied automatically.

**Remote model serving** — route inference to an AWS SageMaker endpoint or a custom REST API from a DataStream operator, with configurable retry and circuit-breaker policies.

---

## Flink SQL — Shaded JAR Note

When deploying to the Flink SQL Gateway, you must use the **shaded classifier JAR** produced by the `maven-shade-plugin`:

```bash
cd otter-stream-sql
mvn clean package -DskipTests

# Use this file — note the -flink-udf classifier:
ls target/otter-stream-sql-1.0.17-flink-udf.jar
```

The shaded JAR bundles `ml-inference-core` and all other runtime dependencies except Flink and SLF4J. Without it, the Flink SQL Gateway raises `NoClassDefFoundError: InferenceException` when it reflects over the `eval()` method signature during UDF registration.

---

## Compatibility

| Otter Streams | Flink | Java | ONNX Runtime | TensorFlow Java |
|---------------|-------|------|--------------|-----------------|
| **1.0.17** | 1.17 – 1.20 | 11, 17 | 1.23.2 | 0.5.0 |
| 1.0.15 | 1.17 – 1.18 | 11, 17 | 1.23.2 | 0.5.0 |
| 1.0.0 | 1.16 | 11 | 1.14.0 | 0.4.2 |

---

## Contributing

Contributions of all kinds are welcome — bug reports, documentation improvements, new engine adapters, or test coverage.

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

Apache License 2.0 — see [LICENSE](LICENSE) for details.

---

## Author

Built with passion by **Nestor Martourez Abiangang A.**

- GitHub: [@martourez21](https://github.com/martourez21)
- LinkedIn: [linkedin.com/in/nestor-abiangang](https://www.linkedin.com/in/nestor-abiangang/)
- Email: [nestorabiawuh@gmail.com](mailto:nestorabiawuh@gmail.com)

Special thanks to the Apache Flink community and all open-source ML framework maintainers.

---

<div align="center">

[Documentation](https://martourez21.github.io/otter-streams/docs/index.html) &nbsp;·&nbsp;
[DataStream Guide](https://martourez21.github.io/otter-streams/docs/datastream.html) &nbsp;·&nbsp;
[Studio Demo](https://martourez21.github.io/otter-streams/docs/studio-demo.html) &nbsp;·&nbsp;
[Discussions](https://github.com/martourez21/otter-streams/discussions) &nbsp;·&nbsp;
[Star the project](https://github.com/martourez21/otter-streams/stargazers)

</div>