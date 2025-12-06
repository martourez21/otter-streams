# Otter Stream - Technical Documentation

## Architecture Overview

Otter Stream is a production-grade ML inference library for Apache Flink, designed for real-time machine learning predictions in streaming data pipelines.

### High-Level Architecture

```mermaid
graph TB
    subgraph "Data Sources"
        A[Kafka] --> B[Flink Stream]
        C[File System] --> B
        D[Database CDC] --> B
    end
    
    subgraph "Otter Stream SDK"
        B --> E[Feature Extractor]
        E --> F[AsyncModelInferenceFunction]
        
        subgraph "Inference Engine Layer"
            F --> G{Model Format?}
            G -->|ONNX| H[OnnxInferenceEngine]
            G -->|TensorFlow| I[TensorFlowInferenceEngine]
            G -->|PyTorch| J[TorchScriptInferenceEngine]
            G -->|XGBoost| K[XGBoostInferenceEngine]
            G -->|Remote| L[RemoteInferenceEngine]
        end
        
        subgraph "Optimization Layer"
            M[ModelCache]
            N[BatchingStrategy]
            O[MetricsCollector]
        end
        
        H --> M
        I --> M
        J --> M
        K --> M
        L --> N
        
        M --> P[InferenceResult]
        N --> P
        P --> O
    end
    
    subgraph "Downstream"
        P --> Q[Alert System]
        P --> R[Database]
        P --> S[Kafka Output]
    end
    
    style E fill:#e1f5ff
    style F fill:#fff4e1
    style M fill:#e8f5e9
    style N fill:#e8f5e9
    style O fill:#e8f5e9
```

### Component Architecture

```mermaid
classDiagram
    class InferenceEngine {
        <<interface>>
        +initialize()
        +predict(input)
        +predictAsync(input)
        +predictBatch(inputs)
        +warmup(iterations)
        +isHealthy()
        +getMetadata()
        +close()
    }
    
    class InferenceConfig {
        -ModelFormat modelFormat
        -String modelPath
        -int batchSize
        -Duration timeout
        -boolean enableCaching
        +builder()
    }
    
    class AsyncModelInferenceFunction {
        -InferenceEngine engine
        -ModelCache cache
        -InferenceMetrics metrics
        +open(parameters)
        +asyncInvoke(input, resultFuture)
        +timeout(input, resultFuture)
        +close()
    }
    
    class ModelCache {
        -Cache cache
        -int maxSize
        -long ttlMillis
        +get(input)
        +put(input, result)
        +clear()
    }
    
    class InferenceResult {
        -T prediction
        -double confidence
        -long inferenceTimeMs
        -boolean success
        +getPrediction()
        +getConfidence()
    }
    
    class OnnxInferenceEngine {
        -OrtEnvironment environment
        -OrtSession session
        +initialize()
        +predict(input)
    }
    
    class TensorFlowInferenceEngine {
        -SavedModelBundle model
        -Session session
        +initialize()
        +predict(input)
    }
    
    class XGBoostInferenceEngine {
        -Booster booster
        +initialize()
        +predict(input)
    }
    
    InferenceEngine <|.. OnnxInferenceEngine
    InferenceEngine <|.. TensorFlowInferenceEngine
    InferenceEngine <|.. XGBoostInferenceEngine
    AsyncModelInferenceFunction --> InferenceEngine
    AsyncModelInferenceFunction --> ModelCache
    AsyncModelInferenceFunction --> InferenceConfig
    InferenceEngine --> InferenceResult
```

## Data Flow

### Synchronous Inference Flow

```mermaid
sequenceDiagram
    participant Stream as Flink Stream
    participant Function as AsyncInferenceFunction
    participant Cache as ModelCache
    participant Engine as InferenceEngine
    participant Metrics as MetricsCollector
    
    Stream->>Function: Input Record
    Function->>Cache: Check Cache
    
    alt Cache Hit
        Cache-->>Function: Cached Result
        Function->>Metrics: Record Cache Hit
    else Cache Miss
        Cache-->>Function: null
        Function->>Metrics: Record Cache Miss
        Function->>Engine: predict(input)
        Engine->>Engine: Load Model (if needed)
        Engine->>Engine: Preprocess Input
        Engine->>Engine: Run Inference
        Engine->>Engine: Postprocess Output
        Engine-->>Function: InferenceResult
        Function->>Cache: Store Result
    end
    
    Function->>Metrics: Record Latency
    Function-->>Stream: InferenceResult
```

### Batch Inference Flow

```mermaid
sequenceDiagram
    participant Stream as Flink Stream
    participant Function as AsyncInferenceFunction
    participant Batcher as BatchAccumulator
    participant Engine as InferenceEngine
    
    loop Accumulate Records
        Stream->>Function: Input 1
        Function->>Batcher: Add to Batch
        Stream->>Function: Input 2
        Function->>Batcher: Add to Batch
        Stream->>Function: Input 3
        Function->>Batcher: Add to Batch
    end
    
    alt Batch Full or Timeout
        Batcher->>Engine: predictBatch(inputs)
        Engine->>Engine: Batch Preprocessing
        Engine->>Engine: Single Inference Call
        Engine->>Engine: Batch Postprocessing
        Engine-->>Batcher: List<InferenceResult>
        Batcher-->>Function: Results
        Function-->>Stream: InferenceResult 1
        Function-->>Stream: InferenceResult 2
        Function-->>Stream: InferenceResult 3
    end
```

## Module Structure

### Core Module Architecture

```mermaid
graph LR
    subgraph "otter-stream-core"
        A[model] --> B[config]
        A --> C[engine]
        B --> D[function]
        C --> D
        D --> E[cache]
        D --> F[metrics]
        G[exception]
    end
    
    style A fill:#bbdefb
    style B fill:#c5e1a5
    style C fill:#ffccbc
    style D fill:#fff9c4
    style E fill:#f8bbd0
    style F fill:#e1bee7
    style G fill:#ffcdd2
```

### Module Dependencies

```mermaid
graph TD
    A[otter-stream-parent] --> B[otter-stream-core]
    A --> C[otter-stream-onnx]
    A --> D[otter-stream-tensorflow]
    A --> E[otter-stream-pytorch]
    A --> F[otter-stream-xgboost]
    A --> G[otter-stream-pmml]
    A --> H[otter-stream-remote]
    A --> I[otter-stream-examples]
    
    C --> B
    D --> B
    E --> B
    F --> B
    G --> B
    H --> B
    I --> B
    I --> C
    I --> H
    
    style A fill:#e3f2fd
    style B fill:#fff3e0
    style C fill:#e8f5e9
    style D fill:#e8f5e9
    style E fill:#e8f5e9
    style F fill:#e8f5e9
    style G fill:#e8f5e9
    style H fill:#fce4ec
    style I fill:#f3e5f5
```

## Inference Engine Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Uninitialized
    Uninitialized --> Initializing: initialize()
    Initializing --> LoadingModel: Load model files
    LoadingModel --> ValidatingModel: Validate format
    ValidatingModel --> WarmingUp: Optional warmup
    WarmingUp --> Ready: Complete
    ValidatingModel --> Ready: Skip warmup
    
    Ready --> Inferring: predict()
    Inferring --> Ready: Success
    Inferring --> Error: Exception
    Error --> Ready: Retry
    Error --> Failed: Max retries
    
    Ready --> Closing: close()
    Failed --> Closing: close()
    Closing --> [*]
```

## Remote Inference Architecture

```mermaid
graph TB
    subgraph "Otter Stream"
        A[AsyncInferenceFunction] --> B{Inference Type}
    end
    
    subgraph "Local Inference"
        B -->|Local| C[OnnxEngine]
        B -->|Local| D[TensorFlowEngine]
    end
    
    subgraph "Remote Inference"
        B -->|Remote| E[HttpRemoteClient]
        B -->|Remote| F[gRPC Client]
        B -->|Remote| G[Cloud Clients]
        
        E --> H[REST API]
        F --> I[gRPC Server]
        
        G --> J[AWS SageMaker]
        G --> K[Google Vertex AI]
        G --> L[Azure ML]
    end
    
    subgraph "Retry & Circuit Breaker"
        E --> M[RetryPolicy]
        F --> M
        G --> M
        M --> N[ExponentialBackoff]
    end
    
    style C fill:#e8f5e9
    style D fill:#e8f5e9
    style E fill:#fff3e0
    style F fill:#fff3e0
    style G fill:#fff3e0
```

## Caching Strategy

```mermaid
graph LR
    A[Input Request] --> B{Check Cache}
    B -->|Hit| C[Return Cached Result]
    B -->|Miss| D[Call Inference Engine]
    D --> E[Get Result]
    E --> F[Store in Cache]
    F --> G[Return Result]
    
    subgraph "Cache Eviction"
        H[TTL Expiry] --> I[Remove Entry]
        J[LRU Policy] --> I
        K[Max Size] --> I
    end
    
    C --> L[Update Metrics]
    G --> L
    
    style B fill:#fff9c4
    style C fill:#c8e6c9
    style D fill:#ffccbc
    style F fill:#b3e5fc
```

## Performance Optimization

### Batching Strategy

```mermaid
graph TD
    A[Input Stream] --> B[Batch Accumulator]
    
    subgraph "Batch Decision"
        B --> C{Batch Full?}
        B --> D{Timeout?}
    end
    
    C -->|Yes| E[Execute Batch Inference]
    D -->|Yes| E
    C -->|No| B
    D -->|No| B
    
    E --> F[Single Model Call]
    F --> G[Distribute Results]
    
    G --> H[Result 1]
    G --> I[Result 2]
    G --> J[Result N]
    
    style E fill:#ffecb3
    style F fill:#c5e1a5
```

### Parallel Processing

```mermaid
graph TB
    subgraph "Flink Parallelism"
        A[Source: Parallelism 4] --> B[Inference: Parallelism 8]
        B --> C[Sink: Parallelism 4]
    end
    
    subgraph "Inference Instance 1"
        D[Thread Pool]
        E[Model 1]
        F[Cache 1]
    end
    
    subgraph "Inference Instance 2"
        G[Thread Pool]
        H[Model 2]
        I[Cache 2]
    end
    
    B --> D
    B --> G
    
    style A fill:#e3f2fd
    style B fill:#fff3e0
    style C fill:#e3f2fd
```

## Metrics Collection

```mermaid
graph LR
    A[Inference Event] --> B[MetricsCollector]
    
    B --> C[Counters]
    B --> D[Histograms]
    B --> E[Meters]
    
    C --> F[total_inferences]
    C --> G[successful_inferences]
    C --> H[failed_inferences]
    C --> I[cache_hits]
    
    D --> J[latency_ms]
    
    E --> K[inference_rate]
    
    subgraph "Flink Metrics System"
        F --> L[MetricGroup]
        G --> L
        H --> L
        I --> L
        J --> L
        K --> L
    end
    
    L --> M[Prometheus]
    L --> N[InfluxDB]
    L --> O[Grafana]
    
    style B fill:#fff9c4
    style L fill:#c8e6c9
```

## Error Handling Flow

```mermaid
flowchart TD
    A[Inference Request] --> B{Try Inference}
    B -->|Success| C[Return Result]
    B -->|Exception| D{Retryable?}
    
    D -->|Yes| E{Retry Count < Max?}
    D -->|No| F[Return Error Result]
    
    E -->|Yes| G[Wait with Backoff]
    E -->|No| H{Fail on Error?}
    
    G --> B
    
    H -->|Yes| I[Throw Exception]
    H -->|No| J[Return Default Prediction]
    
    F --> K[Log Error]
    I --> K
    J --> K
    
    K --> L[Update Metrics]
    
    style B fill:#fff9c4
    style D fill:#ffccbc
    style E fill:#ffccbc
    style H fill:#ffccbc
    style C fill:#c8e6c9
```

## Deployment Architecture

### Local Deployment

```mermaid
graph TB
    subgraph "Flink Cluster"
        A[JobManager] --> B[TaskManager 1]
        A --> C[TaskManager 2]
        A --> D[TaskManager N]
        
        B --> E[Otter Stream + Model]
        C --> F[Otter Stream + Model]
        D --> G[Otter Stream + Model]
    end
    
    H[Shared Storage] --> E
    H --> F
    H --> G
    
    style A fill:#e3f2fd
    style E fill:#c8e6c9
    style F fill:#c8e6c9
    style G fill:#c8e6c9
    style H fill:#fff3e0
```

### Cloud Deployment

```mermaid
graph TB
    subgraph "Flink Application"
        A[Flink Job] --> B[Otter Stream SDK]
    end
    
    subgraph "Model Storage"
        C[S3/GCS/Azure Blob] --> B
    end
    
    subgraph "Model Serving"
        B --> D[Local Models]
        B --> E[Remote Endpoints]
    end
    
    subgraph "Remote Endpoints"
        E --> F[AWS SageMaker]
        E --> G[Vertex AI]
        E --> H[Azure ML]
    end
    
    subgraph "Monitoring"
        B --> I[CloudWatch/Stackdriver/Azure Monitor]
        B --> J[Prometheus]
    end
    
    style A fill:#e3f2fd
    style B fill:#c8e6c9
    style C fill:#fff3e0
```

## Configuration Flow

```mermaid
flowchart LR
    A[Application Code] --> B[InferenceConfig.builder]
    
    B --> C{Model Type?}
    
    C -->|Local| D[Set Model Path]
    C -->|Remote| E[Set Endpoint URL]
    
    D --> F[Configure Performance]
    E --> F
    
    F --> G[Set Batch Size]
    F --> H[Set Timeout]
    F --> I[Enable Caching]
    F --> J[Enable Metrics]
    
    G --> K[Build Config]
    H --> K
    I --> K
    J --> K
    
    K --> L[Create InferenceFunction]
    L --> M[Apply to Stream]
    
    style B fill:#fff9c4
    style F fill:#c8e6c9
    style L fill:#ffccbc
```

## Model Loading Strategies

```mermaid
graph TD
    A[Model Path] --> B{Source Type?}
    
    B -->|file://| C[Local File System]
    B -->|s3://| D[AWS S3]
    B -->|hdfs://| E[HDFS]
    B -->|http://| F[HTTP Download]
    B -->|classpath:| G[JAR Resource]
    
    C --> H[Validate Format]
    D --> I[Download to Temp]
    E --> I
    F --> I
    G --> I
    
    I --> H
    H --> J{Format Valid?}
    
    J -->|Yes| K[Load into Engine]
    J -->|No| L[Throw Exception]
    
    K --> M[Model Ready]
    
    style B fill:#fff9c4
    style H fill:#ffccbc
    style K fill:#c8e6c9
```

## Thread Safety Model

```mermaid
graph TB
    subgraph "Flink Task Manager"
        A[Main Thread] --> B[Async Executor Thread Pool]
        
        subgraph "Thread-Safe Components"
            C[InferenceEngine - Thread Local]
            D[ModelCache - Concurrent]
            E[MetricsCollector - Atomic]
        end
        
        B --> F[Thread 1] --> C
        B --> G[Thread 2] --> C
        B --> H[Thread N] --> C
        
        F --> D
        G --> D
        H --> D
        
        F --> E
        G --> E
        H --> E
    end
    
    style C fill:#c8e6c9
    style D fill:#c8e6c9
    style E fill:#c8e6c9
```

## Extension Points

```mermaid
classDiagram
    class InferenceEngine {
        <<interface>>
        +predict()
        +predictAsync()
    }
    
    class CustomEngine {
        +predict()
        +predictAsync()
    }
    
    class ModelLoader {
        +loadModel()
    }
    
    class CustomLoader {
        +loadModel()
    }
    
    class CacheStrategy {
        +shouldCache()
        +evict()
    }
    
    class CustomCacheStrategy {
        +shouldCache()
        +evict()
    }
    
    InferenceEngine <|.. CustomEngine : implement
    ModelLoader <|-- CustomLoader : extend
    CacheStrategy <|.. CustomCacheStrategy : implement
    
    note for CustomEngine "Add support for\nnew ML frameworks"
    note for CustomLoader "Add support for\nnew storage systems"
    note for CustomCacheStrategy "Implement custom\ncaching policies"
```

## Performance Characteristics

### Latency Breakdown

```mermaid
gantt
    title Inference Latency Components
    dateFormat X
    axisFormat %L ms
    
    section Single Inference
    Input Deserialization :0, 5
    Cache Lookup :5, 10
    Model Inference :10, 100
    Output Serialization :100, 105
    
    section Batch Inference (32 items)
    Input Deserialization :0, 10
    Cache Lookup :10, 20
    Batch Model Inference :20, 150
    Output Serialization :150, 160
```

### Throughput Comparison

```mermaid
graph LR
    A[Configuration] --> B{Strategy}
    
    B -->|No Batching| C[1000 req/sec]
    B -->|Batch Size 32| D[8000 req/sec]
    B -->|Batch + Cache| E[15000 req/sec]
    
    style C fill:#ffccbc
    style D fill:#fff9c4
    style E fill:#c8e6c9
```

## Technical Specifications

### Supported Java Versions
- Java 11 (LTS) - Primary
- Java 17 (LTS) - Supported

### Flink Compatibility
- Apache Flink 1.18.x
- Compatible with Flink 1.17.x and 1.19.x

### Concurrency Model
- Async I/O with Flink's Async Function API
- Configurable thread pools per operator
- Thread-safe model instances

### Memory Management
- Model loaded once per task manager
- Configurable cache sizes with LRU eviction
- Automatic memory cleanup on task failure

### Fault Tolerance
- Checkpointing support
- Model reload on task restart
- Exactly-once semantics (with Kafka/Pulsar)

## Integration Patterns

### Pattern 1: Feature Store Integration

```mermaid
sequenceDiagram
    participant Kafka
    participant Flink
    participant FeatureStore
    participant OtterStream
    participant OutputSink
    
    Kafka->>Flink: Raw Event
    Flink->>FeatureStore: Lookup Features
    FeatureStore-->>Flink: Feature Vector
    Flink->>OtterStream: predict(features)
    OtterStream-->>Flink: Prediction
    Flink->>OutputSink: Enriched Event + Prediction
```

### Pattern 2: Model A/B Testing

```mermaid
graph LR
    A[Input Stream] --> B{Random Split}
    B -->|90%| C[Model A v1.0]
    B -->|10%| D[Model B v2.0]
    
    C --> E[Results with Model Tag]
    D --> E
    
    E --> F[Metrics Analysis]
    
    style C fill:#c8e6c9
    style D fill:#fff9c4
```

### Pattern 3: Ensemble Models

```mermaid
graph TD
    A[Input Features] --> B[Model 1: ONNX]
    A --> C[Model 2: XGBoost]
    A --> D[Model 3: TensorFlow]
    
    B --> E[Aggregator]
    C --> E
    D --> E
    
    E --> F{Ensemble Strategy}
    F -->|Average| G[Final Prediction]
    F -->|Weighted| G
    F -->|Voting| G
    
    style E fill:#fff9c4
    style G fill:#c8e6c9
```

## Glossary

- **Inference**: The process of using a trained ML model to make predictions
- **Async I/O**: Non-blocking I/O operations that don't block Flink's main thread
- **Batching**: Accumulating multiple requests to process together for efficiency
- **Cache Hit**: When a prediction is found in cache without invoking the model
- **Model Warmup**: Running initial predictions to optimize JIT compilation
- **Backpressure**: Flink's flow control mechanism to prevent overwhelming downstream
- **Watermark**: Flink's mechanism for tracking event time progress
- **Task Manager**: Flink worker process that executes tasks
- **Job Manager**: Flink coordinator that manages the cluster and jobs

## References

- [Apache Flink Documentation](https://flink.apache.org/docs/)
- [ONNX Runtime Java API](https://onnxruntime.ai/docs/api/java/)
- [TensorFlow Java](https://www.tensorflow.org/jvm)
- [Deep Java Library (DJL)](https://djl.ai/)