# 🏗️ Otter Streams Architecture

This document describes the architecture and design principles behind Otter Streams, helping you understand how the system works and how to extend it.

##  Design Philosophy

Otter Streams is built on these core principles:

1. **Modular Design**: Each component is independent and replaceable
2. **Async-First**: Non-blocking operations for maximum throughput
3. **Extensible**: Easy to add new model formats and inference engines
4. **Production-Ready**: Built-in monitoring, caching, and fault tolerance
5. **Resource Efficient**: Intelligent batching and memory management

## 📊 System Overview

```mermaid
graph TB
    subgraph "Flink Application"
        A[DataStream] --> B[Async Inference Function]
    end
    
    subgraph "Inference Core"
        B --> C[Model Router]
        C --> D[ONNX Engine]
        C --> E[TensorFlow Engine]
        C --> F[PyTorch Engine]
        C --> G[XGBoost Engine]
        C --> H[PMML Engine]
        C --> I[Remote Engine]
    end
    
    subgraph "Infrastructure"
        J[(Model Cache)] --> C
        K[(Result Cache)] --> B
        L[Metrics Collector] --> M[Monitoring]
        N[Configuration Manager]
    end
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style C fill:#fff3e0
    style J fill:#e8f5e8
```

##  Core Components

### 1. AsyncModelInferenceFunction
The main entry point for integrating ML inference with Flink streams.

```java
public class AsyncModelInferenceFunction<IN, OUT> 
    extends RichAsyncFunction<IN, OUT> {
    
    @Override
    public void asyncInvoke(IN input, ResultFuture<OUT> resultFuture) {
        // Async inference logic
    }
    
    @Override
    public void open(Configuration parameters) {
        // Initialize engine and caches
    }
}
```

**Key Features**:
- Non-blocking async operations
- Automatic batching
- Result caching
- Error handling and retries

### 2. InferenceEngine Interface
The abstraction for all inference engines:

```java
public interface InferenceEngine {
    void initialize(ModelConfig config);
    InferenceOutput execute(Object input);
    InferenceOutput executeBatch(List<Object> inputs);
    Map<String, String> getMetrics();
    void close();
}
```

### 3. Model Configuration System
Centralized configuration management:

```java
@Builder
public class ModelConfig {
    private String modelId;
    private String modelPath;
    private ModelFormat format;
    private Map<String, String> modelOptions;
    private Map<String, String> engineOptions;
    private String signatureName;
    private List<String> inputNames;
    private List<String> outputNames;
}
```

##  Inference Engines

### ONNX Runtime Engine

**Architecture**:
```mermaid
graph LR
    A[ONNX Model] --> B[Session Initialization]
    B --> C[Memory Allocation]
    C --> D[Tensor Creation]
    D --> E[Session Run]
    E --> F[Result Extraction]
    F --> G[Memory Cleanup]
    
    style B fill:#e3f2fd
    style E fill:#f3e5f5
```

**Features**:
- GPU acceleration support
- Thread pool management
- Memory optimization
- Multiple execution providers

**Configuration**:
```java
ModelConfig.builder()
    .modelOptions(Map.of(
        "executionProvider", "CUDA",
        "intraOpThreads", "4",
        "interOpThreads", "2",
        "optimizationLevel", "ALL"
    ))
    .build();
```

### TensorFlow Engine

**Architecture**:
```mermaid
graph TB
    A[SavedModel Directory] --> B[Model Loading]
    B --> C[Signature Discovery]
    C --> D[Graph Optimization]
    D --> E[Session Creation]
    E --> F[GPU/CPU Allocation]
    F --> G[Inference Execution]
    
    style B fill:#e3f2fd
    style G fill:#f3e5f5
```

**Features**:
- SavedModel format support
- Automatic signature detection
- GPU memory management
- TensorFlow Serving compatibility

### PyTorch Engine (Deep Java Library)

**Architecture**:
```mermaid
graph LR
    A[TorchScript Model] --> B[Model Loading]
    B --> C[Device Detection]
    C --> D[GPU: CUDA]
    C --> E[CPU: MKL]
    D --> F[Inference]
    E --> F
    
    style B fill:#e3f2fd
    style C fill:#fff3e0
```

## 🗄️ Caching System

### Multi-Level Cache Architecture

```mermaid
graph TB
    A[Inference Request] --> B{Model Cached?}
    B -->|Yes| C[Use Cached Model]
    B -->|No| D[Load from Disk]
    D --> E[Cache Model]
    
    C --> F{Result Cached?}
    F -->|Yes| G[Return Cached Result]
    F -->|No| H[Execute Inference]
    H --> I[Cache Result]
    I --> J[Return Result]
    
    style B fill:#fff3e0
    style F fill:#fff3e0
```

### Cache Implementation

```java
public class InferenceCache {
    private Cache<String, InferenceEngine> modelCache;
    private Cache<String, InferenceOutput> resultCache;
    
    // Model cache: TTL based, LRU eviction
    // Result cache: Input hash based, configurable TTL
}
```

## 📊 Monitoring & Metrics

### Metrics Collection Architecture

```mermaid
graph TB
    A[Inference Engine] --> B[Metrics Collector]
    B --> C[Throughput Metrics]
    B --> D[Latency Metrics]
    B --> E[Error Metrics]
    B --> F[Cache Metrics]
    
    C --> G[Micrometer Registry]
    D --> G
    E --> G
    F --> G
    
    G --> H[Prometheus]
    G --> I[Graphite]
    G --> J[CloudWatch]
    
    style B fill:#e3f2fd
    style G fill:#f3e5f5
```

### Available Metrics

```java
public interface InferenceMetrics {
    // Throughput
    String INFERENCE_COUNT = "inference.count";
    String INFERENCE_RATE = "inference.rate";
    
    // Latency
    String INFERENCE_LATENCY = "inference.latency";
    String P50_LATENCY = "inference.latency.p50";
    String P95_LATENCY = "inference.latency.p95";
    String P99_LATENCY = "inference.latency.p99";
    
    // Cache
    String CACHE_HITS = "cache.hits";
    String CACHE_MISSES = "cache.misses";
    String CACHE_HIT_RATIO = "cache.hit.ratio";
    
    // Errors
    String ERROR_COUNT = "error.count";
    String ERROR_RATE = "error.rate";
}
```

##  Data Flow

### End-to-End Flow

```mermaid
sequenceDiagram
    participant F as Flink Task
    participant IC as Inference Cache
    participant IE as Inference Engine
    participant M as ML Model
    participant MC as Metrics Collector
    
    F->>IC: Check model cache
    IC-->>F: Model not cached
    F->>IE: Load model from disk
    IE->>M: Initialize model
    M-->>IE: Model ready
    IE->>IC: Cache model
    IC-->>F: Model cached
    
    loop For each input
        F->>IC: Check result cache
        alt Cache hit
            IC-->>F: Return cached result
        else Cache miss
            F->>IE: Execute inference
            IE->>M: Run model
            M-->>IE: Inference result
            IE->>MC: Record metrics
            MC-->>F: Update statistics
            IE->>IC: Cache result
            IC-->>F: Return result
        end
    end
```

##  Performance Optimization

### Batching Strategy

```java
public class SmartBatchingStrategy {
    private int batchSize;
    private Duration batchTimeout;
    private int maxBatchSize;
    
    public List<Object> createBatch(List<Object> inputs) {
        // Dynamic batching based on:
        // 1. Batch size limit
        // 2. Timeout expiration
        // 3. Input similarity
        // 4. System load
    }
}
```

### Memory Management

```java
public class MemoryManager {
    private long maxMemoryBytes;
    private MemoryPool memoryPool;
    
    public void allocateForModel(String modelId, long requiredBytes) {
        // Intelligent allocation with:
        // - Memory pooling
        // - LRU eviction
        // - Fragmentation prevention
    }
}
```

## 🔌 Extension Points

### Creating a Custom Engine

```java
public class CustomInferenceEngine implements InferenceEngine {
    
    @Override
    public void initialize(ModelConfig config) {
        // Initialize your engine
    }
    
    @Override
    public InferenceOutput execute(Object input) {
        // Execute inference
    }
    
    @Override
    public InferenceOutput executeBatch(List<Object> inputs) {
        // Batch execution
    }
}
```

### Registering a New Engine

```java
public class EngineRegistry {
    private static final Map<ModelFormat, Supplier<InferenceEngine>> engines = new HashMap<>();
    
    static {
        engines.put(ModelFormat.ONNX, OnnxInferenceEngine::new);
        engines.put(ModelFormat.TENSORFLOW, TensorFlowInferenceEngine::new);
        // Register your custom engine
        engines.put(ModelFormat.CUSTOM, CustomInferenceEngine::new);
    }
}
```

## 🏗️ Module Dependencies

```mermaid
graph TD
    A[User Application] --> B[ml-inference-core]
    
    B --> C[Async Processing]
    B --> D[Configuration]
    B --> E[Caching]
    B --> F[Metrics]
    
    C --> G[Flink Runtime]
    D --> H[Config Files]
    E --> I[Cache Stores]
    F --> J[Metrics Backends]
    
    B --> K[otter-stream-onnx]
    B --> L[otter-stream-tensorflow]
    B --> M[otter-stream-pytorch]
    B --> N[otter-stream-xgboost]
    B --> O[otter-stream-pmml]
    B --> P[otter-stream-remote]
    
    K --> Q[ONNX Runtime]
    L --> R[TensorFlow Java]
    M --> S[Deep Java Library]
    N --> T[XGBoost4J]
    O --> U[JPMML]
    P --> V[HTTP Client]
    
    style B fill:#e3f2fd
    style K fill:#fff3e0
    style L fill:#e8f5e8
```

## 🔮 Future Architecture

### Planned Improvements

1. **Distributed Model Serving**
    - Model sharding across nodes
    - Load balancing
    - Automatic scaling

2. **Feature Store Integration**
    - Real-time feature computation
    - Feature versioning
    - Feature monitoring

3. **A/B Testing Framework**
    - Model version routing
    - Experiment management
    - Performance comparison

---

**Need to extend the architecture?** Check out our [Contributing Guide](CONTRIBUTING.md) for details on adding new features.
