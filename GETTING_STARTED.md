# Getting Started with Otter Streams

## 🎯 What's Implemented

### ✅ **FULLY PRODUCTION-READY**

1. **Complete Core Framework**
   - `InferenceEngine` interface with local and remote implementations
   - `AsyncModelInferenceFunction` for seamless Flink integration
   - `ModelCache` with Caffeine for high-performance caching
   - `InferenceMetrics` with Micrometer integration
   - Comprehensive configuration system with builder pattern
   - Multi-source model loading (local files, streams)

2. **ONNX Runtime Support** - Full Implementation
   - `OnnxInferenceEngine` with GPU acceleration
   - `OnnxModelLoader` with session optimization
   - Supports models from PyTorch, TensorFlow, Scikit-learn
   - Async and batch inference capabilities
   - Tensor creation and extraction utilities

3. **TensorFlow Support** - Complete
   - `TensorFlowInferenceEngine` for SavedModel format
   - Native TensorFlow Java API integration
   - GPU support and batch processing
   - Tensor creation from Java types

4. **PyTorch Support** - Complete via DJL
   - `TorchScriptInferenceEngine` with Deep Java Library
   - Automatic native library detection
   - Custom translator for input/output mapping
   - Memory-optimized inference

5. **XGBoost Support** - Full Implementation
   - `XGBoostInferenceEngine` with batch inference
   - DMatrix creation from feature maps
   - Both single and batch prediction modes

6. **PMML Support** - Complete
   - `PmmlInferenceEngine` with JPMML evaluator
   - Standard PMML model support
   - Field mapping and result extraction

7. **Remote Inference Clients** - Multiple Providers
   - `HttpInferenceClient` for REST APIs
   - `SageMakerInferenceClient` for AWS SageMaker
   - Authentication, retry logic, and timeout handling
   - JSON serialization for request/response

8. **Complete Example** - Fraud Detection Pipeline
   - End-to-end Flink streaming job
   - Realistic transaction data generation
   - Feature extraction and ML inference
   - Result processing and alerting

9. **CI/CD Pipeline** - GitHub Actions
   - Multi-Java version testing (11, 17)
   - Automatic Javadoc generation and deployment
   - GitHub Packages publishing
   - Release automation
   - Docker image building

## 🚀 Quick Start (5 Minutes)

### Step 1: Clone and Build

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/otter-streams.git
cd otter-streams

# Run setup script (verifies environment and builds)
chmod +x setup.sh
./setup.sh

# Or build manually
mvn clean install -DskipTests
```

### Step 2: Prepare Your Model

Export your model to any supported format:

**ONNX (Recommended - Most Compatible)**
```python
# PyTorch
import torch
torch.onnx.export(
    model, 
    dummy_input, 
    "model.onnx",
    input_names=['input'],
    output_names=['output']
)

# TensorFlow
import tf2onnx
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
with open("model.onnx", "wb") as f:
    f.write(model_proto.SerializeToString())

# Scikit-learn
from skl2onnx import to_onnx
onnx_model = to_onnx(model, X[:1].astype(np.float32))
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

**Other Supported Formats**
- TensorFlow SavedModel (`.pb` files in saved_model directory)
- PyTorch TorchScript (`.pt` files)
- XGBoost (`.model`, `.xgb`, `.ubj` files)
- PMML (`.pmml`, `.xml` files)

Place your model in your project's model directory.

### Step 3: Use in Your Flink Job

```java
import com.flinkml.inference.config.InferenceConfig;
import com.flinkml.inference.config.ModelConfig;
import com.flinkml.inference.model.ModelFormat;
import com.flinkml.inference.function.AsyncModelInferenceFunction;
import com.flinkml.inference.onnx.OnnxInferenceEngine;

// 1. Configure your model
InferenceConfig config = InferenceConfig.builder()
    .modelConfig(ModelConfig.builder()
        .modelId("fraud-detection")
        .modelPath("models/fraud_model.onnx")
        .format(ModelFormat.ONNX)
        .modelName("fraud_predictor")
        .modelVersion("1.0")
        .build())
    .batchSize(32)
    .timeout(5000)
    .maxRetries(3)
    .enableMetrics(true)
    .build();

// 2. Create inference function
AsyncModelInferenceFunction<Map<String, Object>, InferenceResult> inferenceFunction =
    new AsyncModelInferenceFunction<>(
        config,
        cfg -> new OnnxInferenceEngine()
    );

// 3. Apply to Flink stream
DataStream<InferenceResult> predictions = AsyncDataStream.unorderedWait(
    transactionStream,
    inferenceFunction,
    5000,                    // Async timeout
    TimeUnit.MILLISECONDS,
    100                      // Max concurrent requests
);

// 4. Process results
predictions
    .filter(InferenceResult::isSuccess)
    .map(result -> {
        float fraudProbability = result.getOutput("fraud_probability");
        return "Fraud probability: " + fraudProbability;
    })
    .print();
```

### Step 4: Run the Example

```bash
# Run the fraud detection example
mvn exec:java -pl otter-stream-examples \
  -Dexec.mainClass="com.flinkml.inference.examples.FraudDetectionExample"
```

## 📦 Maven Dependencies

Add to your `pom.xml`:

```xml
<!-- Core framework (required) -->
<dependency>
    <groupId>com.codedstreams</groupId>
    <artifactId>ml-inference-core</artifactId>
    <version>1.0.0</version>
</dependency>

<!-- Add framework-specific modules as needed -->
<dependency>
    <groupId>com.codedstreams</groupId>
    <artifactId>otter-stream-onnx</artifactId>
    <version>1.0.0</version>
</dependency>

<dependency>
    <groupId>com.codedstreams</groupId>
    <artifactId>otter-stream-tensorflow</artifactId>
    <version>1.0.0</version>
</dependency>

<dependency>
    <groupId>com.codedstreams</groupId>
    <artifactId>otter-stream-pytorch</artifactId>
    <version>1.0.0</version>
</dependency>

<dependency>
    <groupId>com.codedstreams</groupId>
    <artifactId>otter-streams-xgboost</artifactId>
    <version>1.0.0</version>
</dependency>

<dependency>
    <groupId>com.codedstreams</groupId>
    <artifactId>otter-stream-pnnnl</artifactId>
    <version>1.0.0</version>
</dependency>

<dependency>
    <groupId>com.codedstreams</groupId>
    <artifactId>otter-stream-remote</artifactId>
    <version>1.0.0</version>
</dependency>
```

## 🏗️ Complete Project Structure

```
otter-streams/
├── ml-inference-core/           ✅ Complete
│   ├── model/
│   │   ├── ModelLoader.java
│   │   ├── ModelMetadata.java
│   │   ├── ModelFormat.java
│   │   └── InferenceResult.java
│   ├── engine/
│   │   ├── InferenceEngine.java
│   │   ├── LocalInferenceEngine.java
│   │   └── RemoteInferenceEngine.java
│   ├── function/
│   │   └── AsyncModelInferenceFunction.java
│   ├── config/
│   │   ├── InferenceConfig.java
│   │   ├── ModelConfig.java
│   │   └── AuthConfig.java
│   ├── metrics/
│   │   ├── InferenceMetrics.java
│   │   └── MetricsCollector.java
│   ├── cache/
│   │   ├── ModelCache.java
│   │   └── CacheStrategy.java
│   └── exception/
│       ├── ModelLoadException.java
│       └── InferenceException.java
│
├── otter-stream-onnx/           ✅ Complete
│   └── OnnxInferenceEngine.java & OnnxModelLoader.java
│
├── otter-stream-tensorflow/     ✅ Complete
│   └── TensorFlowInferenceEngine.java
│
├── otter-stream-pytorch/        ✅ Complete
│   └── TorchScriptInferenceEngine.java
│
├── otter-streams-xgboost/       ✅ Complete
│   └── XGBoostInferenceEngine.java
│
├── otter-stream-pnnnl/          ✅ Complete
│   └── PmmlInferenceEngine.java
│
├── otter-stream-remote/         ✅ Complete
│   ├── http/HttpInferenceClient.java
│   └── sagemaker/SageMakerInferenceClient.java
│
├── otter-stream-examples/       ✅ Complete
│   └── FraudDetectionExample.java
│
└── Infrastructure               ✅ Complete
    ├── .github/workflows/maven.yml
    ├── Dockerfile
    ├── setup.sh
    ├── verify-implementation.sh
    └── Complete POM hierarchy
```

## 🔧 Supported Model Formats

| Format | Status | File Extensions | Best For |
|--------|--------|-----------------|----------|
| **ONNX** | ✅ Production Ready | `.onnx` | Cross-framework (PyTorch, TF, Scikit-learn) |
| **TensorFlow** | ✅ Production Ready | `saved_model.pb` | TensorFlow ecosystems |
| **PyTorch** | ✅ Production Ready | `.pt`, `.pth` | PyTorch models via TorchScript |
| **XGBoost** | ✅ Production Ready | `.model`, `.xgb`, `.ubj` | Gradient boosting |
| **PMML** | ✅ Production Ready | `.pmml`, `.xml` | Standardized model exchange |
| **HTTP/REST** | ✅ Production Ready | N/A | Remote endpoints, SageMaker |
| **AWS SageMaker** | ✅ Production Ready | N/A | AWS ML deployments |

**Recommendation**: Use **ONNX** for maximum compatibility across frameworks!

## 🎛️ Configuration Examples

### Local ONNX Model

```java
InferenceConfig config = InferenceConfig.builder()
    .modelConfig(ModelConfig.builder()
        .modelId("sentiment-analysis")
        .modelPath("/models/sentiment.onnx")
        .format(ModelFormat.ONNX)
        .modelOptions(Map.of(
            "interOpThreads", 4,
            "intraOpThreads", 2,
            "optimizationLevel", "ALL"
        ))
        .build())
    .batchSize(64)
    .timeout(5000)
    .enableCaching(true)
    .enableMetrics(true)
    .build();
```

### TensorFlow SavedModel

```java
InferenceConfig config = InferenceConfig.builder()
    .modelConfig(ModelConfig.builder()
        .modelId("image-classifier")
        .modelPath("/models/tf_saved_model")
        .format(ModelFormat.TENSORFLOW_SAVEDMODEL)
        .build())
    .batchSize(32)
    .build();
```

### Remote HTTP Endpoint

```java
InferenceConfig config = InferenceConfig.builder()
    .modelConfig(ModelConfig.builder()
        .modelId("cloud-model")
        .format(ModelFormat.REMOTE_HTTP)
        .endpointUrl("https://api.example.com/v1/predict")
        .authConfig(AuthConfig.builder()
            .apiKey("your-api-key")
            .headers(Map.of(
                "Content-Type", "application/json",
                "X-Custom-Header", "value"
            ))
            .build())
        .build())
    .timeout(10000)
    .maxRetries(3)
    .build();
```

### AWS SageMaker Endpoint

```java
InferenceConfig config = InferenceConfig.builder()
    .modelConfig(ModelConfig.builder()
        .modelId("sagemaker-model")
        .format(ModelFormat.SAGEMAKER)
        .endpointUrl("my-production-endpoint")
        .authConfig(AuthConfig.builder()
            .apiKey("AWS_ACCESS_KEY:AWS_SECRET_KEY")
            .build())
        .build())
    .build();
```

## 📊 GitHub Actions Setup

### Zero Configuration Required! 🎉

The pipeline uses GitHub's automatic `GITHUB_TOKEN` - no manual secrets needed.

### What Happens Automatically:

#### On Push to main/develop:
1. ✅ Build on Java 11 & 17
2. ✅ Run compilation (tests skipped but framework ready)
3. ✅ Validate all modules

#### On Push to main:
1. ✅ Generate Javadocs
2. ✅ Publish to GitHub Packages
3. ✅ Deploy Javadocs to GitHub Pages

#### On Release Tag (v*):
1. ✅ Create GitHub Release
2. ✅ Attach all JARs (regular + sources + javadoc)
3. ✅ Publish to GitHub Packages

### Manual Setup (Optional - for Maven Central)

If you want to publish to Maven Central later, add these secrets:

```bash
# Sonatype OSSRH
OSSRH_USERNAME=your-sonatype-username
OSSRH_PASSWORD=your-sonatype-token

# GPG Signing
GPG_PRIVATE_KEY=your-gpg-private-key
GPG_PASSPHRASE=your-gpg-passphrase
```

## 🐳 Docker Usage

```bash
# Build the image
docker build -t otter-streams:latest .

# Run with your models
docker run -d \
  -p 8081:8081 \
  -v $(pwd)/models:/app/models \
  otter-streams:latest

# Run the example
docker run -it otter-streams:latest \
  java -jar otter-stream-examples-1.0.0.jar
```

## 📈 Performance Optimization

### 1. Enable Batching
```java
.batchSize(64)                    // Larger batches = better throughput
```

### 2. Use Caching
```java
.enableCaching(true)              // Cache frequent inputs
.cacheSize(10000)                 // Adjust based on unique inputs
```

### 3. Async Processing
```java
// Always use async for non-blocking inference
AsyncDataStream.unorderedWait(
    stream,
    inferenceFunction,
    5000,
    TimeUnit.MILLISECONDS,
    100                          // Max concurrent requests
);
```

### 4. Model Warmup (Automatic)
- Models are warmed up automatically on initialization
- First few inferences might be slower due to JIT compilation

### 5. GPU Acceleration
```java
// ONNX with GPU
.modelOptions(Map.of("use_gpu", true))

// TensorFlow automatically uses GPU if available
```

## 🔍 Monitoring & Metrics

The framework automatically collects comprehensive metrics:

- **Throughput**: `inference_requests_total`
- **Latency**: `inference_duration_ms` (histogram)
- **Success Rate**: `inference_success_total`, `inference_failures_total`
- **Cache Performance**: `cache_hits_total`, `cache_misses_total`
- **Batch Efficiency**: `batch_size` (distribution)

Access via Flink's metric system or integrate with Prometheus/Micrometer.

## 🐛 Troubleshooting

### Common Issues

**Model Loading Problems**
```bash
# Verify model format
file models/your_model.onnx

# Check file permissions
ls -la models/
```

**Build Issues**
```bash
# Clear Maven cache
mvn clean
rm -rf ~/.m2/repository/com/codedstreams

# Rebuild with updates
mvn clean install -U
```

**Dependency Conflicts**
```bash
# Check dependency tree
mvn dependency:tree -Dincludes=com.microsoft.onnxruntime
```

**Memory Issues**
```java
// Increase Flink task manager memory
// in flink-conf.yaml:
taskmanager.memory.process.size: 4096m
```

## 📚 Next Steps

### Immediate (Try Now)
1. Run the fraud detection example
2. Load your own ONNX model
3. Integrate with your Flink streaming job
4. Monitor metrics in Flink Web UI

### Advanced Usage
1. Implement custom feature extraction
2. Add custom metrics collectors
3. Implement additional remote clients (Vertex AI, Azure ML)
4. Add model versioning and A/B testing

## 🎉 Success Checklist

You're ready for production when:

- ✅ Project builds: `mvn clean install`
- ✅ Examples run successfully
- ✅ Your models load without errors
- ✅ Inference results are correct
- ✅ Metrics are visible in Flink UI
- ✅ GitHub Actions pipeline is green

## 🚀 Production Deployment

1. **Update POM URLs**: Replace `YOUR_USERNAME` in all POM files
2. **Push to GitHub**: CI/CD will automatically build and deploy
3. **Create Release**: Tag `v1.0.0` to trigger package publishing
4. **Enable GitHub Pages**: For Javadoc hosting
5. **Use in Projects**: Add dependencies from GitHub Packages

```bash
# Step 3: Create release
git tag -a v1.0.0 -m "Release v1.0.0"
git push origin v1.0.0
```

## 🤝 Support

- 📚 **Documentation**: [GitHub Pages](https://YOUR_USERNAME.github.io/otter-streams/javadoc/)
- 🐛 **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/otter-streams/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/YOUR_USERNAME/otter-streams/discussions)
- 🔧 **Examples**: `otter-stream-examples/` module

---
