#  Getting Started with Otter Streams

Welcome to Otter Streams! This guide will help you set up and run your first machine learning inference pipeline with Apache Flink.

## 📋 Prerequisites

### Required Software
- **Java**: JDK 11 or later
- **Apache Flink**: Version 1.17 or later
- **Maven**: Version 3.6 or later

### Optional Dependencies
- **Docker**: For running example projects and integration tests
- **Python 3.8+**: For training example models (optional)

## 📦 Installation

### 1. Add Dependencies to Your Project

Add the core dependency to your `pom.xml`:

```xml
<dependency>
    <groupId>com.codedstreams</groupId>
    <artifactId>ml-inference-core</artifactId>
    <version>1.0.0</version>
</dependency>
```

### 2. Add Framework-Specific Modules

Choose based on your ML framework:

```xml
<!-- ONNX Runtime -->
<dependency>
    <groupId>com.codedstreams</groupId>
    <artifactId>otter-stream-onnx</artifactId>
    <version>1.0.0</version>
</dependency>

<!-- TensorFlow -->
<dependency>
    <groupId>com.codedstreams</groupId>
    <artifactId>otter-stream-tensorflow</artifactId>
    <version>1.0.0</version>
</dependency>

<!-- PyTorch -->
<dependency>
    <groupId>com.codedstreams</groupId>
    <artifactId>otter-stream-pytorch</artifactId>
    <version>1.0.0</version>
</dependency>

<!-- XGBoost -->
<dependency>
    <groupId>com.codedstreams</groupId>
    <artifactId>otter-stream-xgboost</artifactId>
    <version>1.0.0</version>
</dependency>
```

### 3. Verify Installation

```bash
mvn clean compile
```

## 🎯 Your First Inference Pipeline

### Example 1: Simple ONNX Model Inference

```java
import com.codedstreams.ml.inference.*;
import com.codedstreams.ml.inference.config.*;
import org.apache.flink.streaming.api.datastream.*;
import org.apache.flink.streaming.api.environment.*;

public class FirstInferencePipeline {
    public static void main(String[] args) throws Exception {
        // 1. Set up Flink environment
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        
        // 2. Configure model inference
        InferenceConfig config = InferenceConfig.builder()
            .modelConfig(ModelConfig.builder()
                .modelId("sentiment-model")
                .modelPath("/models/sentiment.onnx")
                .format(ModelFormat.ONNX)
                .modelName("sentiment_predictor")
                .build())
            .batchSize(32)
            .timeout(Duration.ofSeconds(5))
            .enableMetrics(true)
            .build();
        
        // 3. Create inference function
        AsyncModelInferenceFunction<TextInput, SentimentScore> inferenceFunction =
            new AsyncModelInferenceFunction<>(
                config,
                cfg -> new OnnxInferenceEngine()
            );
        
        // 4. Set up data stream (example data)
        DataStream<TextInput> textStream = env
            .fromElements(
                new TextInput("I love this product!"),
                new TextInput("This is terrible."),
                new TextInput("It works perfectly.")
            );
        
        // 5. Apply inference
        DataStream<SentimentScore> predictions = AsyncDataStream.unorderedWait(
            textStream,
            inferenceFunction,
            5000,
            TimeUnit.MILLISECONDS,
            100
        );
        
        // 6. Print results
        predictions.print();
        
        // 7. Execute pipeline
        env.execute("First Inference Pipeline");
    }
}
```

### Example 2: Fraud Detection with Real Data

```java
public class FraudDetectionPipeline {
    public static void main(String[] args) throws Exception {
        StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
        
        // Read from Kafka (example)
        Properties properties = new Properties();
        properties.setProperty("bootstrap.servers", "localhost:9092");
        properties.setProperty("group.id", "fraud-detection");
        
        DataStream<Transaction> transactionStream = env
            .addSource(new FlinkKafkaConsumer<>(
                "transactions",
                new TransactionDeserializer(),
                properties
            ));
        
        // Configure fraud detection model
        InferenceConfig fraudConfig = InferenceConfig.builder()
            .modelConfig(ModelConfig.builder()
                .modelId("fraud-detection")
                .modelPath("models/fraud_model.onnx")
                .format(ModelFormat.ONNX)
                .inputNames(new String[]{"features"})
                .outputNames(new String[]{"fraud_probability"})
                .build())
            .batchSize(64)
            .enableCaching(true)
            .cacheSize(10000)
            .build();
        
        // Create fraud detection function
        AsyncModelInferenceFunction<Transaction, FraudScore> fraudDetection =
            new AsyncModelInferenceFunction<>(
                fraudConfig,
                cfg -> new OnnxInferenceEngine(),
                transaction -> extractFeatures(transaction),  // Feature extraction
                output -> parseFraudScore(output)            // Result parsing
            );
        
        // Apply fraud detection
        DataStream<FraudScore> fraudScores = AsyncDataStream.unorderedWait(
            transactionStream,
            fraudDetection,
            10000,
            TimeUnit.MILLISECONDS,
            200
        );
        
        // Alert on high fraud probability
        DataStream<Alert> alerts = fraudScores
            .filter(score -> score.getProbability() > 0.9)
            .map(score -> new Alert(
                "HIGH_FRAUD_RISK",
                score.getTransactionId(),
                score.getProbability()
            ));
        
        // Send alerts to output
        alerts.addSink(new AlertSink());
        
        env.execute("Real-time Fraud Detection");
    }
    
    private static float[] extractFeatures(Transaction transaction) {
        // Extract features from transaction
        return new float[]{
            transaction.getAmount(),
            transaction.getHourOfDay(),
            transaction.getLocationDistance(),
            // ... more features
        };
    }
    
    private static FraudScore parseFraudScore(InferenceOutput output) {
        float probability = output.getOutput("fraud_probability")[0];
        return new FraudScore(probability);
    }
}
```

## 📁 Project Structure

Set up your project like this:

```
my-flink-ml-project/
├── src/
│   ├── main/
│   │   ├── java/
│   │   │   └── com/
│   │   │       └── mycompany/
│   │   │           └── mlpipeline/
│   │   │               ├── MainPipeline.java
│   │   │               ├── models/
│   │   │               └── utils/
│   │   └── resources/
│   │       └── models/           # Your ML models here
│   │           ├── sentiment.onnx
│   │           └── fraud_model.onnx
│   └── test/
│       └── java/
├── pom.xml
└── README.md
```

## 🔧 Configuration Examples

### Basic Configuration

```java
InferenceConfig.builder()
    .modelConfig(ModelConfig.builder()
        .modelId("my-model")
        .modelPath("models/my_model.onnx")
        .format(ModelFormat.ONNX)
        .modelVersion("1.0.0")
        .build())
    .batchSize(32)
    .timeout(Duration.ofSeconds(10))
    .maxRetries(3)
    .build();
```

### Advanced Configuration with Optimization

```java
InferenceConfig.builder()
    .modelConfig(ModelConfig.builder()
        .modelId("optimized-model")
        .modelPath("models/optimized.onnx")
        .format(ModelFormat.ONNX)
        .modelOptions(Map.of(
            "interOpThreads", "4",
            "intraOpThreads", "2",
            "executionMode", "SEQUENTIAL",
            "optimizationLevel", "ALL"
        ))
        .build())
    .batchSize(128)
    .batchTimeout(Duration.ofMillis(50))
    .enableCaching(true)
    .cacheSize(50000)
    .cacheTtl(Duration.ofMinutes(15))
    .enableMetrics(true)
    .metricsPrefix("app.ml.inference")
    .parallelism(4)
    .queueSize(1000)
    .build();
```

## 🧪 Testing Your Setup

Create a simple test to verify everything works:

```java
@Test
public void testBasicInference() {
    // Create test configuration
    InferenceConfig config = InferenceConfig.builder()
        .modelConfig(ModelConfig.builder()
            .modelId("test-model")
            .modelPath("src/test/resources/test_model.onnx")
            .format(ModelFormat.ONNX)
            .build())
        .build();
    
    // Create inference engine
    InferenceEngine engine = new OnnxInferenceEngine();
    engine.initialize(config);
    
    // Create test input
    float[][] input = {{1.0f, 2.0f, 3.0f}};
    
    // Execute inference
    InferenceOutput output = engine.execute(input);
    
    // Verify output
    assertNotNull(output);
    assertTrue(output.getOutput().length > 0);
}
```

## 🚨 Common Issues & Solutions

### Issue 1: Model Not Found
**Error**: `Model file not found: /models/my_model.onnx`
**Solution**: Ensure the model path is correct and the file exists. Use relative paths:

```java
// Use classpath or absolute paths
.modelPath("classpath:/models/my_model.onnx")
// or
.modelPath(new File("src/main/resources/models/my_model.onnx").getAbsolutePath())
```

### Issue 2: Memory Issues
**Error**: `OutOfMemoryError`
**Solution**: Increase Flink memory and configure batching:

```java
// In Flink config
env.getConfig().setTaskManagerMemoryMB(4096);

// In inference config
InferenceConfig.builder()
    .batchSize(16)  // Reduce batch size
    .queueSize(500) // Reduce queue size
    .build();
```

### Issue 3: Slow Performance
**Solution**: Enable caching and optimize configuration:

```java
InferenceConfig.builder()
    .enableCaching(true)
    .cacheSize(10000)
    .batchSize(64)  // Optimal for most models
    .parallelism(2) // Increase parallelism
    .build();
```

## 📚 Next Steps

1. **Explore Examples**: Check out the `otter-stream-examples` module
2. **Read Documentation**: Visit [martourez21.github.io/otter-streams](https://martourez21.github.io/otter-streams/)
3. **Try Different Models**: Experiment with TensorFlow, PyTorch, or XGBoost models
4. **Monitor Performance**: Enable metrics and monitor your inference pipeline

## 🆘 Need Help?

- Check the [GitHub Issues](https://github.com/martourez21/otter-streams/issues)
- Join [GitHub Discussions](https://github.com/martourez21/otter-streams/discussions)
- Email: nestorabiawuh@gmail.com

---

**Ready for more?** Check out the [Architecture Guide](ARCHITECTURE.md) to understand how Otter Streams works internally.
