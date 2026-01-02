#  Otter-Streams

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

## ✨ What is Otter Streams?

Otter Streams is an open-source library that brings production-grade machine learning inference to Apache Flink streaming applications. Deploy your ML models externally or directly into Flink pipelines with enterprise-grade performance, reliability, and monitoring.

### Why choose Otter Streams?

- **🚀 Real-time ML at scale** - Perform inference on streaming data with millisecond latency
- **🔌 Framework agnostic** - Support for ONNX, TensorFlow, PyTorch, XGBoost, and PMML
- **🏢 Production ready** - Built-in monitoring, caching, and error handling
- **☁️ Deployment flexibility** - Local execution, cloud services, or hybrid deployments
- **📊 Full observability** - Comprehensive metrics and monitoring out of the box

## 🎯 Quick Start

### Add to Your Project

```xml
<dependency>
    <groupId>com.codedstreams</groupId>
    <artifactId>ml-inference-core</artifactId>
    <version>1.0.0</version>
</dependency>
```

### Basic Example

```java
// Add ML inference to your Flink stream in minutes
DataStream<FraudScore> predictions = transactionStream
    .map(new AsyncModelInferenceFunction<>(config));
```

**📖 [See the Getting Started Guide](GETTING_STARTED.md)** for detailed instructions.

## 📚 Documentation

- **[📖 Getting Started](GETTING_STARTED.md)** - Your first inference pipeline
- **[🏗️ Architecture Overview](ARCHITECTURE.md)** - System design and components
- **[🎯 Examples & Use Cases](EXAMPLES.md)** - Real-world implementation patterns
- **[🔧 API Reference](https://martourez21.github.io/otter-streams/javadoc/)** - Complete API documentation
- **[📊 Performance Guide](PERFORMANCE.md)** - Optimization and tuning

## 🌟 Key Features

| Feature | Description |
|---------|-------------|
| **Multi-Framework Support** | Run models from ONNX, TensorFlow, PyTorch, XGBoost, and PMML |
| **Async & High Performance** | Non-blocking execution with intelligent batching |
| **Enterprise Monitoring** | Built-in metrics, logging, and health checks |
| **Flexible Deployment** | Local, cloud, or hybrid inference strategies |
| **Production Resilience** | Retry logic, caching, and error handling |

## 🏢 Use Cases

### Real-time Fraud Detection
```java
// Detect fraudulent transactions as they occur
DataStream<FraudScore> scores = transactionStream
    .process(new FraudDetectionModel());
```

### Personalized Recommendations
```java
// Generate personalized content in real-time
DataStream<Recommendation> recs = userBehaviorStream
    .process(new RecommendationModel());
```

### Anomaly Detection
```java
// Monitor systems and detect anomalies immediately
DataStream<AnomalyScore> anomalies = sensorStream
    .process(new AnomalyDetectionModel());
```

**🔍 [Explore more use cases](EXAMPLES.md)**

## 🛠 Supported Frameworks

<table>
<tr>
<td align="center" width="20%">
<br/>
<strong>ONNX Runtime</strong>
</td>
<td align="center" width="20%">
<br/>
<strong>TensorFlow</strong>
</td>
<td align="center" width="20%">
<br/>
<strong>PyTorch</strong>
</td>
<td align="center" width="20%">
<br/>
<strong>XGBoost</strong>
</td>
<td align="center" width="20%">
<br/>
<strong>PMML</strong>
</td>
</tr>
</table>

**🔗 [See Framework Integration Details](ARCHITECTURE.md)**

## 📦 Project Structure

```
otter-streams/
├── ml-inference-core/          # Core inference engine
├── otter-stream-onnx/         # ONNX Runtime integration
├── otter-stream-tensorflow/   # TensorFlow SavedModel support
├── otter-stream-pytorch/      # PyTorch model inference
├── otter-stream-xgboost/      # XGBoost integration
├── otter-stream-pmml/         # PMML model support
├── otter-stream-remote/       # Remote inference service
├── otter-stream-examples/     # Usage examples
└── docs/                      # Documentation
```

**🏗️ [Learn about the architecture](ARCHITECTURE.md)**

## 🤝 Community & Support

### Get Help
- **📖 [Documentation](https://martourez21.github.io/otter-streams/)** - Complete user guide
- **💬 [GitHub Discussions](https://github.com/martourez21/otter-streams/discussions)** - Questions and ideas
- **🐛 [Issue Tracker](https://github.com/martourez21/otter-streams/issues)** - Bug reports and feature requests
- **📧 [Email Support](mailto:nestorabiawuh@gmail.com)** - Direct contact

### Stay Updated
- ⭐ **Star the repository** to show your support
- 👀 **Watch releases** to get notifications
- 🔄 **Follow updates** on GitHub

## 👥 Contributing

We love our contributors! Whether you're fixing bugs, improving documentation, or adding new features, all contributions are welcome.

**📋 [Read our Contributing Guide](CONTRIBUTING.md)**

### Quick Start for Contributors
```bash
# 1. Fork and clone
git clone https://github.com/your-username/otter-streams.git

# 2. Build the project
mvn clean install

# 3. Run tests
mvn test
```

### Ways to Contribute
- 🐛 **Report bugs** and issues
- 💡 **Suggest features** and improvements
- 📚 **Improve documentation**
- 🔧 **Fix issues** labeled "good first issue"
- 🧪 **Add tests** and examples
- 🌍 **Help others** in discussions

## 📄 License

Otter Streams is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

Built with Passion by [Nestor Martourez](https://github.com/martourez21) and the open-source community. Special thanks to:

- The Apache Flink community
- All our contributors and users
- Open-source ML framework maintainers

---

<div align="center">

**Ready to add ML to your streaming pipelines?**

[📖 Get Started](GETTING_STARTED.md) · [💬 Join Discussions](https://github.com/martourez21/otter-streams/discussions) · [⭐ Star the Project](https://github.com/martourez21/otter-streams/stargazers)

</div>