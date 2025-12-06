# Multi-stage build for Flink ML Inference SDK

# Stage 1: Build
FROM maven:3.9-eclipse-temurin-11 AS builder

WORKDIR /build

# Copy pom files first for better layer caching
COPY pom.xml .
COPY flink-ml-inference-core/pom.xml flink-ml-inference-core/
COPY flink-ml-inference-onnx/pom.xml flink-ml-inference-onnx/
COPY flink-ml-inference-tensorflow/pom.xml flink-ml-inference-tensorflow/
COPY flink-ml-inference-pytorch/pom.xml flink-ml-inference-pytorch/
COPY flink-ml-inference-xgboost/pom.xml flink-ml-inference-xgboost/
COPY flink-ml-inference-pmml/pom.xml flink-ml-inference-pmml/
COPY flink-ml-inference-remote/pom.xml flink-ml-inference-remote/
COPY flink-ml-inference-examples/pom.xml flink-ml-inference-examples/

# Download dependencies (cached layer)
RUN mvn dependency:go-offline -B

# Copy source code
COPY flink-ml-inference-core/src flink-ml-inference-core/src
COPY flink-ml-inference-onnx/src flink-ml-inference-onnx/src
COPY flink-ml-inference-tensorflow/src flink-ml-inference-tensorflow/src
COPY flink-ml-inference-pytorch/src flink-ml-inference-pytorch/src
COPY flink-ml-inference-xgboost/src flink-ml-inference-xgboost/src
COPY flink-ml-inference-pmml/src flink-ml-inference-pmml/src
COPY flink-ml-inference-remote/src flink-ml-inference-remote/src
COPY flink-ml-inference-examples/src flink-ml-inference-examples/src

# Build the project
RUN mvn clean package -DskipTests -B

# Stage 2: Runtime
FROM flink:1.18-java11

# Install additional tools
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create directories
RUN mkdir -p /opt/flink-ml-inference/lib \
             /opt/flink-ml-inference/models \
             /opt/flink-ml-inference/config

# Copy built JARs from builder stage
COPY --from=builder /build/flink-ml-inference-core/target/*.jar /opt/flink-ml-inference/lib/
COPY --from=builder /build/flink-ml-inference-onnx/target/*.jar /opt/flink-ml-inference/lib/
COPY --from=builder /build/flink-ml-inference-remote/target/*.jar /opt/flink-ml-inference/lib/

# Copy to Flink lib directory
RUN cp /opt/flink-ml-inference/lib/*.jar /opt/flink/lib/

# Set environment variables
ENV FLINK_ML_INFERENCE_HOME=/opt/flink-ml-inference
ENV MODELS_PATH=/opt/flink-ml-inference/models

# Create non-root user
RUN groupadd -r flink-ml && useradd -r -g flink-ml flink-ml
RUN chown -R flink-ml:flink-ml /opt/flink-ml-inference

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD curl -f http://localhost:8081/ || exit 1

# Switch to non-root user
USER flink-ml

# Expose Flink ports
EXPOSE 8081 6123

# Labels
LABEL maintainer="your-email@example.com"
LABEL version="1.0.0"
LABEL description="Flink ML Inference SDK Runtime"

# Default command
CMD ["/docker-entrypoint.sh", "taskmanager"]