#!/bin/bash

# Verification script for Flink ML Inference SDK implementation
# Checks that all required files are in place

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

MISSING_FILES=0
TOTAL_FILES=0

check_file() {
    TOTAL_FILES=$((TOTAL_FILES + 1))
    if [ -f "$1" ]; then
        echo -e "${GREEN}✓${NC} $1"
    else
        echo -e "${RED}✗${NC} $1 ${YELLOW}(MISSING)${NC}"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo -e "${GREEN}✓${NC} Directory: $1"
    else
        echo -e "${YELLOW}!${NC} Directory: $1 ${YELLOW}(MISSING - will be created)${NC}"
        mkdir -p "$1"
    fi
}

echo "================================================"
echo "Flink ML Inference SDK - Implementation Check"
echo "================================================"
echo ""

echo "📁 Checking Root Files..."
check_file "pom.xml"
check_file "README.md"
check_file "BUILD.md"
check_file "GETTING_STARTED.md"
check_file "IMPLEMENTATION_CHECKLIST.md"
check_file "COMPLETE_FILE_LIST.md"
check_file "Dockerfile"
check_file "setup.sh"
check_file ".github/workflows/ci.yml"

echo ""
echo "📦 Checking Core Module..."
check_dir "flink-ml-inference-core/src/main/java/com/flinkml/inference"
check_file "flink-ml-inference-core/pom.xml"
check_file "flink-ml-inference-core/src/main/java/com/flinkml/inference/model/ModelFormat.java"
check_file "flink-ml-inference-core/src/main/java/com/flinkml/inference/model/InferenceResult.java"
check_file "flink-ml-inference-core/src/main/java/com/flinkml/inference/model/ModelLoader.java"
check_file "flink-ml-inference-core/src/main/java/com/flinkml/inference/config/InferenceConfig.java"
check_file "flink-ml-inference-core/src/main/java/com/flinkml/inference/engine/InferenceEngine.java"
check_file "flink-ml-inference-core/src/main/java/com/flinkml/inference/function/AsyncModelInferenceFunction.java"
check_file "flink-ml-inference-core/src/main/java/com/flinkml/inference/cache/ModelCache.java"
check_file "flink-ml-inference-core/src/main/java/com/flinkml/inference/metrics/InferenceMetrics.java"
check_file "flink-ml-inference-core/src/main/java/com/flinkml/inference/exception/InferenceException.java"

echo ""
echo "🔷 Checking ONNX Module..."
check_dir "flink-ml-inference-onnx/src/main/java/com/flinkml/inference/onnx"
check_file "flink-ml-inference-onnx/pom.xml"
check_file "flink-ml-inference-onnx/src/main/java/com/flinkml/inference/onnx/OnnxInferenceEngine.java"

echo ""
echo "🔶 Checking TensorFlow Module..."
check_dir "flink-ml-inference-tensorflow/src/main/java/com/flinkml/inference/tensorflow"
check_file "flink-ml-inference-tensorflow/pom.xml"
check_file "flink-ml-inference-tensorflow/src/main/java/com/flinkml/inference/tensorflow/TensorFlowInferenceEngine.java"
check_file "flink-ml-inference-tensorflow/src/main/java/com/flinkml/inference/tensorflow/TensorFlowModelLoader.java"

echo ""
echo "🔥 Checking PyTorch Module..."
check_dir "flink-ml-inference-pytorch/src/main/java/com/flinkml/inference/pytorch"
check_file "flink-ml-inference-pytorch/pom.xml"
check_file "flink-ml-inference-pytorch/src/main/java/com/flinkml/inference/pytorch/TorchScriptInferenceEngine.java"
check_file "flink-ml-inference-pytorch/src/main/java/com/flinkml/inference/pytorch/TorchScriptModelLoader.java"

echo ""
echo "🌲 Checking XGBoost Module..."
check_dir "flink-ml-inference-xgboost/src/main/java/com/flinkml/inference/xgboost"
check_file "flink-ml-inference-xgboost/pom.xml"
check_file "flink-ml-inference-xgboost/src/main/java/com/flinkml/inference/xgboost/XGBoostInferenceEngine.java"
check_file "flink-ml-inference-xgboost/src/main/java/com/flinkml/inference/xgboost/XGBoostModelLoader.java"

echo ""
echo "📊 Checking PMML Module..."
check_dir "flink-ml-inference-pmml/src/main/java/com/flinkml/inference/pmml"
check_file "flink-ml-inference-pmml/pom.xml"
check_file "flink-ml-inference-pmml/src/main/java/com/flinkml/inference/pmml/PmmlInferenceEngine.java"
check_file "flink-ml-inference-pmml/src/main/java/com/flinkml/inference/pmml/PmmlModelLoader.java"

echo ""
echo "🌐 Checking Remote Module..."
check_dir "flink-ml-inference-remote/src/main/java/com/flinkml/inference/remote"
check_file "flink-ml-inference-remote/pom.xml"
check_file "flink-ml-inference-remote/src/main/java/com/flinkml/inference/remote/http/HttpRemoteInferenceEngine.java"
check_file "flink-ml-inference-remote/src/main/java/com/flinkml/inference/remote/grpc/GrpcInferenceClient.java"
check_file "flink-ml-inference-remote/src/main/java/com/flinkml/inference/remote/sagemaker/SageMakerInferenceClient.java"

echo ""
echo "💡 Checking Examples Module..."
check_dir "flink-ml-inference-examples/src/main/java/com/flinkml/inference/examples"
check_file "flink-ml-inference-examples/pom.xml"
check_file "flink-ml-inference-examples/src/main/java/com/flinkml/inference/examples/fraud/FraudDetectionExample.java"

echo ""
echo "================================================"
echo "📊 Summary"
echo "================================================"
echo -e "Total files checked: ${TOTAL_FILES}"

if [ $MISSING_FILES -eq 0 ]; then
    echo -e "${GREEN}✅ All files present! Implementation is complete.${NC}"
    echo ""
    echo "🚀 Next steps:"
    echo "   1. Copy all file contents to respective locations"
    echo "   2. Run: mvn clean install"
    echo "   3. Check: mvn verify"
    echo "   4. Deploy: git tag v1.0.0 && git push origin v1.0.0"
    exit 0
else
    echo -e "${RED}❌ Missing ${MISSING_FILES} file(s)${NC}"
    echo ""
    echo "⚠️  Action required:"
    echo "   1. Create missing files using the provided artifacts"
    echo "   2. Copy content from code artifacts above"
    echo "   3. Re-run this script to verify"
    exit 1
fi