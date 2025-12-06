package com.codedstream.otterstream.inference.model;

/**
 * Supported model formats in the inference framework.
 */
public enum ModelFormat {
    ONNX("onnx", "Open Neural Network Exchange"),
    TENSORFLOW_SAVEDMODEL("savedmodel", "TensorFlow SavedModel"),
    TENSORFLOW_GRAPHDEF("pb", "TensorFlow GraphDef"),
    PYTORCH_TORCHSCRIPT("pt", "PyTorch TorchScript"),
    XGBOOST_BINARY("bin", "XGBoost Binary"),
    PMML("pmml", "Predictive Model Markup Language"),
    REMOTE_HTTP("http", "Remote HTTP Endpoint"),
    REMOTE_GRPC("grpc", "Remote gRPC Endpoint"),
    SAGEMAKER("sagemaker", "AWS SageMaker"),
    VERTEX_AI("vertexai", "Google Vertex AI"),
    AZURE_ML("azureml", "Azure Machine Learning");

    private final String extension;
    private final String description;

    ModelFormat(String extension, String description) {
        this.extension = extension;
        this.description = description;
    }

    public String getExtension() { return extension; }
    public String getDescription() { return description; }
}