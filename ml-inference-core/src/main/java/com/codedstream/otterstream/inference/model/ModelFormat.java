package com.codedstream.otterstream.inference.model;

import com.codedstream.otterstream.inference.config.ModelConfig;

/**
 * Enumeration of supported ML model formats in Otter Stream.
 *
 * <p>Defines all model formats that can be loaded and executed by the framework.
 * Each format corresponds to a specific inference engine implementation.
 *
 * <h2>Local Model Formats:</h2>
 * <ul>
 *   <li><b>ONNX:</b> Cross-platform neural network format</li>
 *   <li><b>TENSORFLOW_SAVEDMODEL:</b> TensorFlow 2.x SavedModel format</li>
 *   <li><b>TENSORFLOW_GRAPHDEF:</b> TensorFlow 1.x frozen graph (.pb)</li>
 *   <li><b>PYTORCH_TORCHSCRIPT:</b> PyTorch TorchScript models</li>
 *   <li><b>XGBOOST_BINARY:</b> XGBoost binary model files</li>
 *   <li><b>PMML:</b> Predictive Model Markup Language</li>
 * </ul>
 *
 * <h2>Remote Inference Formats:</h2>
 * <ul>
 *   <li><b>REMOTE_HTTP:</b> Generic REST API endpoints</li>
 *   <li><b>REMOTE_GRPC:</b> gRPC endpoints</li>
 *   <li><b>SAGEMAKER:</b> AWS SageMaker endpoints</li>
 *   <li><b>VERTEX_AI:</b> Google Cloud Vertex AI</li>
 *   <li><b>AZURE_ML:</b> Azure Machine Learning</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * ModelFormat format = ModelFormat.ONNX;
 * System.out.println(format.getDescription());  // "Open Neural Network Exchange"
 * System.out.println(format.getExtension());    // "onnx"
 * }</pre>
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 * @see ModelConfig
 */
public enum ModelFormat {
    /** Open Neural Network Exchange format - cross-platform standard */
    ONNX("onnx", "Open Neural Network Exchange"),

    /** TensorFlow SavedModel format - TensorFlow 2.x default */
    TENSORFLOW_SAVEDMODEL("savedmodel", "TensorFlow SavedModel"),

    /** TensorFlow GraphDef format - TensorFlow 1.x frozen graphs */
    TENSORFLOW_GRAPHDEF("pb", "TensorFlow GraphDef"),

    /** PyTorch TorchScript format - serialized PyTorch models */
    PYTORCH_TORCHSCRIPT("pt", "PyTorch TorchScript"),

    /** XGBoost binary format - gradient boosting models */
    XGBOOST_BINARY("bin", "XGBoost Binary"),

    XGBOOST_JSON("json", "XGBoost JSON"),


    /** Predictive Model Markup Language - XML-based standard */
    PMML("pmml", "Predictive Model Markup Language"),

    /** Remote HTTP/REST endpoint */
    REMOTE_HTTP("http", "Remote HTTP Endpoint"),

    /** Remote gRPC endpoint */
    REMOTE_GRPC("grpc", "Remote gRPC Endpoint"),

    /** AWS SageMaker hosting service */
    SAGEMAKER("sagemaker", "AWS SageMaker"),

    /** Google Cloud Vertex AI service */
    VERTEX_AI("vertexai", "Google Vertex AI"),

    /** Azure Machine Learning service */
    AZURE_ML("azureml", "Azure Machine Learning");

    private final String extension;
    private final String description;

    /**
     * Constructs a model format.
     *
     * @param extension file extension for this format
     * @param description human-readable description
     */
    ModelFormat(String extension, String description) {
        this.extension = extension;
        this.description = description;
    }

    /**
     * Gets the file extension for this model format.
     *
     * @return file extension (without dot)
     */
    public String getExtension() { return extension; }

    /**
     * Gets the human-readable description.
     *
     * @return format description
     */
    public String getDescription() { return description; }
}
