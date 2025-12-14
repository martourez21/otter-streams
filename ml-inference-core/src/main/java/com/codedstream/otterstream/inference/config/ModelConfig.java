package com.codedstream.otterstream.inference.config;

import com.codedstream.otterstream.inference.model.ModelFormat;
import java.util.Map;
import java.util.Objects;

/**
 * Configuration for ML models in Otter Stream inference framework.
 *
 * <p>Defines how to locate, load, and identify a model. Supports both local
 * models (file system, HDFS, S3) and remote models (REST endpoints, cloud services).
 *
 * <h2>Model Sources:</h2>
 * <ul>
 *   <li><b>Local:</b> file://, hdfs://, s3://</li>
 *   <li><b>Remote:</b> HTTP/HTTPS endpoints</li>
 *   <li><b>Cloud:</b> AWS SageMaker, Google Vertex AI, Azure ML</li>
 * </ul>
 *
 * <h2>Usage Examples:</h2>
 *
 * <h3>Local TensorFlow Model:</h3>
 * <pre>{@code
 * ModelConfig config = ModelConfig.builder()
 *     .modelId("fraud-detector")
 *     .modelPath("file:///models/fraud_detection")
 *     .format(ModelFormat.TENSORFLOW_SAVEDMODEL)
 *     .modelVersion("2.0")
 *     .build();
 * }</pre>
 *
 * <h3>Remote SageMaker Endpoint:</h3>
 * <pre>{@code
 * AuthConfig auth = AuthConfig.builder()
 *     .apiKey("AWS_ACCESS_KEY")
 *     .build();
 *
 * ModelConfig config = ModelConfig.builder()
 *     .modelId("recommendation-engine")
 *     .format(ModelFormat.SAGEMAKER)
 *     .endpointUrl("https://runtime.sagemaker.us-east-1.amazonaws.com/...")
 *     .authConfig(auth)
 *     .build();
 * }</pre>
 *
 * <h3>ONNX Model with Options:</h3>
 * <pre>{@code
 * ModelConfig config = ModelConfig.builder()
 *     .modelId("image-classifier")
 *     .modelPath("s3://my-bucket/models/resnet50.onnx")
 *     .format(ModelFormat.ONNX)
 *     .modelOptions(Map.of(
 *         "providers", List.of("CUDAExecutionProvider"),  // Use GPU
 *         "intra_op_num_threads", 4
 *     ))
 *     .build();
 * }</pre>
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 * @see ModelFormat
 * @see AuthConfig
 */
public class ModelConfig {
    private final String modelId;
    private final String modelPath;
    private final ModelFormat format;
    private final String modelName;
    private final String modelVersion;
    private final Map<String, Object> modelOptions;
    private final String endpointUrl;
    private final AuthConfig authConfig;

    /**
     * Constructs model configuration.
     *
     * @param modelId unique identifier for this model
     * @param modelPath file system or URL path to model
     * @param format model format (TensorFlow, ONNX, PyTorch, etc.)
     * @param modelName human-readable model name
     * @param modelVersion version string
     * @param modelOptions engine-specific model options
     * @param endpointUrl remote endpoint URL (for remote models)
     * @param authConfig authentication configuration (for remote models)
     */
    public ModelConfig(String modelId, String modelPath, ModelFormat format,
                       String modelName, String modelVersion, Map<String, Object> modelOptions,
                       String endpointUrl, AuthConfig authConfig) {
        this.modelId = Objects.requireNonNull(modelId);
        this.modelPath = modelPath;
        this.format = Objects.requireNonNull(format);
        this.modelName = modelName;
        this.modelVersion = modelVersion;
        this.modelOptions = Map.copyOf(modelOptions);
        this.endpointUrl = endpointUrl;
        this.authConfig = authConfig;
    }

    /**
     * Creates a new builder for ModelConfig.
     *
     * @return a new builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    public String getModelId() { return modelId; }
    public String getModelPath() { return modelPath; }
    public ModelFormat getFormat() { return format; }
    public String getModelName() { return modelName; }
    public String getModelVersion() { return modelVersion; }
    public Map<String, Object> getModelOptions() { return modelOptions; }
    public String getEndpointUrl() { return endpointUrl; }
    public AuthConfig getAuthConfig() { return authConfig; }

    /**
     * Checks if this is a remote model configuration.
     *
     * @return true if model is accessed via remote endpoint
     */
    public boolean isRemote() { return endpointUrl != null && !endpointUrl.isEmpty(); }

    /**
     * Builder for creating ModelConfig instances.
     */
    public static class Builder {
        private String modelId;
        private String modelPath;
        private ModelFormat format;
        private String modelName = "default";
        private String modelVersion = "1.0";
        private Map<String, Object> modelOptions = Map.of();
        private String endpointUrl;
        private AuthConfig authConfig;

        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        public Builder modelPath(String modelPath) {
            this.modelPath = modelPath;
            return this;
        }

        public Builder format(ModelFormat format) {
            this.format = format;
            return this;
        }

        public Builder modelName(String modelName) {
            this.modelName = modelName;
            return this;
        }

        public Builder modelVersion(String modelVersion) {
            this.modelVersion = modelVersion;
            return this;
        }

        public Builder modelOptions(Map<String, Object> modelOptions) {
            this.modelOptions = Map.copyOf(modelOptions);
            return this;
        }

        public Builder endpointUrl(String endpointUrl) {
            this.endpointUrl = endpointUrl;
            return this;
        }

        public Builder authConfig(AuthConfig authConfig) {
            this.authConfig = authConfig;
            return this;
        }

        /**
         * Builds the ModelConfig instance.
         *
         * @return configured ModelConfig
         * @throws NullPointerException if modelId or format is not set
         */
        public ModelConfig build() {
            return new ModelConfig(modelId, modelPath, format, modelName, modelVersion,
                    modelOptions, endpointUrl, authConfig);
        }
    }
}