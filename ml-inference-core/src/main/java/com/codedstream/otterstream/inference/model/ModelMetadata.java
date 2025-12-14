package com.codedstream.otterstream.inference.model;

import com.codedstream.otterstream.inference.config.ModelConfig;

import java.util.Map;
import java.util.HashMap;
import java.util.Objects;

/**
 * Immutable metadata container for machine learning models.
 *
 * <p>This class captures essential information about a loaded ML model including
 * its identity, format, structure, and loading characteristics. Metadata is
 * typically extracted by {@link ModelLoader} implementations after successful
 * model loading.
 *
 * <h2>Key Attributes:</h2>
 * <ul>
 *   <li><b>Model Identity:</b> Name and version for model identification</li>
 *   <li><b>Format:</b> The model format (ONNX, TensorFlow, PyTorch, etc.)</li>
 *   <li><b>Schema:</b> Input and output structure definitions</li>
 *   <li><b>Size:</b> Memory footprint of the loaded model</li>
 *   <li><b>Timing:</b> When the model was loaded into memory</li>
 * </ul>
 *
 * <h2>Builder Pattern:</h2>
 * <pre>{@code
 * ModelMetadata metadata = ModelMetadata.builder()
 *     .modelName("bert-classifier")
 *     .modelVersion("v2.1.0")
 *     .format(ModelFormat.ONNX)
 *     .inputSchema(Map.of("input_ids", "int32[1,512]"))
 *     .outputSchema(Map.of("logits", "float32[1,2]"))
 *     .modelSize(450_000_000L) // 450MB
 *     .build();
 * }</pre>
 *
 * <h2>Thread Safety:</h2>
 * <p>Instances of this class are immutable and therefore thread-safe.
 * All getters return either primitive values or defensive copies of collections.
 *
 * <h2>Usage with ModelLoader:</h2>
 * <pre>{@code
 * ModelLoader<OrtSession> loader = new OnnxModelLoader();
 * OrtSession model = loader.loadModel(config);
 * ModelMetadata metadata = loader.getModelMetadata(model);
 *
 * // Access metadata
 * String name = metadata.getModelName();
 * Map<String, Object> inputs = metadata.getInputSchema();
 * }</pre>
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 * @see ModelLoader
 * @see ModelFormat
 * @see ModelConfig
 */
public class ModelMetadata {
    private final String modelName;
    private final String modelVersion;
    private final ModelFormat format;
    private final Map<String, Object> inputSchema;
    private final Map<String, Object> outputSchema;
    private final long modelSize;
    private final long loadTimestamp;

    /**
     * Constructs a new ModelMetadata instance with the specified attributes.
     *
     * @param modelName the name of the model (required)
     * @param modelVersion the version of the model (required)
     * @param format the format of the model (required)
     * @param inputSchema map describing model inputs (nullable, defaults to empty)
     * @param outputSchema map describing model outputs (nullable, defaults to empty)
     * @param modelSize size of the model in bytes
     * @param loadTimestamp when the model was loaded (epoch milliseconds)
     * @throws NullPointerException if modelName, modelVersion, or format is null
     */
    public ModelMetadata(String modelName, String modelVersion, ModelFormat format,
                         Map<String, Object> inputSchema, Map<String, Object> outputSchema,
                         long modelSize, long loadTimestamp) {
        this.modelName = Objects.requireNonNull(modelName, "modelName cannot be null");
        this.modelVersion = Objects.requireNonNull(modelVersion, "modelVersion cannot be null");
        this.format = Objects.requireNonNull(format, "format cannot be null");
        this.inputSchema = inputSchema != null ? Map.copyOf(inputSchema) : Map.of();
        this.outputSchema = outputSchema != null ? Map.copyOf(outputSchema) : Map.of();
        this.modelSize = modelSize;
        this.loadTimestamp = loadTimestamp;
    }

    /**
     * Gets the name of the model.
     *
     * @return model name, never null
     */
    public String getModelName() { return modelName; }

    /**
     * Gets the version of the model.
     *
     * @return model version, never null
     */
    public String getModelVersion() { return modelVersion; }

    /**
     * Gets the format of the model.
     *
     * @return model format, never null
     */
    public ModelFormat getFormat() { return format; }

    /**
     * Gets the input schema describing model inputs.
     * <p>Returns a defensive copy of the internal map.
     *
     * @return immutable map of input names to their schema definitions
     */
    public Map<String, Object> getInputSchema() { return inputSchema; }

    /**
     * Gets the output schema describing model outputs.
     * <p>Returns a defensive copy of the internal map.
     *
     * @return immutable map of output names to their schema definitions
     */
    public Map<String, Object> getOutputSchema() { return outputSchema; }

    /**
     * Gets the size of the model in bytes.
     *
     * @return model size in bytes
     */
    public long getModelSize() { return modelSize; }

    /**
     * Gets the timestamp when the model was loaded.
     *
     * @return load timestamp in epoch milliseconds
     */
    public long getLoadTimestamp() { return loadTimestamp; }

    /**
     * Creates a new Builder instance for constructing ModelMetadata.
     *
     * @return a new Builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Builder for constructing {@link ModelMetadata} instances.
     *
     * <p>Provides a fluent API for setting metadata attributes with sensible defaults:
     * <ul>
     *   <li>modelVersion: "unknown"</li>
     *   <li>inputSchema: empty map</li>
     *   <li>outputSchema: empty map</li>
     *   <li>modelSize: 0L</li>
     *   <li>loadTimestamp: current system time</li>
     * </ul>
     *
     * <h2>Example:</h2>
     * <pre>{@code
     * ModelMetadata metadata = ModelMetadata.builder()
     *     .modelName("resnet50")
     *     .format(ModelFormat.ONNX)
     *     .modelSize(1024 * 1024 * 100) // 100MB
     *     .build();
     * }</pre>
     */
    public static class Builder {
        private String modelName;
        private String modelVersion = "unknown";
        private ModelFormat format;
        private Map<String, Object> inputSchema = new HashMap<>();
        private Map<String, Object> outputSchema = new HashMap<>();
        private long modelSize = 0L;
        private long loadTimestamp = System.currentTimeMillis();

        /**
         * Sets the model name.
         *
         * @param modelName the model name
         * @return this builder for method chaining
         */
        public Builder modelName(String modelName) {
            this.modelName = modelName;
            return this;
        }

        /**
         * Sets the model version.
         * <p>Defaults to "unknown" if not specified.
         *
         * @param modelVersion the model version
         * @return this builder for method chaining
         */
        public Builder modelVersion(String modelVersion) {
            this.modelVersion = modelVersion;
            return this;
        }

        /**
         * Sets the model format.
         *
         * @param format the model format
         * @return this builder for method chaining
         */
        public Builder format(ModelFormat format) {
            this.format = format;
            return this;
        }

        /**
         * Sets the input schema.
         * <p>Creates a defensive copy of the provided map.
         *
         * @param inputSchema map of input names to schema definitions
         * @return this builder for method chaining
         */
        public Builder inputSchema(Map<String, Object> inputSchema) {
            this.inputSchema = inputSchema != null ? new HashMap<>(inputSchema) : new HashMap<>();
            return this;
        }

        /**
         * Sets the output schema.
         * <p>Creates a defensive copy of the provided map.
         *
         * @param outputSchema map of output names to schema definitions
         * @return this builder for method chaining
         */
        public Builder outputSchema(Map<String, Object> outputSchema) {
            this.outputSchema = outputSchema != null ? new HashMap<>(outputSchema) : new HashMap<>();
            return this;
        }

        /**
         * Sets the model size in bytes.
         * <p>Defaults to 0L if not specified.
         *
         * @param modelSize size in bytes
         * @return this builder for method chaining
         */
        public Builder modelSize(long modelSize) {
            this.modelSize = modelSize;
            return this;
        }

        /**
         * Sets the load timestamp.
         * <p>Defaults to current system time if not specified.
         *
         * @param loadTimestamp timestamp in epoch milliseconds
         * @return this builder for method chaining
         */
        public Builder loadTimestamp(long loadTimestamp) {
            this.loadTimestamp = loadTimestamp;
            return this;
        }

        /**
         * Builds a new ModelMetadata instance.
         *
         * @return a new immutable ModelMetadata instance
         * @throws NullPointerException if modelName or format is null
         */
        public ModelMetadata build() {
            return new ModelMetadata(
                    modelName,
                    modelVersion,
                    format,
                    inputSchema,
                    outputSchema,
                    modelSize,
                    loadTimestamp
            );
        }
    }

    /**
     * Returns a string representation of the model metadata.
     *
     * @return string representation including all metadata fields
     */
    @Override
    public String toString() {
        return "ModelMetadata{" +
                "modelName='" + modelName + '\'' +
                ", modelVersion='" + modelVersion + '\'' +
                ", format=" + format +
                ", inputSchema=" + inputSchema +
                ", outputSchema=" + outputSchema +
                ", modelSize=" + modelSize +
                ", loadTimestamp=" + loadTimestamp +
                '}';
    }

    /**
     * Compares this metadata to another object for equality.
     *
     * @param o the object to compare
     * @return true if all fields are equal
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ModelMetadata that = (ModelMetadata) o;
        return modelSize == that.modelSize &&
                loadTimestamp == that.loadTimestamp &&
                Objects.equals(modelName, that.modelName) &&
                Objects.equals(modelVersion, that.modelVersion) &&
                format == that.format &&
                Objects.equals(inputSchema, that.inputSchema) &&
                Objects.equals(outputSchema, that.outputSchema);
    }

    /**
     * Returns a hash code based on all metadata fields.
     *
     * @return hash code value
     */
    @Override
    public int hashCode() {
        return Objects.hash(modelName, modelVersion, format, inputSchema,
                outputSchema, modelSize, loadTimestamp);
    }
}