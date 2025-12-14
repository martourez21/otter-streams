package com.codedstream.otterstream.onnx;

import ai.onnxruntime.*;
import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.exception.ModelLoadException;
import com.codedstream.otterstream.inference.model.ModelFormat;
import com.codedstream.otterstream.inference.model.ModelLoader;
import com.codedstream.otterstream.inference.model.ModelMetadata;
import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.TensorInfo;

import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

/**
 * {@link ModelLoader} implementation for loading ONNX models.
 *
 * <p>This loader handles the specifics of loading ONNX models from various sources
 * (file system, input streams) and configuring ONNX Runtime sessions with optimized
 * settings. It extracts model metadata and validates loaded models.
 *
 * <h2>Loading Sources:</h2>
 * <ul>
 *   <li><b>File Path:</b> Load models from local filesystem or network paths</li>
 *   <li><b>InputStream:</b> Load models from memory streams or network streams</li>
 * </ul>
 *
 * <h2>Session Configuration:</h2>
 * <p>Supports configuration through {@link ModelConfig} model options:
 * <ul>
 *   <li><b>interOpThreads:</b> Number of threads for inter-operation parallelism</li>
 *   <li><b>intraOpThreads:</b> Number of threads for intra-operation parallelism</li>
 *   <li><b>optimizationLevel:</b> "disable", "basic", "extended", or "all"</li>
 *   <li><b>useGpu:</b> Boolean flag to enable GPU execution (CUDA)</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * OnnxModelLoader loader = new OnnxModelLoader();
 *
 * // Configure with options
 * ModelConfig config = ModelConfig.builder()
 *     .modelPath("model.onnx")
 *     .modelOption("interOpThreads", 2)
 *     .modelOption("intraOpThreads", 4)
 *     .modelOption("optimizationLevel", "all")
 *     .modelOption("useGpu", true)
 *     .build();
 *
 * // Load model
 * InferenceSession session = loader.loadModel(config);
 *
 * // Validate and get metadata
 * if (loader.validateModel(session, config)) {
 *     ModelMetadata metadata = loader.getModelMetadata(session);
 * }
 * }</pre>
 *
 * <h2>Supported Formats:</h2>
 * <p>This loader only supports {@link ModelFormat#ONNX} format.
 *
 * <h2>Model Validation:</h2>
 * <p>Validates that loaded models have at least one input and one output.
 * More comprehensive validation can be added by extending this class.
 *
 * <h2>Error Handling:</h2>
 * <p>All loading failures throw {@link ModelLoadException} with detailed error
 * messages and root causes.
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 * @see ModelLoader
 * @see InferenceSession
 * @see OnnxInferenceEngine
 */
public class OnnxModelLoader implements ModelLoader<InferenceSession> {

    /**
     * Loads an ONNX model from the path specified in configuration.
     *
     * @param config model configuration containing path and options
     * @return loaded {@link InferenceSession} instance
     * @throws ModelLoadException if loading fails
     */
    @Override
    public InferenceSession loadModel(ModelConfig config) throws ModelLoadException {
        try {
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions opts = new OrtSession.SessionOptions();

            configureSessionOptions(opts, config);

            return new InferenceSession(config.getModelPath(), opts, env);
        } catch (Exception e) {
            throw new ModelLoadException("Failed to load ONNX model from path: " + config.getModelPath(), e);
        }
    }

    /**
     * Loads an ONNX model from an input stream.
     * <p>Useful for loading models from memory or network sources.
     *
     * @param inputStream stream containing ONNX model data
     * @param config model configuration
     * @return loaded {@link InferenceSession} instance
     * @throws ModelLoadException if loading fails
     */
    @Override
    public InferenceSession loadModel(InputStream inputStream, ModelConfig config) throws ModelLoadException {
        try {
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions opts = new OrtSession.SessionOptions();

            configureSessionOptions(opts, config);

            byte[] bytes = inputStream.readAllBytes();
            return new InferenceSession(bytes, opts, env);
        } catch (Exception e) {
            throw new ModelLoadException("Failed to load ONNX model from stream", e);
        }
    }

    /**
     * Validates that a loaded ONNX model matches the configuration.
     * <p>Basic validation checks that the model has at least one input and one output.
     *
     * @param model the loaded {@link InferenceSession}
     * @param config model configuration (not used in basic validation)
     * @return true if model has both inputs and outputs
     */
    @Override
    public boolean validateModel(InferenceSession model, ModelConfig config) {
        try {
            return !model.getInputMetadata().isEmpty() && !model.getOutputMetadata().isEmpty();
        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Gets the model formats supported by this loader.
     *
     * @return array containing only {@link ModelFormat#ONNX}
     */
    @Override
    public ModelFormat[] getSupportedFormats() {
        return new ModelFormat[]{ModelFormat.ONNX};
    }

    /**
     * Extracts metadata from a loaded ONNX model.
     * <p>Extracts input and output schemas from model metadata. If extraction fails,
     * returns minimal metadata with empty schemas.
     *
     * @param model the loaded {@link InferenceSession}
     * @return model metadata including input/output schemas
     */
    @Override
    public ModelMetadata getModelMetadata(InferenceSession model) {
        try {
            Map<String, Object> inputSchema = extractSchema(model.getInputMetadata());
            Map<String, Object> outputSchema = extractSchema(model.getOutputMetadata());

            return new ModelMetadata(
                    "onnx_model",
                    "1.0",
                    ModelFormat.ONNX,
                    inputSchema,
                    outputSchema,
                    0,
                    System.currentTimeMillis()
            );

        } catch (Exception e) {
            return new ModelMetadata(
                    "onnx_model",
                    "1.0",
                    ModelFormat.ONNX,
                    Map.of(),
                    Map.of(),
                    0,
                    System.currentTimeMillis()
            );
        }
    }

    /**
     * Configures ONNX Runtime session options based on model configuration.
     *
     * @param opts session options to configure
     * @param config model configuration with options
     * @throws OrtException if configuration fails
     * @throws IllegalArgumentException for invalid option values
     */
    private void configureSessionOptions(OrtSession.SessionOptions opts, ModelConfig config) throws OrtException {
        Map<String, Object> modelOptions = config.getModelOptions();

        if (modelOptions.containsKey("interOpThreads")) {
            opts.setInterOpNumThreads((Integer) modelOptions.get("interOpThreads"));
        }

        if (modelOptions.containsKey("intraOpThreads")) {
            opts.setIntraOpNumThreads((Integer) modelOptions.get("intraOpThreads"));
        }

        if (modelOptions.containsKey("optimizationLevel")) {
            String level = String.valueOf(modelOptions.get("optimizationLevel")).toLowerCase();
            switch (level) {
                case "disable":
                    opts.setOptimizationLevel(
                            OrtSession.SessionOptions.OptLevel.NO_OPT
                    );
                    break;

                case "basic":
                    opts.setOptimizationLevel(
                            OrtSession.SessionOptions.OptLevel.BASIC_OPT
                    );
                    break;

                case "extended":
                    opts.setOptimizationLevel(
                            OrtSession.SessionOptions.OptLevel.EXTENDED_OPT
                    );
                    break;

                case "all":
                    opts.setOptimizationLevel(
                            OrtSession.SessionOptions.OptLevel.ALL_OPT
                    );
                    break;

                default:
                    throw new IllegalArgumentException(
                            "Unknown optimization level: " + level
                    );
            }

        }

        if (Boolean.TRUE.equals(modelOptions.get("useGpu"))) {
            try {
                opts.addCUDA(0);
            } catch (Exception ignored) {
            }
        }
    }

    /**
     * Extracts schema information from ONNX model metadata.
     *
     * @param metadata ONNX node metadata map
     * @return map of node names to schema information
     */
    private Map<String, Object> extractSchema(Map<String, NodeInfo> metadata) {
        Map<String, Object> schema = new HashMap<>();
        if (metadata == null) return schema;

        for (Map.Entry<String, NodeInfo> entry : metadata.entrySet()) {

            NodeInfo node = entry.getValue();

            Map<String, Object> nodeSchema = new HashMap<>();

            Object info = node.getInfo();

            nodeSchema.put("info", info != null ? info.toString() : "unknown");

            nodeSchema.put("rawType", node.toString());

            schema.put(entry.getKey(), nodeSchema);
        }

        return schema;
    }
}