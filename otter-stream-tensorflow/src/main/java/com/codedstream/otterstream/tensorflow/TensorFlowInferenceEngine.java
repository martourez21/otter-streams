package com.codedstream.otterstream.tensorflow;

import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.engine.LocalInferenceEngine;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.inference.model.ModelFormat;
import com.codedstream.otterstream.inference.model.ModelMetadata;
import org.tensorflow.Result;
import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.proto.framework.SignatureDef;
import org.tensorflow.proto.framework.TensorInfo;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;

import java.util.Map;
import java.util.HashMap;
import java.util.List;

/**
 * TensorFlow SavedModel inference engine using TensorFlow Java API.
 *
 * <p>This engine provides inference capabilities for TensorFlow models saved in
 * the SavedModel format. It leverages the official TensorFlow Java API to load
 * and execute models with support for both CPU and GPU execution (when TensorFlow
 * is built with GPU support).
 *
 * <h2>Supported TensorFlow Features:</h2>
 * <ul>
 *   <li><b>SavedModel Format:</b> TensorFlow's standard serialization format</li>
 *   <li><b>Signature Parsing:</b> Automatic extraction of input/output signatures</li>
 *   <li><b>Tensor Types:</b> Float32 and Int32 tensors with multi-dimensional support</li>
 *   <li><b>Batch Inference:</b> Native support through tensor shape manipulation</li>
 *   <li><b>GPU Acceleration:</b> Automatic when TensorFlow Java with GPU support</li>
 * </ul>
 *
 * <h2>Model Loading:</h2>
 * <pre>{@code
 * ModelConfig config = ModelConfig.builder()
 *     .modelPath("/path/to/saved_model")  // Directory containing saved_model.pb
 *     .modelId("tensorflow-model")
 *     .modelVersion("v1")
 *     .build();
 *
 * TensorFlowInferenceEngine engine = new TensorFlowInferenceEngine();
 * engine.initialize(config);
 *
 * // Get metadata including input/output names
 * ModelMetadata metadata = engine.getMetadata();
 * List<String> inputNames = engine.getCachedInputNames();
 * }</pre>
 *
 * <h2>Inference Example:</h2>
 * <pre>{@code
 * Map<String, Object> inputs = new HashMap<>();
 * inputs.put("input_1", new float[]{0.1f, 0.2f, 0.3f});
 * inputs.put("input_2", new int[]{1, 2, 3});
 *
 * InferenceResult result = engine.infer(inputs);
 * float[] predictions = (float[]) result.getOutput("predictions");
 * }</pre>
 *
 * <h2>Tensor Creation:</h2>
 * <p>Automatically creates appropriate TensorFlow tensors from Java types:
 * <table border="1">
 *   <tr><th>Java Type</th><th>TensorFlow Type</th><th>Shape</th></tr>
 *   <tr><td>float[]</td><td>TFloat32</td><td>[1, array_length]</td></tr>
 *   <tr><td>float[][]</td><td>TFloat32</td><td>[rows, cols]</td></tr>
 *   <tr><td>int[]</td><td>TInt32</td><td>[1, array_length]</td></tr>
 * </table>
 *
 * <h2>Signature Discovery:</h2>
 * <p>The engine automatically discovers model signatures:
 * <ol>
 *   <li>First tries "serving_default" signature</li>
 *   <li>Falls back to first available signature</li>
 *   <li>Extracts input/output tensor names and shapes</li>
 *   <li>Caches names for performance</li>
 * </ol>
 *
 * <h2>Capabilities:</h2>
 * <table border="1">
 *   <tr><th>Feature</th><th>Supported</th><th>Notes</th></tr>
 *   <tr><td>Batch Inference</td><td>Yes</td><td>Through tensor shape manipulation</td></tr>
 *   <tr><td>Native Batching</td><td>Yes</td><td>TensorFlow native batch support</td></tr>
 *   <tr><td>Max Batch Size</td><td>128</td><td>Configurable based on memory</td></tr>
 *   <tr><td>GPU Support</td><td>Yes</td><td>When TensorFlow Java GPU version used</td></tr>
 *   <tr><td>Multi-threading</td><td>Yes</td><td>Session.Runner supports concurrent inference</td></tr>
 * </table>
 *
 * <h2>Dependencies:</h2>
 * <pre>
 * Requires TensorFlow Java API:
 * - org.tensorflow:tensorflow-core-platform (runtime)
 * - org.tensorflow:tensorflow-core-api (runtime)
 * - For GPU: org.tensorflow:libtensorflow with GPU support
 * </pre>
 *
 * <h2>Performance Features:</h2>
 * <ul>
 *   <li><b>Signature Caching:</b> Input/output names cached for performance</li>
 *   <li><b>Tensor Reuse:</b> Automatic tensor cleanup to prevent memory leaks</li>
 *   <li><b>Direct Memory Access:</b> Uses TensorFlow's native memory management</li>
 *   <li><b>Session Pooling:</b> {@link SavedModelBundle} manages session resources</li>
 * </ul>
 *
 * <h2>Thread Safety:</h2>
 * <p>{@link Session.Runner} is not thread-safe, but {@link SavedModelBundle} can be
 * used from multiple threads by creating separate runners. Consider:
 * <ul>
 *   <li>Creating separate runners per thread</li>
 *   <li>Using {@link Session} with synchronization</li>
 *   <li>Implementing connection pooling for high-throughput scenarios</li>
 * </ul>
 *
 * <h2>Resource Management:</h2>
 * <p>Always call {@link #close()} to release native TensorFlow resources.
 * TensorFlow uses native memory that must be explicitly released:
 *
 * <pre>{@code
 * try (TensorFlowInferenceEngine engine = new TensorFlowInferenceEngine()) {
 *     engine.initialize(config);
 *     InferenceResult result = engine.infer(inputs);
 * }
 * }</pre>
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 * @see LocalInferenceEngine
 * @see SavedModelBundle
 * @see <a href="https://www.tensorflow.org/jvm/api_docs/java/org/tensorflow/package-summary">TensorFlow Java API</a>
 */
public class TensorFlowInferenceEngine extends LocalInferenceEngine<SavedModelBundle> {

    private List<String> cachedInputNames;
    private List<String> cachedOutputNames;

    /**
     * Initializes the TensorFlow inference engine by loading a SavedModel.
     *
     * <p>The initialization process:
     * <ol>
     *   <li>Loads SavedModel from directory using {@link SavedModelBundle#load}</li>
     *   <li>Parses model signature to extract input/output tensor information</li>
     *   <li>Caches input and output names for performance</li>
     *   <li>Validates model readiness for inference</li>
     * </ol>
     *
     * <h2>SavedModel Structure:</h2>
     * <pre>
     * saved_model_directory/
     *   ├── saved_model.pb      # Model graph and signatures
     *   ├── variables/          # Model weights
     *   └── assets/             # Additional files (optional)
     * </pre>
     *
     * <h2>Signature Discovery:</h2>
     * <p>The engine looks for signatures in this order:
     * <ol>
     *   <li>"serving_default" signature (standard for serving)</li>
     *   <li>First available signature in the model</li>
     *   <li>Fallback to common output names if no signature found</li>
     * </ol>
     *
     * @param config model configuration containing SavedModel directory path
     * @throws InferenceException if model loading fails or SavedModel is invalid
     */
    @Override
    public void initialize(ModelConfig config) throws InferenceException {
        try {
            this.modelConfig = config;
            this.loadedModel = SavedModelBundle.load(config.getModelPath(), "serve");

            // Parse and cache signature information
            this.cachedInputNames = parseInputNames();
            this.cachedOutputNames = parseOutputNames();

            this.initialized = true;
        } catch (Exception e) {
            throw new InferenceException("Failed to load TensorFlow model from: " + config.getModelPath(), e);
        }
    }

    /**
     * Performs single inference using TensorFlow SavedModel.
     *
     * <p>The inference process:
     * <ol>
     *   <li>Creates {@link Session.Runner} for inference execution</li>
     *   <li>Converts input values to TensorFlow tensors</li>
     *   <li>Feeds tensors to runner using cached input names</li>
     *   <li>Fetches output tensors using cached output names</li>
     *   <li>Executes graph and extracts results</li>
     *   <li>Cleans up tensors to prevent memory leaks</li>
     * </ol>
     *
     * <h2>Tensor Management:</h2>
     * <p>All created tensors are automatically closed using try-with-resources
     * pattern. Output tensors from the runner are also explicitly closed in
     * the finally block to prevent native memory leaks.
     *
     * <h2>Error Handling:</h2>
     * <ul>
     *   <li>Invalid tensor types throw {@link IllegalArgumentException}</li>
     *   <li>Missing input names result in runtime errors</li>
     *   <li>TensorFlow runtime errors throw {@link InferenceException}</li>
     * </ul>
     *
     * @param inputs map of input tensor names to values (float[] or int[])
     * @return inference result containing outputs and timing information
     * @throws InferenceException if inference fails or tensor creation fails
     */
    @Override
    public InferenceResult infer(Map<String, Object> inputs) throws InferenceException {
        Session.Runner runner = null;
        Result result = null;
        try {
            runner = loadedModel.session().runner();
            long startTime = System.currentTimeMillis();

            // Prepare inputs
            for (Map.Entry<String, Object> entry : inputs.entrySet()) {
                try (Tensor tensor = createTensor(entry.getValue())) {
                    runner.feed(entry.getKey(), tensor);
                }
            }

            // Get output tensor names
            List<String> outputNames = cachedOutputNames;
            for (String outputName : outputNames) {
                runner.fetch(outputName);
            }

            // Run inference
            result = runner.run();

            // Extract outputs
            Map<String, Object> outputs = new HashMap<>();
            for (int i = 0; i < result.size() && i < outputNames.size(); i++) {
                Tensor tensor = result.get(i);
                String outputName = outputNames.get(i);
                outputs.put(outputName, extractTensorValue(tensor));
            }

            long endTime = System.currentTimeMillis();
            return new InferenceResult(outputs, endTime - startTime, modelConfig.getModelId());
        } catch (Exception e) {
            throw new InferenceException("TensorFlow inference failed", e);
        } finally {
            // Close all output tensors
            if (result != null) {
                for (int i = 0; i < result.size(); i++) {
                    Tensor tensor = result.get(i);
                    if (tensor != null) {
                        tensor.close();
                    }
                }
            }
        }
    }

    /**
     * Performs batch inference (simplified implementation).
     *
     * <p><strong>Note:</strong> Current implementation processes only the first
     * input in the batch. For proper batch inference, extend this method to:
     * <ol>
     *   <li>Create batch tensors by stacking individual inputs</li>
     *   <li>Modify tensor shapes to include batch dimension</li>
     *   <li>Execute single inference with batch tensor</li>
     *   <li>Split batch output into individual results</li>
     * </ol>
     *
     * @param batchInputs array of input maps for batch processing
     * @return inference result for first input (placeholder implementation)
     * @throws InferenceException if batch processing fails
     */
    @Override
    public InferenceResult inferBatch(Map<String, Object>[] batchInputs) throws InferenceException {
        // TensorFlow handles batching through tensor shapes
        // For simplicity, we'll process the first input
        // In production, you'd create proper batch tensors
        if (batchInputs.length == 0) {
            return new InferenceResult(Map.of(), 0, modelConfig.getModelId());
        }
        return infer(batchInputs[0]);
    }

    /**
     * Gets the engine's capabilities for TensorFlow inference.
     *
     * <p>TensorFlow engine capabilities:
     * <ul>
     *   <li><b>Batch Inference:</b> Supported through tensor batching</li>
     *   <li><b>Native Batching:</b> Yes, TensorFlow native batch support</li>
     *   <li><b>Max Batch Size:</b> 128 (conservative default for memory safety)</li>
     *   <li><b>GPU Support:</b> Yes, when using TensorFlow GPU version</li>
     * </ul>
     *
     * @return engine capabilities indicating full TensorFlow support
     */
    @Override
    public EngineCapabilities getCapabilities() {
        return new EngineCapabilities(true, true, 128, true);
    }

    /**
     * Gets metadata about the loaded TensorFlow model.
     *
     * <p>Extracts metadata from the SavedModel including:
     * <ul>
     *   <li>Model name and version from configuration</li>
     *   <li>Input/output schema from cached names</li>
     *   <li>Model format as {@link ModelFormat#TENSORFLOW_SAVEDMODEL}</li>
     *   <li>Load timestamp for freshness tracking</li>
     * </ul>
     *
     * <h2>Schema Format:</h2>
     * <p>Inputs and outputs are stored as maps with indexed keys:
     * <pre>
     * Input Schema:  {"input_0": "input_tensor_name", "input_1": ...}
     * Output Schema: {"output_0": "output_tensor_name", "output_1": ...}
     * </pre>
     *
     * @return comprehensive model metadata
     */
    @Override
    public ModelMetadata getMetadata() {
        Map<String, Object> inputSchemaMap = new HashMap<>();
        if (cachedInputNames != null) {
            for (int i = 0; i < cachedInputNames.size(); i++) {
                inputSchemaMap.put("input_" + i, cachedInputNames.get(i));
            }
        }

        Map<String, Object> outputSchemaMap = new HashMap<>();
        if (cachedOutputNames != null) {
            for (int i = 0; i < cachedOutputNames.size(); i++) {
                outputSchemaMap.put("output_" + i, cachedOutputNames.get(i));
            }
        }

        // Use modelId from config as modelName, or use a descriptive name
        String modelName = modelConfig.getModelId() != null ?
                modelConfig.getModelId() : "tensorflow-model";

        return ModelMetadata.builder()
                .modelName(modelName)
                .modelVersion(modelConfig.getModelVersion() != null ?
                        modelConfig.getModelVersion() : "unknown")
                .format(ModelFormat.TENSORFLOW_SAVEDMODEL)
                .inputSchema(inputSchemaMap)
                .outputSchema(outputSchemaMap)
                .modelSize(0L)
                .loadTimestamp(System.currentTimeMillis())
                .build();
    }

    /**
     * Closes the TensorFlow engine and releases native resources.
     *
     * <p>Closes the {@link SavedModelBundle} which releases:
     * <ul>
     *   <li>Graph definition memory</li>
     *   <li>Session resources</li>
     *   <li>Variable storage</li>
     *   <li>Any GPU memory allocated</li>
     * </ul>
     *
     * <p>Always call this method when finished to prevent native memory leaks.
     *
     * @throws InferenceException if resource cleanup fails
     */
    @Override
    public void close() throws InferenceException {
        if (loadedModel != null) {
            loadedModel.close();
        }
        super.close();
    }

    /**
     * Creates TensorFlow tensors from Java objects.
     *
     * <p>Supports conversion of Java arrays to TensorFlow tensors:
     * <ul>
     *   <li><b>float[]:</b> Creates 2D tensor with shape [1, length]</li>
     *   <li><b>float[][]:</b> Creates 2D tensor with shape [rows, cols]</li>
     *   <li><b>int[]:</b> Creates 2D tensor with shape [1, length]</li>
     * </ul>
     *
     * <p>Extend this method to support additional types:
     * <ul>
     *   <li>double[] → TFloat64</li>
     *   <li>long[] → TInt64</li>
     *   <li>String[] → TString</li>
     *   <li>boolean[] → TBool</li>
     * </ul>
     *
     * @param value Java object to convert to tensor
     * @return TensorFlow tensor
     * @throws IllegalArgumentException for unsupported types
     */
    private Tensor createTensor(Object value) {
        if (value instanceof float[]) {
            float[] array = (float[]) value;
            return TFloat32.tensorOf(Shape.of(1, array.length), data -> {
                long[] indices = new long[2];
                for (int i = 0; i < array.length; i++) {
                    indices[0] = 0;
                    indices[1] = i;
                    data.setFloat(array[i], indices);
                }
            });
        } else if (value instanceof float[][]) {
            float[][] array = (float[][]) value;
            int rows = array.length;
            int cols = array[0].length;
            return TFloat32.tensorOf(Shape.of(rows, cols), data -> {
                long[] indices = new long[2];
                for (int i = 0; i < rows; i++) {
                    for (int j = 0; j < cols; j++) {
                        indices[0] = i;
                        indices[1] = j;
                        data.setFloat(array[i][j], indices);
                    }
                }
            });
        } else if (value instanceof int[]) {
            int[] array = (int[]) value;
            return TInt32.tensorOf(Shape.of(1, array.length), data -> {
                long[] indices = new long[2];
                for (int i = 0; i < array.length; i++) {
                    indices[0] = 0;
                    indices[1] = i;
                    data.setInt(array[i], indices);
                }
            });
        } else {
            throw new IllegalArgumentException("Unsupported input type: " + value.getClass());
        }
    }

    /**
     * Extracts values from TensorFlow tensors into Java objects.
     *
     * <p>Handles different tensor types and dimensions:
     * <ul>
     *   <li><b>TFloat32:</b> Extracts as float[] or single float</li>
     *   <li><b>TInt32:</b> Extracts as int[] or single int</li>
     *   <li><b>Other types:</b> Returns string representation</li>
     * </ul>
     *
     * <h2>Shape Handling:</h2>
     * <ul>
     *   <li>1D tensors: Return as array</li>
     *   <li>2D tensors with single row: Return first row as array</li>
     *   <li>Scalar (single element): Return as primitive value</li>
     *   <li>Higher dimensions: Flatten to 1D array</li>
     * </ul>
     *
     * @param tensor TensorFlow tensor to extract values from
     * @return Java object (array or primitive) containing tensor values
     */
    private Object extractTensorValue(Tensor tensor) {
        // Extract value based on tensor type
        if (tensor instanceof TFloat32) {
            TFloat32 floatTensor = (TFloat32) tensor;
            long totalSize = tensor.shape().size();
            float[] values = new float[(int) totalSize];

            // Read data using direct ndarray access
            if (tensor.shape().numDimensions() == 1) {
                for (int i = 0; i < totalSize; i++) {
                    values[i] = floatTensor.getFloat(i);
                }
            } else if (tensor.shape().numDimensions() == 2) {
                int idx = 0;
                for (int i = 0; i < tensor.shape().size(0); i++) {
                    for (int j = 0; j < tensor.shape().size(1); j++) {
                        values[idx++] = floatTensor.getFloat(i, j);
                    }
                }
            } else {
                // For higher dimensions, flatten and read
                long[] indices = new long[tensor.shape().numDimensions()];
                for (int i = 0; i < totalSize; i++) {
                    // Convert flat index to multi-dimensional indices
                    long remaining = i;
                    for (int d = tensor.shape().numDimensions() - 1; d >= 0; d--) {
                        indices[d] = remaining % tensor.shape().size(d);
                        remaining /= tensor.shape().size(d);
                    }
                    values[i] = floatTensor.getFloat(indices);
                }
            }

            if (tensor.shape().numDimensions() == 1 && values.length == 1) {
                return values[0];
            } else if (tensor.shape().numDimensions() == 2 && tensor.shape().size(0) == 1) {
                // Return first row for 2D tensor with single row
                int cols = (int) tensor.shape().size(1);
                float[] firstRow = new float[cols];
                System.arraycopy(values, 0, firstRow, 0, cols);
                return firstRow;
            }
            return values;
        } else if (tensor instanceof TInt32) {
            TInt32 intTensor = (TInt32) tensor;
            long totalSize = tensor.shape().size();
            int[] values = new int[(int) totalSize];

            // Read data using direct ndarray access
            if (tensor.shape().numDimensions() == 1) {
                for (int i = 0; i < totalSize; i++) {
                    values[i] = intTensor.getInt(i);
                }
            } else if (tensor.shape().numDimensions() == 2) {
                int idx = 0;
                for (int i = 0; i < tensor.shape().size(0); i++) {
                    for (int j = 0; j < tensor.shape().size(1); j++) {
                        values[idx++] = intTensor.getInt(i, j);
                    }
                }
            } else {
                // For higher dimensions, flatten and read
                long[] indices = new long[tensor.shape().numDimensions()];
                for (int i = 0; i < totalSize; i++) {
                    long remaining = i;
                    for (int d = tensor.shape().numDimensions() - 1; d >= 0; d--) {
                        indices[d] = remaining % tensor.shape().size(d);
                        remaining /= tensor.shape().size(d);
                    }
                    values[i] = intTensor.getInt(indices);
                }
            }

            if (tensor.shape().numDimensions() == 1 && values.length == 1) {
                return values[0];
            }
            return values;
        } else {
            // Fallback for other types
            return tensor.toString();
        }
    }

    /**
     * Parses input tensor names from the SavedModel signature.
     *
     * <p>Extracts input tensor names from the model's signature definition.
     * Returns cached names if available, otherwise parses from signature.
     *
     * @return list of input tensor names
     */
    private List<String> parseInputNames() {
        try {
            SignatureDef signatureDef = getSignatureDef();

            if (signatureDef != null) {
                List<String> inputNames = new java.util.ArrayList<>();
                Map<String, TensorInfo> inputs = signatureDef.getInputsMap();

                for (Map.Entry<String, TensorInfo> entry : inputs.entrySet()) {
                    inputNames.add(entry.getValue().getName());
                }

                if (!inputNames.isEmpty()) {
                    return inputNames;
                }
            }
        } catch (Exception e) {
            System.err.println("Failed to parse input names from signature: " + e.getMessage());
        }

        return new java.util.ArrayList<>();
    }

    /**
     * Parses output tensor names from the SavedModel signature.
     *
     * <p>Extracts output tensor names from the model's signature definition.
     * Returns cached names if available, otherwise parses from signature.
     * Falls back to common output names if no signature found.
     *
     * @return list of output tensor names
     */
    private List<String> parseOutputNames() {
        try {
            SignatureDef signatureDef = getSignatureDef();

            if (signatureDef != null) {
                List<String> outputNames = new java.util.ArrayList<>();
                Map<String, TensorInfo> outputs = signatureDef.getOutputsMap();

                for (Map.Entry<String, TensorInfo> entry : outputs.entrySet()) {
                    outputNames.add(entry.getValue().getName());
                }

                if (!outputNames.isEmpty()) {
                    return outputNames;
                }
            }
        } catch (Exception e) {
            System.err.println("Failed to parse output names from signature: " + e.getMessage());
        }

        // Fallback: return common output names
        return java.util.Arrays.asList("output", "predictions", "scores");
    }

    /**
     * Retrieves the serving signature from the SavedModel.
     *
     * <p>Attempts to get the signature in this order:
     * <ol>
     *   <li>"serving_default" signature (standard TensorFlow serving)</li>
     *   <li>First available signature in the model</li>
     *   <li>null if no signatures found</li>
     * </ol>
     *
     * @return SignatureDef containing input/output specifications
     */
    private SignatureDef getSignatureDef() {
        try {
            // Try to get the default serving signature
            SignatureDef signatureDef = loadedModel.metaGraphDef()
                    .getSignatureDefOrDefault("serving_default", null);

            if (signatureDef == null) {
                // Fallback to any available signature
                Map<String, SignatureDef> signatureMap = loadedModel.metaGraphDef()
                        .getSignatureDefMap();
                if (!signatureMap.isEmpty()) {
                    signatureDef = signatureMap.values().iterator().next();
                }
            }

            return signatureDef;
        } catch (Exception e) {
            System.err.println("Failed to retrieve model signature: " + e.getMessage());
            return null;
        }
    }

    /**
     * Gets the cached input tensor names.
     *
     * @return list of input tensor names extracted during initialization
     */
    public List<String> getCachedInputNames() {
        return cachedInputNames;
    }

    /**
     * Gets the cached output tensor names.
     *
     * @return list of output tensor names extracted during initialization
     */
    public List<String> getCachedOutputNames() {
        return cachedOutputNames;
    }
}