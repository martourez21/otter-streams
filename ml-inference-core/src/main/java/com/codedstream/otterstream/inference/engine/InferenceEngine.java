package com.codedstream.otterstream.inference.engine;

import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.inference.model.ModelMetadata;
import java.util.Map;

/**
 * Core interface for ML inference engines in Otter Stream.
 *
 * <p>All inference engines (TensorFlow, ONNX, PyTorch, XGBoost, etc.) implement
 * this interface to provide a uniform API for model loading and prediction.
 *
 * <h2>Lifecycle:</h2>
 * <ol>
 *   <li>{@link #initialize(ModelConfig)} - Load and prepare the model</li>
 *   <li>{@link #isReady()} - Verify engine is ready for inference</li>
 *   <li>{@link #infer(Map)} or {@link #inferBatch(Map[])} - Make predictions</li>
 *   <li>{@link #close()} - Release resources</li>
 * </ol>
 *
 * <h2>Implementation Example:</h2>
 * <pre>{@code
 * public class MyCustomEngine implements InferenceEngine<MyModel> {
 *     private MyModel model;
 *
 *     @Override
 *     public void initialize(ModelConfig config) {
 *         this.model = loadModel(config.getModelPath());
 *     }
 *
 *     @Override
 *     public InferenceResult infer(Map<String, Object> inputs) {
 *         // Run prediction and return result
 *     }
 * }
 * }</pre>
 *
 * @param <T> the underlying model type (e.g., SavedModelBundle for TensorFlow)
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 * @see LocalInferenceEngine
 */
public interface InferenceEngine<T> {

    /**
     * Initializes the inference engine with the given configuration.
     * <p>Loads the model and prepares the engine for inference operations.
     *
     * @param config model configuration
     * @throws InferenceException if initialization fails
     */
    void initialize(ModelConfig config) throws InferenceException;

    /**
     * Performs inference on a single input.
     *
     * @param inputs map of input name to input value
     * @return inference result containing predictions
     * @throws InferenceException if inference fails
     */
    InferenceResult infer(Map<String, Object> inputs) throws InferenceException;

    /**
     * Performs batch inference on multiple inputs.
     * <p>Batch inference is typically more efficient than multiple single inferences.
     *
     * @param batchInputs array of input maps
     * @return inference result containing batch predictions
     * @throws InferenceException if inference fails
     */
    InferenceResult inferBatch(Map<String, Object>[] batchInputs) throws InferenceException;

    /**
     * Gets the capabilities of this inference engine.
     *
     * @return engine capabilities (batching, GPU support, etc.)
     */
    EngineCapabilities getCapabilities();

    /**
     * Closes the inference engine and releases all resources.
     * <p>After calling this method, the engine should not be used again.
     *
     * @throws InferenceException if cleanup fails
     */
    void close() throws InferenceException;

    /**
     * Checks if the engine is ready for inference operations.
     *
     * @return true if engine is initialized and ready
     */
    boolean isReady();

    /**
     * Gets metadata about the loaded model.
     *
     * @return model metadata including inputs, outputs, and format
     */
    ModelMetadata getMetadata();

    /**
     * Gets the configuration used to initialize this engine.
     *
     * @return model configuration
     */
    ModelConfig getModelConfig();

    /**
     * Describes the capabilities of an inference engine.
     *
     * <p>Use this to determine what features are supported before using them.
     */
    class EngineCapabilities {
        private final boolean supportsBatching;
        private final boolean supportsGPU;
        private final int maxBatchSize;
        private final boolean supportsStreaming;

        /**
         * Constructs engine capabilities.
         *
         * @param supportsBatching whether batch inference is supported
         * @param supportsGPU whether GPU acceleration is available
         * @param maxBatchSize maximum batch size (0 if unlimited)
         * @param supportsStreaming whether streaming inference is supported
         */
        public EngineCapabilities(boolean supportsBatching, boolean supportsGPU,
                                  int maxBatchSize, boolean supportsStreaming) {
            this.supportsBatching = supportsBatching;
            this.supportsGPU = supportsGPU;
            this.maxBatchSize = maxBatchSize;
            this.supportsStreaming = supportsStreaming;
        }

        public boolean supportsBatching() { return supportsBatching; }
        public boolean supportsGPU() { return supportsGPU; }
        public int getMaxBatchSize() { return maxBatchSize; }
        public boolean supportsStreaming() { return supportsStreaming; }
    }
}
