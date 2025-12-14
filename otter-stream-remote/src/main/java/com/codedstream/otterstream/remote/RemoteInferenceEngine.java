package com.codedstream.otterstream.remote;

import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;

import java.util.Map;

/**
 * Abstract base class for remote inference engines communicating with external model endpoints.
 *
 * <p>Provides common functionality for remote inference implementations including:
 * <ul>
 *   <li>Basic initialization with endpoint URL</li>
 *   <li>Default batch inference implementation (sequential processing)</li>
 *   <li>Connection state management</li>
 *   <li>Resource cleanup framework</li>
 * </ul>
 *
 * <h2>Extending This Class:</h2>
 * <p>Subclasses should implement:
 * <ol>
 *   <li>{@link #initialize(ModelConfig)} - Set up remote connection</li>
 *   <li>{@link #infer(Map)} - Send single inference request</li>
 *   <li>{@link #validateConnection()} - Verify endpoint availability</li>
 *   <li>{@link #close()} - Clean up resources</li>
 * </ol>
 *
 * <h2>Batch Inference:</h2>
 * <p>The default {@link #inferBatch(Map[])} implementation processes requests sequentially.
 * Subclasses should override this method if the remote endpoint supports native batch processing
 * for better performance.
 *
 * <h2>Usage Pattern:</h2>
 * <pre>{@code
 * public class MyRemoteEngine extends RemoteInferenceEngine {
 *     @Override
 *     public void initialize(ModelConfig config) throws InferenceException {
 *         super.initialize(config); // Sets endpointUrl
 *         // Custom initialization
 *     }
 *
 *     @Override
 *     public InferenceResult infer(Map<String, Object> inputs) throws InferenceException {
 *         // Send request to endpointUrl
 *     }
 *
 *     @Override
 *     public boolean validateConnection() throws InferenceException {
 *         // Test connection to endpointUrl
 *     }
 *
 *     @Override
 *     public void close() throws InferenceException {
 *         // Clean up resources
 *         super.close(); // Marks as not initialized
 *     }
 * }
 * }</pre>
 *
 * <h2>State Management:</h2>
 * <p>The engine tracks initialization state via {@link #initialized} flag. Always call
 * {@link #initialize(ModelConfig)} before inference and {@link #close()} when finished.
 *
 * <h2>Thread Safety:</h2>
 * <p>This base class is not thread-safe. Subclasses must implement their own synchronization
 * if needed. Consider creating separate engine instances for concurrent inference.
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 * @see InferenceEngine
 * @see HttpInferenceClient
 * @see SageMakerInferenceClient
 */
public abstract class RemoteInferenceEngine implements InferenceEngine<Void> {
    protected ModelConfig modelConfig;
    protected boolean initialized = false;
    protected String endpointUrl;

    /**
     * Initializes the remote inference engine with endpoint configuration.
     *
     * <p>Basic initialization sets:
     * <ul>
     *   <li>{@link #modelConfig} - Configuration for the remote model</li>
     *   <li>{@link #endpointUrl} - URL from {@link ModelConfig#getEndpointUrl()}</li>
     *   <li>{@link #initialized} - Marks engine as ready for inference</li>
     * </ul>
     *
     * <p>Subclasses should call {@code super.initialize(config)} first, then
     * perform their own initialization (HTTP client setup, authentication, etc.).
     *
     * @param config model configuration containing endpoint URL
     * @throws InferenceException if initialization fails
     */
    @Override
    public void initialize(ModelConfig config) throws InferenceException {
        this.modelConfig = config;
        this.endpointUrl = config.getEndpointUrl();
        this.initialized = true;
    }

    /**
     * Performs single inference on remote endpoint (abstract).
     *
     * <p>Subclasses must implement this method to send inference requests to
     * the remote endpoint and parse responses.
     *
     * @param inputs map of input names to values
     * @return inference result containing outputs and timing
     * @throws InferenceException if inference fails
     */
    @Override
    public abstract InferenceResult infer(Map<String, Object> inputs) throws InferenceException;

    /**
     * Performs batch inference using sequential processing.
     *
     * <p>Default implementation processes batch inputs sequentially by calling
     * {@link #infer(Map)} for each input. Outputs are aggregated with index
     * suffixes to maintain traceability.
     *
     * <h2>Output Format:</h2>
     * <pre>
     * {
     *   "output1_0": value1,
     *   "output2_0": value2,
     *   "output1_1": value3,
     *   "output2_1": value4,
     *   ...
     * }
     * </pre>
     *
     * <p><strong>Performance Note:</strong> Sequential processing is inefficient
     * for large batches. Override this method if the remote endpoint supports
     * native batch processing.
     *
     * @param batchInputs array of input maps for batch processing
     * @return aggregated inference result containing all batch outputs
     * @throws InferenceException if any inference in the batch fails
     */
    @Override
    public InferenceResult inferBatch(Map<String, Object>[] batchInputs) throws InferenceException {
        // Default implementation - process sequentially
        // Override for batch-optimized remote endpoints
        Map<String, Object> batchOutputs = new java.util.HashMap<>();
        long totalTime = 0;

        for (int i = 0; i < batchInputs.length; i++) {
            InferenceResult result = infer(batchInputs[i]);
            totalTime += result.getInferenceTimeMs();

            // Store results with index
            for (Map.Entry<String, Object> entry : result.getOutputs().entrySet()) {
                String batchKey = entry.getKey() + "_" + i;
                batchOutputs.put(batchKey, entry.getValue());
            }
        }

        return new InferenceResult(batchOutputs, totalTime, modelConfig.getModelId());
    }

    /**
     * Checks if the engine is ready for inference.
     *
     * @return true if initialized and endpoint URL is set
     */
    @Override
    public boolean isReady() {
        return initialized && endpointUrl != null;
    }

    /**
     * Closes the engine and releases resources.
     *
     * <p>Base implementation marks engine as not initialized and clears endpoint URL.
     * Subclasses should override to clean up their own resources (close HTTP clients,
     * database connections, etc.), then call {@code super.close()}.
     *
     * @throws InferenceException if cleanup fails
     */
    @Override
    public void close() throws InferenceException {
        this.initialized = false;
        this.endpointUrl = null;
    }

    /**
     * Validates connection to remote endpoint (abstract).
     *
     * <p>Subclasses must implement this method to test connectivity to the
     * remote endpoint. This is useful for health checks and connection pooling.
     *
     * @return true if connection is successful
     * @throws InferenceException if validation fails (network error, timeout)
     */
    public abstract boolean validateConnection() throws InferenceException;

    /**
     * Gets engine capabilities for remote endpoints.
     *
     * <p>Default capabilities indicate:
     * <ul>
     *   <li>Not a local engine (remote = false)</li>
     *   <li>No native batch support (batch = false)</li>
     *   <li>Max batch size of 1 (sequential processing)</li>
     *   <li>Async operations supported (async = true)</li>
     * </ul>
     *
     * <p>Subclasses should override to provide accurate capabilities for their
     * specific remote endpoint.
     *
     * @return default engine capabilities for remote inference
     */
    @Override
    public EngineCapabilities getCapabilities() {
        return new EngineCapabilities(false, false, 1, true);
    }
}