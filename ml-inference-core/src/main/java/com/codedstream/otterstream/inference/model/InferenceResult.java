package com.codedstream.otterstream.inference.model;

import java.util.Map;
import java.util.Objects;

/**
 * Container for ML inference results including predictions and metadata.
 *
 * <p>Represents the output from an inference operation, containing the model's
 * predictions, timing information, and success status.
 *
 * <h2>Result Types:</h2>
 * <ul>
 *   <li><b>Success:</b> Contains predictions in outputs map</li>
 *   <li><b>Failure:</b> Contains error message and empty outputs</li>
 * </ul>
 *
 * <h2>Usage Examples:</h2>
 *
 * <h3>Creating Success Result:</h3>
 * <pre>{@code
 * Map<String, Object> outputs = Map.of(
 *     "probability", 0.92,
 *     "class", "fraudulent",
 *     "confidence", 0.88
 * );
 * InferenceResult result = new InferenceResult(outputs, 45, "fraud-detector");
 * }</pre>
 *
 * <h3>Creating Failure Result:</h3>
 * <pre>{@code
 * InferenceResult result = new InferenceResult(
 *     "fraud-detector",
 *     "Model execution timeout",
 *     5000
 * );
 * }</pre>
 *
 * <h3>Consuming Results:</h3>
 * <pre>{@code
 * if (result.isSuccess()) {
 *     double prob = result.getOutput("probability");
 *     String prediction = result.getOutput("class");
 *     System.out.println("Prediction: " + prediction + " (" + prob + ")");
 * } else {
 *     log.error("Inference failed: {}", result.getErrorMessage());
 * }
 * }</pre>
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 */
public class InferenceResult {
    private final Map<String, Object> outputs;
    private final long inferenceTimeMs;
    private final String modelId;
    private final boolean success;
    private final String errorMessage;

    /**
     * Creates a successful inference result.
     *
     * @param outputs map of output name to predicted value
     * @param inferenceTimeMs inference duration in milliseconds
     * @param modelId identifier of the model that made the prediction
     */
    public InferenceResult(Map<String, Object> outputs, long inferenceTimeMs, String modelId) {
        this.outputs = Map.copyOf(Objects.requireNonNull(outputs));
        this.inferenceTimeMs = inferenceTimeMs;
        this.modelId = Objects.requireNonNull(modelId);
        this.success = true;
        this.errorMessage = null;
    }

    /**
     * Creates a failed inference result.
     *
     * @param modelId identifier of the model that failed
     * @param errorMessage description of the failure
     * @param inferenceTimeMs time spent before failure in milliseconds
     */
    public InferenceResult(String modelId, String errorMessage, long inferenceTimeMs) {
        this.outputs = Map.of();
        this.inferenceTimeMs = inferenceTimeMs;
        this.modelId = Objects.requireNonNull(modelId);
        this.success = false;
        this.errorMessage = errorMessage;
    }

    /**
     * Gets all output predictions.
     *
     * @return immutable map of output name to value
     */
    public Map<String, Object> getOutputs() { return outputs; }

    /**
     * Gets inference duration.
     *
     * @return duration in milliseconds
     */
    public long getInferenceTimeMs() { return inferenceTimeMs; }

    /**
     * Gets model identifier.
     *
     * @return model ID
     */
    public String getModelId() { return modelId; }

    /**
     * Checks if inference was successful.
     *
     * @return true if successful, false if failed
     */
    public boolean isSuccess() { return success; }

    /**
     * Gets error message for failed inferences.
     *
     * @return error message, or null if successful
     */
    public String getErrorMessage() { return errorMessage; }

    /**
     * Gets a specific output value by name.
     * <p>Convenience method with type casting.
     *
     * @param <T> expected type of the output
     * @param key output name
     * @return output value cast to type T, or null if not found
     * @throws ClassCastException if type T doesn't match actual type
     */
    @SuppressWarnings("unchecked")
    public <T> T getOutput(String key) {
        return (T) outputs.get(key);
    }
}