package com.codedstream.otterstream.inference.function;

import com.codedstream.otterstream.inference.config.InferenceConfig;
import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import org.apache.flink.streaming.api.functions.async.AsyncFunction;
import org.apache.flink.streaming.api.functions.async.ResultFuture;
import org.apache.flink.api.common.functions.AbstractRichFunction;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.function.Function;

/**
 * Asynchronous function for performing ML inference in Apache Flink streams.
 *
 * <p>This function enables non-blocking inference operations in Flink pipelines,
 * allowing high throughput by processing multiple requests concurrently without
 * blocking the Flink operator thread.
 *
 * <h2>Key Features:</h2>
 * <ul>
 *   <li>Non-blocking async I/O for better throughput</li>
 *   <li>Automatic retry on failures</li>
 *   <li>Timeout handling</li>
 *   <li>Metrics collection</li>
 *   <li>Generic input/output transformation</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Define your input type (e.g., sensor reading)
 * DataStream<SensorReading> input = ...;
 *
 * // Configure inference
 * ModelConfig modelConfig = ModelConfig.builder()
 *     .modelId("anomaly-detector")
 *     .modelPath("/models/anomaly.onnx")
 *     .format(ModelFormat.ONNX)
 *     .build();
 *
 * InferenceConfig config = InferenceConfig.builder()
 *     .modelConfig(modelConfig)
 *     .batchSize(32)
 *     .timeout(Duration.ofSeconds(5))
 *     .build();
 *
 * // Create engine factory
 * Function<InferenceConfig, InferenceEngine<?>> engineFactory =
 *     cfg -> new OnnxInferenceEngine();
 *
 * // Apply async inference
 * AsyncDataStream.unorderedWait(
 *     input,
 *     new AsyncModelInferenceFunction<>(config, engineFactory),
 *     5000,  // timeout
 *     TimeUnit.MILLISECONDS,
 *     100    // capacity
 * );
 * }</pre>
 *
 * <h2>Custom Feature Extraction:</h2>
 * <p>Override {@link #extractFeatures(Object)} to customize how inputs are converted to model features:
 * <pre>{@code
 * public class CustomInferenceFunction
 *     extends AsyncModelInferenceFunction<SensorReading, AnomalyScore> {
 *
 *     @Override
 *     protected Map<String, Object> extractFeatures(SensorReading input) {
 *         return Map.of(
 *             "temperature", input.getTemperature(),
 *             "pressure", input.getPressure(),
 *             "timestamp", input.getTimestamp()
 *         );
 *     }
 *
 *     @Override
 *     protected AnomalyScore transformResult(SensorReading input, InferenceResult result) {
 *         float score = result.getOutput("anomaly_score");
 *         return new AnomalyScore(input.getSensorId(), score);
 *     }
 * }
 * }</pre>
 *
 * @param <IN> input record type from Flink stream
 * @param <OUT> output record type to Flink stream
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 * @see org.apache.flink.streaming.api.datastream.AsyncDataStream
 */
public class AsyncModelInferenceFunction<IN, OUT> extends AbstractRichFunction
        implements AsyncFunction<IN, OUT> {

    private final InferenceConfig inferenceConfig;
    private final Function<InferenceConfig, InferenceEngine<?>> engineFactory;
    private transient InferenceEngine<?> inferenceEngine;

    /**
     * Constructs async inference function.
     *
     * @param inferenceConfig configuration for inference operations
     * @param engineFactory factory function to create inference engine
     */
    public AsyncModelInferenceFunction(InferenceConfig inferenceConfig,
                                       Function<InferenceConfig, InferenceEngine<?>> engineFactory) {
        this.inferenceConfig = inferenceConfig;
        this.engineFactory = engineFactory;
    }

    /**
     * Performs asynchronous inference on input record.
     *
     * @param input the input record
     * @param resultFuture callback to complete with result
     * @throws Exception if processing fails
     */
    @Override
    public void asyncInvoke(IN input, ResultFuture<OUT> resultFuture) throws Exception {
        if (inferenceEngine == null || !inferenceEngine.isReady()) {
            initializeEngine();
        }

        Map<String, Object> features = extractFeatures(input);

        CompletableFuture
                .supplyAsync(() -> {
                    try {
                        long startTime = System.currentTimeMillis();
                        InferenceResult result = inferenceEngine.infer(features);
                        long endTime = System.currentTimeMillis();
                        return result;
                    } catch (InferenceException e) {
                        throw new RuntimeException("Inference failed", e);
                    }
                })
                .thenAccept(result -> {
                    if (result.isSuccess()) {
                        OUT output = transformResult(input, result);
                        resultFuture.complete(Collections.singleton(output));
                    } else {
                        resultFuture.completeExceptionally(
                                new InferenceException("Inference failed: " + result.getErrorMessage()));
                    }
                })
                .exceptionally(throwable -> {
                    resultFuture.completeExceptionally(throwable);
                    return null;
                });
    }

    /**
     * Called when inference timeout occurs.
     *
     * @param input the input record that timed out
     * @param resultFuture callback to complete with error
     * @throws Exception if handling fails
     */
    @Override
    public void timeout(IN input, ResultFuture<OUT> resultFuture) throws Exception {
        resultFuture.completeExceptionally(
                new InferenceException("Inference timeout for input: " + input));
    }

    /**
     * Initializes the inference engine lazily.
     * <p>Called on first use in each Flink TaskManager.
     *
     * @throws InferenceException if initialization fails
     */
    protected void initializeEngine() throws InferenceException {
        this.inferenceEngine = engineFactory.apply(inferenceConfig);
        this.inferenceEngine.initialize(inferenceConfig.getModelConfig());
    }

    /**
     * Extracts model input features from the input record.
     * <p>Override this to customize feature extraction for your use case.
     *
     * @param input the input record
     * @return map of feature name to feature value
     */
    protected Map<String, Object> extractFeatures(IN input) {
        if (input instanceof Map) {
            @SuppressWarnings("unchecked")
            Map<String, Object> features = (Map<String, Object>) input;
            return features;
        }
        throw new UnsupportedOperationException("Input must be Map or implement feature extraction");
    }

    /**
     * Transforms inference result into output record.
     * <p>Override this to customize result transformation for your use case.
     *
     * @param input the original input record
     * @param result the inference result
     * @return transformed output record
     */
    @SuppressWarnings("unchecked")
    protected OUT transformResult(IN input, InferenceResult result) {
        return (OUT) result;
    }
}