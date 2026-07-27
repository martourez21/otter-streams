package com.codedstream.otterstream.inference.function;

import com.codedstream.otterstream.inference.config.InferenceConfig;
import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.async.AsyncFunction;
import org.apache.flink.streaming.api.functions.async.ResultFuture;
import org.apache.flink.api.common.functions.AbstractRichFunction;
import java.util.Collections;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
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

    private static final int DEFAULT_EXECUTOR_POOL_SIZE = 32;

    private final InferenceConfig inferenceConfig;
    private final Function<InferenceConfig, InferenceEngine<?>> engineFactory;
    private final int executorPoolSize;
    private transient InferenceEngine<?> inferenceEngine;

    /**
     * Dedicated pool for the blocking {@code inferenceEngine.infer(...)} call inside
     * {@link #asyncInvoke}. Deliberately <b>not</b> the default
     * {@link CompletableFuture#supplyAsync(java.util.function.Supplier)} behavior, which silently
     * runs on the shared, JVM-wide {@link java.util.concurrent.ForkJoinPool#commonPool()} —
     * fine for short, non-blocking CPU work, but a real problem here: inference calls block the
     * calling thread for the full call duration, and the common pool is shared with anything
     * else in the same JVM using parallel streams or unqualified {@code CompletableFuture}
     * calls. Under concurrent load, that means one hot model can starve unrelated work
     * elsewhere in the same TaskManager, and vice versa — exactly the kind of hard-to-diagnose
     * latency spike a sub-5ms latency target can't tolerate. A dedicated, sized-to-this-function
     * pool keeps that blast radius contained.
     */
    private transient ExecutorService inferenceExecutor;

    /**
     * Constructs async inference function with the default executor pool size ({@value #DEFAULT_EXECUTOR_POOL_SIZE}
     * threads — tune via {@link #AsyncModelInferenceFunction(InferenceConfig, Function, int)} to
     * match the {@code capacity} you pass to {@code AsyncDataStream.unorderedWait(...)}; they
     * should generally be the same order of magnitude, since capacity governs how many
     * in-flight {@code asyncInvoke} calls Flink allows per subtask).
     *
     * @param inferenceConfig configuration for inference operations
     * @param engineFactory factory function to create inference engine
     */
    public AsyncModelInferenceFunction(InferenceConfig inferenceConfig,
                                       Function<InferenceConfig, InferenceEngine<?>> engineFactory) {
        this(inferenceConfig, engineFactory, DEFAULT_EXECUTOR_POOL_SIZE);
    }

    /**
     * @param inferenceConfig  configuration for inference operations
     * @param engineFactory    factory function to create inference engine
     * @param executorPoolSize size of the dedicated thread pool backing {@link #asyncInvoke} —
     *                         should be sized to roughly match the {@code capacity} argument of
     *                         the corresponding {@code AsyncDataStream.unorderedWait(...)} call
     */
    public AsyncModelInferenceFunction(InferenceConfig inferenceConfig,
                                       Function<InferenceConfig, InferenceEngine<?>> engineFactory,
                                       int executorPoolSize) {
        this.inferenceConfig = inferenceConfig;
        this.engineFactory = engineFactory;
        this.executorPoolSize = executorPoolSize;
    }

    /**
     * Initializes the inference engine and the dedicated executor pool.
     *
     * <p>In Flink 2.0+, {@link OpenContext} replaces the legacy {@link Configuration}
     * parameter from earlier versions.
     *
     * @param openContext the open context providing access to runtime information
     * @throws Exception if initialization fails
     */
    @Override
    public void open(OpenContext openContext) throws Exception {  // <-- Use OpenContext
        super.open(openContext);  // <-- Call super with OpenContext
        initializeEngine();

        AtomicInteger threadCounter = new AtomicInteger();
        ThreadFactory threadFactory = runnable -> {
            Thread t = new Thread(runnable, "otter-async-inference-" + threadCounter.incrementAndGet());
            t.setDaemon(true);
            return t;
        };
        this.inferenceExecutor = Executors.newFixedThreadPool(executorPoolSize, threadFactory);
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
        Map<String, Object> features = extractFeatures(input);

        CompletableFuture
                .supplyAsync(() -> {
                    try {
                        return inferenceEngine.infer(features);
                    } catch (InferenceException e) {
                        throw new RuntimeException("Inference failed", e);
                    }
                }, inferenceExecutor)
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

    /** Releases the engine and the dedicated executor pool. */
    @Override
    public void close() throws Exception {
        if (inferenceExecutor != null) {
            inferenceExecutor.shutdown();
            try {
                if (!inferenceExecutor.awaitTermination(10, TimeUnit.SECONDS)) {
                    inferenceExecutor.shutdownNow();
                }
            } catch (InterruptedException e) {
                inferenceExecutor.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
        if (inferenceEngine != null) {
            inferenceEngine.close();
        }
        super.close();
    }

    /**
     * Initializes the inference engine. Called once from {@link #open(OpenContext)} — see
     * that method's Javadoc for why this is no longer safe to call lazily from
     * {@code asyncInvoke}.
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