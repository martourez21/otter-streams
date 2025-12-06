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

public class AsyncModelInferenceFunction<IN, OUT> extends AbstractRichFunction
        implements AsyncFunction<IN, OUT> {

    private final InferenceConfig inferenceConfig;
    private final Function<InferenceConfig, InferenceEngine<?>> engineFactory;
    private transient InferenceEngine<?> inferenceEngine;

    public AsyncModelInferenceFunction(InferenceConfig inferenceConfig,
                                       Function<InferenceConfig, InferenceEngine<?>> engineFactory) {
        this.inferenceConfig = inferenceConfig;
        this.engineFactory = engineFactory;
    }

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

    @Override
    public void timeout(IN input, ResultFuture<OUT> resultFuture) throws Exception {
        resultFuture.completeExceptionally(
                new InferenceException("Inference timeout for input: " + input));
    }

    protected void initializeEngine() throws InferenceException {
        this.inferenceEngine = engineFactory.apply(inferenceConfig);
        this.inferenceEngine.initialize(inferenceConfig.getModelConfig());
    }

    protected Map<String, Object> extractFeatures(IN input) {
        if (input instanceof Map) {
            @SuppressWarnings("unchecked")
            Map<String, Object> features = (Map<String, Object>) input;
            return features;
        }
        throw new UnsupportedOperationException("Input must be Map or implement feature extraction");
    }

    @SuppressWarnings("unchecked")
    protected OUT transformResult(IN input, InferenceResult result) {
        return (OUT) result;
    }
}