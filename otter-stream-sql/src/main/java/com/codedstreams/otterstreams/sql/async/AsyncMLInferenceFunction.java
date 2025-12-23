package com.codedstreams.otterstreams.sql.async;

import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstreams.otterstreams.sql.loader.ModelCache;
import org.apache.flink.streaming.api.functions.async.AsyncFunction;
import org.apache.flink.streaming.api.functions.async.ResultFuture;

import java.util.Collections;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.function.Function;

/**
 * Asynchronous ML inference function for Flink streams.
 *
 * @param <T> Input event type
 * @param <R> Output enriched event type
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 */
public class AsyncMLInferenceFunction<T, R> implements AsyncFunction<T, R> {

    private final String modelName;
    private final Function<T, Map<String, Object>> featureExtractor;
    private final Function<InferenceResult, R> resultMapper;

    private transient ModelCache modelCache;

    public AsyncMLInferenceFunction(
            String modelName,
            Function<T, Map<String, Object>> featureExtractor,
            Function<InferenceResult, R> resultMapper) {

        this.modelName = modelName;
        this.featureExtractor = featureExtractor;
        this.resultMapper = resultMapper;
    }

    @Override
    public void asyncInvoke(T input, ResultFuture<R> resultFuture) {
        if (modelCache == null) {
            modelCache = ModelCache.getInstance();
        }

        InferenceEngine<?> engine = modelCache.getEngine(modelName);
        if (engine == null) {
            resultFuture.complete(Collections.emptyList());
            return;
        }

        CompletableFuture
                .supplyAsync(() -> {
                    try {
                        return engine.infer(featureExtractor.apply(input));
                    } catch (InferenceException e) {
                        throw new RuntimeException(e);
                    }
                })
                .thenAccept(result -> {
                    if (result.isSuccess()) {
                        resultFuture.complete(
                                Collections.singleton(resultMapper.apply(result))
                        );
                    } else {
                        resultFuture.complete(Collections.emptyList());
                    }
                })
                .exceptionally(ex -> {
                    resultFuture.complete(Collections.emptyList());
                    return null;
                });
    }
}
