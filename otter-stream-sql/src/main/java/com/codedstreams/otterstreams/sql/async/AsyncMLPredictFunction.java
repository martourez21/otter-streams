package com.codedstreams.otterstreams.sql.async;

import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstreams.otterstreams.sql.loader.ModelCache;
import org.apache.flink.api.common.functions.OpenContext;
import org.apache.flink.streaming.api.functions.async.ResultFuture;
import org.apache.flink.streaming.api.functions.async.RichAsyncFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

/**
 * Async function for non-blocking inference.
 * Uses ModelCache singleton for thread-safe model management.
 */
public class AsyncMLPredictFunction extends RichAsyncFunction<Map<String, Object>, Double> {
    private static final Logger LOG = LoggerFactory.getLogger(AsyncMLPredictFunction.class);

    private final String modelName;
    private transient ExecutorService executor;
    private transient ModelCache modelCache;

    public AsyncMLPredictFunction(String modelName) {
        this.modelName = modelName;
    }

    @Override
    public void open(OpenContext openContext) throws Exception {
        this.modelCache = ModelCache.getInstance();
        this.executor = Executors.newFixedThreadPool(10);
        LOG.info("AsyncMLPredictFunction initialized for model: {}", modelName);
    }

    @Override
    public void asyncInvoke(Map<String, Object> input, ResultFuture<Double> resultFuture) {
        CompletableFuture.supplyAsync(() -> {
            try {
                // Get engine from cache
                InferenceEngine<?> engine = modelCache.getEngine(modelName);
                if (engine == null) {
                    LOG.warn("Model engine not found in cache for: {}", modelName);
                    return null;
                }

                InferenceResult result = engine.infer(input);
                if (result.isSuccess()) {
                    Object pred = result.getOutputs().values().iterator().next();
                    return ((Number) pred).doubleValue();
                } else {
                    LOG.warn("Inference failed for model: {}, error: {}",
                            modelName, result.getErrorMessage());
                    return null;
                }
            } catch (Exception e) {
                LOG.error("Async inference failed for model: {}", modelName, e);
                return null;
            }
        }, executor).thenAccept(result -> {
            if (result != null) {
                resultFuture.complete(Collections.singleton(result));
            } else {
                resultFuture.completeExceptionally(new RuntimeException("Inference failed for model: " + modelName));
            }
        });
    }

    @Override
    public void close() throws Exception {
        if (executor != null) {
            executor.shutdown();
        }
        LOG.info("AsyncMLPredictFunction closed for model: {}", modelName);
    }
}