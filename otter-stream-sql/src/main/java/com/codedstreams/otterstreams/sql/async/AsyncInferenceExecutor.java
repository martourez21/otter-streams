package com.codedstreams.otterstreams.sql.async;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Supplier;

/**
 * Manages async inference execution with thread pool.
 */
public class AsyncInferenceExecutor {
    private final ExecutorService executor;
    private final int poolSize;

    public AsyncInferenceExecutor(int poolSize) {
        this.poolSize = poolSize;
        this.executor = Executors.newFixedThreadPool(poolSize);
    }

    public <T> CompletableFuture<T> submit(Supplier<T> task) {
        return CompletableFuture.supplyAsync(task, executor);
    }

    public void shutdown() {
        executor.shutdown();
    }
}

