package com.codedstreams.otterstreams.sql.async;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Supplier;
import java.util.List;
import java.util.stream.Collectors;

/**
 * {@code AsyncInferenceExecutor} provides a simple abstraction for executing tasks asynchronously
 * using a fixed-size thread pool. It is designed for running inference or other long-running tasks
 * concurrently without blocking the main thread.
 *
 * <p>This class manages an internal {@link ExecutorService} and allows submission of tasks
 * as {@link Supplier} instances. Results of tasks are returned as {@link CompletableFuture}
 * objects, enabling asynchronous processing and easy composition of multiple tasks.</p>
 *
 * <p>Typical usage:</p>
 * <pre>{@code
 * AsyncInferenceExecutor executor = new AsyncInferenceExecutor(4);
 *
 * // Submit a single task
 * CompletableFuture<ResultType> future = executor.submit(() -> performInference(input));
 * future.thenAccept(result -> handleResult(result))
 *       .exceptionally(ex -> { handleError(ex); return null; });
 *
 * // Submit multiple tasks and combine results
 * List<Supplier<ResultType>> tasks = List.of(
 *     () -> performInference(input1),
 *     () -> performInference(input2),
 *     () -> performInference(input3)
 * );
 *
 * List<CompletableFuture<ResultType>> futures = tasks.stream()
 *     .map(task -> executor.submit(task)
 *                         .exceptionally(ex -> {
 *                             handleError(ex); // Handle exception for this task
 *                             return null; // Or return a default value
 *                         }))
 *     .collect(Collectors.toList());
 *
 * CompletableFuture<Void> allDone = CompletableFuture.allOf(
 *     futures.toArray(new CompletableFuture[0])
 * );
 *
 * allDone.thenRun(() -> {
 *     List<ResultType> results = futures.stream()
 *         .map(CompletableFuture::join) // join will return null for failed tasks
 *         .collect(Collectors.toList());
 *     handleResults(results);
 * });
 *
 * executor.shutdown();
 * }</pre>
 *
 * <p>Notes:</p>
 * <ul>
 *   <li>Exceptions thrown by individual tasks can be handled using {@link CompletableFuture#exceptionally}.</li>
 *   <li>Using {@link CompletableFuture#join} will propagate unchecked exceptions; returning a default value in {@code exceptionally} is a safe way to avoid stopping the whole batch.</li>
 *   <li>The {@link #shutdown()} method should be called when no further tasks need to be submitted.</li>
 * </ul>
 */
public class AsyncInferenceExecutor {

    /**
     * The internal thread pool used to execute tasks asynchronously.
     */
    private final ExecutorService executor;

    /**
     * The number of threads in the fixed thread pool.
     */
    private final int poolSize;

    /**
     * Creates a new {@code AsyncInferenceExecutor} with a fixed-size thread pool.
     *
     * @param poolSize the number of threads to maintain in the pool
     */
    public AsyncInferenceExecutor(int poolSize) {
        this.poolSize = poolSize;
        this.executor = Executors.newFixedThreadPool(poolSize);
    }

    /**
     * Submits a task for asynchronous execution.
     *
     * <p>The task is executed on the internal thread pool, and the result is returned as
     * a {@link CompletableFuture}. Any exception thrown during execution will be captured
     * by the {@link CompletableFuture} and can be handled using
     * {@link CompletableFuture#exceptionally(java.util.function.Function)}.</p>
     *
     * @param <T>  the type of the result produced by the task
     * @param task a {@link Supplier} representing the task to execute
     * @return a {@link CompletableFuture} representing the pending completion of the task
     */
    public <T> CompletableFuture<T> submit(Supplier<T> task) {
        return CompletableFuture.supplyAsync(task, executor);
    }

    /**
     * Shuts down the executor, preventing new tasks from being submitted.
     *
     * <p>Previously submitted tasks continue to execute. Use this method when you no longer
     * need to submit any tasks to the executor. For immediate shutdown of running tasks,
     * consider using {@link ExecutorService#shutdownNow()} on a custom implementation.</p>
     */
    public void shutdown() {
        executor.shutdown();
    }
}
