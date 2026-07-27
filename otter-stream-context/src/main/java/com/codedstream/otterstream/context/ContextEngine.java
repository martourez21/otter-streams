package com.codedstream.otterstream.context;

import com.codedstream.otterstream.context.spi.ContextProvider;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Assembles context from every configured {@link ContextProvider} in parallel before inference
 * — the entry point for {@code otter-stream-context}.
 *
 * <h2>Built for concurrency and scale, specifically</h2>
 * <ul>
 *   <li><b>Dedicated executor, never the shared common pool.</b> Every provider fetch runs on a
 *       sized, named thread pool created for this engine — not
 *       {@link CompletableFuture#supplyAsync(java.util.function.Supplier)}'s default
 *       {@link java.util.concurrent.ForkJoinPool#commonPool()}. This is the exact bug fixed in
 *       {@code AsyncModelInferenceFunction} (see {@code PERFORMANCE.md}): a blocking network
 *       call on the shared JVM-wide pool can starve unrelated work elsewhere in the same
 *       process under load. Under many concurrent users, this matters far more than it did for
 *       a single async function, since every inference request now fans out to N provider
 *       calls instead of one.</li>
 *   <li><b>Per-provider timeout, not a global one.</b> One slow provider (a Redis instance under
 *       load, a flaky REST API) degrades to a partial {@link Context} — its
 *       {@link ContextResult#succeeded()} is false, timestamped, with a reason — rather than
 *       blocking the whole assembly or, worse, silently waiting forever. Every other provider's
 *       result is still returned on time.</li>
 *   <li><b>Bounded caching, not unbounded memoization.</b> The optional {@link ContextCache} is
 *       Caffeine-backed with a hard maximum size — see its class Javadoc for why that matters
 *       under high entity/session cardinality.</li>
 * </ul>
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * ContextEngine engine = ContextEngine.builder()
 *         .provider(new FeatureProviderContextAdapter(redisFeatureProvider))
 *         .provider(new ConversationMemoryProvider(50, Duration.ofHours(2)))
 *         .provider(pineconeContextProvider)
 *         .cache(new ContextCache(100_000, 30))
 *         .parallelism(64)
 *         .perProviderTimeout(Duration.ofMillis(300))
 *         .build();
 *
 * Context context = engine.assemble(userId, Map.of("query", userMessage, "topK", 5));
 * Map<String, Object> modelInput = context.flatten();
 * InferenceResult result = runtime.infer("rag-model", modelInput);
 * }</pre>
 *
 * @since 0.1.0
 */
public class ContextEngine implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(ContextEngine.class);

    private final List<ContextProvider> providers;
    private final ExecutorService executor;
    private final long perProviderTimeoutMillis;
    private final ContextCache cache;

    private ContextEngine(Builder builder) {
        this.providers = List.copyOf(builder.providers);
        this.perProviderTimeoutMillis = builder.perProviderTimeout.toMillis();
        this.cache = builder.cache;

        AtomicInteger threadCounter = new AtomicInteger();
        ThreadFactory threadFactory = runnable -> {
            Thread t = new Thread(runnable, "otter-context-engine-" + threadCounter.incrementAndGet());
            t.setDaemon(true);
            return t;
        };
        this.executor = Executors.newFixedThreadPool(builder.parallelism, threadFactory);
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Assembles context for {@code key}, fanning out to every configured provider in parallel.
     * Serves from {@link ContextCache} first if one is configured and holds a fresh entry.
     *
     * @param key     the entity/session/query key to assemble context for
     * @param request extra request-scoped parameters passed through to every provider
     * @return the assembled context — check {@link Context#isComplete()} if partial results
     *         (one or more providers failed/timed out) matter for your use case
     */
    public Context assemble(String key, Map<String, Object> request) {
        Objects.requireNonNull(key, "key cannot be null");
        Map<String, Object> safeRequest = request != null ? request : Map.of();

        if (cache != null) {
            Context cached = cache.get(key);
            if (cached != null) {
                return new Context(cached, true);
            }
        }

        List<CompletableFuture<ContextResult>> futures = new ArrayList<>(providers.size());
        for (ContextProvider provider : providers) {
            futures.add(fetchWithTimeout(provider, key, safeRequest));
        }

        List<ContextResult> results = new ArrayList<>(futures.size());
        for (CompletableFuture<ContextResult> future : futures) {
            results.add(future.join());
        }

        Context context = new Context(key, results, System.currentTimeMillis(), false);
        if (cache != null) {
            cache.put(key, context);
        }
        return context;
    }

    private CompletableFuture<ContextResult> fetchWithTimeout(ContextProvider provider, String key, Map<String, Object> request) {
        long start = System.nanoTime();
        return CompletableFuture
                .supplyAsync(() -> {
                    try {
                        Map<String, Object> data = provider.fetch(key, request);
                        return ContextResult.success(provider.getProviderId(), data, elapsedMicros(start));
                    } catch (Exception e) {
                        LOG.debug("Context provider '{}' failed for key '{}': {}", provider.getProviderId(), key, e.getMessage());
                        return ContextResult.failure(provider.getProviderId(), elapsedMicros(start), e.getMessage());
                    }
                }, executor)
                .orTimeout(perProviderTimeoutMillis, TimeUnit.MILLISECONDS)
                .exceptionally(throwable -> {
                    String reason = throwable instanceof TimeoutException
                            ? "timed out after " + perProviderTimeoutMillis + "ms"
                            : throwable.getMessage();
                    LOG.debug("Context provider '{}' failed for key '{}': {}", provider.getProviderId(), key, reason);
                    return ContextResult.failure(provider.getProviderId(), elapsedMicros(start), reason);
                });
    }

    private static long elapsedMicros(long startNanos) {
        return (System.nanoTime() - startNanos) / 1000;
    }

    public ContextCache getCache() {
        return cache;
    }

    /** Shuts down the dedicated executor. The engine should not be reused afterward. */
    @Override
    public void close() {
        executor.shutdown();
        try {
            if (!executor.awaitTermination(10, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            executor.shutdownNow();
            Thread.currentThread().interrupt();
        }
    }

    public static class Builder {
        private final List<ContextProvider> providers = new ArrayList<>();
        private int parallelism = 32;
        private Duration perProviderTimeout = Duration.ofMillis(500);
        private ContextCache cache;

        public Builder provider(ContextProvider provider) {
            providers.add(Objects.requireNonNull(provider));
            return this;
        }

        /** Size of the dedicated thread pool backing every {@code assemble()} call. Default 32 — tune to your expected concurrent request count times provider count. */
        public Builder parallelism(int parallelism) {
            this.parallelism = parallelism;
            return this;
        }

        /** Per-provider fetch timeout. Default 500ms. A slow provider past this contributes a failed {@link ContextResult}, not a blocked request. */
        public Builder perProviderTimeout(Duration timeout) {
            this.perProviderTimeout = Objects.requireNonNull(timeout);
            return this;
        }

        /** Optional. Without one, every {@code assemble()} call fans out fresh — fine for low QPS or highly dynamic context, wasteful otherwise. */
        public Builder cache(ContextCache cache) {
            this.cache = cache;
            return this;
        }

        public ContextEngine build() {
            if (providers.isEmpty()) {
                throw new IllegalStateException("ContextEngine requires at least one provider");
            }
            return new ContextEngine(this);
        }
    }
}
