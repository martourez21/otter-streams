package com.codedstream.otterstream.context;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import java.util.concurrent.TimeUnit;

/**
 * Bounded, TTL-based cache for assembled {@link Context}s, keyed by the request key (entity id,
 * session id, etc.) — the "Context Cache" layer: avoid re-assembling context (parallel
 * provider fan-out, each with its own network round trip) on every request when the underlying
 * context hasn't changed since the last one.
 *
 * <p>Uses Caffeine, the same library {@code ModelCache} already uses elsewhere in this project
 * — same battle-tested bounded-size + time-based eviction, not a hand-rolled cache. Deliberately
 * bounded by {@code maximumSize}: unlike a naively-unbounded {@code Map}, this can't be grown
 * into an OOM by a workload with high/unbounded key cardinality (many distinct users/sessions)
 * — exactly the safety property {@code SlidingWindowFeatureProvider}'s own cardinality caveat
 * calls out as needing caller-driven pruning; here it's built in from the start via Caffeine's
 * own eviction.
 *
 * @since 0.1.0
 */
public class ContextCache {

    private final Cache<String, Context> cache;

    /**
     * @param maximumSize             maximum number of distinct keys (entities/sessions) to cache
     * @param expireAfterWriteSeconds how long a cached context stays valid before it's considered
     *                                stale and must be reassembled
     */
    public ContextCache(long maximumSize, long expireAfterWriteSeconds) {
        this.cache = Caffeine.newBuilder()
                .maximumSize(maximumSize)
                .expireAfterWrite(expireAfterWriteSeconds, TimeUnit.SECONDS)
                .build();
    }

    Context get(String key) {
        return cache.getIfPresent(key);
    }

    void put(String key, Context context) {
        cache.put(key, context);
    }

    /** Forces a key to be reassembled on next request — call after anything that invalidates cached context (e.g. a known state update). */
    public void invalidate(String key) {
        cache.invalidate(key);
    }

    public void invalidateAll() {
        cache.invalidateAll();
    }

    public long estimatedSize() {
        return cache.estimatedSize();
    }
}
