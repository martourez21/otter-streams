package com.codedstream.otterstream.inference.cache;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import java.util.concurrent.TimeUnit;

/**
 * Thread-safe LRU cache for storing ML model predictions and inference results.
 *
 * <p>This cache uses Caffeine for high-performance caching with automatic eviction
 * based on size and time constraints. Ideal for caching inference results in
 * streaming applications to reduce latency and computational overhead.
 *
 * <h2>Features:</h2>
 * <ul>
 *   <li>Thread-safe operations for concurrent Flink streams</li>
 *   <li>Automatic eviction based on LRU policy</li>
 *   <li>Time-based expiration for stale data</li>
 *   <li>Configurable maximum size</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Create cache for 10,000 entries, expiring after 30 minutes
 * ModelCache<String, InferenceResult> cache =
 *     new ModelCache<>(10000, 30);
 *
 * // Store prediction
 * cache.put("input-hash-123", predictionResult);
 *
 * // Retrieve cached prediction
 * InferenceResult cached = cache.get("input-hash-123");
 * if (cached != null) {
 *     // Use cached result
 * }
 * }</pre>
 *
 * @param <K> the type of cache keys (typically String hashes)
 * @param <V> the type of cached values (typically InferenceResult)
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 */
public class ModelCache<K, V> {
    private final Cache<K, V> cache;

    /**
     * Creates a new model cache with specified size and expiration policy.
     *
     * @param maximumSize maximum number of entries to store
     * @param expireAfterWriteMinutes time in minutes before entries expire
     */
    public ModelCache(long maximumSize, long expireAfterWriteMinutes) {
        this.cache = Caffeine.newBuilder()
                .maximumSize(maximumSize)
                .expireAfterWrite(expireAfterWriteMinutes, TimeUnit.MINUTES)
                .build();
    }

    /**
     * Retrieves a cached value if present.
     *
     * @param key the cache key
     * @return the cached value, or null if not found or expired
     */
    public V get(K key) {
        return cache.getIfPresent(key);
    }

    /**
     * Stores a value in the cache.
     *
     * @param key the cache key
     * @param value the value to cache
     */
    public void put(K key, V value) {
        cache.put(key, value);
    }

    /**
     * Removes a specific entry from the cache.
     *
     * @param key the key to invalidate
     */
    public void invalidate(K key) {
        cache.invalidate(key);
    }

    /**
     * Clears all entries from the cache.
     */
    public void invalidateAll() {
        cache.invalidateAll();
    }

    /**
     * Returns the approximate current size of the cache.
     *
     * @return estimated number of entries in cache
     */
    public long size() {
        return cache.estimatedSize();
    }
}