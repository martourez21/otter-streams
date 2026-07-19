package com.codedstream.otterstream.redis;

import com.codedstream.otterstream.runtime.spi.FeatureProvider;
import java.io.Closeable;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import redis.clients.jedis.Jedis;
import redis.clients.jedis.JedisPool;
import redis.clients.jedis.JedisPoolConfig;

/**
 * {@link FeatureProvider} backed by Redis, where each entity's features are stored as fields in
 * a Redis hash keyed by (an optional prefix plus) the entity id — the common "online feature
 * store" pattern (e.g. a table synced from Feast, Tecton, or a custom feature pipeline).
 *
 * <h2>Expected data shape:</h2>
 * <pre>
 * HSET features:user:42 age 31 country "US" ltv_score 0.87
 * </pre>
 *
 * <h2>Usage:</h2>
 * <pre>{@code
 * RedisFeatureProvider features = new RedisFeatureProvider("localhost", 6379, "features:user:");
 * Map<String, Object> values = features.fetch("42", List.of("age", "country", "ltv_score"));
 * // values = {"age": "31", "country": "US", "ltv_score": "0.87"}
 * }</pre>
 *
 * <p>Redis hash values are always strings; callers are responsible for any further parsing
 * (numeric conversion, etc.) required by the downstream inference input schema.
 *
 * <p>Thread-safe: backed by a {@link JedisPool}, one connection borrowed per {@link #fetch}
 * call.
 *
 * @since 0.1.0
 */
public class RedisFeatureProvider implements FeatureProvider, Closeable {

    private final JedisPool pool;
    private final String keyPrefix;
    private final boolean ownsPool;

    /**
     * Connects to a standalone Redis instance with default pool settings.
     *
     * @param host Redis host
     * @param port Redis port
     */
    public RedisFeatureProvider(String host, int port) {
        this(host, port, "");
    }

    /**
     * Connects to a standalone Redis instance with default pool settings.
     *
     * @param host      Redis host
     * @param port      Redis port
     * @param keyPrefix prefix prepended to every entity id to form the Redis hash key
     *                  (e.g. {@code "features:user:"}); pass {@code ""} for no prefix
     */
    public RedisFeatureProvider(String host, int port, String keyPrefix) {
        this(new JedisPool(new JedisPoolConfig(), host, port), keyPrefix, true);
    }

    /**
     * Uses a caller-supplied, already-configured {@link JedisPool} (for TLS, auth, cluster mode,
     * custom pool sizing, etc.). This provider will not close the pool on {@link #close()};
     * the caller retains ownership.
     *
     * @param pool      pre-configured Jedis connection pool
     * @param keyPrefix prefix prepended to every entity id to form the Redis hash key
     */
    public RedisFeatureProvider(JedisPool pool, String keyPrefix) {
        this(pool, keyPrefix, false);
    }

    private RedisFeatureProvider(JedisPool pool, String keyPrefix, boolean ownsPool) {
        this.pool = Objects.requireNonNull(pool, "pool cannot be null");
        this.keyPrefix = keyPrefix != null ? keyPrefix : "";
        this.ownsPool = ownsPool;
    }

    @Override
    public String getProviderId() {
        return "redis";
    }

    /**
     * Fetches feature values from the Redis hash for {@code keyPrefix + entityId}.
     *
     * @param entityId     the entity to fetch features for
     * @param featureNames the hash fields to retrieve; if null or empty, every field in the
     *                     hash is returned via {@code HGETALL}
     * @return a map of feature name to raw string value from Redis; a requested field that
     *         doesn't exist on the hash is omitted (not present as a null-valued entry)
     */
    @Override
    public Map<String, Object> fetch(String entityId, List<String> featureNames) throws Exception {
        Objects.requireNonNull(entityId, "entityId cannot be null");
        String key = keyPrefix + entityId;

        try (Jedis jedis = pool.getResource()) {
            Map<String, Object> result = new LinkedHashMap<>();

            if (featureNames == null || featureNames.isEmpty()) {
                Map<String, String> all = jedis.hgetAll(key);
                result.putAll(all);
                return result;
            }

            String[] fields = featureNames.toArray(new String[0]);
            List<String> values = jedis.hmget(key, fields);
            for (int i = 0; i < fields.length; i++) {
                String value = values.get(i);
                if (value != null) {
                    result.put(fields[i], value);
                }
            }
            return result;
        }
    }

    /**
     * Closes the underlying {@link JedisPool} — only if this provider created it itself
     * (i.e. via the {@code (host, port[, keyPrefix])} constructors). Pools supplied via
     * {@link #RedisFeatureProvider(JedisPool, String)} are left open for the caller to manage.
     */
    @Override
    public void close() {
        if (ownsPool) {
            pool.close();
        }
    }
}
