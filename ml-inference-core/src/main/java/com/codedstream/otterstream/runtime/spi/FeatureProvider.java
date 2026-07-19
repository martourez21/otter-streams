package com.codedstream.otterstream.runtime.spi;

import java.util.List;
import java.util.Map;

/**
 * Service Provider Interface (SPI) for fetching feature values used as inference inputs.
 *
 * <p><b>Status:</b> this interface defines the extension point only. It is introduced now
 * (alongside {@link InferenceProvider} and {@link ModelRegistry}) so the runtime's plugin
 * surface is stable and complete, but concrete implementations (Redis, JDBC, Feast, REST,
 * Iceberg) are intentionally out of scope until the Feature Store Integrations milestone.
 * {@link com.codedstream.otterstream.runtime.OtterRuntime} does not yet wire this SPI into
 * the inference path.
 *
 * <h2>Intended shape:</h2>
 * <pre>{@code
 * public class RedisFeatureProvider implements FeatureProvider {
 *     public String getProviderId() { return "redis"; }
 *     public Map<String, Object> fetch(String entityId, List<String> featureNames) {
 *         // look up featureNames for entityId in Redis
 *     }
 * }
 * }</pre>
 *
 * @since 0.1.0 (interface only; implementations land in a later milestone)
 */
public interface FeatureProvider {

    /**
     * A stable, unique identifier for this provider (e.g. {@code "redis"}, {@code "jdbc"}).
     *
     * @return provider id
     */
    String getProviderId();

    /**
     * Fetches the requested feature values for a given entity.
     *
     * @param entityId     the entity to fetch features for (e.g. a user or account id)
     * @param featureNames the names of the features to retrieve
     * @return a map of feature name to feature value
     * @throws Exception if the underlying feature store is unreachable or the lookup fails
     */
    Map<String, Object> fetch(String entityId, List<String> featureNames) throws Exception;
}
