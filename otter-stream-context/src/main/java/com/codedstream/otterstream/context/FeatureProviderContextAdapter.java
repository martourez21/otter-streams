package com.codedstream.otterstream.context;

import com.codedstream.otterstream.context.spi.ContextProvider;
import com.codedstream.otterstream.runtime.spi.FeatureProvider;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Adapts an existing {@code FeatureProvider} (Redis/JDBC/Feast from earlier work) into a
 * {@link ContextProvider}, so none of that client code needs duplicating for the Context
 * Engine. The request map's {@code "featureNames"} entry (a {@code List<String>}, if present)
 * is passed through as the underlying provider's requested feature list; omit it to fetch
 * everything the provider returns by default (matching each {@code FeatureProvider}'s own
 * "empty/null list" convention).
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * FeatureProvider redis = new RedisFeatureProvider("localhost", 6379, "features:user:");
 * ContextProvider asContext = new FeatureProviderContextAdapter(redis);
 *
 * ContextEngine engine = ContextEngine.builder().provider(asContext).build();
 * }</pre>
 *
 * @since 0.1.0
 */
public final class FeatureProviderContextAdapter implements ContextProvider {

    private final FeatureProvider delegate;

    public FeatureProviderContextAdapter(FeatureProvider delegate) {
        this.delegate = Objects.requireNonNull(delegate, "delegate cannot be null");
    }

    @Override
    public String getProviderId() {
        return delegate.getProviderId();
    }

    @Override
    @SuppressWarnings("unchecked")
    public Map<String, Object> fetch(String key, Map<String, Object> request) throws Exception {
        List<String> featureNames = List.of();
        Object requested = request != null ? request.get("featureNames") : null;
        if (requested instanceof List<?> list) {
            featureNames = (List<String>) list;
        }
        return delegate.fetch(key, featureNames);
    }
}
