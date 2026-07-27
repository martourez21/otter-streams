package com.codedstream.otterstream.runtime.feature;

import com.codedstream.otterstream.runtime.spi.FeatureProvider;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Supplier;

/**
 * Wraps any {@link FeatureProvider}, stamping every fetch result with a {@link FeatureVersion}
 * under the reserved {@code "_featureVersion"} key — the "feature versioning" piece of the
 * Feature Store Integration roadmap item. See {@link FeatureVersion}'s Javadoc for exactly what
 * this does and doesn't cover (fetch-time stamping, not point-in-time historical correctness).
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * // Static version tag (e.g. a deployed feature-pipeline release)
 * FeatureProvider versioned = new VersioningFeatureProvider(redis, () -> "v3");
 *
 * Map<String, Object> values = versioned.fetch("42", List.of("age", "country"));
 * FeatureVersion version = (FeatureVersion) values.get(VersioningFeatureProvider.VERSION_KEY);
 * }</pre>
 *
 * @since 0.1.0
 */
public class VersioningFeatureProvider implements FeatureProvider {

    /** Reserved key under which the {@link FeatureVersion} is stamped into the fetch result. */
    public static final String VERSION_KEY = "_featureVersion";

    private final FeatureProvider delegate;
    private final Supplier<String> versionTagSupplier;

    /**
     * @param delegate           the provider to wrap
     * @param versionTagSupplier called once per {@code fetch()} to obtain the current version
     *                           tag — a {@code Supplier} rather than a fixed string so the tag
     *                           can change over time (e.g. read from a config value, an
     *                           environment variable, or a deployment marker) without
     *                           reconstructing this wrapper
     */
    public VersioningFeatureProvider(FeatureProvider delegate, Supplier<String> versionTagSupplier) {
        this.delegate = Objects.requireNonNull(delegate, "delegate cannot be null");
        this.versionTagSupplier = Objects.requireNonNull(versionTagSupplier, "versionTagSupplier cannot be null");
    }

    @Override
    public String getProviderId() {
        return delegate.getProviderId();
    }

    @Override
    public Map<String, Object> fetch(String entityId, List<String> featureNames) throws Exception {
        Map<String, Object> result = delegate.fetch(entityId, featureNames);
        Map<String, Object> stamped = new LinkedHashMap<>(result);
        stamped.put(VERSION_KEY, new FeatureVersion(versionTagSupplier.get(), System.currentTimeMillis()));
        return stamped;
    }
}
