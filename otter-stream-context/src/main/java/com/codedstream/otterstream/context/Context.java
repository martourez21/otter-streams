package com.codedstream.otterstream.context;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The assembled output of {@link ContextEngine#assemble}: every configured provider's result,
 * namespaced by provider id so two providers can never silently clobber each other's keys, plus
 * a flattened merged view for the common case where callers just want one map to feed into
 * feature extraction.
 *
 * @since 0.1.0
 */
public final class Context {

    private final String key;
    private final Map<String, ContextResult> resultsByProvider;
    private final long assembledAtMillis;
    private final boolean fromCache;

    Context(String key, List<ContextResult> results, long assembledAtMillis, boolean fromCache) {
        this.key = Objects.requireNonNull(key, "key cannot be null");
        Map<String, ContextResult> map = new LinkedHashMap<>();
        for (ContextResult result : results) {
            map.put(result.providerId(), result);
        }
        this.resultsByProvider = Map.copyOf(map);
        this.assembledAtMillis = assembledAtMillis;
        this.fromCache = fromCache;
    }

    /** Copy constructor used by {@link ContextEngine} to return a cache-hit Context with {@code fromCache=true} without mutating the cached instance. */
    Context(Context original, boolean fromCache) {
        this.key = original.key;
        this.resultsByProvider = original.resultsByProvider;
        this.assembledAtMillis = original.assembledAtMillis;
        this.fromCache = fromCache;
    }

    /** @return the entity/session/query key this context was assembled for */
    public String getKey() {
        return key;
    }

    /** @return one provider's result, or null if that provider wasn't part of this assembly */
    public ContextResult getProviderResult(String providerId) {
        return resultsByProvider.get(providerId);
    }

    public Map<String, ContextResult> getAllProviderResults() {
        return resultsByProvider;
    }

    /**
     * Flattens every successful provider's data into one map, namespaced as
     * {@code "<providerId>.<key>"} to avoid collisions between providers that happen to use the
     * same field name (e.g. two providers both returning a field called {@code "score"}).
     */
    public Map<String, Object> flatten() {
        Map<String, Object> flattened = new LinkedHashMap<>();
        for (ContextResult result : resultsByProvider.values()) {
            if (!result.succeeded()) continue;
            for (Map.Entry<String, Object> entry : result.data().entrySet()) {
                flattened.put(result.providerId() + "." + entry.getKey(), entry.getValue());
            }
        }
        return flattened;
    }

    /** @return true if every configured provider succeeded */
    public boolean isComplete() {
        return resultsByProvider.values().stream().allMatch(ContextResult::succeeded);
    }

    public long getAssembledAtMillis() {
        return assembledAtMillis;
    }

    /** @return true if this Context was served from {@link ContextCache} rather than freshly assembled */
    public boolean isFromCache() {
        return fromCache;
    }
}
