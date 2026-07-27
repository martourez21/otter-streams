package com.codedstream.otterstream.context.spi;

import java.util.Map;

/**
 * A source of context to assemble before inference — the generalized parent abstraction behind
 * {@code otter-stream-context}. Where
 * {@code com.codedstream.otterstream.runtime.spi.FeatureProvider} is a single-shot
 * entity→feature lookup, {@code ContextProvider} is the broader shape: session/conversation
 * memory, vector search results, cached state, or a feature lookup are all just different
 * implementations of "given a request, produce some context." An existing
 * {@code FeatureProvider} adapts trivially into one — see {@code FeatureProviderContextAdapter}
 * — so none of the Redis/JDBC/Feast work from earlier needs to be duplicated.
 *
 * <p>Implementations must be safe for concurrent use — {@link ContextEngine} calls every
 * configured provider in parallel for a single request, so a provider blocking one caller must
 * not block another.
 *
 * @since 0.1.0
 * @see ContextEngine
 */
public interface ContextProvider {

    /** A stable, unique identifier — becomes the namespace key in the assembled {@link Context}. */
    String getProviderId();

    /**
     * Produces context for one request.
     *
     * @param key     the entity/session/query key this request is for (a user id, a session id,
     *                a raw query string for vector search — meaning is provider-specific)
     * @param request additional request-scoped parameters a provider might need (e.g. how many
     *                vector matches to return, which conversation turns to include)
     * @return the context this provider contributes, as a flat map
     * @throws Exception if the underlying lookup/computation fails
     */
    Map<String, Object> fetch(String key, Map<String, Object> request) throws Exception;
}
