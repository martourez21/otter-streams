package com.codedstream.otterstream.context.vector;

import java.util.Map;

/**
 * One scored match from a vector search.
 *
 * @param id       the matched document/vector's id
 * @param score    similarity score — higher-is-better convention (cosine similarity, dot
 *                 product), not a distance metric; a provider using a distance-based index
 *                 should convert before returning
 * @param metadata whatever metadata the vector store returns alongside the match (source text,
 *                 document title, chunk index — provider-specific)
 * @since 0.1.0
 */
public record VectorMatch(String id, double score, Map<String, Object> metadata) {
    public VectorMatch {
        metadata = metadata == null ? Map.of() : Map.copyOf(metadata);
    }
}

