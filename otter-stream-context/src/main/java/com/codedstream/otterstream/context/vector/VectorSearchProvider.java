package com.codedstream.otterstream.context.vector;

import com.codedstream.otterstream.context.spi.ContextProvider;
import java.util.List;
import java.util.Map;

/**
 * A {@link ContextProvider} specialized for vector/embedding search (Pinecone, Milvus,
 * OpenSearch's vector fields, or any other similarity-search backend) — retrieval-augmented
 * generation's core dependency: given a query embedding, find the most relevant stored
 * documents/chunks to include as context.
 *
 * <p>{@link #search} is the typed entry point; {@link #fetch} (inherited from
 * {@link ContextProvider}, so this still composes with {@link
 * com.codedstream.otterstream.context.ContextEngine} alongside every other provider type) reads
 * the query embedding from the request map's {@code "embedding"} key
 * ({@code List<Double>}/{@code float[]}) and {@code "topK"} key (int, default 5), and returns
 * {@code {"matches": List<VectorMatch>}}.
 *
 * @since 0.1.0
 */
public interface VectorSearchProvider extends ContextProvider {

    /**
     * @param embedding the query embedding vector
     * @param topK      how many matches to return
     * @return the top-K matches, highest score first
     * @throws Exception if the underlying vector store call fails
     */
    List<VectorMatch> search(float[] embedding, int topK) throws Exception;

    @Override
    @SuppressWarnings("unchecked")
    default Map<String, Object> fetch(String key, Map<String, Object> request) throws Exception {
        Object embeddingObj = request.get("embedding");
        float[] embedding = toFloatArray(embeddingObj);
        int topK = request.get("topK") instanceof Integer i ? i : 5;
        return Map.of("matches", search(embedding, topK));
    }

    private static float[] toFloatArray(Object embeddingObj) {
        if (embeddingObj instanceof float[] arr) {
            return arr;
        }
        if (embeddingObj instanceof List<?> list) {
            float[] arr = new float[list.size()];
            for (int i = 0; i < list.size(); i++) {
                arr[i] = ((Number) list.get(i)).floatValue();
            }
            return arr;
        }
        throw new IllegalArgumentException(
                "VectorSearchProvider requires request[\"embedding\"] to be a float[] or List<Number>, got: "
                        + (embeddingObj == null ? "null" : embeddingObj.getClass()));
    }
}
