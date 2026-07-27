package com.codedstream.otterstream.context.vector;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * {@link VectorSearchProvider} backed by Pinecone's REST query API. Uses the JDK's own
 * {@link HttpClient} and {@code jackson-databind} (already transitively available via
 * {@code ml-inference-core}) — no Pinecone SDK dependency, matching the same
 * zero-new-dependency reasoning as {@code RestDecisionEngineConnector} in
 * {@code otter-stream-rules}.
 *
 * <p><b>Index host, stated plainly:</b> Pinecone's query endpoint host is per-index
 * ({@code https://{index}-{project}.svc.{environment}.pinecone.io/query} for pod-based indexes,
 * a different host shape for serverless indexes) — pass your index's actual query host as
 * {@code indexHost}; this class doesn't derive it for you, since that mapping is account/plan
 * -specific and not something to guess at. For Milvus or OpenSearch instead, implement
 * {@link VectorSearchProvider} directly against their own query APIs — this class is Pinecone
 * -specific by design, the same "one concrete example, generic interface for the rest" approach
 * used for {@code DecisionEngineConnector}'s REST connector.
 *
 * <h2>Usage</h2>
 * <pre>{@code
 * VectorSearchProvider pinecone = new PineconeContextProvider(
 *         "pinecone-docs",
 *         "https://my-index-abc123.svc.us-east-1-aws.pinecone.io",
 *         System.getenv("PINECONE_API_KEY"),
 *         "default");
 *
 * List<VectorMatch> matches = pinecone.search(queryEmbedding, 5);
 * }</pre>
 *
 * @since 0.1.0
 */
public class PineconeContextProvider implements VectorSearchProvider {

    private final String providerId;
    private final String queryUrl;
    private final String apiKey;
    private final String namespace;
    private final HttpClient httpClient;
    private final ObjectMapper mapper = new ObjectMapper();
    private final Duration requestTimeout;

    public PineconeContextProvider(String providerId, String indexHost, String apiKey, String namespace) {
        this(providerId, indexHost, apiKey, namespace, Duration.ofSeconds(5));
    }

    public PineconeContextProvider(
            String providerId, String indexHost, String apiKey, String namespace, Duration requestTimeout) {
        this.providerId = Objects.requireNonNull(providerId, "providerId cannot be null");
        this.queryUrl = Objects.requireNonNull(indexHost, "indexHost cannot be null").replaceAll("/$", "") + "/query";
        this.apiKey = Objects.requireNonNull(apiKey, "apiKey cannot be null");
        this.namespace = namespace;
        this.requestTimeout = requestTimeout;
        this.httpClient = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();
    }

    @Override
    public String getProviderId() {
        return providerId;
    }

    @Override
    @SuppressWarnings("unchecked")
    public List<VectorMatch> search(float[] embedding, int topK) throws Exception {
        Map<String, Object> body = new LinkedHashMap<>();
        List<Float> vector = new ArrayList<>(embedding.length);
        for (float f : embedding) vector.add(f);
        body.put("vector", vector);
        body.put("topK", topK);
        body.put("includeMetadata", true);
        if (namespace != null && !namespace.isBlank()) {
            body.put("namespace", namespace);
        }

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(queryUrl))
                .timeout(requestTimeout)
                .header("Content-Type", "application/json")
                .header("Api-Key", apiKey)
                .POST(HttpRequest.BodyPublishers.ofByteArray(mapper.writeValueAsBytes(body)))
                .build();

        HttpResponse<byte[]> response = httpClient.send(request, HttpResponse.BodyHandlers.ofByteArray());
        if (response.statusCode() / 100 != 2) {
            throw new IOException("Pinecone query failed: HTTP " + response.statusCode());
        }

        JsonNode root = mapper.readTree(response.body());
        JsonNode matchesNode = root.path("matches");
        List<VectorMatch> matches = new ArrayList<>();
        for (JsonNode matchNode : matchesNode) {
            String id = matchNode.path("id").asText();
            double score = matchNode.path("score").asDouble();
            Map<String, Object> metadata = matchNode.has("metadata")
                    ? mapper.convertValue(matchNode.get("metadata"), Map.class)
                    : Map.of();
            matches.add(new VectorMatch(id, score, metadata));
        }
        return matches;
    }
}
