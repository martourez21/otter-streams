package com.codedstream.otterstream.feast;

import com.codedstream.otterstream.runtime.spi.FeatureProvider;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import okhttp3.MediaType;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;
import okhttp3.ResponseBody;

/**
 * {@link FeatureProvider} backed by a <a href="https://docs.feast.dev/reference/feature-servers/python-feature-server">
 * Feast HTTP feature server</a>'s online-serving endpoint ({@code POST /get-online-features}).
 *
 * <p>This targets Feast's REST feature server, not its gRPC serving API — REST keeps this
 * module dependency-free beyond an HTTP client and JSON, at the cost of slightly higher latency
 * than gRPC. If you need the gRPC path, implement {@link FeatureProvider} directly against
 * Feast's generated Java stubs; the interface is intentionally minimal to make that easy.
 *
 * <h2>Request/response shape</h2>
 * Sends:
 * <pre>{@code
 * POST {baseUrl}/get-online-features
 * {
 *   "features": ["driver_hourly_stats:conv_rate", "driver_hourly_stats:acc_rate"],
 *   "entities": { "driver_id": ["1001"] }
 * }
 * }</pre>
 * and reads Feast's {@code metadata.feature_names[]} / {@code results[].values[]} response shape
 * into a flat {@code Map<String, Object>}. The exact response schema can vary slightly across
 * Feast versions/deployments — if yours differs, this class is a small, single-purpose starting
 * point to fork from.
 *
 * @since 0.1.0
 */
public class FeastFeatureProvider implements FeatureProvider {

    private static final MediaType JSON = MediaType.get("application/json; charset=utf-8");

    private final OkHttpClient client;
    private final ObjectMapper mapper = new ObjectMapper();
    private final String baseUrl;
    private final String entityColumnName;

    /**
     * @param baseUrl          base URL of the Feast HTTP feature server, e.g.
     *                         {@code "http://localhost:6566"} (no trailing slash)
     * @param entityColumnName the entity join key Feast expects, e.g. {@code "driver_id"}
     */
    public FeastFeatureProvider(String baseUrl, String entityColumnName) {
        this(baseUrl, entityColumnName, new OkHttpClient.Builder()
                .connectTimeout(5, TimeUnit.SECONDS)
                .readTimeout(10, TimeUnit.SECONDS)
                .build());
    }

    /**
     * @param baseUrl          base URL of the Feast HTTP feature server (no trailing slash)
     * @param entityColumnName the entity join key Feast expects, e.g. {@code "driver_id"}
     * @param client           a caller-configured {@link OkHttpClient} (custom timeouts,
     *                         interceptors for auth headers, connection pooling, etc.)
     */
    public FeastFeatureProvider(String baseUrl, String entityColumnName, OkHttpClient client) {
        this.baseUrl = Objects.requireNonNull(baseUrl, "baseUrl cannot be null");
        this.entityColumnName = Objects.requireNonNull(entityColumnName, "entityColumnName cannot be null");
        this.client = Objects.requireNonNull(client, "client cannot be null");
    }

    @Override
    public String getProviderId() {
        return "feast";
    }

    /**
     * @param entityId     the entity id to look up (sent as the single value for
     *                     {@code entityColumnName})
     * @param featureNames fully-qualified Feast feature references, e.g.
     *                     {@code "driver_hourly_stats:conv_rate"}
     * @return a map of feature reference to its returned value
     */
    @Override
    public Map<String, Object> fetch(String entityId, List<String> featureNames) throws Exception {
        Objects.requireNonNull(entityId, "entityId cannot be null");
        if (featureNames == null || featureNames.isEmpty()) {
            return Collections.emptyMap();
        }

        Map<String, Object> requestBody = new LinkedHashMap<>();
        requestBody.put("features", featureNames);
        Map<String, Object> entities = new LinkedHashMap<>();
        entities.put(entityColumnName, List.of(entityId));
        requestBody.put("entities", entities);

        RequestBody body = RequestBody.create(mapper.writeValueAsBytes(requestBody), JSON);
        Request request = new Request.Builder()
                .url(baseUrl + "/get-online-features")
                .post(body)
                .build();

        try (Response response = client.newCall(request).execute()) {
            ResponseBody responseBody = response.body();
            if (!response.isSuccessful() || responseBody == null) {
                throw new IOException("Feast feature server request failed: HTTP " + response.code());
            }
            return parseResponse(responseBody.bytes());
        }
    }

    private Map<String, Object> parseResponse(byte[] bytes) throws IOException {
        JsonNode root = mapper.readTree(bytes);
        JsonNode featureNamesNode = root.path("metadata").path("feature_names");
        JsonNode resultsNode = root.path("results");

        Map<String, Object> out = new LinkedHashMap<>();
        for (int i = 0; i < featureNamesNode.size() && i < resultsNode.size(); i++) {
            String name = featureNamesNode.get(i).asText();
            JsonNode valuesNode = resultsNode.get(i).path("values");
            Object value = null;
            if (valuesNode.isArray() && valuesNode.size() > 0) {
                value = mapper.treeToValue(valuesNode.get(0), Object.class);
            }
            out.put(name, value);
        }
        return out;
    }
}
