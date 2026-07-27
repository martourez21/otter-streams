package com.codedstream.otterstream.rules.connector;

import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.rules.model.Decision;
import com.codedstream.otterstream.rules.model.RuleEvaluationException;
import com.codedstream.otterstream.rules.spi.DecisionEngineConnector;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A configurable {@link DecisionEngineConnector} that talks to any external decision engine
 * exposing a REST endpoint — Red Hat Decision Manager / KIE Server, Camunda DMN, IBM ODM
 * Decision Server, or an in-house decision service. See {@link DecisionEngineConnector}'s
 * Javadoc for why this is one configurable connector rather than one class per vendor.
 *
 * <p>Uses the JDK's built-in {@link HttpClient} (stable since Java 11) rather than a bundled
 * HTTP library — deliberately zero new runtime dependencies for this connector, consistent
 * with {@code otter-stream-rules}'s minimal-footprint design; JSON (de)serialization reuses
 * {@code jackson-databind}, already a transitive dependency via {@code ml-inference-core}, not
 * a new one this module introduces.
 *
 * <h2>Request/response mapping</h2>
 * The request body sent is:
 * <pre>{@code
 * {
 *   "modelId": "...",
 *   "outputs": { ...InferenceResult.getOutputs()... },
 *   "context": { ...caller-supplied context... }
 * }
 * }</pre>
 * The response is expected to be a JSON object with at least a {@code flag} field; {@code
 * category}, {@code confidence}, and {@code matchedRuleIds} are read if present. Engines with a
 * different native response shape (e.g. raw KIE Server XML/JSON envelopes) should front this
 * connector with a small adapter/gateway that normalizes to the shape above — kept deliberately
 * out of this connector so it doesn't accumulate vendor-specific response parsing.
 *
 * @since 0.1.0
 */
public final class RestDecisionEngineConnector implements DecisionEngineConnector {

    private final String connectorId;
    private final URI endpoint;
    private final String bearerToken;
    private final HttpClient httpClient;
    private final ObjectMapper mapper = new ObjectMapper();
    private final Duration requestTimeout;

    public RestDecisionEngineConnector(String connectorId, URI endpoint, String bearerToken) {
        this(connectorId, endpoint, bearerToken, Duration.ofSeconds(5));
    }

    public RestDecisionEngineConnector(String connectorId, URI endpoint, String bearerToken, Duration requestTimeout) {
        this.connectorId = Objects.requireNonNull(connectorId, "connectorId cannot be null");
        this.endpoint = Objects.requireNonNull(endpoint, "endpoint cannot be null");
        this.bearerToken = bearerToken;
        this.requestTimeout = requestTimeout;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(5))
                .build();
    }

    @Override
    public String getConnectorId() {
        return connectorId;
    }

    @Override
    public Decision evaluate(InferenceResult inferenceResult, Map<String, Object> context) throws RuleEvaluationException {
        try {
            Map<String, Object> body = new LinkedHashMap<>();
            body.put("modelId", inferenceResult.getModelId());
            body.put("outputs", inferenceResult.getOutputs());
            body.put("context", context != null ? context : Map.of());

            HttpRequest.Builder requestBuilder = HttpRequest.newBuilder()
                    .uri(endpoint)
                    .timeout(requestTimeout)
                    .header("Content-Type", "application/json")
                    .POST(HttpRequest.BodyPublishers.ofByteArray(mapper.writeValueAsBytes(body)));
            if (bearerToken != null && !bearerToken.isBlank()) {
                requestBuilder.header("Authorization", "Bearer " + bearerToken);
            }

            HttpResponse<byte[]> response =
                    httpClient.send(requestBuilder.build(), HttpResponse.BodyHandlers.ofByteArray());

            if (response.statusCode() / 100 != 2) {
                throw new RuleEvaluationException(
                        "Decision engine '" + connectorId + "' returned HTTP " + response.statusCode());
            }
            return parseDecision(response.body());
        } catch (IOException | InterruptedException e) {
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            throw new RuleEvaluationException("Failed to reach decision engine '" + connectorId + "'", e);
        }
    }

    private Decision parseDecision(byte[] responseBody) throws RuleEvaluationException {
        try {
            JsonNode root = mapper.readTree(responseBody);
            if (!root.has("flag")) {
                throw new RuleEvaluationException(
                        "Decision engine '" + connectorId + "' response missing required 'flag' field");
            }
            String flag = root.get("flag").asText();
            String category = root.has("category") && !root.get("category").isNull()
                    ? root.get("category").asText()
                    : null;
            double confidence = root.has("confidence") ? root.get("confidence").asDouble() : 0.0;

            List<String> matchedRuleIds = List.of();
            if (root.has("matchedRuleIds") && root.get("matchedRuleIds").isArray()) {
                List<String> ids = new java.util.ArrayList<>();
                root.get("matchedRuleIds").forEach(node -> ids.add(node.asText()));
                matchedRuleIds = ids;
            }

            return new Decision(
                    flag, category, confidence, matchedRuleIds, System.currentTimeMillis(),
                    Map.of("connectorId", connectorId));
        } catch (IOException e) {
            throw new RuleEvaluationException(
                    "Failed to parse decision engine '" + connectorId + "' response as JSON", e);
        }
    }
}
