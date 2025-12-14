package com.codedstream.otterstream.remote.http;

import com.codedstream.otterstream.inference.config.InferenceConfig;
import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.inference.model.ModelMetadata;
import com.codedstream.otterstream.remote.RemoteInferenceEngine;
import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.*;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * HTTP-based remote inference client for REST API model endpoints.
 *
 * <p>This engine sends inference requests to remote HTTP endpoints using REST APIs.
 * It supports authentication headers, configurable timeouts, and JSON-based request/response
 * formats. Ideal for integrating with model serving frameworks like TensorFlow Serving REST API,
 * TorchServe, or custom model endpoints.
 *
 * <h2>Supported Features:</h2>
 * <ul>
 *   <li><b>REST API Integration:</b> Standard HTTP POST with JSON payloads</li>
 *   <li><b>Authentication:</b> Custom headers via {@link com.codedstream.otterstream.inference.config.AuthConfig}</li>
 *   <li><b>Timeout Configuration:</b> Connect, read, and write timeouts</li>
 *   <li><b>Connection Validation:</b> HEAD requests to verify endpoint availability</li>
 *   <li><b>JSON Serialization:</b> Automatic Java Map ↔ JSON conversion</li>
 * </ul>
 *
 * <h2>Configuration Example:</h2>
 * <pre>{@code
 * ModelConfig config = ModelConfig.builder()
 *     .modelId("http-model")
 *     .endpointUrl("https://api.modelserver.com/v1/predict")
 *     .modelOption("connectTimeout", "10")
 *     .modelOption("readTimeout", "30")
 *     .modelOption("writeTimeout", "30")
 *     .authConfig(AuthConfig.builder()
 *         .addHeader("Authorization", "Bearer token123")
 *         .addHeader("X-API-Key", "key456")
 *         .build())
 *     .build();
 *
 * HttpInferenceClient client = new HttpInferenceClient();
 * client.initialize(config);
 * }</pre>
 *
 * <h2>Inference Example:</h2>
 * <pre>{@code
 * Map<String, Object> inputs = new HashMap<>();
 * inputs.put("feature1", 0.5);
 * inputs.put("feature2", "text");
 * inputs.put("feature3", new float[]{0.1f, 0.2f, 0.3f});
 *
 * InferenceResult result = client.infer(inputs);
 * Map<String, Object> predictions = result.getOutputs();
 * }</pre>
 *
 * <h2>HTTP Request Details:</h2>
 * <table border="1">
 *   <tr><th>Method</th><th>POST</th></tr>
 *   <tr><td>Content-Type</td><td>application/json</td></tr>
 *   <tr><td>Timeout</td><td>30 seconds (configurable)</td></tr>
 *   <tr><td>Authentication</td><td>Custom headers</td></tr>
 * </table>
 *
 * <h2>Error Handling:</h2>
 * <ul>
 *   <li>Non-2xx HTTP responses throw {@link InferenceException}</li>
 *   <li>Connection timeouts throw {@link InferenceException} with root cause</li>
 *   <li>JSON serialization errors are wrapped in {@link InferenceException}</li>
 * </ul>
 *
 * <h2>Performance Considerations:</h2>
 * <ul>
 *   <li>OkHttp connection pooling for HTTP/1.1 and HTTP/2</li>
 *   <li>Configurable timeouts to prevent hanging requests</li>
 *   <li>Single-threaded by default (use async patterns for high throughput)</li>
 * </ul>
 *
 * <h2>Thread Safety:</h2>
 * <p>{@link OkHttpClient} is thread-safe and can be shared across threads.
 * However, each {@link HttpInferenceClient} instance should be used from a single
 * thread or synchronized externally.
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 * @see RemoteInferenceEngine
 * @see okhttp3.OkHttpClient
 * @see com.fasterxml.jackson.databind.ObjectMapper
 */
public class HttpInferenceClient extends RemoteInferenceEngine {

    private OkHttpClient httpClient;
    private ObjectMapper objectMapper;

    private InferenceConfig inferenceConfig;

    private ModelMetadata metadata;

    private MediaType JSON_MEDIA_TYPE = MediaType.parse("application/json");

    /**
     * Initializes the HTTP inference client with connection configuration.
     *
     * <p>Configures {@link OkHttpClient} with timeout settings from model options:
     * <ul>
     *   <li><b>connectTimeout:</b> Connection establishment timeout (default: 30s)</li>
     *   <li><b>readTimeout:</b> Response read timeout (default: 30s)</li>
     *   <li><b>writeTimeout:</b> Request write timeout (default: 30s)</li>
     * </ul>
     *
     * <p>Also initializes {@link ObjectMapper} for JSON serialization.
     *
     * @param config model configuration containing endpoint URL and timeout options
     * @throws InferenceException if initialization fails
     */
    @Override
    public void initialize(ModelConfig config) throws InferenceException {
        super.initialize(config);

        OkHttpClient.Builder clientBuilder = new OkHttpClient.Builder()
                .connectTimeout(config.getModelOptions().containsKey("connectTimeout") ?
                        Long.parseLong(config.getModelOptions().get("connectTimeout").toString()) : 30, TimeUnit.SECONDS)
                .readTimeout(config.getModelOptions().containsKey("readTimeout") ?
                        Long.parseLong(config.getModelOptions().get("readTimeout").toString()) : 30, TimeUnit.SECONDS)
                .writeTimeout(config.getModelOptions().containsKey("writeTimeout") ?
                        Long.parseLong(config.getModelOptions().get("writeTimeout").toString()) : 30, TimeUnit.SECONDS);

        this.httpClient = clientBuilder.build();
        this.objectMapper = new ObjectMapper();
    }

    /**
     * Sends inference request to remote HTTP endpoint.
     *
     * <p>Request flow:
     * <ol>
     *   <li>Serialize inputs to JSON using {@link ObjectMapper}</li>
     *   <li>Create HTTP POST request with JSON body</li>
     *   <li>Add authentication headers from {@link com.codedstream.otterstream.inference.config.AuthConfig}</li>
     *   <li>Execute request with configured timeouts</li>
     *   <li>Parse JSON response back to Map</li>
     *   <li>Return {@link InferenceResult} with timing information</li>
     * </ol>
     *
     * @param inputs map of input names to values (must be JSON-serializable)
     * @return inference result containing outputs and request timing
     * @throws InferenceException if HTTP request fails, times out, or response parsing fails
     */
    @Override
    public InferenceResult infer(Map<String, Object> inputs) throws InferenceException {
        try {
            long startTime = System.currentTimeMillis();

            String jsonBody = objectMapper.writeValueAsString(inputs);

            Request.Builder requestBuilder = new Request.Builder()
                    .url(endpointUrl)
                    .post(RequestBody.create(jsonBody, JSON_MEDIA_TYPE));

            // Add authentication headers
            if (modelConfig.getAuthConfig() != null) {
                Map<String, String> headers = modelConfig.getAuthConfig().getHeaders();
                for (Map.Entry<String, String> header : headers.entrySet()) {
                    requestBuilder.addHeader(header.getKey(), header.getValue());
                }
            }

            Request request = requestBuilder.build();

            try (Response response = httpClient.newCall(request).execute()) {
                if (!response.isSuccessful()) {
                    throw new InferenceException("HTTP request failed: " + response.code() + " - " + response.message());
                }

                String responseBody = response.body().string();
                @SuppressWarnings("unchecked")
                Map<String, Object> outputs = objectMapper.readValue(responseBody, Map.class);

                long endTime = System.currentTimeMillis();
                return new InferenceResult(outputs, endTime - startTime, modelConfig.getModelId());
            }
        } catch (Exception e) {
            throw new InferenceException("HTTP inference failed", e);
        }
    }

    /**
     * Validates connection to remote endpoint using HTTP HEAD request.
     *
     * <p>Sends a lightweight HEAD request to verify:
     * <ul>
     *   <li>Endpoint is reachable</li>
     *   <li>Endpoint responds to HTTP requests</li>
     *   <li>Authentication works (if configured)</li>
     * </ul>
     *
     * @return true if HEAD request succeeds (2xx status)
     * @throws InferenceException if validation request fails (network error, timeout)
     */
    @Override
    public boolean validateConnection() throws InferenceException {
        try {
            Request request = new Request.Builder()
                    .url(endpointUrl)
                    .head()
                    .build();

            try (Response response = httpClient.newCall(request).execute()) {
                return response.isSuccessful();
            }
        } catch (Exception e) {
            throw new InferenceException("Connection validation failed", e);
        }
    }

    /**
     * Closes the HTTP client and releases resources.
     *
     * <p>Shuts down OkHttp dispatcher thread pool and evicts all connections.
     * Always call this method when finished with the client to release threads
     * and connections.
     *
     * @throws InferenceException if resource cleanup fails
     */
    @Override
    public void close() throws InferenceException {
        if (httpClient != null) {
            httpClient.dispatcher().executorService().shutdown();
            httpClient.connectionPool().evictAll();
        }
        super.close();
    }

    /**
     * Gets metadata about the remote model.
     *
     * @return model metadata (currently returns null, override for implementation)
     */
    @Override
    public ModelMetadata getMetadata() {
        return metadata;
    }

    /**
     * Gets the model configuration.
     *
     * @return the model configuration used for initialization
     */
    @Override
    public ModelConfig getModelConfig() {
        return inferenceConfig.getModelConfig();
    }
}