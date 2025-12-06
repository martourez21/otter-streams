package com.codedstream.otterstream.remote.http;

import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.fasterxml.jackson.databind.ObjectMapper;
import okhttp3.*;
import java.util.Map;
import java.util.concurrent.TimeUnit;

public class HttpInferenceClient extends RemoteInferenceEngine {

    private OkHttpClient httpClient;
    private ObjectMapper objectMapper;
    private MediaType JSON_MEDIA_TYPE = MediaType.parse("application/json");

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

    @Override
    public void close() throws InferenceException {
        if (httpClient != null) {
            httpClient.dispatcher().executorService().shutdown();
            httpClient.connectionPool().evictAll();
        }
        super.close();
    }
}
