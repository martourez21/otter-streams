package com.codedstream.otterstream.remote.vertex;


import com.fasterxml.jackson.databind.ObjectMapper;
import com.flinkml.inference.config.InferenceConfig;
import com.flinkml.inference.engine.InferenceEngine;
import com.flinkml.inference.engine.ModelMetadata;
import com.flinkml.inference.model.InferenceResult;
import com.google.cloud.aiplatform.v1.EndpointName;
import com.google.cloud.aiplatform.v1.PredictRequest;
import com.google.cloud.aiplatform.v1.PredictResponse;
import com.google.cloud.aiplatform.v1.PredictionServiceClient;
import com.google.cloud.aiplatform.v1.PredictionServiceSettings;
import com.google.protobuf.Value;
import com.google.protobuf.util.JsonFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.util.*;
import java.util.concurrent.CompletableFuture;
import java.util.stream.Collectors;

/**
 * Google Cloud Vertex AI inference client.
 * Supports Vertex AI prediction endpoints.
 */
public class VertexAIInferenceClient implements InferenceEngine<Map<String, Object>, Map<String, Object>> {
    private static final Logger LOG = LoggerFactory.getLogger(VertexAIInferenceClient.class);
    private static final long serialVersionUID = 1L;

    private final InferenceConfig config;
    private transient PredictionServiceClient predictionClient;
    private transient ObjectMapper objectMapper;
    private transient ModelMetadata metadata;
    private transient EndpointName endpointName;

    public VertexAIInferenceClient(InferenceConfig config) {
        this.config = config;
    }

    @Override
    public void initialize() throws Exception {
        LOG.info("Initializing Vertex AI inference client for endpoint: {}", config.getEndpointUrl());

        try {
            // Parse endpoint name: projects/{project}/locations/{location}/endpoints/{endpoint}
            // Or accept direct endpoint name
            String endpoint = config.getEndpointUrl();

            if (endpoint.startsWith("projects/")) {
                String[] parts = endpoint.split("/");
                if (parts.length >= 6) {
                    String project = parts[1];
                    String location = parts[3];
                    String endpointId = parts[5];
                    endpointName = EndpointName.of(project, location, endpointId);
                } else {
                    throw new IllegalArgumentException("Invalid Vertex AI endpoint format: " + endpoint);
                }
            } else {
                // Extract from config options
                String project = (String) config.getFrameworkOptions().get("project_id");
                String location = (String) config.getFrameworkOptions().getOrDefault("location", "us-central1");
                endpointName = EndpointName.of(project, location, endpoint);
            }

            // Create prediction client
            PredictionServiceSettings settings = PredictionServiceSettings.newBuilder()
                    .setEndpoint(String.format("%s-aiplatform.googleapis.com:443", endpointName.getLocation()))
                    .build();

            predictionClient = PredictionServiceClient.create(settings);
            objectMapper = new ObjectMapper();

            metadata = new ModelMetadata(
                    endpointName.toString(),
                    config.getModelVersion(),
                    "Google Vertex AI",
                    new ArrayList<>(),
                    new ArrayList<>()
            );

            LOG.info("Vertex AI client initialized successfully");

        } catch (IOException e) {
            LOG.error("Failed to initialize Vertex AI client", e);
            throw new Exception("Vertex AI initialization failed", e);
        }
    }

    @Override
    public InferenceResult<Map<String, Object>> predict(Map<String, Object> input) throws Exception {
        long startTime = System.nanoTime();

        try {
            // Convert input to Value proto
            String jsonInput = objectMapper.writeValueAsString(input);
            Value.Builder valueBuilder = Value.newBuilder();
            JsonFormat.parser().merge(jsonInput, valueBuilder);

            // Create predict request
            PredictRequest request = PredictRequest.newBuilder()
                    .setEndpoint(endpointName.toString())
                    .addInstances(valueBuilder.build())
                    .build();

            // Make prediction
            PredictResponse response = predictionClient.predict(request);

            // Parse response
            if (response.getPredictionsCount() > 0) {
                Value prediction = response.getPredictions(0);
                String jsonOutput = JsonFormat.printer().print(prediction);
                Map<String, Object> output = objectMapper.readValue(jsonOutput, Map.class);

                long endTime = System.nanoTime();
                long inferenceTimeMs = (endTime - startTime) / 1_000_000;

                return InferenceResult.<Map<String, Object>>builder()
                        .prediction(output)
                        .inferenceTimeMs(inferenceTimeMs)
                        .modelVersion(config.getModelVersion())
                        .success(true)
                        .addMetadata("deployed_model_id", response.getDeployedModelId())
                        .build();
            } else {
                throw new Exception("No predictions returned from Vertex AI");
            }

        } catch (Exception e) {
            LOG.error("Vertex AI inference failed", e);
            throw e;
        }
    }

    @Override
    public CompletableFuture<InferenceResult<Map<String, Object>>> predictAsync(
            Map<String, Object> input) {

        return CompletableFuture.supplyAsync(() -> {
            try {
                return predict(input);
            } catch (Exception e) {
                throw new RuntimeException("Async inference failed", e);
            }
        });
    }

    @Override
    public List<InferenceResult<Map<String, Object>>> predictBatch(
            List<Map<String, Object>> inputs) throws Exception {

        long startTime = System.nanoTime();

        try {
            // Convert all inputs to Value protos
            List<Value> instances = new ArrayList<>();
            for (Map<String, Object> input : inputs) {
                String jsonInput = objectMapper.writeValueAsString(input);
                Value.Builder valueBuilder = Value.newBuilder();
                JsonFormat.parser().merge(jsonInput, valueBuilder);
                instances.add(valueBuilder.build());
            }

            // Create batch predict request
            PredictRequest request = PredictRequest.newBuilder()
                    .setEndpoint(endpointName.toString())
                    .addAllInstances(instances)
                    .build();

            // Make batch prediction
            PredictResponse response = predictionClient.predict(request);

            long endTime = System.nanoTime();
            long totalTimeMs = (endTime - startTime) / 1_000_000;
            long avgTimeMs = response.getPredictionsCount() > 0 ?
                    totalTimeMs / response.getPredictionsCount() : 0;

            // Parse all predictions
            List<InferenceResult<Map<String, Object>>> results = new ArrayList<>();
            for (Value prediction : response.getPredictionsList()) {
                String jsonOutput = JsonFormat.printer().print(prediction);
                Map<String, Object> output = objectMapper.readValue(jsonOutput, Map.class);

                results.add(InferenceResult.<Map<String, Object>>builder()
                        .prediction(output)
                        .inferenceTimeMs(avgTimeMs)
                        .modelVersion(config.getModelVersion())
                        .success(true)
                        .build());
            }

            return results;

        } catch (Exception e) {
            LOG.error("Vertex AI batch inference failed", e);
            throw e;
        }
    }

    @Override
    public CompletableFuture<List<InferenceResult<Map<String, Object>>>> predictBatchAsync(
            List<Map<String, Object>> inputs) {

        return CompletableFuture.supplyAsync(() -> {
            try {
                return predictBatch(inputs);
            } catch (Exception e) {
                throw new RuntimeException("Async batch inference failed", e);
            }
        });
    }

    @Override
    public void warmup(int iterations) throws Exception {
        LOG.info("Warming up Vertex AI endpoint with {} iterations", iterations);

        Map<String, Object> dummyInput = Map.of("features", new float[]{0.0f});

        for (int i = 0; i < iterations; i++) {
            try {
                predict(dummyInput);
            } catch (Exception e) {
                LOG.warn("Warmup iteration {} failed: {}", i, e.getMessage());
            }
        }

        LOG.info("Endpoint warmup completed");
    }

    @Override
    public boolean isHealthy() {
        return predictionClient != null && !predictionClient.isShutdown();
    }

    @Override
    public ModelMetadata getMetadata() {
        return metadata;
    }

    @Override
    public void close() {
        LOG.info("Closing Vertex AI inference client");

        if (predictionClient != null) {
            predictionClient.close();
        }
    }
}
