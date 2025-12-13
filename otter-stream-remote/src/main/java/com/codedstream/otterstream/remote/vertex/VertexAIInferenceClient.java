package com.codedstream.otterstream.remote.vertex;

import com.codedstream.otterstream.inference.config.InferenceConfig;
import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.inference.model.ModelFormat;
import com.codedstream.otterstream.inference.model.ModelMetadata;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.google.cloud.aiplatform.v1.EndpointName;
import com.google.cloud.aiplatform.v1.PredictRequest;
import com.google.cloud.aiplatform.v1.PredictResponse;
import com.google.cloud.aiplatform.v1.PredictionServiceClient;
import com.google.cloud.aiplatform.v1.PredictionServiceSettings;
import com.google.protobuf.Value;
import com.google.protobuf.util.JsonFormat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * Google Vertex AI remote inference engine.
 */
public class VertexAIInferenceClient
        implements InferenceEngine<PredictionServiceClient> {

    private static final Logger LOG =
            LoggerFactory.getLogger(VertexAIInferenceClient.class);

    private final InferenceConfig inferenceConfig;

    private PredictionServiceClient predictionClient;
    private ObjectMapper objectMapper;
    private ModelMetadata metadata;
    private EndpointName endpointName;
    private boolean initialized = false;

    public VertexAIInferenceClient(InferenceConfig inferenceConfig) {
        this.inferenceConfig = inferenceConfig;
    }

    // ---------------------------------------------------------------------
    // Initialization
    // ---------------------------------------------------------------------

    @Override
    public void initialize(ModelConfig modelConfig) throws InferenceException {
        LOG.info("Initializing Vertex AI inference engine");

        try {
            Map<String, Object> options = inferenceConfig.getEngineOptions();
            String endpoint = (String) options.get("endpoint");

            if (endpoint == null) {
                throw new IllegalArgumentException("Vertex AI endpoint not configured");
            }

            if (endpoint.startsWith("projects/")) {
                String[] parts = endpoint.split("/");
                if (parts.length < 6) {
                    throw new IllegalArgumentException(
                            "Invalid Vertex AI endpoint format: " + endpoint);
                }
                endpointName = EndpointName.of(parts[1], parts[3], parts[5]);
            } else {
                String project = (String) options.get("project_id");
                String location = (String) options.getOrDefault("location", "us-central1");
                endpointName = EndpointName.of(project, location, endpoint);
            }

            PredictionServiceSettings settings =
                    PredictionServiceSettings.newBuilder()
                            .setEndpoint(
                                    endpointName.getLocation()
                                            + "-aiplatform.googleapis.com:443")
                            .build();

            this.predictionClient = PredictionServiceClient.create(settings);
            this.objectMapper = new ObjectMapper();

            this.metadata = new ModelMetadata(
                    modelConfig.getModelName(),
                    modelConfig.getModelVersion(),
                    ModelFormat.VERTEX_AI,
                    Map.of(), // input schema (optional)
                    Map.of(), // output schema (optional)
                    -1L,      // model size unknown
                    System.currentTimeMillis()
            );

            this.initialized = true;
            LOG.info("Vertex AI inference engine initialized");

        } catch (Exception e) {
            throw new InferenceException("Failed to initialize Vertex AI engine", e);
        }
    }

    // ---------------------------------------------------------------------
    // Inference
    // ---------------------------------------------------------------------

    @Override
    public InferenceResult infer(Map<String, Object> inputs)
            throws InferenceException {

        ensureInitialized();

        long start = System.currentTimeMillis();

        try {
            PredictRequest request = PredictRequest.newBuilder()
                    .setEndpoint(endpointName.toString())
                    .addInstances(toValue(inputs))
                    .build();

            PredictResponse response = predictionClient.predict(request);

            if (response.getPredictionsCount() == 0) {
                throw new InferenceException("No predictions returned from Vertex AI");
            }

            Map<String, Object> output =
                    fromValue(response.getPredictions(0));

            return new InferenceResult(
                    output,
                    System.currentTimeMillis() - start,
                    metadata.getModelVersion()
            );

        } catch (Exception e) {
            throw new InferenceException("Vertex AI inference failed", e);
        }
    }

    @Override
    public InferenceResult inferBatch(Map<String, Object>[] batchInputs)
            throws InferenceException {

        ensureInitialized();

        long start = System.currentTimeMillis();

        try {
            List<Value> instances = new ArrayList<>();
            for (Map<String, Object> input : batchInputs) {
                instances.add(toValue(input));
            }

            PredictRequest request = PredictRequest.newBuilder()
                    .setEndpoint(endpointName.toString())
                    .addAllInstances(instances)
                    .build();

            PredictResponse response = predictionClient.predict(request);

            Map<String, Object> aggregated = new HashMap<>();
            int i = 0;
            for (Value prediction : response.getPredictionsList()) {
                aggregated.put("result_" + i++, fromValue(prediction));
            }

            return new InferenceResult(
                    aggregated,
                    System.currentTimeMillis() - start,
                    metadata.getModelVersion()
            );

        } catch (Exception e) {
            throw new InferenceException("Vertex AI batch inference failed", e);
        }
    }

    // ---------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------

    private Value toValue(Map<String, Object> input) throws Exception {
        String json = objectMapper.writeValueAsString(input);
        Value.Builder builder = Value.newBuilder();
        JsonFormat.parser().merge(json, builder);
        return builder.build();
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> fromValue(Value value) throws Exception {
        String json = JsonFormat.printer().print(value);
        return objectMapper.readValue(json, Map.class);
    }

    private void ensureInitialized() throws InferenceException {
        if (!initialized || predictionClient == null) {
            throw new InferenceException("Vertex AI engine not initialized");
        }
    }

    // ---------------------------------------------------------------------
    // Engine Info
    // ---------------------------------------------------------------------

    @Override
    public boolean isReady() {
        return initialized;
    }

    @Override
    public ModelMetadata getMetadata() {
        return metadata;
    }

    @Override
    public ModelConfig getModelConfig() {
        return inferenceConfig.getModelConfig();
    }

    @Override
    public EngineCapabilities getCapabilities() {
        return new EngineCapabilities(
                true,   // remote
                true,   // batch
                inferenceConfig.getBatchSize(),
                true    // async supported by platform
        );
    }


    @Override
    public void close() {
        if (predictionClient != null) {
            predictionClient.close();
        }
        initialized = false;
    }
}
