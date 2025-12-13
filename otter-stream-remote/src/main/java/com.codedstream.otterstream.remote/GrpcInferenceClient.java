package com.codedstream.otterstream.remote;

import com.codedstream.otterstream.inference.config.InferenceConfig;
import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.inference.model.ModelFormat;
import com.codedstream.otterstream.inference.model.ModelMetadata;
import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.StatusRuntimeException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.TimeUnit;

/**
 * gRPC inference client for TensorFlow Serving, Triton, etc.
 * NOTE: This is a template — actual gRPC stubs must be generated from proto files.
 */
public class GrpcInferenceClient
        implements InferenceEngine<ManagedChannel> {

    private static final Logger LOG =
            LoggerFactory.getLogger(GrpcInferenceClient.class);

    private final InferenceConfig inferenceConfig;

    private ManagedChannel channel;
    private ModelMetadata metadata;
    private boolean initialized = false;

    public GrpcInferenceClient(InferenceConfig inferenceConfig) {
        this.inferenceConfig = inferenceConfig;
    }

    // ---------------------------------------------------------------------
    // Initialization
    // ---------------------------------------------------------------------

    @Override
    public void initialize(ModelConfig modelConfig) throws InferenceException {
        LOG.info("Initializing gRPC inference client");

        try {
            Map<String, Object> options = inferenceConfig.getEngineOptions();
            String endpoint = (String) options.get("endpoint");

            if (endpoint == null) {
                throw new IllegalArgumentException("gRPC endpoint not configured");
            }

            String[] parts = endpoint.split(":");
            String host = parts[0];
            int port = parts.length > 1 ? Integer.parseInt(parts[1]) : 8500;

            this.channel = ManagedChannelBuilder
                    .forAddress(host, port)
                    .usePlaintext() // 🔐 use TLS in production
                    .build();

            this.metadata = new ModelMetadata(
                    modelConfig.getModelName(),
                    modelConfig.getModelVersion(),
                    ModelFormat.REMOTE_GRPC,
                    Map.of(), // input schema (optional)
                    Map.of(), // output schema (optional)
                    -1L,      // model size unknown
                    System.currentTimeMillis()
            );

            this.initialized = true;
            LOG.info("gRPC inference client initialized");

        } catch (Exception e) {
            throw new InferenceException("Failed to initialize gRPC inference client", e);
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
            /*
             * TODO: Replace this placeholder with real gRPC call:
             *
             * PredictRequest request = ...
             * PredictResponse response = blockingStub
             *      .withDeadlineAfter(inferenceConfig.getTimeoutMs(), TimeUnit.MILLISECONDS)
             *      .predict(request);
             */

            // Placeholder output
            Map<String, Object> output = Map.of("prediction", 0.5);

            return new InferenceResult(
                    output,
                    System.currentTimeMillis() - start,
                    metadata.getModelVersion()
            );

        } catch (StatusRuntimeException e) {
            throw new InferenceException(
                    "gRPC inference failed: " + e.getStatus(), e);
        }
    }

    @Override
    public InferenceResult inferBatch(Map<String, Object>[] batchInputs)
            throws InferenceException {

        ensureInitialized();

        long start = System.currentTimeMillis();

        try {
            Map<String, Object> aggregated = new HashMap<>();

            for (int i = 0; i < batchInputs.length; i++) {
                // Sequential fallback; replace with real batch call if supported
                InferenceResult result = infer(batchInputs[i]);
                aggregated.put("result_" + i, result.getOutputs());
            }

            return new InferenceResult(
                    aggregated,
                    System.currentTimeMillis() - start,
                    metadata.getModelVersion()
            );

        } catch (Exception e) {
            throw new InferenceException("gRPC batch inference failed", e);
        }
    }

    // ---------------------------------------------------------------------
    // Engine Info
    // ---------------------------------------------------------------------

    @Override
    public boolean isReady() {
        return initialized && channel != null && !channel.isShutdown();
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
                true,   // batch supported (engine-level)
                inferenceConfig.getBatchSize(),
                false   // async depends on stub implementation
        );
    }

    // ---------------------------------------------------------------------
    // Shutdown
    // ---------------------------------------------------------------------

    @Override
    public void close() {
        LOG.info("Closing gRPC inference client");

        if (channel != null) {
            try {
                channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                LOG.warn("Channel shutdown interrupted", e);
                channel.shutdownNow();
            }
        }
        initialized = false;
    }

    // ---------------------------------------------------------------------
    // Internal
    // ---------------------------------------------------------------------

    private void ensureInitialized() throws InferenceException {
        if (!initialized || channel == null) {
            throw new InferenceException("gRPC inference client not initialized");
        }
    }
}
