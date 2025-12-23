package com.codedstreams.otterstreams.sql.runtime;

import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.engine.LocalInferenceEngine;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.inference.model.ModelMetadata;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.proto.framework.GraphDef;

import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.HashMap;
import java.util.Map;

/**
 * TensorFlow GraphDef (frozen graph) inference engine.
 */
public class TensorFlowGraphDefEngine extends LocalInferenceEngine<Graph> {
    private static final Logger LOG = LoggerFactory.getLogger(TensorFlowGraphDefEngine.class);

    private Session session;

    @Override
    protected void loadModelDirectly(ModelConfig config) throws InferenceException {
        try {
            String modelPath = config.getModelPath();
            LOG.info("Loading TensorFlow GraphDef from: {}", modelPath);

            byte[] graphDef = Files.readAllBytes(Paths.get(modelPath));

            this.loadedModel = new Graph();
            this.loadedModel.importGraphDef(GraphDef.parseFrom(graphDef));
            this.session = new Session(loadedModel);

            LOG.info("GraphDef loaded successfully");
        } catch (Exception e) {
            throw new InferenceException("Failed to load GraphDef", e);
        }
    }

    @Override
    public InferenceResult infer(Map<String, Object> inputs) throws InferenceException {
        long startTime = System.currentTimeMillis();

        try {
            // Similar to SavedModel but with different tensor names
            Map<String, Object> results = new HashMap<>();
            results.put("output", 0.5); // Placeholder

            long endTime = System.currentTimeMillis();
            return new InferenceResult(results, endTime - startTime, modelConfig.getModelId());
        } catch (Exception e) {
            throw new InferenceException("Inference failed", e);
        }
    }

    @Override
    public InferenceResult inferBatch(Map<String, Object>[] batchInputs) throws InferenceException {
        return infer(batchInputs[0]); // Simplified
    }

    @Override
    public EngineCapabilities getCapabilities() {
        return new EngineCapabilities(true, false, 128, false);
    }

    @Override
    public ModelMetadata getMetadata() {
        return ModelMetadata.builder()
                .modelName(modelConfig.getModelId())
                .format(modelConfig.getFormat())
                .build();
    }

    @Override
    public void close() throws InferenceException {
        if (session != null) session.close();
        if (loadedModel != null) loadedModel.close();
        super.close();
    }
}

