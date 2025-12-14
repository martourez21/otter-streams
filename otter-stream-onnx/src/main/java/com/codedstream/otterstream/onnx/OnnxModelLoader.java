package com.codedstream.otterstream.onnx;

import ai.onnxruntime.*;
import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.exception.ModelLoadException;
import com.codedstream.otterstream.inference.model.ModelFormat;
import com.codedstream.otterstream.inference.model.ModelLoader;
import com.codedstream.otterstream.inference.model.ModelMetadata;
import ai.onnxruntime.NodeInfo;

import ai.onnxruntime.TensorInfo;

import java.io.InputStream;
import java.util.HashMap;
import java.util.Map;

public class OnnxModelLoader implements ModelLoader<InferenceSession> {

    @Override
    public InferenceSession loadModel(ModelConfig config) throws ModelLoadException {
        try {
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions opts = new OrtSession.SessionOptions();

            configureSessionOptions(opts, config);

            return new InferenceSession(config.getModelPath(), opts, env);
        } catch (Exception e) {
            throw new ModelLoadException("Failed to load ONNX model from path: " + config.getModelPath(), e);
        }
    }

    @Override
    public InferenceSession loadModel(InputStream inputStream, ModelConfig config) throws ModelLoadException {
        try {
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            OrtSession.SessionOptions opts = new OrtSession.SessionOptions();

            configureSessionOptions(opts, config);

            byte[] bytes = inputStream.readAllBytes();
            return new InferenceSession(bytes, opts, env);
        } catch (Exception e) {
            throw new ModelLoadException("Failed to load ONNX model from stream", e);
        }
    }

    @Override
    public boolean validateModel(InferenceSession model, ModelConfig config) {
        try {
            return !model.getInputMetadata().isEmpty() && !model.getOutputMetadata().isEmpty();
        } catch (Exception e) {
            return false;
        }
    }

    @Override
    public ModelFormat[] getSupportedFormats() {
        return new ModelFormat[]{ModelFormat.ONNX};
    }

    @Override
    public ModelMetadata getModelMetadata(InferenceSession model) {
        try {
            Map<String, Object> inputSchema = extractSchema(model.getInputMetadata());
            Map<String, Object> outputSchema = extractSchema(model.getOutputMetadata());

            return new ModelMetadata(
                    "onnx_model",
                    "1.0",
                    ModelFormat.ONNX,
                    inputSchema,
                    outputSchema,
                    0,
                    System.currentTimeMillis()
            );

        } catch (Exception e) {
            return new ModelMetadata(
                    "onnx_model",
                    "1.0",
                    ModelFormat.ONNX,
                    Map.of(),
                    Map.of(),
                    0,
                    System.currentTimeMillis()
            );
        }
    }

    private void configureSessionOptions(OrtSession.SessionOptions opts, ModelConfig config) throws OrtException {
        Map<String, Object> modelOptions = config.getModelOptions();

        if (modelOptions.containsKey("interOpThreads")) {
            opts.setInterOpNumThreads((Integer) modelOptions.get("interOpThreads"));
        }

        if (modelOptions.containsKey("intraOpThreads")) {
            opts.setIntraOpNumThreads((Integer) modelOptions.get("intraOpThreads"));
        }

        if (modelOptions.containsKey("optimizationLevel")) {
            String level = String.valueOf(modelOptions.get("optimizationLevel")).toLowerCase();
            switch (level) {
                case "disable":
                    opts.setOptimizationLevel(
                            OrtSession.SessionOptions.OptLevel.NO_OPT
                    );
                    break;

                case "basic":
                    opts.setOptimizationLevel(
                            OrtSession.SessionOptions.OptLevel.BASIC_OPT
                    );
                    break;

                case "extended":
                    opts.setOptimizationLevel(
                            OrtSession.SessionOptions.OptLevel.EXTENDED_OPT
                    );
                    break;

                case "all":
                    opts.setOptimizationLevel(
                            OrtSession.SessionOptions.OptLevel.ALL_OPT
                    );
                    break;

                default:
                    throw new IllegalArgumentException(
                            "Unknown optimization level: " + level
                    );
            }

        }

        if (Boolean.TRUE.equals(modelOptions.get("useGpu"))) {
            try {
                opts.addCUDA(0);
            } catch (Exception ignored) {
            }
        }
    }

    private Map<String, Object> extractSchema(Map<String, NodeInfo> metadata) {
        Map<String, Object> schema = new HashMap<>();
        if (metadata == null) return schema;

        for (Map.Entry<String, NodeInfo> entry : metadata.entrySet()) {

            NodeInfo node = entry.getValue();

            Map<String, Object> nodeSchema = new HashMap<>();

            Object info = node.getInfo();

            nodeSchema.put("info", info != null ? info.toString() : "unknown");

            nodeSchema.put("rawType", node.toString());

            schema.put(entry.getKey(), nodeSchema);
        }

        return schema;
    }
}
