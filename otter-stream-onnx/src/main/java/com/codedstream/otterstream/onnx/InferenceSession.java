package com.codedstream.otterstream.onnx;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.util.Map;

public class InferenceSession {

    private final OrtEnvironment environment;
    private final OrtSession session;

    // Load from file path
    public InferenceSession(String modelPath, OrtSession.SessionOptions options, OrtEnvironment environment) throws Exception {
        this.environment = environment;
        this.session = environment.createSession(modelPath, options);
    }

    // Load from bytes
    public InferenceSession(byte[] modelBytes, OrtSession.SessionOptions options, OrtEnvironment environment) throws Exception {
        this.environment = environment;
        this.session = environment.createSession(modelBytes, options);
    }

    public Map<String, NodeInfo> getInputMetadata() throws OrtException {
        return session.getInputInfo();
    }

    public Map<String, NodeInfo> getOutputMetadata() throws OrtException {
        return session.getOutputInfo();
    }

    public OrtSession getSession() {
        return session;
    }

    public void close() {
        try {
            session.close();
            environment.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}

