package com.codedstream.otterstream.inference.model;

import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.exception.ModelLoadException;

import java.io.InputStream;
import java.util.Map;

public interface ModelLoader<T> {
    T loadModel(ModelConfig config) throws ModelLoadException;
    T loadModel(InputStream inputStream, ModelConfig config) throws ModelLoadException;
    boolean validateModel(T model, ModelConfig config);
    ModelFormat[] getSupportedFormats();
    ModelMetadata getModelMetadata(T model);
}