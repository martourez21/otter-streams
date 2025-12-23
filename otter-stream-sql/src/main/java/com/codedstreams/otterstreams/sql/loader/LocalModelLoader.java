package com.codedstreams.otterstreams.sql.loader;

import com.codedstreams.otterstreams.sql.config.ModelSourceConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.InputStream;

/**
 * Loads models from local filesystem or HDFS.
 */
public class LocalModelLoader implements ModelLoader {
    private static final Logger LOG = LoggerFactory.getLogger(LocalModelLoader.class);
    private final ModelSourceConfig config;

    public LocalModelLoader(ModelSourceConfig config) {
        this.config = config;
    }

    @Override
    public InputStream loadModel() throws Exception {
        String path = config.getModelPath();

        // Remove file:// prefix if present
        if (path.startsWith("file://")) {
            path = path.substring(7);
        }

        File modelFile = new File(path);
        if (!modelFile.exists()) {
            throw new IllegalArgumentException("Model file not found: " + path);
        }

        LOG.info("Loading model from local filesystem: {}", path);
        return new FileInputStream(modelFile);
    }

    @Override
    public String getModelPath() {
        return config.getModelPath();
    }
}
