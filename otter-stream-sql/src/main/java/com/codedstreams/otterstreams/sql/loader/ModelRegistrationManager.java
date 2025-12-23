package com.codedstreams.otterstreams.sql.loader;

import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstreams.otterstreams.sql.config.SqlInferenceConfig;
import com.codedstreams.otterstreams.sql.metadata.ModelDescriptor;
import com.codedstreams.otterstreams.sql.metadata.ModelRegistry;
import com.codedstreams.otterstreams.sql.runtime.InferenceEngineFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.InputStream;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Manages model registration, loading, and lifecycle.
 *
 * @author Nestor Martourez Abiangang A.
 * @since 1.0.0
 */
public class ModelRegistrationManager {
    private static final Logger LOG = LoggerFactory.getLogger(ModelRegistrationManager.class);
    private static final ModelRegistrationManager INSTANCE = new ModelRegistrationManager();

    private final ModelCache modelCache;
    private final ModelRegistry modelRegistry;
    private final ConcurrentHashMap<String, ModelLoadingTask> loadingTasks;

    private ModelRegistrationManager() {
        this.modelCache = ModelCache.getInstance();
        this.modelRegistry = ModelRegistry.getInstance();
        this.loadingTasks = new ConcurrentHashMap<>();
    }

    public static ModelRegistrationManager getInstance() {
        return INSTANCE;
    }

    /**
     * Registers and loads a model from configuration.
     */
    public synchronized InferenceEngine<?> registerAndLoadModel(SqlInferenceConfig config) throws Exception {
        String modelName = config.getModelName();

        // Check if already loaded
        InferenceEngine<?> existingEngine = modelCache.getEngine(modelName);
        if (existingEngine != null && existingEngine.isReady()) {
            LOG.info("Model already loaded: {}", modelName);
            return existingEngine;
        }

        // Check if currently loading
        ModelLoadingTask existingTask = loadingTasks.get(modelName);
        if (existingTask != null && !existingTask.isComplete()) {
            LOG.info("Model currently loading: {}, waiting...", modelName);
            return existingTask.waitForCompletion();
        }

        // Start new loading task
        ModelLoadingTask task = new ModelLoadingTask(modelName, config);
        loadingTasks.put(modelName, task);

        try {
            InferenceEngine<?> engine = task.execute();
            modelCache.putEngine(modelName, engine);

            // Register metadata
            ModelDescriptor descriptor = createDescriptor(config, engine);
            modelRegistry.registerModel(modelName, descriptor);

            LOG.info("Model successfully registered and loaded: {}", modelName);
            return engine;
        } finally {
            loadingTasks.remove(modelName);
        }
    }

    /**
     * Unregisters and closes a model.
     */
    public void unregisterModel(String modelName) {
        try {
            InferenceEngine<?> engine = modelCache.getEngine(modelName);
            if (engine != null) {
                engine.close();
            }
            modelCache.invalidate(modelName);
            modelRegistry.unregisterModel(modelName);
            LOG.info("Model unregistered: {}", modelName);
        } catch (Exception e) {
            LOG.error("Error unregistering model: {}", modelName, e);
        }
    }

    /**
     * Reloads a model with new configuration.
     */
    public InferenceEngine<?> reloadModel(String modelName, SqlInferenceConfig newConfig) throws Exception {
        LOG.info("Reloading model: {}", modelName);
        unregisterModel(modelName);
        return registerAndLoadModel(newConfig);
    }

    private ModelDescriptor createDescriptor(SqlInferenceConfig config, InferenceEngine<?> engine) {
        return new ModelDescriptor(
                config.getModelName(),
                config.getModelName(),
                config.getModelVersion(),
                config.getModelSource().getModelFormat(),
                config.getModelSource().getModelPath(),
                null,  // schema will be extracted from engine
                config.getAdditionalOptions()
        );
    }

    /**
     * Internal class to manage async model loading.
     */
    private static class ModelLoadingTask {
        private final String modelName;
        private final SqlInferenceConfig config;
        private volatile InferenceEngine<?> result;
        private volatile Exception error;
        private volatile boolean complete = false;

        ModelLoadingTask(String modelName, SqlInferenceConfig config) {
            this.modelName = modelName;
            this.config = config;
        }

        InferenceEngine<?> execute() throws Exception {
            try {
                // Load model from source
                ModelLoader loader = ModelLoaderFactory.create(config.getModelSource());
                InputStream modelStream = loader.loadModel();

                // Create inference engine
                ModelConfig modelConfig = ModelConfig.builder()
                        .modelId(config.getModelName())
                        .modelPath(config.getModelSource().getModelPath())
                        .format(config.getModelSource().getModelFormat())
                        .modelVersion(config.getModelVersion())
                        .build();

                InferenceEngine<?> engine = InferenceEngineFactory.createEngine(modelConfig);
                engine.initialize(modelConfig);

                this.result = engine;
                this.complete = true;
                return engine;
            } catch (Exception e) {
                this.error = e;
                this.complete = true;
                throw e;
            }
        }

        boolean isComplete() {
            return complete;
        }

        InferenceEngine<?> waitForCompletion() throws Exception {
            while (!complete) {
                Thread.sleep(100);
            }
            if (error != null) {
                throw error;
            }
            return result;
        }
    }
}