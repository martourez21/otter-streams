package com.codedstreams.otterstreams.sql.loader;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.ConcurrentHashMap;

/**
 * Tracks progress of model loading operations.
 *
 * @author Nestor Martourez Abiangang A.
 * @since 1.0.0
 */
public class ModelLoadingProgressTracker {
    private static final Logger LOG = LoggerFactory.getLogger(ModelLoadingProgressTracker.class);
    private static final ModelLoadingProgressTracker INSTANCE = new ModelLoadingProgressTracker();

    private final ConcurrentHashMap<String, LoadingProgress> progressMap;

    private ModelLoadingProgressTracker() {
        this.progressMap = new ConcurrentHashMap<>();
    }

    public static ModelLoadingProgressTracker getInstance() {
        return INSTANCE;
    }

    public void startLoading(String modelName) {
        progressMap.put(modelName, new LoadingProgress(modelName));
        LOG.info("Started loading model: {}", modelName);
    }

    public void updateProgress(String modelName, String stage, int percentComplete) {
        LoadingProgress progress = progressMap.get(modelName);
        if (progress != null) {
            progress.updateProgress(stage, percentComplete);
            LOG.debug("Model {} loading progress: {} - {}%", modelName, stage, percentComplete);
        }
    }

    public void completeLoading(String modelName, boolean success) {
        LoadingProgress progress = progressMap.get(modelName);
        if (progress != null) {
            progress.complete(success);
            LOG.info("Model {} loading {}", modelName, success ? "completed" : "failed");
        }
    }

    public LoadingProgress getProgress(String modelName) {
        return progressMap.get(modelName);
    }

    public void removeProgress(String modelName) {
        progressMap.remove(modelName);
    }

    /**
     * Represents loading progress for a single model.
     */
    public static class LoadingProgress {
        private final String modelName;
        private final long startTime;
        private volatile String currentStage;
        private volatile int percentComplete;
        private volatile boolean completed;
        private volatile boolean success;
        private volatile long endTime;

        public LoadingProgress(String modelName) {
            this.modelName = modelName;
            this.startTime = System.currentTimeMillis();
            this.currentStage = "INITIALIZING";
            this.percentComplete = 0;
            this.completed = false;
        }

        public void updateProgress(String stage, int percent) {
            this.currentStage = stage;
            this.percentComplete = Math.min(100, Math.max(0, percent));
        }

        public void complete(boolean success) {
            this.completed = true;
            this.success = success;
            this.endTime = System.currentTimeMillis();
            this.percentComplete = 100;
        }

        public String getModelName() { return modelName; }
        public String getCurrentStage() { return currentStage; }
        public int getPercentComplete() { return percentComplete; }
        public boolean isCompleted() { return completed; }
        public boolean isSuccess() { return success; }
        public long getElapsedTime() {
            return (completed ? endTime : System.currentTimeMillis()) - startTime;
        }
    }
}
