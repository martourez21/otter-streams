package com.codedstreams.otterstreams.sql.loader;

import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.TimeUnit;

/**
 * Thread-safe LRU cache for loaded inference engines.
 */
public class ModelCache {
    private static final Logger LOG = LoggerFactory.getLogger(ModelCache.class);
    private static final ModelCache INSTANCE = new ModelCache();

    private final Cache<String, InferenceEngine<?>> engineCache;

    private ModelCache() {
        this.engineCache = Caffeine.newBuilder()
                .maximumSize(100)
                .expireAfterWrite(30, TimeUnit.MINUTES)
                .build();
    }

    public static ModelCache getInstance() {
        return INSTANCE;
    }

    public InferenceEngine<?> getEngine(String modelName) {
        return engineCache.getIfPresent(modelName);
    }

    public void putEngine(String modelName, InferenceEngine<?> engine) {
        engineCache.put(modelName, engine);
        LOG.info("Cached engine for model: {}", modelName);
    }

    public void invalidate(String modelName) {
        engineCache.invalidate(modelName);
    }

    public void invalidateAll() {
        engineCache.invalidateAll();
    }
}
