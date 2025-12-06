package com.codedstream.otterstream.inference.cache;

public enum CacheStrategy {
    NONE("No caching"),
    INPUT_HASH("Cache based on input hash"),
    MODEL_OUTPUT("Cache model outputs"),
    FEATURE_BASED("Cache based on feature values");

    private final String description;

    CacheStrategy(String description) {
        this.description = description;
    }

    public String getDescription() {
        return description;
    }
}
