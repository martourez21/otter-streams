package com.codedstream.otterstream.inference.cache;

/**
 * Defines caching strategies for ML inference operations in Otter Stream.
 *
 * <p>Cache strategies determine how inference results are cached to improve
 * performance and reduce redundant model computations. Choose the appropriate
 * strategy based on your use case and data characteristics.
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * CacheStrategy strategy = CacheStrategy.INPUT_HASH;
 * System.out.println(strategy.getDescription());
 * }</pre>
 *
 * @author Nestor Martourez
 * @author Sr Software and Data Streaming Engineer @ CodedStreams
 * @since 1.0.0
 */
public enum CacheStrategy {

    /** No caching - all inferences go directly to the model */
    NONE("No caching"),

    /** Cache based on hash of input features - best for identical inputs */
    INPUT_HASH("Cache based on input hash"),

    /** Cache model outputs for reuse - useful for frequently accessed predictions */
    MODEL_OUTPUT("Cache model outputs"),

    /** Cache based on specific feature values - granular caching control */
    FEATURE_BASED("Cache based on feature values");

    private final String description;

    /**
     * Constructs a cache strategy with its description.
     *
     * @param description human-readable description of the caching strategy
     */
    CacheStrategy(String description) {
        this.description = description;
    }

    /**
     * Gets the human-readable description of this caching strategy.
     *
     * @return description of the caching approach
     */
    public String getDescription() {
        return description;
    }
}