package com.codedstream.otterstream.runtime.spi;

import java.util.Objects;

/**
 * A pointer to a model held by a {@link ModelRegistry}, independent of where or how
 * the model is physically stored.
 *
 * <p>A reference names a model by its logical {@code modelId} and, optionally, a
 * specific {@code version}. When no version is given, registries resolve to whatever
 * they consider the latest/active version for that model id.
 *
 * <h2>Usage:</h2>
 * <pre>{@code
 * ModelConfig latest = modelRegistry.resolve(ModelReference.of("fraud-detector"));
 * ModelConfig pinned = modelRegistry.resolve(ModelReference.of("fraud-detector", "2.1.0"));
 * }</pre>
 *
 * @since 0.1.0
 * @see ModelRegistry
 */
public final class ModelReference {

    private final String modelId;
    private final String version;

    private ModelReference(String modelId, String version) {
        this.modelId = Objects.requireNonNull(modelId, "modelId cannot be null");
        this.version = version;
    }

    /**
     * Creates a reference to the latest/active version of a model.
     *
     * @param modelId logical model identifier
     * @return an unversioned reference
     */
    public static ModelReference of(String modelId) {
        return new ModelReference(modelId, null);
    }

    /**
     * Creates a reference to a specific model version.
     *
     * @param modelId logical model identifier
     * @param version specific version to resolve
     * @return a versioned reference
     */
    public static ModelReference of(String modelId, String version) {
        return new ModelReference(modelId, version);
    }

    public String getModelId() {
        return modelId;
    }

    public String getVersion() {
        return version;
    }

    /**
     * @return true if this reference pins a specific version rather than "latest"
     */
    public boolean hasVersion() {
        return version != null && !version.isEmpty();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ModelReference)) return false;
        ModelReference that = (ModelReference) o;
        return modelId.equals(that.modelId) && Objects.equals(version, that.version);
    }

    @Override
    public int hashCode() {
        return Objects.hash(modelId, version);
    }

    @Override
    public String toString() {
        return hasVersion() ? modelId + "@" + version : modelId + "@latest";
    }
}
