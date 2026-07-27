package com.codedstream.otterstream.runtime.registry;

import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.exception.ModelLoadException;
import com.codedstream.otterstream.runtime.spi.ModelRegistry;
import com.codedstream.otterstream.runtime.spi.ModelReference;
import java.util.NavigableMap;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentSkipListMap;

/**
 * Simple in-memory {@link ModelRegistry} implementation: configurations are registered
 * programmatically (typically at application startup) and resolved by exact version or,
 * when unversioned, by the lexicographically-highest registered version string.
 *
 * <p>This is the default registry {@link com.codedstream.otterstream.runtime.OtterRuntime}
 * uses when no other {@link ModelRegistry} is supplied. It has no notion of polling an
 * external store for new versions — for that, implement {@link ModelRegistry} against a
 * real backend (MLflow, S3, Nexus, ...) and supply it via {@code OtterRuntime.builder().registry(...)}.
 *
 * <p><b>Version ordering caveat:</b> "highest" is determined by {@link String#compareTo(String)}
 * on the version string, which is <em>not</em> semver-aware (e.g. "10.0" sorts before "9.0").
 * Callers that need semantic versioning should use a custom {@link ModelRegistry} or always
 * resolve by explicit version.
 *
 * @since 0.1.0
 */
public class DefaultModelRegistry implements ModelRegistry {

    private final ConcurrentHashMap<String, NavigableMap<String, ModelConfig>> versionsByModelId =
            new ConcurrentHashMap<>();

    @Override
    public void register(String modelId, ModelConfig config) {
        Objects.requireNonNull(modelId, "modelId cannot be null");
        Objects.requireNonNull(config, "config cannot be null");
        String version = config.getModelVersion() != null ? config.getModelVersion() : "unversioned";
        versionsByModelId
                .computeIfAbsent(modelId, id -> new ConcurrentSkipListMap<>())
                .put(version, config);
    }

    @Override
    public ModelConfig resolve(ModelReference reference) throws ModelLoadException {
        Objects.requireNonNull(reference, "reference cannot be null");
        NavigableMap<String, ModelConfig> versions = versionsByModelId.get(reference.getModelId());
        if (versions == null || versions.isEmpty()) {
            throw new ModelLoadException(
                    "No model registered under id '" + reference.getModelId() + "' in DefaultModelRegistry");
        }

        if (reference.hasVersion()) {
            ModelConfig config = versions.get(reference.getVersion());
            if (config == null) {
                throw new ModelLoadException("Model '" + reference.getModelId()
                        + "' has no registered version '" + reference.getVersion() + "'");
            }
            return config;
        }

        return versions.lastEntry().getValue();
    }
}
