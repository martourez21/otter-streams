package com.codedstream.otterstream.runtime.spi;

import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.exception.ModelLoadException;

/**
 * Service Provider Interface (SPI) for resolving a {@link ModelReference} to a concrete,
 * loadable {@link ModelConfig}.
 *
 * <p>This is the extension point for integrating Otter with external model stores such as
 * MLflow, S3, MinIO, Azure Blob, GCS, or a Nexus repository: implement this interface against
 * the store's API and hand resolved {@link ModelConfig}s back to the runtime. Otter ships a
 * simple in-memory default ({@code DefaultModelRegistry}) suitable for statically-configured
 * deployments; production use cases that need registry polling / auto-discovery of new versions
 * build on top of this SPI.
 *
 * <p><b>Scope note:</b> Otter deliberately stops at "resolve a reference to a model config."
 * Training, hyperparameter tuning, feature engineering, experiment tracking, drift detection,
 * and retraining orchestration are explicitly out of scope and remain the responsibility of the
 * MLOps platform (MLflow, Kubeflow, SageMaker, Vertex AI, etc.) that backs the registry.
 *
 * @since 0.1.0
 * @see ModelReference
 */
public interface ModelRegistry {

    /**
     * Resolves a reference to a concrete model configuration.
     *
     * @param reference the model reference to resolve (versioned or "latest")
     * @return the resolved model configuration
     * @throws ModelLoadException if the model id/version is unknown to this registry, or
     *                            resolution otherwise fails
     */
    ModelConfig resolve(ModelReference reference) throws ModelLoadException;

    /**
     * Registers/updates a model configuration under a given logical model id.
     * <p>Optional operation; registries backed by a read-only external store (e.g. a
     * pre-populated S3 bucket) may not support programmatic registration.
     *
     * @param modelId logical model identifier
     * @param config  the configuration to register
     * @throws UnsupportedOperationException if this registry is read-only
     */
    default void register(String modelId, ModelConfig config) {
        throw new UnsupportedOperationException(
                getClass().getSimpleName() + " does not support programmatic registration");
    }
}
