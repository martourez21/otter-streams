package com.codedstream.otterstream.runtime;

import com.codedstream.otterstream.inference.cache.ModelCache;
import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.exception.DeploymentException;
import com.codedstream.otterstream.inference.exception.InferenceException;
import com.codedstream.otterstream.inference.exception.ModelLoadException;
import com.codedstream.otterstream.inference.metrics.MetricsCollector;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.runtime.lifecycle.DynamicLoader;
import com.codedstream.otterstream.runtime.lifecycle.LifecycleListener;
import com.codedstream.otterstream.runtime.lifecycle.LifecycleManager;
import com.codedstream.otterstream.runtime.lifecycle.ManagedModel;
import com.codedstream.otterstream.runtime.lifecycle.ShadowListener;
import com.codedstream.otterstream.runtime.registry.DefaultModelRegistry;
import com.codedstream.otterstream.runtime.registry.ProviderRegistry;
import com.codedstream.otterstream.runtime.spi.InferenceProvider;
import com.codedstream.otterstream.runtime.spi.ModelReference;
import com.codedstream.otterstream.runtime.spi.ModelRegistry;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The central entry point for Otter Streams — the AI runtime layer that sits on top of the
 * individual {@link com.codedstream.otterstream.inference.engine.InferenceEngine} implementations.
 *
 * <p>Where a bare {@code InferenceEngine} only knows how to load one model and answer
 * {@code infer()} calls, {@code OtterRuntime} ties together:
 * <ul>
 *   <li>{@link ProviderRegistry} — which engine implementations are available (Provider SPI)</li>
 *   <li>{@link ModelRegistry} — where model configurations come from (by id/version)</li>
 *   <li>{@link LifecycleManager} — safe validate → warm → swap → retire deployments</li>
 *   <li>an optional result {@link ModelCache}</li>
 *   <li>an optional {@link MetricsCollector}</li>
 * </ul>
 *
 * <p>This class is purely additive: every existing class (
 * {@link com.codedstream.otterstream.inference.engine.InferenceEngine},
 * {@link ModelConfig}, {@code AsyncModelInferenceFunction}, ...) continues to work exactly as
 * before and does not require {@code OtterRuntime} to function. Using the runtime is opt-in.
 *
 * <h2>Usage:</h2>
 * <pre>{@code
 * OtterRuntime runtime = OtterRuntime.builder()
 *         .metrics(new MetricsCollector())
 *         .cache(new ModelCache<>(10_000, 30))
 *         .build(); // auto-discovers provider modules on the classpath via ServiceLoader
 *
 * ManagedModel model = runtime.deploy(ModelConfig.builder()
 *         .modelId("fraud-detector")
 *         .modelPath("file:///models/fraud_detection.onnx")
 *         .format(ModelFormat.ONNX)
 *         .modelVersion("1.0")
 *         .build());
 *
 * InferenceResult result = runtime.infer("fraud-detector", inputs);
 *
 * // later — hot-swap to a new version, zero code change for callers of infer("fraud-detector", ...)
 * runtime.deploy(v2Config);
 *
 * runtime.close();
 * }</pre>
 *
 * @since 0.1.0
 */
public class OtterRuntime implements AutoCloseable {

    private static final Logger LOG = LoggerFactory.getLogger(OtterRuntime.class);

    private final ProviderRegistry providerRegistry;
    private final ModelRegistry modelRegistry;
    private final LifecycleManager lifecycleManager;
    private final DynamicLoader dynamicLoader;
    private final ModelCache<String, InferenceResult> resultCache;
    private final MetricsCollector metricsCollector;

    private OtterRuntime(ProviderRegistry providerRegistry, ModelRegistry modelRegistry,
                          ModelCache<String, InferenceResult> resultCache, MetricsCollector metricsCollector) {
        this.providerRegistry = providerRegistry;
        this.modelRegistry = modelRegistry;
        this.lifecycleManager = new LifecycleManager(providerRegistry);
        this.dynamicLoader = new DynamicLoader(lifecycleManager, modelRegistry);
        this.resultCache = resultCache;
        this.metricsCollector = metricsCollector;
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Deploys a model version with no warmup probe.
     *
     * @param config model configuration to deploy
     * @return the managed model now serving this version
     * @throws DeploymentException if validation fails
     */
    public ManagedModel deploy(ModelConfig config) throws DeploymentException {
        return lifecycleManager.deploy(config);
    }

    /**
     * Deploys a model version, running the given warmup probe before it takes traffic.
     *
     * @param config      model configuration to deploy
     * @param warmupProbe sample input exercised once before the swap; empty map skips warmup
     * @return the managed model now serving this version
     * @throws DeploymentException if validation or warmup fails
     */
    public ManagedModel deploy(ModelConfig config, Map<String, Object> warmupProbe) throws DeploymentException {
        return lifecycleManager.deploy(config, warmupProbe);
    }

    /**
     * Resolves a model reference via the configured {@link ModelRegistry} and deploys it.
     *
     * @param reference the model reference to resolve and deploy
     * @return the managed model now serving this version
     * @throws ModelLoadException  if the reference cannot be resolved to a config
     * @throws DeploymentException if validation fails once resolved
     */
    public ManagedModel deployFromRegistry(ModelReference reference) throws ModelLoadException, DeploymentException {
        ModelConfig config = modelRegistry.resolve(reference);
        return deploy(config);
    }

    /**
     * Convenience for the common case: run inference against whatever version is currently
     * active for {@code modelId}.
     *
     * @param modelId the deployed model's logical id
     * @param inputs  inference inputs
     * @return inference result
     * @throws InferenceException if nothing is deployed under this id, or the engine call fails
     */
    public InferenceResult infer(String modelId, Map<String, Object> inputs) throws InferenceException {
        return lifecycleManager.getManagedModel(modelId).infer(inputs);
    }

    public ManagedModel getManagedModel(String modelId) {
        return lifecycleManager.getManagedModel(modelId);
    }

    public boolean isDeployed(String modelId) {
        return lifecycleManager.isDeployed(modelId);
    }

    public void undeploy(String modelId) {
        lifecycleManager.undeploy(modelId);
    }

    // ------------------------------------------------------------------
    // Milestone 4 — Dynamic model loading (registry polling)
    // ------------------------------------------------------------------

    /**
     * Starts polling the configured {@link ModelRegistry} for new versions of {@code modelId}
     * and automatically deploys them as they appear. Requires the model to already be resolvable
     * via the registry (see {@link ModelRegistry#resolve}); the first poll deploys it if it
     * isn't already deployed.
     *
     * @param modelId      the model id to watch
     * @param pollInterval how often to check the registry for a new version
     */
    public void watch(String modelId, Duration pollInterval) {
        dynamicLoader.watch(modelId, pollInterval);
    }

    /**
     * Stops automatic polling/deployment for a model id previously passed to {@link #watch}.
     */
    public void unwatch(String modelId) {
        dynamicLoader.unwatch(modelId);
    }

    public boolean isWatching(String modelId) {
        return dynamicLoader.isWatching(modelId);
    }

    // ------------------------------------------------------------------
    // Milestone 5 — Rollback (graceful draining is automatic on every deploy/swap)
    // ------------------------------------------------------------------

    /**
     * Re-deploys the last previously-active version for a model, undoing the most recent hot
     * swap. Goes through the full validate/warm flow again.
     *
     * @param modelId the model id to roll back
     * @return the managed model, now back on the rolled-back version
     * @throws DeploymentException  if the rollback deployment fails validation/warmup
     * @throws IllegalStateException if there is no prior version to roll back to
     */
    public ManagedModel rollback(String modelId) throws DeploymentException {
        return lifecycleManager.rollback(modelId);
    }

    // ------------------------------------------------------------------
    // Milestone 6 — Shadow & canary deployments
    // ------------------------------------------------------------------

    /**
     * Deploys a candidate version alongside the current primary, receiving
     * {@code trafficPercent}% of traffic.
     *
     * @throws DeploymentException if validation/warmup of the canary fails
     */
    public ManagedModel deployCanary(ModelConfig config, int trafficPercent) throws DeploymentException {
        return lifecycleManager.deployCanary(config, trafficPercent);
    }

    /**
     * Promotes the current canary to primary and retires the old primary.
     *
     * @throws IllegalStateException if no canary is currently deployed
     */
    public void promoteCanary(String modelId) {
        lifecycleManager.promoteCanary(modelId);
    }

    /**
     * Discards the current canary without touching the primary.
     *
     * @throws IllegalStateException if no canary is currently deployed
     */
    public void rollbackCanary(String modelId) {
        lifecycleManager.rollbackCanary(modelId);
    }

    /**
     * Deploys a shadow version: a sampled copy of traffic is mirrored to it asynchronously for
     * comparison, never affecting what callers of {@link #infer} receive.
     *
     * @param sampleRate fraction of traffic to mirror, from 0.0 (off) to 1.0 (all)
     * @throws DeploymentException if validation/warmup of the shadow fails
     */
    public ManagedModel deployShadow(String modelId, ModelConfig config, double sampleRate) throws DeploymentException {
        return lifecycleManager.deployShadow(modelId, config, sampleRate);
    }

    /**
     * Stops shadowing traffic for a model. No-op if no shadow is deployed.
     */
    public void stopShadow(String modelId) {
        lifecycleManager.stopShadow(modelId);
    }

    /**
     * Registers a listener notified whenever a shadow inference call completes (or fails).
     */
    public void addShadowListener(ShadowListener listener) {
        lifecycleManager.addShadowListener(listener);
    }

    /**
     * Registers a listener notified of deployment lifecycle events (validating/warming/
     * activated/retired/failed) across all models.
     */
    public void addLifecycleListener(LifecycleListener listener) {
        lifecycleManager.addListener(listener);
    }

    public ProviderRegistry getProviderRegistry() {
        return providerRegistry;
    }

    public ModelRegistry getModelRegistry() {
        return modelRegistry;
    }

    public LifecycleManager getLifecycleManager() {
        return lifecycleManager;
    }

    /**
     * @return the configured result cache, or null if none was set on the builder
     */
    public ModelCache<String, InferenceResult> getResultCache() {
        return resultCache;
    }

    /**
     * @return the configured metrics collector, or null if none was set on the builder
     */
    public MetricsCollector getMetricsCollector() {
        return metricsCollector;
    }

    /**
     * Closes every currently-deployed model's active engine. The runtime instance should not
     * be reused after calling this.
     */
    @Override
    public void close() {
        dynamicLoader.shutdown();
        List<String> modelIds = new ArrayList<>(lifecycleManager.getDeployedModelIds());
        for (String modelId : modelIds) {
            lifecycleManager.undeploy(modelId);
        }
        lifecycleManager.shutdown();
        LOG.info("OtterRuntime closed, {} model(s) undeployed", modelIds.size());
    }

    /**
     * Builder for {@link OtterRuntime}.
     */
    public static class Builder {
        private ProviderRegistry providerRegistry;
        private ModelRegistry modelRegistry;
        private ModelCache<String, InferenceResult> resultCache;
        private MetricsCollector metricsCollector;
        private boolean autoDiscoverProviders = true;
        private final List<InferenceProvider> manualProviders = new ArrayList<>();

        /**
         * Supplies a fully-configured {@link ProviderRegistry}. If set, this takes precedence
         * over {@link #autoDiscoverProviders(boolean)} and {@link #provider(InferenceProvider)}
         * (though those still apply on top of the supplied registry).
         */
        public Builder providerRegistry(ProviderRegistry providerRegistry) {
            this.providerRegistry = providerRegistry;
            return this;
        }

        /**
         * Registers a single provider manually. Can be called multiple times.
         */
        public Builder provider(InferenceProvider provider) {
            this.manualProviders.add(Objects.requireNonNull(provider));
            return this;
        }

        /**
         * Whether to scan the classpath for providers via {@link java.util.ServiceLoader}.
         * Defaults to true.
         */
        public Builder autoDiscoverProviders(boolean autoDiscoverProviders) {
            this.autoDiscoverProviders = autoDiscoverProviders;
            return this;
        }

        /**
         * Supplies the {@link ModelRegistry} used by {@link #deployFromRegistry(ModelReference)}.
         * Defaults to an empty {@link DefaultModelRegistry} (in-memory, register-then-resolve).
         */
        public Builder registry(ModelRegistry modelRegistry) {
            this.modelRegistry = modelRegistry;
            return this;
        }

        /**
         * Supplies an optional result cache. Not wired into the inference path automatically
         * (callers decide caching policy); exposed via {@link OtterRuntime#getResultCache()}.
         */
        public Builder cache(ModelCache<String, InferenceResult> resultCache) {
            this.resultCache = resultCache;
            return this;
        }

        /**
         * Supplies an optional metrics collector, exposed via {@link OtterRuntime#getMetricsCollector()}.
         */
        public Builder metrics(MetricsCollector metricsCollector) {
            this.metricsCollector = metricsCollector;
            return this;
        }

        public OtterRuntime build() {
            ProviderRegistry registry = this.providerRegistry != null ? this.providerRegistry : new ProviderRegistry();
            if (autoDiscoverProviders) {
                registry.discoverProviders();
            }
            for (InferenceProvider provider : manualProviders) {
                registry.register(provider);
            }
            ModelRegistry resolvedModelRegistry = this.modelRegistry != null ? this.modelRegistry : new DefaultModelRegistry();
            return new OtterRuntime(registry, resolvedModelRegistry, resultCache, metricsCollector);
        }
    }
}
