package com.codedstream.otterstream.runtime.registry;

import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.model.ModelFormat;
import com.codedstream.otterstream.runtime.spi.InferenceProvider;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.ServiceLoader;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Holds the set of {@link InferenceProvider}s known to an {@link com.codedstream.otterstream.runtime.OtterRuntime}
 * and resolves the right provider for a given {@link ModelFormat} — this is the "Provider SPI"
 * / plugin-architecture piece of the runtime (Milestone 2).
 *
 * <p>Providers can be added in two ways:
 * <ul>
 *   <li>{@link #discoverProviders()} — scans the classpath via {@link ServiceLoader}, picking up
 *       every provider module present (otter-stream-onnx, otter-stream-tensorflow, ...)</li>
 *   <li>{@link #register(InferenceProvider)} — manual registration, useful for custom/internal
 *       providers that aren't shipped as a separate module</li>
 * </ul>
 *
 * <p>Thread-safe: backed by {@link ConcurrentHashMap}, safe to read from concurrently while a
 * background thread registers new providers.
 *
 * @since 0.1.0
 */
public class ProviderRegistry {

    private static final Logger LOG = LoggerFactory.getLogger(ProviderRegistry.class);

    private final Map<String, InferenceProvider> providersById = new ConcurrentHashMap<>();
    private final Map<ModelFormat, List<InferenceProvider>> providersByFormat = new ConcurrentHashMap<>();

    /**
     * Creates an empty registry with no providers registered.
     */
    public ProviderRegistry() {
    }

    /**
     * Creates a registry pre-populated via {@link ServiceLoader} discovery.
     *
     * @return a new registry with all classpath providers registered
     */
    public static ProviderRegistry discover() {
        ProviderRegistry registry = new ProviderRegistry();
        registry.discoverProviders();
        return registry;
    }

    /**
     * Scans the classpath for {@link InferenceProvider} implementations declared under
     * {@code META-INF/services/com.codedstream.otterstream.runtime.spi.InferenceProvider}
     * and registers each of them. Safe to call multiple times (re-registration overwrites
     * by provider id).
     */
    public void discoverProviders() {
        ServiceLoader<InferenceProvider> loader =
                ServiceLoader.load(InferenceProvider.class, ProviderRegistry.class.getClassLoader());
        int count = 0;
        for (InferenceProvider provider : loader) {
            register(provider);
            count++;
        }
        LOG.info("Discovered {} InferenceProvider(s) via ServiceLoader", count);
    }

    /**
     * Manually registers a provider, indexing it by every {@link ModelFormat} it supports.
     * If multiple providers support the same format, they're tried in descending
     * {@link InferenceProvider#getPriority()} order and the highest-priority one wins ties
     * via first-registered.
     *
     * @param provider the provider to register
     */
    public void register(InferenceProvider provider) {
        Objects.requireNonNull(provider, "provider cannot be null");
        InferenceProvider previous = providersById.put(provider.getProviderId(), provider);
        if (previous != null) {
            LOG.warn("Replacing InferenceProvider registered under id '{}'", provider.getProviderId());
        }
        for (ModelFormat format : provider.getSupportedFormats()) {
            List<InferenceProvider> candidates =
                    providersByFormat.computeIfAbsent(format, f -> new ArrayList<>());
            synchronized (candidates) {
                candidates.removeIf(p -> p.getProviderId().equals(provider.getProviderId()));
                candidates.add(provider);
                candidates.sort(Comparator.comparingInt(InferenceProvider::getPriority).reversed());
            }
        }
    }

    /**
     * Finds the highest-priority provider registered for a given format.
     *
     * @param format the model format to resolve
     * @return the matching provider, if any
     */
    public Optional<InferenceProvider> findProvider(ModelFormat format) {
        List<InferenceProvider> candidates = providersByFormat.get(format);
        if (candidates == null || candidates.isEmpty()) {
            return Optional.empty();
        }
        synchronized (candidates) {
            return Optional.of(candidates.get(0));
        }
    }

    /**
     * Looks up a provider by its id.
     *
     * @param providerId provider id (e.g. {@code "onnx"})
     * @return the provider, if registered
     */
    public Optional<InferenceProvider> getById(String providerId) {
        return Optional.ofNullable(providersById.get(providerId));
    }

    /**
     * Resolves the given format to a provider and creates a fresh, uninitialized engine.
     *
     * @param format the model format to create an engine for
     * @return a new engine instance
     * @throws IllegalStateException if no provider is registered for this format
     */
    public InferenceEngine<?> createEngine(ModelFormat format) {
        return findProvider(format)
                .orElseThrow(() -> new IllegalStateException(
                        "No InferenceProvider registered for format: " + format
                                + ". Did you forget to add the corresponding otter-stream-* dependency,"
                                + " or call ProviderRegistry.discoverProviders()/register(...)?"))
                .createEngine();
    }

    /**
     * @return all currently registered providers, keyed by nothing in particular (id order not guaranteed)
     */
    public Collection<InferenceProvider> getAllProviders() {
        return Collections.unmodifiableCollection(providersById.values());
    }

    /**
     * @return true if no providers are registered
     */
    public boolean isEmpty() {
        return providersById.isEmpty();
    }
}
