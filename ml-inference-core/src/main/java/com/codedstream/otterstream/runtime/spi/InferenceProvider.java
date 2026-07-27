package com.codedstream.otterstream.runtime.spi;

import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.model.ModelFormat;
import java.util.Set;

/**
 * Service Provider Interface (SPI) that lets a model-format implementation
 * (ONNX, TensorFlow, PyTorch, XGBoost, PMML, remote HTTP/gRPC/SageMaker/Vertex AI, ...)
 * register itself with {@link com.codedstream.otterstream.runtime.OtterRuntime} without the
 * runtime needing a compile-time dependency on that implementation.
 *
 * <p>Providers are discovered in one of two ways:
 * <ul>
 *   <li><b>Automatic:</b> via {@link java.util.ServiceLoader}. Implementations should be
 *       declared in {@code META-INF/services/com.codedstream.otterstream.runtime.spi.InferenceProvider}
 *       in the provider module's jar.</li>
 *   <li><b>Manual:</b> via {@code OtterRuntime.builder().provider(myProvider)}.</li>
 * </ul>
 *
 * <h2>Implementation Example:</h2>
 * <pre>{@code
 * public class OnnxProvider implements InferenceProvider {
 *     public String getProviderId() { return "onnx"; }
 *     public Set<ModelFormat> getSupportedFormats() { return Set.of(ModelFormat.ONNX); }
 *     public InferenceEngine<?> createEngine() { return new OnnxInferenceEngine(); }
 * }
 * }</pre>
 *
 * Then, in {@code META-INF/services/com.codedstream.otterstream.runtime.spi.InferenceProvider}:
 * <pre>{@code
 * com.codedstream.otterstream.onnx.OnnxProvider
 * }</pre>
 *
 * <p>Note this is purely a <em>factory</em> abstraction: it says "given this format, here is
 * a fresh, uninitialized {@link InferenceEngine} that can handle it." The existing
 * {@link InferenceEngine}/{@link com.codedstream.otterstream.inference.model.ModelLoader}
 * contracts are unchanged; engines are still initialized via
 * {@link InferenceEngine#initialize(com.codedstream.otterstream.inference.config.ModelConfig)}.
 *
 * @since 0.1.0
 * @see com.codedstream.otterstream.runtime.registry.ProviderRegistry
 */
public interface InferenceProvider {

    /**
     * A stable, unique identifier for this provider (e.g. {@code "onnx"}, {@code "tensorflow"}).
     *
     * @return provider id
     */
    String getProviderId();

    /**
     * The {@link ModelFormat}s this provider can create engines for.
     *
     * @return non-empty set of supported formats
     */
    Set<ModelFormat> getSupportedFormats();

    /**
     * Whether this provider can handle the given format.
     *
     * @param format model format to check
     * @return true if supported
     */
    default boolean supports(ModelFormat format) {
        return getSupportedFormats().contains(format);
    }

    /**
     * Creates a new, uninitialized {@link InferenceEngine} instance.
     * <p>The caller is responsible for calling
     * {@link InferenceEngine#initialize(com.codedstream.otterstream.inference.config.ModelConfig)}.
     * A fresh instance must be returned on every call; engines are stateful once initialized
     * and must never be shared across concurrent deployments.
     *
     * @return a new engine instance ready to be initialized
     */
    InferenceEngine<?> createEngine();

    /**
     * Resolution priority when multiple providers support the same format.
     * Higher wins. Defaults to 0.
     *
     * @return priority value
     */
    default int getPriority() {
        return 0;
    }
}
