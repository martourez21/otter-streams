package com.codedstream.otterstream.benchmarks;

import com.codedstream.otterstream.benchmarks.support.NoOpInferenceProvider;
import com.codedstream.otterstream.inference.config.ModelConfig;
import com.codedstream.otterstream.inference.model.ModelFormat;
import com.codedstream.otterstream.runtime.lifecycle.LifecycleManager;
import com.codedstream.otterstream.runtime.lifecycle.ManagedModel;
import com.codedstream.otterstream.runtime.registry.ProviderRegistry;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Threads;
import org.openjdk.jmh.annotations.Warmup;

/**
 * Measures {@code ManagedModel.infer()}'s own overhead — canary-routing check, in-flight
 * counter increment/decrement, shadow sample-rate check — using {@link
 * com.codedstream.otterstream.benchmarks.support.NoOpInferenceEngine} so the measured time is
 * (as close as achievable) purely Otter's routing machinery, not any real model's compute time.
 *
 * <p><b>What this does and doesn't tell you:</b> if this benchmark reports, say, 200ns per call,
 * that's the ceiling on how much latency {@code ManagedModel} itself adds on top of whatever
 * your actual model takes — it says nothing about your model's own latency, which is what
 * actually dominates the sub-5ms target in practice. See this module's README for how to
 * measure that separately.
 *
 * <p>{@link Threads @Threads(4)} runs this concurrently — the routing path is designed to
 * support concurrent callers (see {@code ManagedModel}'s class Javadoc), so a meaningful
 * benchmark needs to exercise that, not just single-threaded throughput.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
@Threads(4)
public class ManagedModelRoutingBenchmark {

    private ManagedModel managedModel;
    private Map<String, Object> inputs;

    @Setup(Level.Trial)
    public void setup() throws Exception {
        ProviderRegistry registry = new ProviderRegistry();
        registry.register(new NoOpInferenceProvider());

        LifecycleManager lifecycleManager = new LifecycleManager(registry);
        ModelConfig config = ModelConfig.builder()
                .modelId("bench-model")
                .modelPath("n/a")
                .format(ModelFormat.ONNX)
                .modelVersion("1")
                .build();

        this.managedModel = lifecycleManager.deploy(config);
        this.inputs = Map.of("x", 1.0);
    }

    @Benchmark
    public Object infer() throws Exception {
        return managedModel.infer(inputs);
    }
}
