package com.codedstream.otterstream.benchmarks;

import com.codedstream.otterstream.benchmarks.support.NoOpInferenceEngine;
import com.codedstream.otterstream.runtime.serving.LeastConnectionsStrategy;
import com.codedstream.otterstream.runtime.serving.ReplicaPool;
import com.codedstream.otterstream.runtime.serving.RoundRobinStrategy;
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
 * Measures {@link ReplicaPool#infer} overhead — replica selection plus in-flight tracking —
 * across a pool of 4 {@link NoOpInferenceEngine} replicas, comparing
 * {@link RoundRobinStrategy} against {@link LeastConnectionsStrategy}. The latter does more
 * work per call (scans every replica's in-flight count) — this benchmark quantifies exactly how
 * much more, so "least connections costs a bit more per call but load-balances better under
 * uneven request cost" is a measured trade-off, not a guess.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
@Threads(4)
public class ReplicaPoolBenchmark {

    private static final int REPLICA_COUNT = 4;

    private ReplicaPool roundRobinPool;
    private ReplicaPool leastConnectionsPool;
    private Map<String, Object> inputs;

    @Setup(Level.Trial)
    public void setup() throws Exception {
        roundRobinPool = new ReplicaPool("bench-model", new RoundRobinStrategy());
        leastConnectionsPool = new ReplicaPool("bench-model", new LeastConnectionsStrategy());

        for (int i = 0; i < REPLICA_COUNT; i++) {
            NoOpInferenceEngine rrEngine = new NoOpInferenceEngine();
            rrEngine.initialize(null);
            roundRobinPool.addReplica(rrEngine);

            NoOpInferenceEngine lcEngine = new NoOpInferenceEngine();
            lcEngine.initialize(null);
            leastConnectionsPool.addReplica(lcEngine);
        }

        inputs = Map.of("x", 1.0);
    }

    @Benchmark
    public Object roundRobin() throws Exception {
        return roundRobinPool.infer(inputs);
    }

    @Benchmark
    public Object leastConnections() throws Exception {
        return leastConnectionsPool.infer(inputs);
    }
}
