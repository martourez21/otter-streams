package com.codedstream.otterstream.benchmarks;

import com.codedstream.otterstream.rules.expr.CompiledCondition;
import com.codedstream.otterstream.rules.expr.ExpressionEvaluator;
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
import org.openjdk.jmh.annotations.Warmup;

/**
 * Directly quantifies the "compile once, evaluate many times" design decision behind
 * {@code ExpressionEvaluator} (see its class Javadoc and {@code DefaultRuleEngine}'s
 * performance notes): {@link #evaluatePrecompiled} evaluates a condition that was compiled once
 * at setup, {@link #compileAndEvaluateEveryTime} re-parses the same condition string on every
 * invocation. The ratio between these two numbers is the actual, measured cost of what would
 * happen if a rule engine parsed conditions per-request instead of per-deployment — this
 * benchmark exists specifically to make that claim checkable rather than asserted.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Thread)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
public class ExpressionEvaluatorBenchmark {

    private static final String CONDITION =
            "output.risk_score > 0.85 && output.country == 'US' || output.account_age_days < 7";

    private CompiledCondition compiled;
    private Map<String, Object> context;

    @Setup(Level.Trial)
    public void setup() throws Exception {
        compiled = ExpressionEvaluator.compile(CONDITION);
        context = Map.of("output", Map.of(
                "risk_score", 0.9,
                "country", "US",
                "account_age_days", 3));
    }

    @Benchmark
    public boolean evaluatePrecompiled() {
        return compiled.evaluate(context);
    }

    @Benchmark
    public boolean compileAndEvaluateEveryTime() throws Exception {
        return ExpressionEvaluator.compile(CONDITION).evaluate(context);
    }
}
