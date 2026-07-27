package com.codedstream.otterstream.benchmarks;

import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstream.otterstream.rules.engine.DefaultRuleEngine;
import com.codedstream.otterstream.rules.engine.YamlRuleSetSource;
import com.codedstream.otterstream.rules.spi.RuleEngine;
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
 * Measures {@code DefaultRuleEngine.evaluate()} against a realistic 4-rule set (the same shape
 * as {@code otter-stream-rules/src/main/resources/rules-example.yaml}), in both SINGLE and
 * MULTIPLE evaluation modes. This is the number that most directly answers "how much does the
 * Rule Engine add to my per-request latency," since condition compilation (the expensive part)
 * happens once at setup, not per invocation — see {@code ExpressionEvaluatorBenchmark} for the
 * isolated compiled-condition-evaluation cost this builds on.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
@Threads(4)
public class RuleEngineBenchmark {

    private static final String RULES_YAML = """
            name: bench-rules
            version: "1"
            evaluationMode: SINGLE
            rules:
              - id: high-risk
                priority: 100
                condition: "output.risk_score > 0.85"
                flag: FRAUD
                color: "#c0392b"
              - id: medium-risk
                priority: 50
                condition: "output.risk_score > 0.5 && output.risk_score <= 0.85"
                flag: REVIEW
                color: "#b45309"
              - id: new-account
                priority: 40
                condition: "output.account_age_days < 7 && output.transaction_amount > 1000"
                flag: REVIEW
                color: "#b45309"
              - id: default-approve
                priority: 0
                condition: "true"
                flag: APPROVE
                color: "#00875a"
            """;

    private static final String RULES_YAML_MULTIPLE = RULES_YAML.replace("evaluationMode: SINGLE", "evaluationMode: MULTIPLE");

    private RuleEngine singleModeEngine;
    private RuleEngine multipleModeEngine;
    private InferenceResult matchingResult;
    private InferenceResult nonMatchingResult;

    @Setup(Level.Trial)
    public void setup() throws Exception {
        singleModeEngine = new DefaultRuleEngine(YamlRuleSetSource.fromString(RULES_YAML));
        multipleModeEngine = new DefaultRuleEngine(YamlRuleSetSource.fromString(RULES_YAML_MULTIPLE));

        matchingResult = new InferenceResult(
                Map.of("risk_score", 0.92, "account_age_days", 3, "transaction_amount", 1500.0), 2, "bench-model");
        nonMatchingResult = new InferenceResult(
                Map.of("risk_score", 0.1, "account_age_days", 400, "transaction_amount", 20.0), 2, "bench-model");
    }

    @Benchmark
    public Object evaluateSingleMode_matching() throws Exception {
        return singleModeEngine.evaluate(matchingResult, Map.of());
    }

    @Benchmark
    public Object evaluateSingleMode_fallthrough() throws Exception {
        return singleModeEngine.evaluate(nonMatchingResult, Map.of());
    }

    @Benchmark
    public Object evaluateMultipleMode_matching() throws Exception {
        return multipleModeEngine.evaluate(matchingResult, Map.of());
    }
}
