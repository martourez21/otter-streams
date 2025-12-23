package com.codedstreams.otterstreams.sql.cep;

import com.codedstream.otterstream.inference.engine.InferenceEngine;
import com.codedstream.otterstream.inference.model.InferenceResult;
import com.codedstreams.otterstreams.sql.loader.ModelCache;
import org.apache.flink.cep.pattern.Pattern;
import org.apache.flink.cep.pattern.conditions.IterativeCondition;

import java.util.Map;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * Integration between Flink CEP and ML inference for pattern-based decisions.
 */
public class CEPInferenceIntegration {

    public static <T> Pattern<T, T> withMLCondition(
            Pattern<T, T> pattern,
            String modelName,
            Function<T, Map<String, Object>> featureExtractor,
            Predicate<Double> scorePredicate) {

        return pattern.where(new MLCondition<>(modelName, featureExtractor, scorePredicate));
    }

    /**
     * CEP condition backed by an ML inference engine.
     */
    static final class MLCondition<T> extends IterativeCondition<T> {

        private final String modelName;
        private final Function<T, Map<String, Object>> featureExtractor;
        private final Predicate<Double> scorePredicate;

        private transient ModelCache modelCache;

        MLCondition(String modelName,
                    Function<T, Map<String, Object>> featureExtractor,
                    Predicate<Double> scorePredicate) {
            this.modelName = modelName;
            this.featureExtractor = featureExtractor;
            this.scorePredicate = scorePredicate;
        }

        @Override
        public boolean filter(T value, Context<T> ctx) throws Exception {
            if (modelCache == null) {
                modelCache = ModelCache.getInstance();
            }

            InferenceEngine<?> engine = modelCache.getEngine(modelName);
            if (engine == null) {
                return false;
            }

            Map<String, Object> features = featureExtractor.apply(value);
            InferenceResult result = engine.infer(features);

            if (!result.isSuccess() || result.getOutputs().isEmpty()) {
                return false;
            }

            Object prediction = result.getOutputs().values().iterator().next();
            if (!(prediction instanceof Number)) {
                return false;
            }

            double score = ((Number) prediction).doubleValue();
            return scorePredicate.test(score);
        }
    }
}
