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
 * Provides integration between Flink CEP (Complex Event Processing) and machine learning inference engines.
 * <p>
 * This utility class allows attaching ML-based conditions to Flink CEP {@link Pattern} objects, so that
 * events can be filtered or matched based on predictions from a pre-trained ML model.
 * </p>
 *
 * <p>
 * Example usage:
 * <pre>{@code
 * Pattern<MyEvent, MyEvent> pattern = Pattern.<MyEvent>begin("start")
 *     .next("next")
 *     .where(CEPInferenceIntegration.withMLCondition(
 *         "myModel",
 *         event -> Map.of("feature1", event.getValue()),
 *         score -> score > 0.8
 *     ));
 * }</pre>
 * </p>
 */
public class CEPInferenceIntegration {

    /**
     * Attaches a machine learning-based condition to a Flink CEP pattern.
     * <p>
     * The condition evaluates each event by extracting features using the provided {@code featureExtractor},
     * sending them to the ML model identified by {@code modelName}, and testing the resulting score with
     * {@code scorePredicate}.
     * </p>
     *
     * @param pattern          The Flink CEP pattern to attach the ML condition to.
     * @param modelName        The name of the ML model to use for inference. The model should be registered
     *                         in the {@link ModelCache}.
     * @param featureExtractor A function that extracts features from an event of type {@code T} as a
     *                         {@link Map} of feature names to values.
     * @param scorePredicate   A predicate that evaluates the numeric prediction score returned by the ML model.
     * @param <T>              The type of events in the CEP pattern.
     * @return The original pattern with the ML-based condition applied.
     */
    public static <T> Pattern<T, T> withMLCondition(
            Pattern<T, T> pattern,
            String modelName,
            Function<T, Map<String, Object>> featureExtractor,
            Predicate<Double> scorePredicate) {

        return pattern.where(new MLCondition<>(modelName, featureExtractor, scorePredicate));
    }

    /**
     * A Flink CEP {@link IterativeCondition} that applies an ML inference engine to each event.
     * <p>
     * This condition:
     * <ol>
     *     <li>Retrieves the ML model engine from the {@link ModelCache} by {@code modelName}.</li>
     *     <li>Extracts features from the event using {@code featureExtractor}.</li>
     *     <li>Performs inference using the ML engine.</li>
     *     <li>Tests the resulting score with {@code scorePredicate}.</li>
     * </ol>
     * </p>
     *
     * @param <T> The type of events this condition evaluates.
     */
    static final class MLCondition<T> extends IterativeCondition<T> {

        /** Name of the ML model used for inference. */
        private final String modelName;

        /** Function to extract features from the event. */
        private final Function<T, Map<String, Object>> featureExtractor;

        /** Predicate to evaluate the numeric score produced by the ML model. */
        private final Predicate<Double> scorePredicate;

        /** Cache for ML model engines. */
        private transient ModelCache modelCache;

        /**
         * Constructs an ML-based CEP condition.
         *
         * @param modelName        The name of the ML model to use.
         * @param featureExtractor Function to extract features from each event.
         * @param scorePredicate   Predicate to evaluate the ML prediction score.
         */
        MLCondition(String modelName,
                    Function<T, Map<String, Object>> featureExtractor,
                    Predicate<Double> scorePredicate) {
            this.modelName = modelName;
            this.featureExtractor = featureExtractor;
            this.scorePredicate = scorePredicate;
        }

        /**
         * Evaluates whether an event satisfies the ML condition.
         * <p>
         * Steps:
         * <ol>
         *     <li>Initialize the {@link ModelCache} if needed.</li>
         *     <li>Retrieve the {@link InferenceEngine} for {@code modelName}.</li>
         *     <li>Extract features from the event.</li>
         *     <li>Run inference and obtain the result.</li>
         *     <li>Check if the result is successful and has a numeric output.</li>
         *     <li>Apply the {@code scorePredicate} to the score.</li>
         * </ol>
         * </p>
         *
         * @param value The event to evaluate.
         * @param ctx   The CEP evaluation context.
         * @return {@code true} if the event passes the ML-based predicate, {@code false} otherwise.
         * @throws Exception if an error occurs during inference or feature extraction.
         */
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
