package com.codedstreams.otterstreams.sql.util;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

/**
 * Postprocesses inference results (denormalization, thresholding, etc.).
 *
 * @author Nestor Martourez Abiangang A.
 * @since 1.0.0
 */
public class ResultPostprocessor {
    private static final Logger LOG = LoggerFactory.getLogger(ResultPostprocessor.class);

    /**
     * Applies softmax to convert logits to probabilities.
     */
    public static double[] softmax(double[] logits) {
        double maxLogit = Double.NEGATIVE_INFINITY;
        for (double logit : logits) {
            if (logit > maxLogit) maxLogit = logit;
        }

        double sum = 0.0;
        double[] probabilities = new double[logits.length];

        for (int i = 0; i < logits.length; i++) {
            probabilities[i] = Math.exp(logits[i] - maxLogit);
            sum += probabilities[i];
        }

        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] /= sum;
        }

        return probabilities;
    }

    /**
     * Applies sigmoid to convert logits to probability.
     */
    public static double sigmoid(double logit) {
        return 1.0 / (1.0 + Math.exp(-logit));
    }

    /**
     * Applies threshold to binary classification.
     */
    public static int applyThreshold(double probability, double threshold) {
        return probability >= threshold ? 1 : 0;
    }

    /**
     * Extracts top-k predictions.
     */
    public static Map<Integer, Double> topK(double[] probabilities, int k) {
        Map<Integer, Double> topK = new HashMap<>();

        for (int iteration = 0; iteration < Math.min(k, probabilities.length); iteration++) {
            int maxIndex = -1;
            double maxValue = Double.NEGATIVE_INFINITY;

            for (int i = 0; i < probabilities.length; i++) {
                if (!topK.containsKey(i) && probabilities[i] > maxValue) {
                    maxValue = probabilities[i];
                    maxIndex = i;
                }
            }

            if (maxIndex >= 0) {
                topK.put(maxIndex, maxValue);
            }
        }

        return topK;
    }

    /**
     * Denormalizes predictions back to original scale.
     */
    public static double denormalize(double normalized, double min, double max) {
        return normalized * (max - min) + min;
    }
}
