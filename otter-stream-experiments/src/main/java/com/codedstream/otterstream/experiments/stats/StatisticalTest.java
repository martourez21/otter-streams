package com.codedstream.otterstream.experiments.stats;

import java.util.List;

/**
 * Two statistical comparisons for experiment outcomes, implemented without a math library
 * dependency (Apache Commons Math, etc.) — consistent with this project's general preference
 * for keeping optional modules dependency-light (see {@code otter-stream-rules}'s
 * hand-rolled expression evaluator for the same reasoning). Both are standard, well-documented
 * techniques, not novel statistics:
 *
 * <ul>
 *   <li>{@link #welchTTest} — for continuous metrics (confidence scores, latency, any real
 *       -valued measurement) where control and variant may have different variances.</li>
 *   <li>{@link #twoProportionZTest} — for binary/proportion metrics (conversion rate, fraud-flag
 *       rate — anything that's fundamentally "did X happen, yes or no" aggregated into a rate).</li>
 * </ul>
 *
 * <p><b>Honesty note on the p-value approximation:</b> {@link #welchTTest} computes the t
 * statistic and Welch–Satterthwaite degrees of freedom correctly, but converts it to a p-value
 * via a normal-distribution approximation rather than the exact Student's t-distribution CDF
 * (which needs a numerical incomplete-beta-function implementation to do without a math
 * library). This approximation is standard practice and accurate for moderate-to-large sample
 * sizes (a rough rule of thumb: each group should have at least ~30 observations) but will be
 * slightly conservative for small samples — call out the sample sizes alongside any p-value this
 * produces, don't just report the number in isolation.
 *
 * @since 0.1.0
 */
public final class StatisticalTest {

    private StatisticalTest() {
    }

    /**
     * Result of a two-sample comparison.
     *
     * @param controlMean       mean of the control group's observations
     * @param variantMean       mean of the variant group's observations
     * @param controlSampleSize number of control observations
     * @param variantSampleSize number of variant observations
     * @param statistic         the test statistic (t or z, depending on which test produced this)
     * @param pValue            two-tailed p-value (approximate for {@link #welchTTest}, see class Javadoc)
     * @param significantAt95   convenience: {@code pValue < 0.05}
     */
    public record ComparisonResult(
            double controlMean,
            double variantMean,
            int controlSampleSize,
            int variantSampleSize,
            double statistic,
            double pValue,
            boolean significantAt95) {
    }

    /**
     * Welch's t-test (unequal variance) comparing the means of two continuous samples.
     *
     * @param control control group's observed metric values
     * @param variant variant group's observed metric values
     * @return the comparison result
     * @throws IllegalArgumentException if either sample has fewer than 2 observations
     */
    public static ComparisonResult welchTTest(List<Double> control, List<Double> variant) {
        requireMinSize(control, "control");
        requireMinSize(variant, "variant");

        double meanA = mean(control);
        double meanB = mean(variant);
        double varA = variance(control, meanA);
        double varB = variance(variant, meanB);
        int nA = control.size();
        int nB = variant.size();

        double seA = varA / nA;
        double seB = varB / nB;
        double standardError = Math.sqrt(seA + seB);

        double t = standardError == 0.0 ? 0.0 : (meanB - meanA) / standardError;

        // Welch–Satterthwaite degrees of freedom — computed correctly; used only to sanity-check
        // sample adequacy, not fed into an exact t-distribution CDF (see class Javadoc).
        double numerator = (seA + seB) * (seA + seB);
        double denominator = (seA * seA) / (nA - 1) + (seB * seB) / (nB - 1);
        double degreesOfFreedom = denominator == 0.0 ? (nA + nB - 2) : numerator / denominator;

        double pValue = twoTailedPValueFromZ(t);
        // Small-sample correction: widen (increase) the p-value slightly when degrees of freedom
        // is low, since the normal approximation understates tail probability there. This is a
        // coarse correction, not a substitute for the exact t-distribution — see class Javadoc.
        if (degreesOfFreedom < 30) {
            pValue = Math.min(1.0, pValue * (30.0 / Math.max(1.0, degreesOfFreedom)));
        }

        return new ComparisonResult(meanA, meanB, nA, nB, t, pValue, pValue < 0.05);
    }

    /**
     * Two-proportion z-test comparing conversion/flag rates between two binary samples.
     *
     * @param controlSuccesses number of "positive" outcomes in the control group (e.g. flagged as fraud)
     * @param controlTotal     total control observations
     * @param variantSuccesses number of "positive" outcomes in the variant group
     * @param variantTotal     total variant observations
     * @return the comparison result (means here are the two proportions)
     */
    public static ComparisonResult twoProportionZTest(
            int controlSuccesses, int controlTotal, int variantSuccesses, int variantTotal) {
        if (controlTotal < 1 || variantTotal < 1) {
            throw new IllegalArgumentException("Both groups need at least 1 observation");
        }
        double pA = (double) controlSuccesses / controlTotal;
        double pB = (double) variantSuccesses / variantTotal;
        double pooled = (double) (controlSuccesses + variantSuccesses) / (controlTotal + variantTotal);

        double standardError = Math.sqrt(pooled * (1 - pooled) * (1.0 / controlTotal + 1.0 / variantTotal));
        double z = standardError == 0.0 ? 0.0 : (pB - pA) / standardError;
        double pValue = twoTailedPValueFromZ(z);

        return new ComparisonResult(pA, pB, controlTotal, variantTotal, z, pValue, pValue < 0.05);
    }

    private static void requireMinSize(List<Double> sample, String label) {
        if (sample == null || sample.size() < 2) {
            throw new IllegalArgumentException(
                    "The " + label + " sample needs at least 2 observations to compute variance, had "
                            + (sample == null ? 0 : sample.size()));
        }
    }

    private static double mean(List<Double> values) {
        double sum = 0.0;
        for (double v : values) sum += v;
        return sum / values.size();
    }

    private static double variance(List<Double> values, double mean) {
        double sumSquares = 0.0;
        for (double v : values) {
            double diff = v - mean;
            sumSquares += diff * diff;
        }
        return sumSquares / (values.size() - 1);
    }

    private static double twoTailedPValueFromZ(double z) {
        double p = 2 * (1 - standardNormalCdf(Math.abs(z)));
        return Math.max(0.0, Math.min(1.0, p));
    }

    /**
     * Standard normal CDF via the Abramowitz &amp; Stegun 7.1.26 erf approximation
     * (max error ~1.5e-7) — avoids pulling in a math library for this one function.
     */
    private static double standardNormalCdf(double x) {
        return 0.5 * (1 + erf(x / Math.sqrt(2)));
    }

    private static double erf(double x) {
        double sign = x < 0 ? -1 : 1;
        x = Math.abs(x);

        double a1 = 0.254829592;
        double a2 = -0.284496736;
        double a3 = 1.421413741;
        double a4 = -1.453152027;
        double a5 = 1.061405429;
        double p = 0.3275911;

        double t = 1.0 / (1.0 + p * x);
        double y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
        return sign * y;
    }
}
