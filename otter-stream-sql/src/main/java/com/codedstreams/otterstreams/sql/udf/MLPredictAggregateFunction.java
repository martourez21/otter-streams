package com.codedstreams.otterstreams.sql.udf;

import org.apache.flink.table.functions.AggregateFunction;
import java.util.ArrayList;
import java.util.List;

/**
 * Aggregate function for batch inference over windows.
 */
public class MLPredictAggregateFunction extends AggregateFunction<Double, MLPredictAggregateFunction.Accumulator> {

    public static class Accumulator {
        public List<String> features = new ArrayList<>();
        public double sumScore = 0.0;
        public int count = 0;
    }

    @Override
    public Accumulator createAccumulator() {
        return new Accumulator();
    }

    public void accumulate(Accumulator acc, String featuresJson) {
        acc.features.add(featuresJson);
        acc.count++;
    }

    @Override
    public Double getValue(Accumulator acc) {
        if (acc.count == 0) return 0.0;
        // Perform batch inference here
        return acc.sumScore / acc.count;
    }
}
