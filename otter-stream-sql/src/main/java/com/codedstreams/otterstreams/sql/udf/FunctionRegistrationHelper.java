package com.codedstreams.otterstreams.sql.udf;

import org.apache.flink.table.api.TableEnvironment;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Helper class for registering SQL functions programmatically.
 *
 * @author Nestor Martourez Abiangang A.
 * @since 1.0.0
 */
public class FunctionRegistrationHelper {
    private static final Logger LOG = LoggerFactory.getLogger(FunctionRegistrationHelper.class);

    /**
     * Registers all Otter Stream SQL functions.
     */
    public static void registerAllFunctions(TableEnvironment tableEnv) {
        try {
            // Register scalar function
            tableEnv.createTemporarySystemFunction(
                    "ML_PREDICT",
                    MLPredictScalarFunction.class
            );
            LOG.info("Registered ML_PREDICT scalar function");

            // Register table function
            tableEnv.createTemporarySystemFunction(
                    "ML_PREDICT_TABLE",
                    MLPredictTableFunction.class
            );
            LOG.info("Registered ML_PREDICT_TABLE table function");

            // Register aggregate function
            tableEnv.createTemporarySystemFunction(
                    "ML_PREDICT_AGG",
                    MLPredictAggregateFunction.class
            );
            LOG.info("Registered ML_PREDICT_AGG aggregate function");

            LOG.info("All Otter Stream SQL functions registered successfully");
        } catch (Exception e) {
            LOG.error("Failed to register SQL functions", e);
            throw new RuntimeException("Function registration failed", e);
        }
    }

    /**
     * Unregisters all Otter Stream SQL functions.
     */
    public static void unregisterAllFunctions(TableEnvironment tableEnv) {
        try {
            tableEnv.dropTemporarySystemFunction("ML_PREDICT");
            tableEnv.dropTemporarySystemFunction("ML_PREDICT_TABLE");
            tableEnv.dropTemporarySystemFunction("ML_PREDICT_AGG");
            LOG.info("All Otter Stream SQL functions unregistered");
        } catch (Exception e) {
            LOG.warn("Error unregistering functions", e);
        }
    }
}
