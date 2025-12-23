package com.codedstreams.otterstreams.sql.udf;

import org.apache.flink.table.catalog.CatalogFunction;
import org.apache.flink.table.factories.FunctionDefinitionFactory;
import org.apache.flink.table.functions.FunctionDefinition;
import org.apache.flink.table.functions.ScalarFunctionDefinition;

import java.util.Collections;
import java.util.Set;

public class MLInferenceFunctionFactory implements FunctionDefinitionFactory {

    public String factoryIdentifier() {
        return "ml_infer";
    }

    // No function requirements are needed
    public Set requirements() {
        return Collections.emptySet();
    }

    @Override
    public FunctionDefinition createFunctionDefinition(
            String name,
            CatalogFunction catalogFunction,
            Context context) {

        // Use the constructor directly
        return new ScalarFunctionDefinition(
                "ml_infer",
                new MLInferenceFunction()
        );
    }
}
