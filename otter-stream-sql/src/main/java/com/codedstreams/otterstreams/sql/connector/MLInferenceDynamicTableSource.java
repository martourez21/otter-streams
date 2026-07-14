package com.codedstreams.otterstreams.sql.connector;

import com.codedstreams.otterstreams.sql.config.SqlInferenceConfig;
import org.apache.flink.table.connector.ChangelogMode;
import org.apache.flink.table.connector.source.DynamicTableSource;
import org.apache.flink.table.connector.source.LookupTableSource;
import org.apache.flink.table.connector.source.lookup.AsyncLookupFunctionProvider;
import org.apache.flink.table.connector.source.lookup.LookupFunctionProvider;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.types.DataType;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Dynamic table source for ML inference with lookup support.
 * Uses ModelCache singleton for model management.
 * Flink 2.0 compatible.
 */
public class MLInferenceDynamicTableSource implements LookupTableSource {

    private static final Logger LOG = LoggerFactory.getLogger(MLInferenceDynamicTableSource.class);

    private final SqlInferenceConfig config;
    private final DataType producedDataType;
    private final boolean asyncEnabled;

    public MLInferenceDynamicTableSource(SqlInferenceConfig config, DataType producedDataType) {
        this(config, producedDataType, true);
    }

    public MLInferenceDynamicTableSource(SqlInferenceConfig config, DataType producedDataType, boolean asyncEnabled) {
        this.config = config;
        this.producedDataType = producedDataType;
        this.asyncEnabled = asyncEnabled;
        LOG.info("MLInferenceDynamicTableSource initialized for model: {}, async: {}",
                config.getModelName(), asyncEnabled);
    }

    @Override
    public LookupRuntimeProvider getLookupRuntimeProvider(LookupContext context) {
        if (asyncEnabled) {
            return AsyncLookupFunctionProvider.of(new AsyncMLInferenceLookupFunction(config));
        } else {
            return LookupFunctionProvider.of(new MLInferenceLookupFunction(config));
        }
    }

    // for LookupTableSource. Only needed if implementing ScanTableSource.
    public ChangelogMode getChangelogMode() {
        return ChangelogMode.insertOnly();
    }

    @Override
    public DynamicTableSource copy() {
        return new MLInferenceDynamicTableSource(config, producedDataType, asyncEnabled);
    }

    @Override
    public String asSummaryString() {
        return "ML Inference Source (model: " + config.getModelName() + ", async: " + asyncEnabled + ")";
    }
}