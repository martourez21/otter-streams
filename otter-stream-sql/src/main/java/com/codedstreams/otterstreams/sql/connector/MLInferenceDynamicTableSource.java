package com.codedstreams.otterstreams.sql.connector;

import com.codedstreams.otterstreams.sql.config.SqlInferenceConfig;
import org.apache.flink.table.connector.ChangelogMode;
import org.apache.flink.table.connector.source.DynamicTableSource;
import org.apache.flink.table.connector.source.LookupTableSource;
import org.apache.flink.table.connector.source.TableFunctionProvider;
import org.apache.flink.table.data.RowData;
import org.apache.flink.table.types.DataType;

/**
 * Dynamic table source for ML inference with lookup support.
 */
public class MLInferenceDynamicTableSource implements LookupTableSource {

    private final SqlInferenceConfig config;
    private final DataType producedDataType;

    public MLInferenceDynamicTableSource(SqlInferenceConfig config, DataType producedDataType) {
        this.config = config;
        this.producedDataType = producedDataType;
    }

    @Override
    public LookupRuntimeProvider getLookupRuntimeProvider(LookupContext context) {
        return TableFunctionProvider.of(new MLInferenceLookupFunction(config));
    }

    public ChangelogMode getChangelogMode() {
        return ChangelogMode.insertOnly();
    }

    @Override
    public DynamicTableSource copy() {
        return new MLInferenceDynamicTableSource(config, producedDataType);
    }

    @Override
    public String asSummaryString() {
        return "ML Inference Source";
    }
}
