package com.codedstreams.otterstreams.sql.connector;

import com.codedstreams.otterstreams.sql.config.SqlInferenceConfig;
import org.apache.flink.configuration.ConfigOption;
import org.apache.flink.configuration.ReadableConfig;
import org.apache.flink.table.connector.source.DynamicTableSource;
import org.apache.flink.table.factories.DynamicTableSourceFactory;
import org.apache.flink.table.factories.FactoryUtil;

import java.util.HashSet;
import java.util.Set;

/**
 * Factory for creating ML inference table sources.
 */
public class MLInferenceDynamicTableFactory implements DynamicTableSourceFactory {

    public static final String IDENTIFIER = "ml-inference";

    @Override
    public String factoryIdentifier() {
        return IDENTIFIER;
    }

    @Override
    public Set<ConfigOption<?>> requiredOptions() {
        Set<ConfigOption<?>> options = new HashSet<>();
        // Add required options
        return options;
    }

    @Override
    public Set<ConfigOption<?>> optionalOptions() {
        Set<ConfigOption<?>> options = new HashSet<>();
        // Add optional options
        return options;
    }

    @Override
    public DynamicTableSource createDynamicTableSource(Context context) {
        FactoryUtil.TableFactoryHelper helper = FactoryUtil.createTableFactoryHelper(this, context);
        helper.validate();

        ReadableConfig options = helper.getOptions();
        SqlInferenceConfig config = SqlInferenceConfig.fromOptions(
                context.getCatalogTable().getOptions()
        );

        return new MLInferenceDynamicTableSource(config, context.getPhysicalRowDataType());
    }
}
