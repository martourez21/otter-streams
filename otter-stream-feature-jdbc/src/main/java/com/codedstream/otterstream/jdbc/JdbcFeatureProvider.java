package com.codedstream.otterstream.jdbc;

import com.codedstream.otterstream.runtime.spi.FeatureProvider;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.ResultSetMetaData;
import java.sql.SQLException;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import javax.sql.DataSource;

/**
 * {@link FeatureProvider} backed by a JDBC-accessible table: one row per entity, one column per
 * feature. Depends only on {@code java.sql} — bring whatever JDBC driver (PostgreSQL, MySQL,
 * Snowflake, Redshift, ...) you need on the runtime classpath yourself.
 *
 * <h2>Expected data shape:</h2>
 * A table such as:
 * <pre>
 * CREATE TABLE user_features (
 *   user_id      VARCHAR PRIMARY KEY,
 *   age          INT,
 *   country      VARCHAR,
 *   ltv_score    DOUBLE PRECISION
 * );
 * </pre>
 *
 * <h2>Usage:</h2>
 * <pre>{@code
 * // simplest: a raw JDBC URL, a new Connection opened per fetch()
 * JdbcFeatureProvider features = new JdbcFeatureProvider(
 *         "jdbc:postgresql://localhost:5432/features", "reader", "secret",
 *         "user_features", "user_id");
 *
 * // production: bring your own pooled DataSource (HikariCP, etc.)
 * JdbcFeatureProvider features = new JdbcFeatureProvider(myDataSource, "user_features", "user_id");
 *
 * Map<String, Object> values = features.fetch("42", List.of("age", "country", "ltv_score"));
 * }</pre>
 *
 * <p><b>SQL injection note:</b> {@code entityId} is always sent as a bind parameter (never
 * concatenated). {@code table}, {@code entityIdColumn}, and the {@code featureNames} passed to
 * {@link #fetch} <em>are</em> interpolated directly into the generated SQL as identifiers —
 * these must come from trusted, developer-supplied configuration, never from end-user input.
 *
 * @since 0.1.0
 */
public class JdbcFeatureProvider implements FeatureProvider {

    private final DataSource dataSource;
    private final String jdbcUrl;
    private final String username;
    private final String password;
    private final String table;
    private final String entityIdColumn;

    /**
     * Opens a new {@link Connection} via {@link DriverManager} on every {@link #fetch} call.
     * Simple and dependency-free, but not pooled — fine for low/moderate QPS or prototyping;
     * for production throughput, prefer the {@link DataSource}-based constructor with a pooled
     * implementation such as HikariCP.
     *
     * @param jdbcUrl        JDBC connection URL (its driver must already be on the classpath)
     * @param username       database username
     * @param password       database password
     * @param table          table (or view) to query — trusted config, not user input
     * @param entityIdColumn column identifying the entity — trusted config, not user input
     */
    public JdbcFeatureProvider(String jdbcUrl, String username, String password, String table, String entityIdColumn) {
        this.dataSource = null;
        this.jdbcUrl = Objects.requireNonNull(jdbcUrl, "jdbcUrl cannot be null");
        this.username = username;
        this.password = password;
        this.table = Objects.requireNonNull(table, "table cannot be null");
        this.entityIdColumn = Objects.requireNonNull(entityIdColumn, "entityIdColumn cannot be null");
    }

    /**
     * Uses a caller-supplied, already-configured {@link DataSource} (recommended for
     * production — typically a connection pool). This provider never closes the DataSource;
     * the caller retains ownership of its lifecycle.
     *
     * @param dataSource     a configured JDBC data source
     * @param table          table (or view) to query — trusted config, not user input
     * @param entityIdColumn column identifying the entity — trusted config, not user input
     */
    public JdbcFeatureProvider(DataSource dataSource, String table, String entityIdColumn) {
        this.dataSource = Objects.requireNonNull(dataSource, "dataSource cannot be null");
        this.jdbcUrl = null;
        this.username = null;
        this.password = null;
        this.table = Objects.requireNonNull(table, "table cannot be null");
        this.entityIdColumn = Objects.requireNonNull(entityIdColumn, "entityIdColumn cannot be null");
    }

    @Override
    public String getProviderId() {
        return "jdbc";
    }

    /**
     * Fetches one row for {@code entityId} and returns the requested columns as a map.
     *
     * @param entityId     the entity id to look up (bound as a SQL parameter)
     * @param featureNames the columns to select; if null or empty, every column is selected
     * @return a map of column name to value, or an empty map if no row matched
     */
    @Override
    public Map<String, Object> fetch(String entityId, List<String> featureNames) throws Exception {
        Objects.requireNonNull(entityId, "entityId cannot be null");
        String columns = (featureNames == null || featureNames.isEmpty())
                ? "*"
                : String.join(", ", featureNames);
        String sql = "SELECT " + columns + " FROM " + table + " WHERE " + entityIdColumn + " = ?";

        try (Connection conn = getConnection();
             PreparedStatement ps = conn.prepareStatement(sql)) {
            ps.setString(1, entityId);
            try (ResultSet rs = ps.executeQuery()) {
                if (!rs.next()) {
                    return Collections.emptyMap();
                }
                ResultSetMetaData meta = rs.getMetaData();
                Map<String, Object> result = new LinkedHashMap<>();
                for (int i = 1; i <= meta.getColumnCount(); i++) {
                    result.put(meta.getColumnLabel(i), rs.getObject(i));
                }
                return result;
            }
        }
    }

    private Connection getConnection() throws SQLException {
        if (dataSource != null) {
            return dataSource.getConnection();
        }
        return DriverManager.getConnection(jdbcUrl, username, password);
    }
}
