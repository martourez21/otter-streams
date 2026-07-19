package com.codedstream.otterstream.rules.engine;

import com.codedstream.otterstream.rules.model.Rule;
import com.codedstream.otterstream.rules.model.RuleEvaluationException;
import com.codedstream.otterstream.rules.model.RuleEvaluationMode;
import com.codedstream.otterstream.rules.model.RuleSet;
import com.codedstream.otterstream.rules.spi.RuleSetSource;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.yaml.snakeyaml.LoaderOptions;
import org.yaml.snakeyaml.Yaml;
import org.yaml.snakeyaml.constructor.SafeConstructor;

/**
 * The standard, default way to configure rules: a YAML file.
 *
 * <h2>Format</h2>
 * <pre>{@code
 * name: fraud-detection-rules
 * version: "3"
 * evaluationMode: SINGLE          # or MULTIPLE — see RuleEvaluationMode
 * rules:
 *   - id: high-risk-score
 *     name: High Risk Score
 *     priority: 100
 *     condition: "output.risk_score > 0.85"
 *     flag: FRAUD
 *     category: HIGH_RISK          # optional
 *     color: "#c0392b"              # optional, hex, dashboard display only
 *     description: "..."            # optional
 *     enabled: true                 # optional, defaults to true
 *   - id: default-approve
 *     priority: 0
 *     condition: "true"
 *     flag: APPROVE
 *     color: "#00875a"
 * }</pre>
 *
 * <p>Uses SnakeYAML's {@link SafeConstructor} explicitly — arbitrary Java object
 * instantiation from YAML tags is disabled, since rule files are configuration that may come
 * from outside the JVM's own trust boundary (an ops team editing a file, a config service).
 *
 * @since 0.1.0
 */
public final class YamlRuleSetSource implements RuleSetSource {

    private final InputStream input;
    private final String sourceDescription;

    private YamlRuleSetSource(InputStream input, String sourceDescription) {
        this.input = input;
        this.sourceDescription = sourceDescription;
    }

    public static YamlRuleSetSource fromPath(Path path) throws RuleEvaluationException {
        try {
            return new YamlRuleSetSource(Files.newInputStream(path), path.toString());
        } catch (IOException e) {
            throw new RuleEvaluationException("Failed to open rule YAML file: " + path, e);
        }
    }

    public static YamlRuleSetSource fromClasspath(String resourcePath) throws RuleEvaluationException {
        InputStream stream = YamlRuleSetSource.class.getClassLoader().getResourceAsStream(resourcePath);
        if (stream == null) {
            throw new RuleEvaluationException("Rule YAML classpath resource not found: " + resourcePath);
        }
        return new YamlRuleSetSource(stream, "classpath:" + resourcePath);
    }

    public static YamlRuleSetSource fromString(String yaml) {
        return new YamlRuleSetSource(
                new java.io.ByteArrayInputStream(yaml.getBytes(java.nio.charset.StandardCharsets.UTF_8)),
                "<inline>");
    }

    @Override
    @SuppressWarnings("unchecked")
    public RuleSet load() throws RuleEvaluationException {
        LoaderOptions loaderOptions = new LoaderOptions();
        Yaml yaml = new Yaml(new SafeConstructor(loaderOptions));

        Object parsed;
        try {
            parsed = yaml.load(input);
        } catch (Exception e) {
            throw new RuleEvaluationException("Failed to parse rule YAML (" + sourceDescription + ")", e);
        }
        if (!(parsed instanceof Map)) {
            throw new RuleEvaluationException("Rule YAML (" + sourceDescription + ") must have a top-level mapping");
        }
        Map<String, Object> root = (Map<String, Object>) parsed;

        String name = requireString(root, "name", sourceDescription);
        String version = stringOrDefault(root, "version", "unversioned");
        RuleEvaluationMode mode = parseMode(stringOrDefault(root, "evaluationMode", "SINGLE"), sourceDescription);

        Object rulesRaw = root.get("rules");
        if (!(rulesRaw instanceof List)) {
            throw new RuleEvaluationException("Rule YAML (" + sourceDescription + ") must have a 'rules' list");
        }

        List<Rule> rules = new ArrayList<>();
        for (Object entry : (List<?>) rulesRaw) {
            if (!(entry instanceof Map)) {
                throw new RuleEvaluationException("Each entry under 'rules' must be a mapping (" + sourceDescription + ")");
            }
            rules.add(parseRule((Map<String, Object>) entry, sourceDescription));
        }

        try {
            return new RuleSet(name, version, mode, rules);
        } catch (IllegalArgumentException e) {
            throw new RuleEvaluationException("Invalid rule set (" + sourceDescription + "): " + e.getMessage(), e);
        }
    }

    private Rule parseRule(Map<String, Object> entry, String sourceDescription) throws RuleEvaluationException {
        String id = requireString(entry, "id", sourceDescription);
        String name = stringOrDefault(entry, "name", id);
        int priority = intOrDefault(entry, "priority", 0);
        String condition = requireString(entry, "condition", sourceDescription);
        String flag = requireString(entry, "flag", sourceDescription);
        String category = entry.get("category") != null ? entry.get("category").toString() : null;
        String color = entry.get("color") != null ? entry.get("color").toString() : null;
        String description = entry.get("description") != null ? entry.get("description").toString() : null;
        boolean enabled = booleanOrDefault(entry, "enabled", true);

        try {
            return new Rule(id, name, priority, condition, flag, category, enabled, color, description);
        } catch (IllegalArgumentException e) {
            throw new RuleEvaluationException(
                    "Invalid rule '" + id + "' (" + sourceDescription + "): " + e.getMessage(), e);
        }
    }

    private static RuleEvaluationMode parseMode(String raw, String sourceDescription) throws RuleEvaluationException {
        try {
            return RuleEvaluationMode.valueOf(raw.trim().toUpperCase());
        } catch (IllegalArgumentException e) {
            throw new RuleEvaluationException(
                    "Invalid evaluationMode '" + raw + "' (" + sourceDescription + "); expected SINGLE or MULTIPLE");
        }
    }

    private static String requireString(Map<String, Object> map, String key, String sourceDescription)
            throws RuleEvaluationException {
        Object value = map.get(key);
        if (value == null) {
            throw new RuleEvaluationException("Missing required field '" + key + "' (" + sourceDescription + ")");
        }
        return value.toString();
    }

    private static String stringOrDefault(Map<String, Object> map, String key, String defaultValue) {
        Object value = map.get(key);
        return value != null ? value.toString() : defaultValue;
    }

    private static int intOrDefault(Map<String, Object> map, String key, int defaultValue) {
        Object value = map.get(key);
        return value instanceof Number number ? number.intValue() : defaultValue;
    }

    private static boolean booleanOrDefault(Map<String, Object> map, String key, boolean defaultValue) {
        Object value = map.get(key);
        return value instanceof Boolean bool ? bool : defaultValue;
    }
}
