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
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Properties;
import java.util.Set;

/**
 * Alternative flat-file rule configuration format, for teams whose config tooling standardizes
 * on {@code .properties} rather than YAML. Functionally equivalent to {@link YamlRuleSetSource};
 * YAML remains the recommended default (nesting reads more naturally for rule lists), but this
 * is a first-class, fully-supported option, not an afterthought.
 *
 * <h2>Format</h2>
 * <pre>{@code
 * name=fraud-detection-rules
 * version=3
 * evaluationMode=SINGLE
 *
 * rules.high-risk-score.priority=100
 * rules.high-risk-score.condition=output.risk_score > 0.85
 * rules.high-risk-score.flag=FRAUD
 * rules.high-risk-score.category=HIGH_RISK
 * rules.high-risk-score.color=#c0392b
 * rules.high-risk-score.enabled=true
 *
 * rules.default-approve.priority=0
 * rules.default-approve.condition=true
 * rules.default-approve.flag=APPROVE
 * rules.default-approve.color=#00875a
 * }</pre>
 *
 * Rule ids are derived from the segment immediately after {@code rules.} — {@code high-risk-score}
 * in the example above, matching {@link Rule#id()}.
 *
 * @since 0.1.0
 */
public final class PropertiesRuleSetSource implements RuleSetSource {

    private static final String RULES_PREFIX = "rules.";

    private final InputStream input;
    private final String sourceDescription;

    private PropertiesRuleSetSource(InputStream input, String sourceDescription) {
        this.input = input;
        this.sourceDescription = sourceDescription;
    }

    public static PropertiesRuleSetSource fromPath(Path path) throws RuleEvaluationException {
        try {
            return new PropertiesRuleSetSource(Files.newInputStream(path), path.toString());
        } catch (IOException e) {
            throw new RuleEvaluationException("Failed to open rule properties file: " + path, e);
        }
    }

    public static PropertiesRuleSetSource fromClasspath(String resourcePath) throws RuleEvaluationException {
        InputStream stream = PropertiesRuleSetSource.class.getClassLoader().getResourceAsStream(resourcePath);
        if (stream == null) {
            throw new RuleEvaluationException("Rule properties classpath resource not found: " + resourcePath);
        }
        return new PropertiesRuleSetSource(stream, "classpath:" + resourcePath);
    }

    @Override
    public RuleSet load() throws RuleEvaluationException {
        Properties props = new Properties();
        try {
            props.load(input);
        } catch (IOException e) {
            throw new RuleEvaluationException("Failed to parse rule properties (" + sourceDescription + ")", e);
        }

        String name = props.getProperty("name");
        if (name == null) {
            throw new RuleEvaluationException("Missing required 'name' property (" + sourceDescription + ")");
        }
        String version = props.getProperty("version", "unversioned");
        String modeRaw = props.getProperty("evaluationMode", "SINGLE");
        RuleEvaluationMode mode;
        try {
            mode = RuleEvaluationMode.valueOf(modeRaw.trim().toUpperCase());
        } catch (IllegalArgumentException e) {
            throw new RuleEvaluationException(
                    "Invalid evaluationMode '" + modeRaw + "' (" + sourceDescription + "); expected SINGLE or MULTIPLE");
        }

        Set<String> ruleIds = new LinkedHashSet<>();
        for (String key : props.stringPropertyNames()) {
            if (key.startsWith(RULES_PREFIX)) {
                String remainder = key.substring(RULES_PREFIX.length());
                int dot = remainder.indexOf('.');
                if (dot > 0) {
                    ruleIds.add(remainder.substring(0, dot));
                }
            }
        }

        List<Rule> rules = new ArrayList<>();
        for (String id : ruleIds) {
            String base = RULES_PREFIX + id + ".";
            String condition = props.getProperty(base + "condition");
            String flag = props.getProperty(base + "flag");
            if (condition == null || flag == null) {
                throw new RuleEvaluationException(
                        "Rule '" + id + "' is missing 'condition' or 'flag' (" + sourceDescription + ")");
            }
            String ruleName = props.getProperty(base + "name", id);
            int priority = parseIntOrDefault(props.getProperty(base + "priority"), 0);
            String category = props.getProperty(base + "category");
            String color = props.getProperty(base + "color");
            String description = props.getProperty(base + "description");
            boolean enabled = Boolean.parseBoolean(props.getProperty(base + "enabled", "true"));

            try {
                rules.add(new Rule(id, ruleName, priority, condition, flag, category, enabled, color, description));
            } catch (IllegalArgumentException e) {
                throw new RuleEvaluationException(
                        "Invalid rule '" + id + "' (" + sourceDescription + "): " + e.getMessage(), e);
            }
        }

        try {
            return new RuleSet(name, version, mode, rules);
        } catch (IllegalArgumentException e) {
            throw new RuleEvaluationException("Invalid rule set (" + sourceDescription + "): " + e.getMessage(), e);
        }
    }

    private static int parseIntOrDefault(String value, int defaultValue) {
        if (value == null) return defaultValue;
        try {
            return Integer.parseInt(value.trim());
        } catch (NumberFormatException e) {
            return defaultValue;
        }
    }
}
