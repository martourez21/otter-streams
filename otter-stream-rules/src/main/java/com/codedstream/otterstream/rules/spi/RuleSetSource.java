package com.codedstream.otterstream.rules.spi;

import com.codedstream.otterstream.rules.model.RuleEvaluationException;
import com.codedstream.otterstream.rules.model.RuleSet;

/**
 * Loads a {@link RuleSet} from wherever it's configured. YAML ({@code YamlRuleSetSource}) is the
 * standard/default path; {@code PropertiesRuleSetSource} is the alternative flat-file format;
 * a project can also implement this interface directly for a fully programmatic rule set
 * (the "create a class to define the rules" option) — there's no separate "class-based" SPI
 * beyond this one, since a hand-written {@code RuleSetSource} that builds and returns a
 * {@link RuleSet} in code *is* that option.
 *
 * @since 0.1.0
 */
@FunctionalInterface
public interface RuleSetSource {
    /**
     * @return the loaded rule set
     * @throws RuleEvaluationException if the source is malformed or unreadable
     */
    RuleSet load() throws RuleEvaluationException;
}
