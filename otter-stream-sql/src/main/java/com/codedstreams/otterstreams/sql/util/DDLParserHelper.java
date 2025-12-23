package com.codedstreams.otterstreams.sql.util;

import java.util.HashMap;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Helper for parsing SQL DDL options.
 *
 * @author Nestor Martourez Abiangang A.
 * @since 1.0.0
 */
public class DDLParserHelper {
    private static final Pattern OPTION_PATTERN = Pattern.compile("'([^']+)'\\s*=\\s*'([^']+)'");

    /**
     * Parses WITH clause options from DDL.
     */
    public static Map<String, String> parseWithClause(String withClause) {
        Map<String, String> options = new HashMap<>();

        Matcher matcher = OPTION_PATTERN.matcher(withClause);
        while (matcher.find()) {
            String key = matcher.group(1);
            String value = matcher.group(2);
            options.put(key, value);
        }

        return options;
    }

    /**
     * Validates option key format.
     */
    public static boolean isValidOptionKey(String key) {
        return key != null && key.matches("[a-zA-Z0-9._-]+");
    }

    /**
     * Converts option key to standard format.
     */
    public static String normalizeOptionKey(String key) {
        return key.toLowerCase().replace('_', '.');
    }
}
