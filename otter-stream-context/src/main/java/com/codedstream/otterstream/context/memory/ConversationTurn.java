package com.codedstream.otterstream.context.memory;

import java.util.Objects;

/**
 * One turn in a conversation — the unit {@link ConversationMemoryProvider} stores and retrieves.
 *
 * @param role            who said it, e.g. {@code "user"} or {@code "assistant"} — free-form,
 *                        not a fixed enum, so this fits whatever role vocabulary your LLM
 *                        integration already uses
 * @param content         the turn's text
 * @param timestampMillis when this turn was recorded
 * @since 0.1.0
 */
public record ConversationTurn(String role, String content, long timestampMillis) {

    public ConversationTurn {
        Objects.requireNonNull(role, "role cannot be null");
        Objects.requireNonNull(content, "content cannot be null");
    }
}
