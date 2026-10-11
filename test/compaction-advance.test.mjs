import assert from "node:assert/strict";
import test from "node:test";

import { didConversationAdvance } from "../dist/index.js";

const anchor = { id: "compact-turn", type: "message", message: { role: "assistant" } };

function branchWith(...tail) {
    return [{ id: "root", type: "message", message: { role: "user" } }, anchor, ...tail];
}

test("allows non-contextual session entries", () => {
    const branch = branchWith(
        { id: "extension-state", type: "custom" },
        { id: "checkpoint", type: "label" },
        { id: "session-name", type: "session_info" },
        { id: "model", type: "model_change" },
        { id: "thinking", type: "thinking_level_change" },
    );

    assert.equal(didConversationAdvance(branch, anchor.id), false);
});

test("cancels for messages and unflagged custom_message; a flagged passive notice does not", () => {
    for (const role of ["user", "assistant", "toolResult", "bashExecution"]) {
        assert.equal(
            didConversationAdvance(
                branchWith({ id: `message-${role}`, type: "message", message: { role } }),
                anchor.id,
            ),
            true,
            role,
        );
    }
    // A chime or a status line is not a conversational move, but only the writer knows that: an
    // unflagged custom_message is still how another extension injects new context, so it cancels.
    for (const notice of [
        { id: "chime", type: "custom_message", customType: "clock", display: true, details: { passive: true } },
        { id: "reminder", type: "custom_message", customType: "headroom", display: false, details: { passive: true } },
    ]) {
        assert.equal(didConversationAdvance(branchWith(notice), anchor.id), false, notice.id);
    }
    assert.equal(
        didConversationAdvance(
            branchWith({ id: "custom-message", type: "custom_message", content: "New context" }),
            anchor.id,
        ),
        true,
        "an unflagged custom_message still cancels",
    );
});

test("cancels for contextual entries, unknown types, or a missing compact anchor", () => {
    assert.equal(
        didConversationAdvance(branchWith({ id: "branch-summary", type: "branch_summary" }), anchor.id),
        true,
    );
    assert.equal(
        didConversationAdvance(branchWith({ id: "compaction", type: "compaction" }), anchor.id),
        true,
    );
    assert.equal(
        didConversationAdvance(branchWith({ id: "future", type: "future_context" }), anchor.id),
        true,
    );
    assert.equal(didConversationAdvance(branchWith(), "missing"), true);
    assert.equal(didConversationAdvance(branchWith(), null), true);
});
