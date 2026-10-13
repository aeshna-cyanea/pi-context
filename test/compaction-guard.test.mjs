import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";

import registerContext, { postCompactHint } from "../dist/index.js";

// The compaction-target guard (todos 58 + 77). These tests exist because the guard's
// whole job is arithmetic about window, and arithmetic is what a fixture can pin.
// Two failures measured on myself set the shape: a 12,484-char summary replacing 7
// nodes cost +4,625 tokens (2026-10-11), and a checkpoint-target compaction at
// 252,736/262,144 freed nothing and the session died (2026-10-13).

const usage = (percent, tokens, contextWindow = 262_144) => ({
    percent, tokens, contextWindow,
});

const user = (text) => ({ role: "user", content: text, timestamp: 1 });
const assistant = (content) => ({
    role: "assistant", content, api: "anthropic-messages", provider: "fixture",
    model: "offline", usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
    stopReason: "toolUse", timestamp: 1,
});
const text = (t) => [{ type: "text", text: t }];
const toolCall = (id, name, args = {}) => ({ type: "toolCall", id, name, arguments: args });
const bigText = (n) => "x".repeat(n);

function createHarness({ entries, usageBefore = usage(96, 251_000), labels = {} } = {}) {
    const commands = new Map(), tools = new Map(), events = new Map();
    const sent = [], notifications = [], labelled = new Map();
    let leaf = entries.at(-1).id;
    const ctx = {
        sessionManager: {
            getLeafId: () => leaf,
            getLabel: (id) => labels[id] ?? labelled.get(id),
            getBranch: () => entries,
            getTree: () => entries.map((entry) => ({ entry, children: [] })),
            getChildren: () => [],
        },
        getContextUsage: () => usageBefore,
        hasPendingMessages: () => false,
        abort: () => {},
        ui: { notify: (...args) => notifications.push(args) },
    };
    const commandCtx = { ...ctx, waitForIdle: async () => {}, navigateTree: async () => ({ editorText: undefined }) };
    registerContext({
        registerCommand: (name, definition) => commands.set(name, definition),
        registerTool: (definition) => tools.set(definition.name, definition),
        on: (name, handler) => events.set(name, handler),
        setLabel: (id, name) => labelled.set(id, name),
        sendUserMessage: () => { queueMicrotask(() => commands.get("acm").handler("", commandCtx)); },
        sendMessage: (message, options) => { sent.push({ message, options }); },
    });
    const compact = (params) => tools.get("context_compact").execute("compact-call", params, undefined, undefined, ctx);
    const checkpoint = (params) => tools.get("context_checkpoint").execute("cp-call", params, undefined, undefined, ctx);
    const timeline = (params = {}) => tools.get("context_timeline").execute("tl-call", params, undefined, undefined, ctx);
    return { compact, checkpoint, timeline, ctx, sent, notifications, labelled };
}

// root -> summary -> small step -> the compact request itself. Targeting the small
// step folds one short node while a long summary is written: pure loss.
const shallowPath = () => ([
    { id: "rootnode", type: "message", message: user("Root prefix that stays in context.") },
    { id: "summary12", type: "branch_summary", summary: "Earlier handoff, already on this path." },
    { id: "smallstep", type: "message", message: assistant(text("One short recent step.")) },
    { id: "request1", type: "message", message: assistant([toolCall("compact-call", "context_compact")]) },
]);

test("refuses a target that costs more window than it frees, and names the better one", async () => {
    const h = createHarness({ entries: shallowPath() });
    await assert.rejects(
        h.compact({ target: "smallstep", summary: bigText(12_000) }),
        (err) => {
            assert.match(err.message, /refusing this target at 96\.0% \(251k\/262k\) fill/);
            assert.match(err.message, /nets ~-\d/);
            assert.match(err.message, /Better target: summary12 \(previous summary on this path\)/);
            assert.match(err.message, /Recipe for this fold \(this fold is small: one or two lines/);
            return true;
        },
    );
});

test("a huge single-node fold is accepted: the yardstick is tokens, not node count", async () => {
    // autumn's correction, 2026-10-13: compacting one node back past an enormous tool
    // result is a GOOD compaction. A node-count threshold would have refused it.
    const entries = [
        { id: "rootnode", type: "message", message: user("Root prefix that stays in context.") },
        { id: "callread1", type: "message", message: assistant([toolCall("r1", "bash")]) },
        { id: "hugeout1", type: "message", message: {
            role: "toolResult", toolName: "bash", toolCallId: "r1", content: text(bigText(60_000)),
            isError: false, timestamp: 1,
        } },
        { id: "request1", type: "message", message: assistant([toolCall("compact-call", "context_compact")]) },
    ];
    const h = createHarness({ entries });
    const result = await h.compact({ target: "callread1", summary: "The dump established X; keep the conclusion, drop the payload." });
    assert.match(result.content[0].text, /^compact start/);
    assert.doesNotMatch(result.content[0].text, /WARNING/);
    assert.match(result.content[0].text, /folds 2 node\(s\) \/ ~15k tokens/);
    assert.match(result.content[0].text, /net ~\+15k of window/);
});

test("a marginal target under no pressure warns and proceeds instead of blocking", async () => {
    const h = createHarness({ entries: shallowPath(), usageBefore: usage(18, 47_000) });
    const result = await h.compact({ target: "smallstep", summary: bigText(12_000) });
    assert.match(result.content[0].text, /^compact start/);
    assert.match(result.content[0].text, /WARNING: this target barely helps/);
    assert.match(result.content[0].text, /summary only needs to cover the folded region/);
});

test("force: true is the deliberate override, and it is mine to pass", async () => {
    const h = createHarness({ entries: shallowPath() });
    const result = await h.compact({ target: "smallstep", summary: bigText(12_000), force: true });
    assert.match(result.content[0].text, /^compact start/);
    assert.match(result.content[0].text, /net ~-3k of window/);
});

test("'previous-summary' resolves to the newest summary already on the path", async () => {
    // Low pressure here on purpose: this fixture folds only a handful of tokens, so at
    // 96% fill the guard would (correctly) refuse it. The alias is what is under test.
    const h = createHarness({ entries: shallowPath(), usageBefore: usage(18, 47_000) });
    const result = await h.compact({ target: "previous-summary", summary: "Short handoff." });
    // summary12 folds the two nodes after it; the request node plus the small step.
    assert.match(result.content[0].text, /^compact start/);
    assert.match(result.content[0].text, /folds 2 node\(s\)/);  // everything after summary12
});

test("a fold dominated by one huge result is labelled, and the recipe fits it", async () => {
    const entries = [
        { id: "rootnode", type: "message", message: user("Root prefix that stays in context.") },
        { id: "callread1", type: "message", message: assistant([toolCall("r1", "bash")]) },
        { id: "hugeout1", type: "message", message: {
            role: "toolResult", toolName: "bash", toolCallId: "r1", content: text(bigText(60_000)),
            isError: false, timestamp: 1,
        } },
        { id: "request1", type: "message", message: assistant([toolCall("compact-call", "context_compact")]) },
    ];
    const h = createHarness({ entries, labels: { callread1: "before-the-dump" } });
    const tl = await h.timeline({});
    assert.match(tl.content[0].text, /before-the-dump.*reclaim ~15k \(1 huge result\)/);
    const result = await h.compact({ target: "before-the-dump", summary: "The dump proved X; payload is in /tmp/out.txt." });
    assert.match(result.content[0].text, /fold shape: 1 huge result -- one large result dominates this fold/);
    assert.match(result.content[0].text, /cite the file or command instead of restating its content/);
});

test("a long chain of work is labelled as a chain and asked for a lossy-but-correct handoff", async () => {
    const entries = [{ id: "rootnode", type: "message", message: user("Root prefix that stays.") }];
    for (let i = 0; i < 12; i++) {
        entries.push({ id: `step${i}`, type: "message",
            message: assistant(text(`Step ${i} of the work, each carrying a little real content. `.repeat(12))) });
    }
    entries.push({ id: "request1", type: "message", message: assistant([toolCall("compact-call", "context_compact")]) });
    const h = createHarness({ entries, labels: { rootnode: "chain-start" } });
    const tl = await h.timeline({});
    assert.match(tl.content[0].text, /chain-start.*reclaim ~\d+k \(chain\)/);
    const result = await h.compact({ target: "chain-start", summary: "State, decisions, side effects, next step." });
    assert.match(result.content[0].text, /fold shape: chain -- a chain of work is being folded/);
});

test("a small fold is labelled short, so I do not write a three-paragraph handoff for it", async () => {
    const h = createHarness({ entries: shallowPath(), usageBefore: usage(18, 47_000), labels: { smallstep: "just-now" } });
    const tl = await h.timeline({});
    assert.match(tl.content[0].text, /just-now.*reclaim ~\d+ \(short\)/);
});

test("the post-compaction hint is configurable, and a missing file is loud rather than silent", () => {
    const dir = mkdtempSync(join(tmpdir(), "pc-hint-"));
    const file = join(dir, "hint.txt");
    const saved = { ...process.env };
    const warnings = [];
    const realWarn = console.warn;
    try {
        delete process.env.PI_CONTEXT_POST_COMPACT_HINT_FILE;
        delete process.env.PI_CONTEXT_POST_COMPACT_HINT;
        assert.match(postCompactHint(), /still in this session file and searchable/);
        process.env.PI_CONTEXT_POST_COMPACT_HINT = "inline hint";
        assert.equal(postCompactHint(), "inline hint");
        writeFileSync(file, "file hint with a trailing newline\n");
        process.env.PI_CONTEXT_POST_COMPACT_HINT_FILE = file;
        assert.equal(postCompactHint(), "file hint with a trailing newline");   // trimmed, and read per call, so no restart to change it
        console.warn = (...a) => warnings.push(a.join(" "));
        process.env.PI_CONTEXT_POST_COMPACT_HINT_FILE = join(dir, "absent.txt");
        assert.equal(postCompactHint(), "inline hint");
        console.warn = realWarn;
        assert.match(warnings.join("\n"), /cannot read PI_CONTEXT_POST_COMPACT_HINT_FILE/);
    } finally { console.warn = realWarn; process.env = saved; }
});

test("context_checkpoint reports fill and warns when the anchor sits deep", async () => {
    const h = createHarness({ entries: shallowPath() });
    const result = await h.checkpoint({ name: "phase-done" });
    assert.match(result.content[0].text, /Created checkpoint 'phase-done' at/);
    assert.match(result.content[0].text, /Context fill now: 96\.0% \(251k\/262k\)/);
    assert.match(result.content[0].text, /Warning: this anchor sits at 96\.0% fill/);
    assert.match(result.content[0].text, /A checkpoint is not a compaction target/);
    assert.match(result.content[0].text, /target summary12 \(previous summary on this path\)/);
});

test("context_checkpoint stays quiet when there is no pressure to warn about", async () => {
    const h = createHarness({ entries: shallowPath(), usageBefore: usage(12, 31_000) });
    const result = await h.checkpoint({ name: "phase-start" });
    assert.match(result.content[0].text, /Created checkpoint/);
    assert.doesNotMatch(result.content[0].text, /Warning/);
});

test("context_timeline prices each target candidate with its reclaimable size", async () => {
    const h = createHarness({ entries: shallowPath(), labels: { smallstep: "recent-anchor" } });
    const result = await h.timeline({});
    const lines = result.content[0].text.split("\n");
    const summaryLine = lines.find((line) => line.includes("summary12"));
    const anchorLine = lines.find((line) => line.includes("smallstep"));
    assert.match(summaryLine, /reclaim ~\d+[k]?/);
    assert.match(anchorLine, /reclaim ~\d+[k]?/);
    // The summary sits above the anchor, so compacting there must free strictly more.
    const size = (line) => Number(/reclaim ~(\d+)k?/.exec(line)[1]) * (line.includes("reclaim ~") && /\d+k/.test(line) ? 1000 : 1);
    assert.ok(size(summaryLine) >= size(anchorLine), `${summaryLine}\n must reclaim at least ${anchorLine}`);
});
