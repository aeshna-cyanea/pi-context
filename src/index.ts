import { readFileSync } from "node:fs";
import {
    type ExtensionAPI,
    type SessionManager,
    type SessionEntry,
    type ExtensionCommandContext,
    type ContextUsage,
} from "@earendil-works/pi-coding-agent";
import {
    Type,
    type Static,
    type TextContent,
    type ImageContent,
    type ToolCall,
} from "@earendil-works/pi-ai";
import { describeHistoryInterval, estimateHistoryTokens, formatTokens, formatContextUsage, isContextTool as isInternal, parseCheckpointPhase, RetainedCodemodeOutputType } from "./utils.js";

// Define missing types locally as they are not exported from the main entry point
interface SessionTreeNode {
    entry: SessionEntry;
    children: SessionTreeNode[];
    label?: string;
}

const PiContextCustomMessageType = "pi-context";

const PassiveCompactionEntryTypes = new Set<SessionEntry["type"]>([
    "custom",
    "label",
    "session_info",
    "model_change",
    "thinking_level_change",
]);

// Pi >= 0.99 records nested calls on the parent result, not as transcript messages.
// Keep this shape local so direct calls remain compatible with older Pi SDKs.
interface CompactNestedCallRecord {
    calls: { id: string; name: string; status: string }[];
}

// Retain only a successful parent result with one identifiable successful compact.
// Other nested tools/models and omitted arguments are safe because the entire
// parent output is carried forward; missing or ambiguous compact records cancel.
const findCodemodeCompactResult = (
    branch: readonly SessionEntry[],
    compactToolCallId: string,
    codemodeToolCallId: string,
) => {
    const results = branch.filter((entry) => entry.type === "message" &&
        entry.message.role === "toolResult" && entry.message.toolName === "codemode" &&
        entry.message.toolCallId === codemodeToolCallId);
    if (results.length !== 1) return undefined;
    const result = results[0];
    if (result.type !== "message" || result.message.role !== "toolResult" || result.message.isError) return undefined;
    const record = (result.message as typeof result.message & { nestedCalls?: CompactNestedCallRecord }).nestedCalls;
    if (!Array.isArray(record?.calls)) return undefined;
    const compactCalls = record.calls.filter((call) => call.name === "context_compact");
    if (compactCalls.length !== 1 || compactCalls[0].id !== compactToolCallId ||
        compactCalls[0].status !== "ok") return undefined;
    return { entry: result, message: result.message };
};

/**
 * Detect conversation advancement since the compact request, not agent_end.
 * A successful codemode result can be retained in full; other contextual
 * entries still cancel because the handoff summary does not account for them.
 */
/**
 * A display-only notice from an extension - a clock chime, a pressure reminder - is not a
 * conversational move, and treating it as one destroys an authored compaction outright. But a
 * `custom_message` is also how another extension injects genuinely new context, which the handoff
 * summary does need to account for, and the fork cannot tell the two apart from the entry alone.
 * So the writer opts in: `pi.sendMessage({ ..., details: { passive: true } })`. Anything unflagged
 * still cancels, which keeps upstream's protection intact.
 */
const isPassiveNotice = (entry: SessionEntry): boolean =>
    entry.type === "custom_message" &&
    (entry.details as { passive?: unknown } | undefined)?.passive === true;

export const didConversationAdvance = (
    branch: readonly SessionEntry[],
    requestLeaf: string | null,
    compactToolCallId?: string,
    codemodeToolCallId?: string,
): boolean => {
    if (!requestLeaf) return true;

    const requestIndex = branch.findIndex((entry) => entry.id === requestLeaf);
    if (requestIndex === -1) return true;

    const tail = branch.slice(requestIndex + 1);
    const codemodeResult = codemodeToolCallId && compactToolCallId
        ? findCodemodeCompactResult(tail, compactToolCallId, codemodeToolCallId) : undefined;
    if (codemodeToolCallId) {
        const request = branch[requestIndex];
        if (!codemodeResult || request.type !== "message" || request.message.role !== "assistant") return true;
        const calls = request.message.content.filter((block) => block.type === "toolCall");
        if (calls.length !== 1 || calls[0].id !== codemodeToolCallId || calls[0].name !== "codemode") return true;
    }
    return tail.some((entry) => {
        if (PassiveCompactionEntryTypes.has(entry.type)) return false;
        if (isPassiveNotice(entry)) return false;
        if (compactToolCallId && entry.type === "message") {
            const message = entry.message;
            if (message.role === "toolResult" && !message.isError) {
                if (!codemodeToolCallId && message.toolName === "context_compact" &&
                    message.toolCallId === compactToolCallId) return false;
                if (entry === codemodeResult?.entry) return false;
            }
            if (message.role === "assistant" && message.content.length === 0 &&
                (message.stopReason === "aborted" ||
                    (message.stopReason === "error" && message.errorMessage === "This operation was aborted"))) return false;
        }
        return true;
    });
};

// Match codemode's documented <parent id>/<n> IDs against actual transcript calls;
// parent IDs are opaque and may themselves contain slashes. Never fall back to the leaf.
const findCompactRequest = (branch: readonly SessionEntry[], toolCallId: string) => {
    for (let i = branch.length - 1; i >= 0; i--) {
        const entry = branch[i];
        if (entry.type !== "message" || entry.message.role !== "assistant") continue;
        for (const block of entry.message.content) {
            if (block.type !== "toolCall") continue;
            if (block.id === toolCallId && block.name === "context_compact") {
                return { requestLeaf: entry.id, codemodeToolCallId: undefined };
            }
            if (block.name === "codemode" && toolCallId.startsWith(`${block.id}/`) &&
                /^\d+$/.test(toolCallId.slice(block.id.length + 1))) {
                return { requestLeaf: entry.id, codemodeToolCallId: block.id };
            }
        }
    }
};

const resolveTargetId = (sm: SessionManager, target: string): string => {
    if (target.toLowerCase() === "root") {
        const tree = sm.getTree();
        return tree.length > 0 ? tree[0].entry.id : target;
    }

    // If it already looks like an ID, keep it.
    if (/^[0-9a-f]{8,}$/i.test(target)) return target;

    // Iterative DFS to avoid call stack overflows on deep histories.
    const stack: SessionTreeNode[] = [...(sm.getTree() as unknown as SessionTreeNode[])];
    while (stack.length > 0) {
        const n = stack.pop()!;
        if (sm.getLabel(n.entry.id) === target) return n.entry.id;
        if (n.children?.length) stack.push(...n.children);
    }

    // Fallback: let SessionManager deal with invalid targets downstream.
    return target;
};

// ---- compaction-target guard (todos 58 + 77) ---------------------------------
// A compaction replaces only the path between its target and the tip, so a target
// near the head folds almost nothing while the summary text is pure addition.
// Measured twice on myself: +4,625 tokens (2026-10-11, 199k/262k) and a session lost
// at 252,736/262,144 (2026-10-13), where the request right after the compaction
// reported 251,358 cache-hit tokens -- proof the branch still carried everything.
// Both times the target was a checkpoint, which by construction sits at my current
// position. The guard puts the price in the tool result at the moment of the call,
// and refuses to let me buy nothing while I am actually under window pressure.

// The only honest question is net window: what the folded region costs, minus what the
// summary I am writing adds. Node count is decoration -- one enormous tool result is a
// one-node fold worth taking, and thirty small nodes can be a fold worth refusing.
// Measured on myself: a 12,484-char summary replaced 7 nodes and cost +4,625 tokens.
const GUARD_MIN_NET_TOKENS = Number(process.env.PI_CONTEXT_COMPACT_MIN_NET_TOKENS ?? 2_000);
// Below this fill a marginal compaction is wasteful but survivable; at or above it the
// wasteful target is the failure that ends the session, so it stops being a choice I
// get to make by accident.
const GUARD_PRESSURE_PERCENT = Number(process.env.PI_CONTEXT_COMPACT_PRESSURE_PERCENT ?? 50);

/** The summary lands on the new branch and stays there; price it like the region it replaces. */
const summaryTokens = (summary: string) => Math.ceil(summary.length / 4);

interface FoldSize { nodes: number; tokens: number; }

// autumn, 2026-10-13: "you can even give yourself different prompts based on how many
// nodes you're collapsing - 1 big tool output that only needed a few morsels of info
// from (or turned out to be complete junk), vs a long chain of work that needs a lossy
// but correct summary." The summary is an argument I author BEFORE the call, so the
// shape has to be visible where I choose the target -- in the timeline -- and again in
// the result, where it teaches the next one.
type FoldShape = "single-result" | "short" | "chain" | "mixed";

const GUARD_SHORT_TOKENS = Number(process.env.PI_CONTEXT_COMPACT_SHORT_TOKENS ?? 2_000);
const GUARD_CHAIN_NODES = Number(process.env.PI_CONTEXT_COMPACT_CHAIN_NODES ?? 8);

const shapeFold = (region: readonly SessionEntry[]): { shape: FoldShape; tokens: number; share: number } => {
    const sizes = region.map((entry) => estimateHistoryTokens(entry));
    const tokens = sizes.reduce((sum, n) => sum + n, 0);
    const share = tokens > 0 ? Math.max(0, ...sizes) / tokens : 0;
    if (tokens < GUARD_SHORT_TOKENS) return { shape: "short", tokens, share };
    if (region.length <= 3 && share >= 0.75) return { shape: "single-result", tokens, share };
    if (region.length >= GUARD_CHAIN_NODES) return { shape: "chain", tokens, share };
    return { shape: "mixed", tokens, share };
};

const FOLD_RECIPES: Record<FoldShape, string> = {
    "single-result": "one large result dominates this fold: keep only the morsels you actually used, cite the file or command instead of restating its content, and if it turned out to be junk, say so in one line and move on",
    short: "this fold is small: one or two lines saying what changed since the target is enough, because a longer summary costs more window than the fold frees",
    chain: "a chain of work is being folded: write a lossy-but-correct handoff -- state, decisions and constraints, external side effects, validation status, source anchors, explicit next step",
    mixed: "size the summary to the fold: cover what is being discarded, never what already sits above the target",
};

const FOLD_LABEL: Record<FoldShape, string> = {
    "single-result": "1 huge result", short: "short", chain: "chain", mixed: "mixed",
};

// The folded path is not deleted -- it stays in the session JSONL and stays searchable.
// Say so, because "the summary lost something" and "I must re-derive it" are not the only
// two options. Templated, and overridable so this stays upstreamable: bagheera's copy
// names the local index command.
const DEFAULT_POST_COMPACT_HINT =
    "The folded history is not gone: it is still in this session file and searchable. Search it before re-deriving anything.";

/**
 * Long prose does not fit the single-token KEY=value convention that bagheera's env loader
 * uses, so the hint may live in a file. Read it per compaction rather than at load: editing
 * the hint must not require a restart. A named-but-unreadable file falls back to the inline
 * or default text and says so out loud, because a hint that silently goes missing is the
 * same failure the hint exists to prevent. Placeholders {session} and {file} are substituted.
 */
export const postCompactHint = (): string => {
    const file = process.env.PI_CONTEXT_POST_COMPACT_HINT_FILE;
    if (file) {
        try {
            const text = readFileSync(file, "utf8").trim();
            if (text) return text;
            console.warn(`pi-context: ${file} is empty, using the default post-compaction hint`);
        } catch (err) {
            console.warn(`pi-context: cannot read PI_CONTEXT_POST_COMPACT_HINT_FILE ${file}: ` +
                `${err instanceof Error ? err.message : String(err)}; using the default hint`);
        }
    }
    return process.env.PI_CONTEXT_POST_COMPACT_HINT || DEFAULT_POST_COMPACT_HINT;
};

/** What a compaction at `tid` would actually replace: the active path after that node. */
const measureFold = (sm: SessionManager, tid: string): FoldSize | undefined => {
    const branch = sm.getBranch();
    const index = branch.findIndex((entry) => entry.id === tid);
    if (index < 0) return undefined;
    const region = branch.slice(index + 1);
    return { nodes: region.length, tokens: region.reduce((sum, entry) => sum + estimateHistoryTokens(entry), 0) };
};

/** The newest summary on the active path, else root: the target a shallow attempt should have used. */
const betterTarget = (sm: SessionManager): { id: string; kind: string } | undefined => {
    const branch = sm.getBranch();
    for (let i = branch.length - 2; i >= 0; i--) {
        const entry = branch[i];
        if (entry.type === "branch_summary" || entry.type === "compaction") {
            return { id: entry.id, kind: "previous summary on this path" };
        }
    }
    return branch.length ? { id: branch[0].id, kind: "root" } : undefined;
};

const PreviousTargetAliases = new Set(["previous-summary", "previous_summary", "auto"]);

/** Suffix estimates over the active path: reclaim[i] = what compacting *at* branch[i] would free. */
const reclaimByIndex = (branch: readonly SessionEntry[]): number[] => {
    const reclaim = new Array<number>(branch.length).fill(0);
    let running = 0;
    for (let i = branch.length - 1; i >= 0; i--) {
        reclaim[i] = running;
        running += estimateHistoryTokens(branch[i]);
    }
    return reclaim;
};

const ContextTimelineDescription = "Inspect the active conversation path as a structural map: checkpoints, summaries/compactions, branch points, user turns, and current position. Use when orientation or compact target selection depends on the shape of history. Folded intervals show historical token estimates; each SUMMARY and checkpoint line also carries the reclaimable size of compacting at that point, which is the number that decides target choice.";
const ContextTimelineParams = Type.Object({
    limit: Type.Optional(Type.Number({ description: "Maximum visible timeline entries (default: 50)." })),
    verbose: Type.Optional(Type.Boolean({ description: "If true, show all messages including internal context-tool traffic. If false (default), collapse to structural milestones." })),
});

const ContextCompactDescription = "Create a summarized continuation branch from an earlier checkpoint or history node. The selected target is the branch point; the summary must restore the useful state from the compacted path after that target. Only the path between target and tip is replaced, so a target near the head reclaims almost nothing: the tool reports the fold size, and refuses a shallow target while context is tight. This changes conversation history only; it does not modify or roll back disk files or external systems.";
const ContextCompactParams = Type.Object({
    target: Type.String({ description: "Checkpoint name, history node ID, or root to use as the branch point for the summarized continuation. 'previous-summary' selects the newest summary already on the active path, which is usually the target that actually frees window." }),
    summary: Type.String({ description: "Handoff summary injected into the new continuation branch. it replaces ONLY the folded region, so its size follows the fold, which context_timeline labels per candidate. A fold dominated by one huge tool result needs the morsels you actually used -- cite the file or command rather than restating content, or one line if it was junk. A short fold needs one or two lines. A long chain of work needs a lossy-but-correct handoff: task/state at the moment of the call, decisions and constraints, external side effects (changed files, processes, browser/tickets/remote state), validation status, source anchors, explicit next step. Either way, do not restate anything that already sits above the target: the retained prefix is still in context, and duplicated text is pure cost. Do not rely on backupCheckpoint for details needed in the next phase." }),
    backupCheckpoint: Type.Optional(Type.String({ description: "Optional checkpoint name to label the current conversation state before branching. This is only a recovery pointer; the summary must still contain the state needed to continue." })),
    force: Type.Optional(Type.Boolean({ description: "Proceed even when the target folds too little to help. Use only for a compaction whose purpose is a fresh summary rather than reclaimed window." })),
});

const ContextCheckpointDescription = "Create a named anchor by labeling a conversation history node. This does not branch, summarize, or affect external state; it only makes the point easy to find later in timeline or compact target selection.";
const ContextCheckpointParams = Type.Object({
    name: Type.String({ description: "Unique <scope>-<phase> name; phase suffixes: start, done (stable result), pivot (change approach), pause, resume. Keep scope consistent within a phase's lifecycle; other names remain ordinary anchors." }),
    target: Type.Optional(Type.String({ description: "Optional history node ID or checkpoint name to label. Defaults to the current meaningful position near the conversation head." })),
});

interface PendingCompactRequest extends Static<typeof ContextCompactParams> {
    tid: string;
    enrichedMessage: string;
    usageBeforeText: string;
    usageBefore?: ContextUsage;
    fold?: FoldSize;
    net?: number;
    requestLeaf: string;
    toolCallId: string;
    codemodeToolCallId?: string;
    invalidated?: boolean;
}

export default function (pi: ExtensionAPI) {
    let CommandCtx: ExtensionCommandContext | null = null;
    let CompactParams: PendingCompactRequest | null = null;
    let runtimeActive = true;
    let pendingCommandContext: Promise<ExtensionCommandContext> | null = null;
    let resolveCommandContext: ((ctx: ExtensionCommandContext) => void) | undefined;
    let rejectCommandContext: ((error: Error) => void) | undefined;

    // sendUserMessage is fire-and-forget. Resolve from the command handler,
    // rather than assuming command dispatch has completed when it returns.
    async function ensureCommandContext(): Promise<ExtensionCommandContext> {
        if (CommandCtx) return CommandCtx;
        if (pendingCommandContext) return pendingCommandContext;

        let timeout: ReturnType<typeof setTimeout> | undefined;
        pendingCommandContext = new Promise<ExtensionCommandContext>((resolve, reject) => {
            resolveCommandContext = resolve;
            rejectCommandContext = reject;
            timeout = setTimeout(() => reject(new Error(
                "context_compact: automatic command context acquisition timed out. Requires Pi >= 0.84.2 with extension command dispatch support.",
            )), 5000);
            pi.sendUserMessage("/acm", {
                deliverAs: "followUp",
                expandPromptTemplates: true,
            });
        });
        try {
            return await pendingCommandContext;
        } finally {
            clearTimeout(timeout);
            pendingCommandContext = null;
            resolveCommandContext = undefined;
            rejectCommandContext = undefined;
        }
    }

    pi.on("session_shutdown", () => {
        runtimeActive = false;
        CommandCtx = null;
        CompactParams = null;
        rejectCommandContext?.(new Error("context_compact: session closed while acquiring command context."));
    });

    pi.registerCommand("acm", {
        description: "Enable agentic context management for the current session",
        handler: async (args, ctx) => {
            CommandCtx = ctx;
            if (resolveCommandContext) {
                resolveCommandContext(ctx);
            } else {
                ctx.ui.notify("Agentic Context Management enabled.", "info");
            }
            if (args) {
                pi.sendUserMessage(args, { deliverAs: "followUp" });
            }
        }
    });

    // Helper: Check if a checkpoint name already exists in the tree
    // Iterative DFS to avoid call stack overflows on deep histories.
    // Push children in reverse order to preserve left-to-right pre-order semantics.
    const findCheckpointInTree = (sm: SessionManager, nodes: SessionTreeNode[], checkpointName: string): string | null => {
        const stack: SessionTreeNode[] = [...nodes].reverse();
        while (stack.length > 0) {
            const n = stack.pop()!;
            if (sm.getLabel(n.entry.id) === checkpointName) return n.entry.id;
            if (n.children?.length) {
                for (let i = n.children.length - 1; i >= 0; i--) {
                    stack.push(n.children[i]);
                }
            }
        }
        return null;
    };

    pi.registerTool({
        name: "context_checkpoint",
        label: "Context Checkpoint",
        description: ContextCheckpointDescription,
        parameters: ContextCheckpointParams,
        async execute(_id, params: Static<typeof ContextCheckpointParams>, _signal, _onUpdate, ctx) {
            const sm = ctx.sessionManager as SessionManager;

            // Deduplication check: ensure checkpoint name is unique
            const existingCheckpointId = findCheckpointInTree(sm, sm.getTree(), params.name);
            if (existingCheckpointId) {
                return {
                    content: [{
                        type: "text",
                        text: `Error: Checkpoint '${params.name}' already exists at ${existingCheckpointId}. Checkpoint names must be unique. Use a different name or remove the existing one first.`
                    }],
                    details: {}
                };
            }

            let id = params.target ? resolveTargetId(sm, params.target) : undefined;

            if (!id) {
                // Auto-resolve: Find the last interesting node to checkpoint.
                // We skip ToolResults that look awkward when checkpointed and internal-only assistant messages that look empty.
                const branch = sm.getBranch();
                for (let i = branch.length - 1; i >= 0; i--) {
                    const entry = branch[i];

                    // 1. Check ToolResults
                    if (entry.type === 'message' && entry.message.role === 'toolResult') {
                        const tr = entry.message as any;
                        if (isInternal(tr.toolName)) continue;

                        // Public tool result is a valid target
                        id = entry.id;
                        break;
                    }

                    // 2. Check Assistant messages for visibility
                    if (entry.type === 'message' && entry.message.role === 'assistant') {
                        const m = entry.message;
                        const hasInternalTool = m.content.some(c => c.type === 'toolCall' && isInternal(c.name));

                        if (!hasInternalTool) {
                            id = entry.id;
                            break;
                        }
                    }

                    id = entry.id;
                    break;
                }
                // Fallback to leaf if search failed
                if (!id) id = sm.getLeafId() ?? "";
            }

            pi.setLabel(id, params.name);
            // A checkpoint is an anchor for finding a place later, never a compaction
            // target: it sits at the current position by construction, and compacting
            // from the current position folds nothing. Say the cost while it is cheap.
            const usage = ctx.getContextUsage();
            const fill = usage?.percent;
            const report = [`Created checkpoint '${params.name}' at ${id}.`,
                `Context fill now: ${formatContextUsage(usage, true)}.`];
            if (fill != null && fill >= GUARD_PRESSURE_PERCENT) {
                const alt = betterTarget(sm);
                report.push(`Warning: this anchor sits at ${fill.toFixed(1)}% fill. A checkpoint is not a compaction target -- compacting from here would fold almost nothing.` +
                    (alt ? ` To reclaim window, target ${alt.id} (${alt.kind}) instead.` : ""));
            }
            return {
                content: [{ type: "text", text: report.join(" ") }],
                details: {}
            };
        },
    });

    pi.registerTool({
        name: "context_timeline",
        label: "Context Timeline",
        description: ContextTimelineDescription,
        parameters: ContextTimelineParams,
        async execute(_id, params: Static<typeof ContextTimelineParams>, _signal, _onUpdate, ctx) {
            const sm = ctx.sessionManager as SessionManager;
            const branch = sm.getBranch();
            const currentLeafId = sm.getLeafId();
            const verbose = params.verbose ?? false;
            const limit = params.limit ?? 50;

            const backboneIds = new Set(branch.map((e) => e.id));
            const sequence: SessionEntry[] = [];

            branch.forEach((entry) => {
                sequence.push(entry);

                // Preserve side-summary logic: Show branch summaries/compactions that are off-path
                const children = sm.getChildren(entry.id);
                children.forEach((child) => {
                    if ((child.type === "branch_summary" || child.type === "compaction") && !backboneIds.has(child.id)) {
                        sequence.push(child);
                    }
                });
            });

            const getMsgContent = (entry: SessionEntry): string => {
                if (entry.type === "branch_summary" || entry.type === "compaction") {
                    const e = entry;
                    return e.summary || "[No summary provided]";
                }
                if (entry.type === "label") {
                    return `checkpoint: ${entry.label}`;
                }
                if (entry.type === "custom_message" && entry.customType === RetainedCodemodeOutputType) {
                    const text = typeof entry.content === "string" ? entry.content
                        : entry.content.map((block) => block.type === "text" ? block.text : "[image]").join(" ");
                    return `(codemode output · ~${formatTokens(estimateHistoryTokens(entry))} tokens) ${text}`;
                }

                if (entry.type === "message") {
                    const msg = entry.message;

                    if (msg.role === "toolResult") {
                        const tr = msg;
                        if (!verbose && isInternal(tr.toolName)) return "";

                        const extractText = (content: (TextContent | ImageContent)[]): string => {
                            return content
                                .map((p) => (p.type === "text" ? p.text : ""))
                                .join(" ")
                                .trim();
                        };

                        let resText = extractText(tr.content);
                        const details = tr.details as Record<string, unknown> | undefined;
                        if ((tr.toolName === "read" || tr.toolName === "edit") && details && "path" in details && typeof details.path === "string") {
                            resText = `${details.path}: ${resText}`;
                        }
                        return `(${tr.toolName}) ${resText}`;
                    }

                    if (msg.role === "bashExecution") {
                        return `[Bash] ${msg.command}`;
                    }

                    if (msg.role === "user" || msg.role === "assistant") {
                        let text = "";
                        if (typeof msg.content === "string") {
                            text = msg.content;
                        } else if (Array.isArray(msg.content)) {
                            text = msg.content
                                .map((p: any) => {
                                    if (typeof p === "object" && p !== null && "text" in p) return (p as TextContent).text;
                                    return "";
                                })
                                .join(" ")
                                .trim();
                        }

                        let toolCallsText = "";
                        if (msg.role === "assistant") {
                            const toolCalls = msg.content.filter((c): c is ToolCall => c.type === "toolCall");

                            toolCallsText = toolCalls
                                .filter((tc) => verbose || !isInternal(tc.name))
                                .map((tc) => `call: ${tc.name}(${JSON.stringify(tc.arguments)})`)
                                .join("; ");
                        }

                        return [text, toolCallsText].filter(Boolean).join(" ");
                    }
                }
                return "";
            };

            const isInteresting = (entry: SessionEntry): boolean => {
                // 1. HEAD and Root
                if (entry.id === currentLeafId) return true;
                if (branch.length > 0 && entry.id === branch[0].id) return true;

                // 2. Explicit checkpoints (labels) - only show the checkpointed node, not the label node itself
                if (sm.getLabel(entry.id)) return true;
                if (entry.type === 'label') return false; // Hide label nodes, they are redundant

                // 3. Structural Milestones (Summaries)
                if (entry.type === 'branch_summary' || entry.type === 'compaction') return true;
                if (entry.type === 'custom_message' && entry.customType === RetainedCodemodeOutputType) return true;

                // 4. Branch Points (Forks)
                if (sm.getChildren(entry.id).length > 1) return true;

                // 5. Natural Milestones (User Messages) - This is the key auto-tagging mechanism
                if (entry.type === 'message' && entry.message.role === 'user') return true;

                return false;
            };

            const visibleSequenceIds = new Set<string>();
            sequence.forEach(e => {
                if (verbose || isInteresting(e)) {
                    visibleSequenceIds.add(e.id);
                }
            });

            let visibleEntries = sequence.filter(e => visibleSequenceIds.has(e.id));
            if (visibleEntries.length > limit) {
                const allowedIds = new Set(visibleEntries.slice(-limit).map(e => e.id));
                visibleSequenceIds.clear();
                allowedIds.forEach(id => visibleSequenceIds.add(id));
            }

            const branchIndexOf = new Map(branch.map((entry, index) => [entry.id, index]));
            const reclaim = reclaimByIndex(branch);
            const lines: string[] = [];
            let hiddenEntries: SessionEntry[] = [];
            const flushHidden = () => {
                if (hiddenEntries.length) lines.push(`  :  ... (${describeHistoryInterval(hiddenEntries)}) ...`);
                hiddenEntries = [];
            };

            sequence.forEach((entry) => {
                if (!visibleSequenceIds.has(entry.id)) {
                    // Off-path summaries never contribute to active-path intervals.
                    if (backboneIds.has(entry.id) && entry.type !== "custom" && entry.type !== "label" &&
                        (entry.type !== "custom_message" || entry.customType === RetainedCodemodeOutputType)) hiddenEntries.push(entry);
                    return;
                }

                flushHidden();

                const isHead = entry.id === currentLeafId;
                const label = sm.getLabel(entry.id);
                const content = getMsgContent(entry).replace(/\s+/g, " ");

                let role = entry.type.toUpperCase();
                if (entry.type === "message") {
                    const m = entry.message;
                    role =
                        m.role === "assistant"
                            ? "AI"
                            : m.role === "user"
                                ? "USER"
                                : m.role === "bashExecution"
                                    ? "BASH"
                                    : "TOOL";
                } else if (entry.type === "branch_summary" || entry.type === "compaction") {
                    role = "SUMMARY";
                } else if (entry.type === "custom_message" && entry.customType === RetainedCodemodeOutputType) {
                    role = "TOOL";
                }

                // hide custom messages
                if (role === "CUSTOM_MESSAGE") {
                    return
                }

                const id = entry.id;
                const isRoot = branch.length > 0 && entry.id === branch[0].id;
                const stage = label ? parseCheckpointPhase(label)?.phase : undefined;
                const meta = [isRoot ? "ROOT" : null, isHead ? "HEAD" : null,
                    !backboneIds.has(id) ? "off-path" : null,
                    label ? `checkpoint: ${label}` : null,
                    stage ? `phase: ${stage}` : null].filter(Boolean).join(", ");

                const body = content.length > 100 ? content.slice(0, 100) + "..." : content;

                const marker = isHead ? "*" : (role === "USER" ? "•" : "|");

                // Target candidates carry their price: compacting here frees this much.
                const isCandidate = entry.type === "branch_summary" || entry.type === "compaction" || !!label;
                const candidateIndex = branchIndexOf.get(id);
                const price = isCandidate && candidateIndex != null
                    ? ` · reclaim ~${formatTokens(reclaim[candidateIndex] ?? 0)} (${FOLD_LABEL[shapeFold(branch.slice(candidateIndex + 1)).shape]})`
                    : "";

                lines.push(`${marker} ${id}${meta ? ` (${meta})` : ""} [${role}]${price} ${body}`);
            });

            flushHidden();
            const hud = `Context: ${formatContextUsage(ctx.getContextUsage(), true)}`;

            return { content: [{ type: "text", text: hud + "\n" + (lines.join("\n") || "(Root Path Only)") }], details: {} };
        },
    });

    pi.registerTool({
        name: "context_compact",
        label: "Context Compact",
        description: ContextCompactDescription,
        parameters: ContextCompactParams,
        async execute(_id, params: Static<typeof ContextCompactParams>, _signal, _onUpdate, ctx) {
            // Anchor at the assistant request (or its codemode script), not the
            // current leaf: sibling tools/hooks may already have appended entries.
            const request = findCompactRequest(ctx.sessionManager.getBranch(), _id);
            if (!request) {
                throw new Error("context_compact: cannot locate the requesting tool call in session history.");
            }
            await ensureCommandContext();
            if (!runtimeActive || _signal?.aborted) {
                throw new Error("context_compact: cancelled before compaction started.");
            }
            const sm = ctx.sessionManager as SessionManager;
            const usageBeforeText = formatContextUsage(ctx.getContextUsage());

            if (CompactParams && (request.codemodeToolCallId || CompactParams.codemodeToolCallId)) {
                CompactParams.invalidated = true;
                throw new Error("context_compact: only one compact request is allowed per codemode script.");
            }
            const requested = PreviousTargetAliases.has(params.target.trim().toLowerCase())
                ? (betterTarget(sm)?.id ?? "root")
                : params.target;
            const tid = resolveTargetId(sm, requested);
            if (request.codemodeToolCallId) {
                const branch = sm.getBranch();
                const targetIndex = branch.findIndex((entry) => entry.id === tid);
                const requestIndex = branch.findIndex((entry) => entry.id === request.requestLeaf);
                if (targetIndex < 0 || targetIndex >= requestIndex) {
                    throw new Error("context_compact: codemode target must precede the script on the active path.");
                }
            }

            const currentLeaf = sm.getLeafId();
            if (currentLeaf === tid) {
                return { content: [{ type: "text", text: `Already at target ${tid}` }], details: {} };
            }

            const fold = measureFold(sm, tid);
            const shape = fold ? shapeFold(sm.getBranch().slice(
                sm.getBranch().findIndex((entry) => entry.id === tid) + 1)) : undefined;
            const usage = ctx.getContextUsage();
            const fill = usage?.percent;
            const added = summaryTokens(params.summary);
            const net = fold ? fold.tokens - added : undefined;
            const marginal = net != null && net < GUARD_MIN_NET_TOKENS;
            if (marginal && !params.force) {
                const detail = `the target folds ${fold!.nodes} node(s) / ~${formatTokens(fold!.tokens)} tokens while the summary adds ~${formatTokens(added)}, so it nets ${net >= 0 ? "+" : ""}~${formatTokens(net)} of window`;
                if (fill != null && fill >= GUARD_PRESSURE_PERCENT) {
                    const alt = betterTarget(sm);
                    const altFold = alt ? measureFold(sm, alt.id) : undefined;
                    throw new Error(
                        `context_compact: refusing this target at ${formatContextUsage(usage, true)} fill -- ${detail}. ` +
                        `A compaction replaces only the path between target and tip, so a target near the head costs more than it frees. ` +
                        (alt && altFold ? `Better target: ${alt.id} (${alt.kind}), which folds ${altFold.nodes} node(s) / ~${formatTokens(altFold.tokens)} tokens. ` : "") +
                        `Recipe for this fold (${shape ? FOLD_RECIPES[shape.shape] : "unknown shape"}). ` +
                        `Pass force: true only if you want a fresh summary rather than reclaimed window.`,
                    );
                }
            }
            if (params.backupCheckpoint && currentLeaf) {
                pi.setLabel(currentLeaf, params.backupCheckpoint);
            }
            const currentLabel = currentLeaf ? sm.getLabel(currentLeaf) : undefined;
            const origin = currentLabel ? `checkpoint: ${currentLabel}` : (currentLeaf || "unknown");

            const enrichedMessage = `(handoff summary from ${origin})\n${params.summary}`;

            // Awaiting this tool only schedules compaction. In codemode, all calls
            // before and after it finish before history switches; its position
            // within the script does not split the output or rerun operations.
            CompactParams = {
                ...params, tid, enrichedMessage, usageBeforeText,
                usageBefore: usage, fold, net,
                requestLeaf: request.requestLeaf,
                toolCallId: _id,
                codemodeToolCallId: request.codemodeToolCallId,
            };

            const notes = ["compact start"];
            if (fold) {
                notes.push(`folds ${fold.nodes} node(s) / ~${formatTokens(fold.tokens)} tokens`);
                notes.push(`summary ~${formatTokens(added)} tokens`);
                notes.push(`net ~${net! >= 0 ? "+" : "-"}${formatTokens(Math.abs(net!))} of window`);
            }
            if (shape) notes.push(`fold shape: ${FOLD_LABEL[shape.shape]} -- ${FOLD_RECIPES[shape.shape]}`);
            if (marginal) {
                const alt = betterTarget(sm);
                const altFold = alt ? measureFold(sm, alt.id) : undefined;
                notes.push(`WARNING: this target barely helps` +
                    (alt && altFold ? `; ${alt.id} (${alt.kind}) folds ~${formatTokens(altFold.tokens)}` : "") +
                    `; and the summary only needs to cover the folded region, not restate the retained prefix`);
            }
            return { content: [{ type: "text", text: notes.join(" · ") }], details: {} };
        },
    });

    pi.on("turn_end", async (_event, ctx) => {
        if (!CompactParams) {
            return
        }
        ctx.abort()
    });

    pi.on("agent_end", async (_event, ctx) => {
        if (!CompactParams) {
            return
        }
        if (!CommandCtx) {
            return
        }

        const sm = ctx.sessionManager as SessionManager;
        const compactParams = CompactParams;
        const commandCtx = CommandCtx;
        CompactParams = null;
        const requestLeaf = compactParams.requestLeaf;

        // `agent_end` is emitted before the core Agent is actually idle. If we
        // call pi.sendMessage({ triggerTurn: true }) inside this handler, pi still
        // sees an active stream and queues the message as steering; after
        // `agent_end` the loop has already stopped, so that queued message is not
        // drained. Defer navigation + continuation until the current run settles.
        setTimeout(async () => {
            try {
                if (!runtimeActive) return;
                await commandCtx.waitForIdle();
                if (!runtimeActive) return;

                // Check unaccounted context after idle, before creating the branch.
                // This is not atomic across Pi's awaited session_before_tree hooks.
                const branch = sm.getBranch();
                if (compactParams.invalidated || ctx.hasPendingMessages() ||
                    didConversationAdvance(branch, requestLeaf, compactParams.toolCallId, compactParams.codemodeToolCallId)) {
                    const reason = compactParams.invalidated
                        ? "more than one compact request was made."
                        : compactParams.codemodeToolCallId
                            ? "conversation advanced or codemode did not finish with a verifiable compact result."
                            : "conversation advanced before compaction completed.";
                    commandCtx.ui.notify(`context_compact cancelled: ${reason}`, "warning");
                    pi.sendMessage({
                        customType: PiContextCustomMessageType,
                        content: [
                            `context_compact cancelled: ${reason}`,
                            "No compaction was applied; continue from the current path. If still useful, inspect timeline and retry with an updated summary.",
                        ].join("\n"),
                        display: false,
                    }, {
                        triggerTurn: true,
                        deliverAs: "followUp",
                    });
                    return;
                }

                const originalLeaf = sm.getLeafId()!;
                let nid: string;
                try {
                    let summaryTarget = compactParams.tid;
                    if (compactParams.codemodeToolCallId) {
                        const result = findCodemodeCompactResult(
                            branch.slice(branch.findIndex((entry) => entry.id === requestLeaf) + 1),
                            compactParams.toolCallId, compactParams.codemodeToolCallId,
                        )!;
                        // Model context: prefix through target -> full output -> summary.
                        // Copy returned text/images as-is, including truncation notices
                        // and paths, not script source or unprinted intermediate values.
                        // A provenance-linked custom message leaves the original calls
                        // recoverable without duplicating billable assistant/tool results.
                        sm.branch(compactParams.tid);
                        sm.appendCustomMessageEntry(RetainedCodemodeOutputType, [
                            { type: "text", text: "Output from the codemode script that already completed before compaction. Do not rerun its completed operations. The following handoff summary covers earlier history, not this output." },
                            ...result.message.content,
                        ], false, { requestEntryId: requestLeaf, resultEntryId: result.entry.id });
                        // Preserve store state with one net delta, not copies of its
                        // write log: repeated old-target compacts must not amplify it.
                        // Delete-then-set matches Pi's replay; Map keeps special keys safe.
                        const targetIndex = branch.findIndex((entry) => entry.id === compactParams.tid);
                        const prefixKeys = new Set<string>();
                        const writes = new Map<string, unknown>(), deletes = new Set<string>();
                        for (let index = 0; index < branch.length; index++) {
                            const entry = branch[index];
                            if (entry.type !== "custom" || entry.customType !== "codemode-store") continue;
                            const data = entry.data;
                            // Ignore malformed entries just as Pi's readCodemodeStore does.
                            if (typeof data !== "object" || data === null ||
                                !("set" in data) || !("delete" in data) ||
                                typeof data.set !== "object" || data.set === null ||
                                !Array.isArray(data.delete) || !data.delete.every((key: unknown) => typeof key === "string")) continue;
                            if (index <= targetIndex) {
                                for (const key of data.delete) prefixKeys.delete(key);
                                for (const key of Object.keys(data.set)) prefixKeys.add(key);
                                continue;
                            }
                            for (const key of data.delete) {
                                writes.delete(key);
                                deletes.add(key);
                            }
                            for (const [key, value] of Object.entries(data.set)) {
                                writes.set(key, value);
                                deletes.delete(key);
                            }
                        }
                        // Keys created and removed after target need no tombstone.
                        const deleted = [...deletes].filter(key => prefixKeys.has(key));
                        if (writes.size || deleted.length) {
                            sm.appendCustomEntry("codemode-store", { set: Object.fromEntries(writes), delete: deleted });
                        }
                        summaryTarget = sm.getLeafId()!;
                        sm.branch(originalLeaf);
                    }
                    nid = sm.branchWithSummary(summaryTarget, compactParams.enrichedMessage);
                } finally {
                    // Rebuild from the original live path. Navigating a custom
                    // message would move it to the editor; the summary is the leaf.
                    sm.branch(originalLeaf);
                }
                const navigation = await commandCtx.navigateTree(nid, { summarize: false });
                if (!runtimeActive) return;
                if (navigation.cancelled) {
                    throw new Error("tree navigation was cancelled; the original path was retained.");
                }

                const usageAfter = commandCtx.getContextUsage();
                const beforeTokens = compactParams.usageBefore?.tokens;
                const afterTokens = usageAfter?.tokens;
                const reclaimed = beforeTokens != null && afterTokens != null ? beforeTokens - afterTokens : undefined;
                const reclaimNote = reclaimed == null
                    ? "context_compact complete; reclamation unmeasured (usage unavailable)."
                    : reclaimed > 0
                        ? `context_compact complete: reclaimed ~${formatTokens(reclaimed)} tokens (${compactParams.usageBeforeText} -> ${formatContextUsage(usageAfter, true)}).`
                        : `context_compact complete but RECLAIMED NOTHING: fill went ${reclaimed === 0 ? "unchanged" : "UP"} by ~${formatTokens(Math.abs(reclaimed))} tokens (${compactParams.usageBeforeText} -> ${formatContextUsage(usageAfter, true)}). The target sat too close to the head; compact from the previous summary on the path to actually free window.`;
                commandCtx.ui.notify([
                    `Compacted to ${compactParams.target}${compactParams.target === compactParams.tid ? "" : `(${compactParams.tid})`}`,
                    `Context Usage: ${compactParams.usageBeforeText} -> ${formatContextUsage(usageAfter)}`,
                    `Backup checkpoint created: ${compactParams.backupCheckpoint || "none"}`,
                    `Summary: ${compactParams.enrichedMessage}`,
                ].join("\n"), "info");

                const hint = postCompactHint()
                    .replaceAll("{session}", sm.getSessionId?.() ?? "")
                    .replaceAll("{file}", sm.getSessionFile?.() ?? "");
                pi.sendMessage({
                    customType: PiContextCustomMessageType,
                    content: compactParams.codemodeToolCallId
                        ? `${reclaimNote} Your previous history was summarized. Read the retained codemode output and the handoff summary. The script already completed; do not repeat its operations or a Next Step it has already satisfied. Continue with the remaining work. ${hint}`
                        : `${reclaimNote} Above is the handoff summary of your previous path. Read it, take it as your current state, and continue with the next step it names. Files, processes, and external services are unchanged. ${hint}`,
                    display: false,
                }, {
                    triggerTurn: true,
                    deliverAs: "followUp",
                });
            } catch (err) {
                if (!runtimeActive) return;
                const message = err instanceof Error ? err.message : String(err);
                commandCtx.ui.notify(`context_compact failed: ${message}`, "error");
                pi.sendMessage({
                    customType: PiContextCustomMessageType,
                    content: [
                        `context_compact failed: ${message}`,
                        "Compaction did not finish. Inspect the current path before retrying; a recoverable summary branch may have been created.",
                    ].join("\n"),
                    display: false,
                }, {
                    triggerTurn: true,
                    deliverAs: "followUp",
                });
            }
        }, 0);
    });
}
