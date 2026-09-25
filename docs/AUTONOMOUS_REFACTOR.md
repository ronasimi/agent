# Autonomous single-model refactor

## Supplied baseline

The source archive was al-agent-coder-role-and-prompt-boundary-fixed(2).zip. Its SHA-256 is:

```text
87485656a71544511675c75780b14f3c1d48fc225e85b545895a1aa6715ec2b7
```

The available copy ending in (1).zip had the same hash. The complete refactored repository is supplied, including tools, Web UI, configuration, scripts, documentation, and tests.

## Concrete defects and repairs

| Finding | Repair |
|---|---|
| The runtime facade hid refresh_history inside keyword arguments, while WebSocket dispatch checked the explicit signature. Fresh turns could omit saved context. | Made the parameter explicit and reload history after acquiring the conversation lock. Recent raw history is independent of legacy compaction watermarks. |
| Isolated tool processes lost the active conversation context. Conversation-scoped writes could land in the default conversation. | Serialize the conversation ID in each worker request and restore its ContextVar before invocation. |
| Global foreground flags allowed one finishing chat to clear another chat's active status. | Track active and waiting turns by unique token in an atomic SQLite update. Remove only the finishing token. |
| Streaming tool-call merging could collapse distinct calls with identical names and arguments. | Use explicit stream call identity/index; preserve unidentified calls separately. |
| Re-registering a tool silently replaced the previous callable/schema. | Reject duplicate registration and snapshot the catalog under a lock for each turn. |
| Per-conversation lock waits had no finite queue bound and could leak a handle on failure. | Add bounded acquisition, cancellation checks, and exception-safe handle cleanup. |
| WebSocket run registration occurred before cleanup covered acknowledgment and workspace preflight. | Put both inside the cleanup scope and reject duplicate active turn IDs. |
| Job-list rendering used a backslash inside an f-string expression, invalid under the declared Python 3.11 minimum. | Escape the title before interpolating it. |

The replacement loop also addresses execution hazards in the former routing/recovery design: partial streams cannot trigger actions; invalid native batches cannot partially execute before argument validation; malformed calls return feedback to the same model; tool results are durably paired with calls; and transport failure never causes an automatic replay.

History normalization matches results by call ID, handles a missing first result correctly, and preserves an unknown-outcome placeholder for interrupted calls. Context fitting preserves the current user request even after synthetic protocol feedback and results have been appended. Schema activation either succeeds within the allowance or leaves the previous active set intact.

## Removed orchestration

Removed the large intent-driven foreground engine and its helper policies, specialist model-role selection, fast-model decision helpers, and separate vision path. The catalog no longer maps user words to tool bundles. The foreground loop no longer forces weather/time/profile tools, recipe execution, structured-plan scheduling, validator calls, escalation, or domain-specific fallback paths.

Ordinary tool implementations, explicit slash commands, durable job state machines, lifecycle checks, and argument validation remain. Their conditionals implement requested operations and execution constraints.

## Autonomous mechanism

The model starts with the complete tool-name inventory and discovery descriptions. It selects tool_search or load_tools to obtain schemas, selects a loaded tool and supplies arguments, then receives an observation. This repeats until it returns a final answer or the runtime reaches a configured bound.

JSON mode uses an action schema derived from active tools and a final-answer variant. It is the default because it does not require a native function-calling chat template. Native mode uses the same model, tools, validation, and storage with provider-native tool messages.

The harness never reads prose as an instruction to invoke a tool. Responses must be complete and structurally valid. Recovery decisions remain model-selected. Successful tool execution is reported separately from the model's final answer; the runtime does not manufacture success after an error.

## One model

The original distilled 4B GGUF backs agent-main:4b. All consumers resolve configuration through tools.config, which normalizes legacy role names onto this model and its main options. Research, compaction, self-optimization proposals, and custom-tool generation use it too. Background report residency swapping was removed, and generation no longer unloads the shared model with keep_alive=0.

Semantic-memory compatibility tools use the existing lexical store, eliminating the embedding-model dependency. Text-only operation is explicit; image display remains supported but there is no visual inference sidecar. Model capability probes no longer choose runtime routes.

## Migration

1. Back up existing workspace and memory data and retain deployment-specific integration settings.
2. Extract this complete source tree into the deployment source location.
3. Merge the agent block from config/config.yaml. Keep tool_protocol=json for the supplied distilled model unless native support is confirmed.
4. Run scripts/create_ollama_aliases.sh to create the default alias, or configure AGENT_MODEL for an already installed equivalent model.
5. Apply the host Ollama settings in ollama.env.example as needed.
6. Rebuild with docker compose up -d --build.
7. Run docker compose exec webui python diagnostics/validate_ollama.py.

No chat-data deletion or database reset is required. New source archives intentionally exclude runtime files and credentials. Old specialized model tags can remain installed; the harness will not select them. Previously configured job types and recipes retain their own behavior.

## Verification

The active offline suite passed: **587 passed, 4 skipped**. Three skips are opt-in live Ollama tests. One sandbox test requires UID remapping unavailable in this environment. There were two dependency deprecation warnings.

Checks performed:

- Actual Ollama SDK request serialization and streamed-response parsing through HTTP mocks in both JSON and native modes.
- Conversation refresh, legacy compaction compatibility, isolated subprocess scope, concurrent foreground tracking, and WebSocket preflight cleanup.
- Model-selected tool sequencing, strict arguments, schema allowance rollback, cancellation, budgets, no automatic partial-stream replay, and uncertain-mutation replay blocking.
- Native call/result pairing and stream identity.
- Existing independent tool, persistence, browser, research, integration, lifecycle, and UI regressions.
- Python compilation, correctness lint across production modules, full default lint on the new core, JavaScript syntax, and shell-script syntax.
- Offline scripted simulator using real calculation dispatch and error feedback.

Tests requiring the removed architecture were replaced or retired. Their original source is preserved as non-collected .py.txt files in docs/legacy-tests. Examples include keyword-selected tool bundles, deterministic no-model execution, forced scheduler recovery, specialist model escalation, and sidecar vision. The reported passing count is for the refactored suite, not a claim that the old architecture's expectations still hold.

## Deployment limits

Live Ollama, the exact GGUF chat template, answer quality, host browser/system integrations, and Docker build/run were not exercised here. The live smoke test checks actual tool discovery and calculation; the opt-in live test module also checks direct answering and current-time retrieval. These tests deliberately fail when the model does not perform the requested behavior.

The model owns semantic choices and can make mistakes. The removed domain-specific fact gates are not replaced by a claim of guaranteed factuality. Prompt policy asks for tool evidence, while the runtime enforces structure, state, and execution bounds.

Context size is estimated from serialized input; it is not exact tokenizer accounting. The turn deadline is cooperative, and long tools use their existing backend timeouts or isolated execution. Unknown-outcome protection is scoped to one turn, not durable exactly-once execution across crashes or user resubmissions. Tool-side idempotency remains necessary for operations requiring that guarantee.
